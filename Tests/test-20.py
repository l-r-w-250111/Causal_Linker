import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. データ生成 (変更なし) ---
def generate_5d_mechanical_data(batch_size=32, seq_len=100):
    n_vars = 5
    data = torch.zeros(batch_size, seq_len, n_vars).to(device)
    for b in range(batch_size):
        for t in range(1, seq_len):
            p = data[b, t-1]
            c = torch.randn(n_vars).to(device) * 0.03
            c[1] += 1.4 * p[0]; c[2] += 1.4 * p[1] 
            c[3] += 1.4 * p[2]; c[4] += 1.4 * p[3] 
            c[2] -= 0.8 * p[4]; c[1] -= 0.6 * p[4]
            c += 0.2 * p 
            data[b, t] = torch.clamp(c, -5, 5)
    return (data - data.mean()) / (data.std() + 1e-6)

# --- 2. ハイブリッド・プロパゲーター・モデル ---
class HybridSharpModel(nn.Module):
    def __init__(self, n_vars=5, d_model=128):
        super().__init__()
        self.n_vars = n_vars
        self.K = 2
        self.raw_S = nn.Parameter(torch.randn(n_vars, n_vars) * 0.1)
        self.raw_phase = nn.Parameter(torch.randn(n_vars, n_vars) * 0.05)
        
        self.mode_gen = nn.Sequential(
            nn.Linear(n_vars * 5, d_model), nn.LayerNorm(d_model),
            nn.ReLU(), nn.Linear(d_model, self.K * 2) 
        )
        self.register_buffer("u", F.normalize(torch.randn(1, n_vars), dim=1))

    @torch.no_grad()
    def update_u(self):
        S = self._get_S_core(0)
        v = F.normalize(torch.mm(self.u, S), dim=1)
        u_new = F.normalize(torch.mm(v, S.t()), dim=1)
        self.u.copy_(u_new)

    def _get_S_core(self, epoch):
        S = torch.tanh(self.raw_S * 3.0)
        tau = min(0.15, epoch / 15000.0)
        S_sparse = torch.sign(S) * torch.relu(torch.abs(S) - tau)
        
        # 自己ループのベース強度
        diag_mask = torch.eye(self.n_vars).to(S.device)
        diag_val = 0.95 + 0.04 * torch.sigmoid(torch.diag(self.raw_S))
        return S_sparse * (1 - diag_mask) + diag_mask * diag_val

    def forward(self, x_2ch, history, phi_global, epoch=2000):
        B = x_2ch.shape[0]
        params = self.mode_gen(history).view(B, self.K, 2)
        
        # [アイデア1] 位相を半固定
        phi_base = torch.tensor([0.0, np.pi/2], device=device)
        phi_curr = phi_base.unsqueeze(0) + torch.tanh(params[:, :, 0]) * (np.pi / 36)
        
        # [アイデア2] モード別ゲイン制約
        r_mode = torch.stack([
            1.2 * torch.sigmoid(params[:, 0, 1]),  # k=0: 伝搬モード
            0.6 * torch.sigmoid(params[:, 1, 1])   # k=1: 保存（遅延）モード
        ], dim=1)
        
        # [アイデア3] 自己ループの役割分担
        S_core = self._get_S_core(epoch)
        # スペクトル正規化
        with torch.no_grad():
            v = F.normalize(torch.mm(self.u, S_core), dim=1)
            sigma = torch.mm(torch.mm(self.u, S_core), v.t())
            factor = 1.05 / torch.max(sigma, torch.tensor(1.05).to(device))
        S = S_core * factor

        self_loop_mask = torch.eye(self.n_vars).to(device)
        
        # 伝搬モード(k=0)は非対角のみ、保存モード(k=1)は対角のみを強調
        # (厳密に0にすると学習が止まるため、ソフトに重み付け)
        w_k0 = (1.0 - self_loop_mask) * 1.0  
        w_k1 = self_loop_mask * 1.0
        
        # 各ノード・各モードの位相
        theta_ij_k0 = self.raw_phase + phi_curr[:, 0].view(B, 1, 1)
        theta_ij_k1 = self.raw_phase + phi_curr[:, 1].view(B, 1, 1)
        
        # 複素演算の統合
        x_real, x_imag = x_2ch[:, :, 0].unsqueeze(1), x_2ch[:, :, 1].unsqueeze(1)
        
        # Mode 0 (Propagator)
        A0 = S * w_k0 * r_mode[:, 0].view(B, 1, 1)
        res0_real = torch.sum(A0 * (torch.cos(theta_ij_k0) * x_real - torch.sin(theta_ij_k0) * x_imag), dim=2)
        res0_imag = torch.sum(A0 * (torch.sin(theta_ij_k0) * x_real + torch.cos(theta_ij_k0) * x_imag), dim=2)
        
        # Mode 1 (Reservoir)
        A1 = S * w_k1 * r_mode[:, 1].view(B, 1, 1)
        res1_real = torch.sum(A1 * (torch.cos(theta_ij_k1) * x_real - torch.sin(theta_ij_k1) * x_imag), dim=2)
        res1_imag = torch.sum(A1 * (torch.sin(theta_ij_k1) * x_real + torch.cos(theta_ij_k1) * x_imag), dim=2)
        
        next_real = 0.999 * (res0_real + res1_real)
        next_imag = 0.999 * (res0_imag + res1_imag)
        
        return torch.stack([next_real, next_imag], dim=-1), phi_curr

# --- 3. 学習と評価 ---
def train():
    model = HybridSharpModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    mask_true = torch.eye(5).to(device)
    for i in range(4): mask_true[i+1, i] = 1.0

    print("Hybrid Propagator-Reservoir Training...")
    for epoch in range(2501):
        model.train(); optimizer.zero_grad(); model.update_u()
        data = generate_5d_mechanical_data(16, 64)
        B, T, V = data.shape
        x_imag_latent = torch.zeros(B, V, 1).to(device)
        phi_latent = torch.tensor([0.0, np.pi/2]).repeat(B, 1).to(device)
        
        loss_mse = 0
        for t in range(T-1):
            x_in = torch.cat([data[:, t, :].unsqueeze(-1), x_imag_latent], dim=-1)
            x_out, phi_latent = model(x_in, data[:, t, :].repeat(1, 5), phi_latent, epoch)
            x_imag_latent = x_out[:, :, 1].unsqueeze(-1)
            loss_mse += F.mse_loss(x_out[:, :, 0], data[:, t+1, :])

        loss = (loss_mse / T) + 0.02 * torch.norm(model._get_S_core(epoch) * (1-mask_true), p=1)
        loss.backward(); optimizer.step()
        if epoch % 500 == 0: print(f"Epoch {epoch:4d} | MSE: {loss_mse.item()/T:.4f}")
    return model

def evaluate_do(model):
    model.eval()
    with torch.no_grad():
        print("\n[Table 1] Hybrid S Map:")
        S = model._get_S_core(2500).cpu().numpy()
        for i, row in enumerate(S):
            print(f"X{i}: "+" ".join([f"{v:6.3f}" if abs(v)>0.01 else "   .  " for v in row]))
        
        print("\n[Table 2] Do(X0=1.0) 10-step Intervention:")
        x_2ch = torch.zeros(1, 5, 2).to(device)
        phi_t = torch.tensor([0.0, np.pi/2]).repeat(1, 1).to(device)
        for s in range(25):
            if s < 10: x_2ch[0, 0, 0] = 1.0
            x_out, phi_t = model(x_2ch, torch.zeros(1, 25).to(device), phi_t, 2500)
            v = x_out[0, :, 0].cpu().numpy()
            if s < 10: v[0] = 1.0
            print(f"Step {s:2d} | {' '.join([f'X{i}:{v[i]:>6.2f}' for i in range(5)])}")
            x_2ch[0, :, 0] = torch.tensor(v).to(device)
            x_2ch[0, :, 1] = x_out[0, :, 1]

if __name__ == "__main__":
    m = train(); evaluate_do(m)
