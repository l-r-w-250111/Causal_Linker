import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. データ生成 (5次元・二次遅れ系) ---
def generate_5d_mechanical_data(batch_size=32, seq_len=100):
    n_vars = 5
    data = torch.zeros(batch_size, seq_len, n_vars).to(device)
    for b in range(batch_size):
        for t in range(1, seq_len):
            p = data[b, t-1]
            c = torch.randn(n_vars).to(device) * 0.03
            c[1] += 1.4 * p[0] 
            c[2] += 1.4 * p[1] 
            c[3] += 1.4 * p[2] 
            c[4] += 1.4 * p[3] 
            c[2] -= 0.8 * p[4]  
            c[1] -= 0.6 * p[4]
            c += 0.2 * p 
            data[b, t] = torch.clamp(c, -5, 5)
    return (data - data.mean()) / (data.std() + 1e-6)

# --- 2. 最終・安定化エネルギー保存モデル ---
class FinalSharpModel(nn.Module):
    def __init__(self, n_vars=5, d_model=128):
        super().__init__()
        self.n_vars = n_vars
        self.K = 2
        self.raw_S = nn.Parameter(torch.randn(n_vars, n_vars) * 0.01)
        self.gamma_ij_k = nn.Parameter(torch.randn(n_vars, n_vars, self.K) * 0.1)
        self.delta_phi_ij_k = nn.Parameter(torch.randn(n_vars, n_vars, self.K) * 0.01)
        
        self.mode_gen = nn.Sequential(
            nn.Linear(n_vars * 5, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, self.K * 2) 
        )
        self.path_weights = nn.Parameter(torch.randn(n_vars, n_vars, self.K))

    def get_S(self, epoch=2000):
        # 鋭利な因果抽出
        S = torch.tanh(self.raw_S * 2.5)
        tau = min(0.12, epoch / 18000.0)
        S = torch.sign(S) * torch.relu(torch.abs(S) - tau)
        
        # 自己ループの慣性強化 (0.85 ~ 0.98)
        diag_mask = torch.eye(self.n_vars).to(S.device)
        diag_val = 0.85 + 0.13 * torch.sigmoid(torch.diag(self.raw_S)) 
        S = S * (1 - diag_mask) + diag_mask * diag_val
        
        # 正規化による発散防止（分母を最小限にし、エネルギー消失を防ぐ）
        S_abs_sum = S.abs().sum(dim=1, keepdim=True) + 1e-6
        S = S / torch.clamp(S_abs_sum, min=1.0) 
        return S

    def forward(self, x_2ch, history, phi_global, epoch=2000):
        B = x_2ch.shape[0]
        params = self.mode_gen(history).view(B, self.K, 2)
        phi_curr = phi_global + torch.tanh(params[:, :, 0]) * (np.pi / 24)
        
        # ゲイン上限を1.1に設定し、正規化損失を補填
        r_mode = 1.1 * torch.sigmoid(params[:, :, 1]) 
        
        w = torch.softmax(self.path_weights, dim=-1)
        S = self.get_S(epoch)
        
        d_phi_raw = torch.einsum('ijk,ijk->ij', w, self.delta_phi_ij_k).unsqueeze(0)
        delta_phi = d_phi_raw * (torch.abs(S) > 0.01).float().unsqueeze(0)
        theta_ij = torch.einsum('ijk,bk->bij', w, phi_curr) + delta_phi
        
        # 伝達効率の最大化
        sigma_ij_k = 1.0 * torch.sigmoid(self.gamma_ij_k)
        damp_ij = torch.einsum('ijk,ijk->ij', w, sigma_ij_k).unsqueeze(0)
        g_ij = damp_ij * torch.einsum('ijk,bk->bij', w, r_mode)
        A_gain = S.unsqueeze(0) * g_ij
        
        cos_t, sin_t = torch.cos(theta_ij), torch.sin(theta_ij)
        x_real, x_imag = x_2ch[:, :, 0].unsqueeze(1), x_2ch[:, :, 1].unsqueeze(1)
        
        # 全体減衰を最小限(0.995)に留め、持続性を確保
        next_real = 0.995 * torch.sum(A_gain * (cos_t * x_real - sin_t * x_imag), dim=2)
        next_imag = 0.995 * torch.sum(A_gain * (sin_t * x_real + cos_t * x_imag), dim=2)
        
        return torch.stack([next_real, next_imag], dim=-1), phi_curr, A_gain

# --- 3. 学習プロセス ---
def train_final():
    n_vars = 5
    model = FinalSharpModel(n_vars).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    
    mask = torch.eye(n_vars).to(device)
    for i in range(n_vars - 1): mask[i+1, i] = 1.0 
    mask[1, 4] = 1.0; mask[2, 4] = 1.0 

    for epoch in range(2501):
        if epoch == 2001:
            model.raw_S.requires_grad_(False)
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0002)

        model.train()
        optimizer.zero_grad()
        data = generate_5d_mechanical_data(16, 64)
        B, T, V = data.shape
        x_pad = torch.cat([torch.zeros(B, 4, V).to(device), data], dim=1)
        x_imag_latent = torch.zeros(B, V, 1).to(device)
        phi_latent = torch.tensor([0.0, np.pi/2]).repeat(B, 1).to(device)
        
        loss_mse = 0
        for t in range(T-1):
            x_in = torch.cat([data[:, t, :].unsqueeze(-1), x_imag_latent], dim=-1)
            x_out, phi_latent, _ = model(x_in, x_pad[:, t:t+5, :].reshape(B, -1), phi_latent, epoch)
            x_imag_latent = x_out[:, :, 1].unsqueeze(-1)
            loss_mse += F.mse_loss(x_out[:, :, 0], data[:, t+1, :])

        S_curr = model.get_S(epoch)
        if epoch < 2001:
            off_diag = 1.0 - torch.eye(n_vars).to(device)
            loss_sparse = 0.05 * torch.norm(S_curr * off_diag, p=1)
            loss_mask = 0.2 * torch.mean(torch.abs(S_curr) * (1 - mask) * off_diag)
            loss = (loss_mse / T) + loss_sparse + loss_mask
        else:
            loss = (loss_mse / T)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        if epoch % 500 == 0:
            # ZeroRate計算バグの修正 (非対角20要素を分母にする)
            num_off_diag = n_vars**2 - n_vars
            zr = (torch.abs(S_curr * (1.0-torch.eye(n_vars).to(device))) < 0.01).float().sum() / num_off_diag
            print(f"Epoch {epoch:4d} | MSE: {loss_mse.item()/T:.4f} | Off-Diag ZeroRate: {zr:.2%}")
            
    return model

def evaluate(model):
    model.eval()
    with torch.no_grad():
        S = model.get_S(2500).cpu().numpy()
        print("\n[Table 1] Final Causal Map S (Stable & Sparse):")
        labels = [f"X{i}" for i in range(5)]
        print("      " + "".join([f"{l:<8}" for l in labels]))
        for i, row in enumerate(S):
            row_str = "".join([f"{v:>8.4f}" if abs(v) > 0.01 else f"{'.':>8}" for v in row])
            print(f"{labels[i]}: {row_str}")

        print("\n[Table 2] Pulse Response Rollout (Energy Tracking):")
        # Step 0: X0に強めのインパルス
        x_2ch = torch.zeros(1, 5, 2).to(device); x_2ch[0, 0, 0] = 3.0
        phi_t = torch.tensor([0.0, np.pi/2]).repeat(1, 1).to(device)
        h_buf = torch.zeros(1, 5, 5).to(device)
        for s in range(20):
            x_out, phi_t, _ = model(x_2ch, h_buf.view(1, -1), phi_t, 2500)
            v = x_out[0, :, 0].cpu().numpy()
            print(f"Step {s:2d} | {' '.join([f'X{i}:{v[i]:>6.2f}' for i in range(5)])}")
            # 入力を切った後の循環を見る
            x_2ch[0, :, 0] = x_out[0, :, 0] if s > 0 else 0.0
            x_2ch[0, :, 1] = x_out[0, :, 1]
            h_buf = torch.roll(h_buf, -1, dims=1); h_buf[0, 4, :] = torch.tensor(v).to(device)

if __name__ == "__main__":
    m = train_final()
    evaluate(m)
