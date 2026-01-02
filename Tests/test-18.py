import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. データ生成 (5次元・負帰還を含む二次遅れ系) ---
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
            c[2] -= 0.8 * p[4]  # 強力な負帰還
            c[1] -= 0.6 * p[4]
            c += 0.2 * p 
            data[b, t] = torch.clamp(c, -5, 5)
    return (data - data.mean()) / (data.std() + 1e-6)

# --- 2. 二段階・離散構造同定モデル ---
class DiscreteTopologyModel(nn.Module):
    def __init__(self, n_vars=5, d_model=128):
        super().__init__()
        self.n_vars = n_vars
        self.K = 2
        # raw_Sの初期値を少し大きめにし、シグモイドの飽和を促す
        self.raw_S = nn.Parameter(torch.randn(n_vars, n_vars) * 0.5)
        
        self.gamma_ij_k = nn.Parameter(torch.randn(n_vars, n_vars, self.K) + 2.0)
        self.delta_phi_ij_k = nn.Parameter(torch.randn(n_vars, n_vars, self.K) * 0.05)
        
        self.mode_gen = nn.Sequential(
            nn.Linear(n_vars * 5, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, self.K * 2) 
        )
        self.path_weights = nn.Parameter(torch.randn(n_vars, n_vars, self.K))

    def get_S(self):
        # [1] 指摘: Sを「ほぼ二値」にするロジック
        return torch.sign(self.raw_S) * torch.sigmoid(torch.abs(self.raw_S))

    def forward(self, x_2ch, history, phi_global):
        B = x_2ch.shape[0]
        params = self.mode_gen(history).view(B, self.K, 2)
        phi_curr = phi_global + torch.tanh(params[:, :, 0]) * (np.pi / 24)
        r_mode = torch.sigmoid(params[:, :, 1])
        
        w = torch.softmax(self.path_weights, dim=-1)
        S = self.get_S()
        
        # [3] 指摘: 位相に従属性をもたせる (|S|が小さいエッジの偏差を抑制)
        d_phi_raw = torch.einsum('ijk,ijk->ij', w, self.delta_phi_ij_k).unsqueeze(0)
        delta_phi = d_phi_raw * torch.abs(S).unsqueeze(0) 
        
        theta_ij = torch.einsum('ijk,bk->bij', w, phi_curr) + delta_phi
        
        sigma_ij_k = torch.sigmoid(self.gamma_ij_k)
        damp_ij = torch.einsum('ijk,ijk->ij', w, sigma_ij_k).unsqueeze(0)
        
        # 強度(振幅)はg_ijに集約
        g_ij = damp_ij * torch.einsum('ijk,bk->bij', w, r_mode)
        A_gain = S.unsqueeze(0) * g_ij
        
        cos_t, sin_t = torch.cos(theta_ij), torch.sin(theta_ij)
        x_real, x_imag = x_2ch[:, :, 0].unsqueeze(1), x_2ch[:, :, 1].unsqueeze(1)
        
        next_real = torch.sum(A_gain * (cos_t * x_real - sin_t * x_imag), dim=2)
        next_imag = torch.sum(A_gain * (sin_t * x_real + cos_t * x_imag), dim=2)
        
        return torch.stack([next_real, next_imag], dim=-1), phi_curr, A_gain, delta_phi

# --- 3. 学習プロセス (二段階) ---
def train_two_stage():
    n_vars = 5
    model = DiscreteTopologyModel(n_vars).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 因果マスク
    mask = torch.zeros(n_vars, n_vars).to(device)
    for i in range(n_vars - 1): mask[i+1, i] = 1.0 
    mask[1, 4] = 1.0; mask[2, 4] = 1.0 
    mask += torch.eye(n_vars).to(device)

    # 1. 構造同定フェーズ
    print("Stage 1: Structural Identification (Epoch 0-2000)")
    for epoch in range(2501):
        # [4] 指摘: Epoch 2000でSを固定
        if epoch == 2001:
            print("\nStage 2: Freezing S and Refining Dynamics (Epoch 2001-2500)")
            model.raw_S.requires_grad_(False)
            # 最適化対象をS以外に絞る
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0005)

        model.train()
        optimizer.zero_grad()
        data = generate_5d_mechanical_data(32, 64)
        B, T, V = data.shape
        x_pad = torch.cat([torch.zeros(B, 4, V).to(device), data], dim=1)
        x_imag_latent = torch.zeros(B, V, 1).to(device)
        phi_latent = torch.tensor([0.0, np.pi/2]).repeat(B, 1).to(device)
        
        loss_mse = 0
        avg_A_gain = 0
        avg_delta_phi = 0
        
        for t in range(T-1):
            x_in = torch.cat([data[:, t, :].unsqueeze(-1), x_imag_latent], dim=-1)
            x_out, phi_latent, A_gain, d_phi = model(x_in, x_pad[:, t:t+5, :].reshape(B, -1), phi_latent)
            x_imag_latent = x_out[:, :, 1].unsqueeze(-1)
            loss_mse += F.mse_loss(x_out[:, :, 0], data[:, t+1, :])
            avg_A_gain += A_gain.abs().mean(0)
            avg_delta_phi += d_phi.abs().mean()

        # [2] 指摘: 全域的スパース圧力 (Sに対して直接)
        S_curr = model.get_S()
        loss_sparse_S = 0.03 * torch.norm(S_curr, p=1) 
        # マスク外への追加圧力
        loss_mask_S = 0.05 * torch.mean(torch.abs(S_curr) * (1 - mask))
        
        # [3] 位相へのスパース圧力
        loss_sparse_phi = 0.01 * (avg_delta_phi / T)
        
        loss = (loss_mse / T) + loss_sparse_S + loss_mask_S + loss_sparse_phi
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:4d} | MSE: {loss_mse.item()/T:.4f} | S_norm: {S_curr.abs().mean().item():.4f}")
            
    return model

# --- 4. 結果の解析 ---
def evaluate(model):
    model.eval()
    with torch.no_grad():
        S = model.get_S().cpu().numpy()
        
        print("\n[Table 1] Final Discrete Structure S (Binary Logic):")
        labels = [f"X{i}" for i in range(5)]
        print("      " + "".join([f"{l:<8}" for l in labels]))
        for i, row in enumerate(S):
            # 0.1以下を視認性のために.として表示
            row_str = "".join([f"{v:>8.4f}" if abs(v) > 0.05 else f"{'.':>8}" for v in row])
            print(f"{labels[i]}: {row_str}")

        print("\n[Table 2] Rollout Stability (Impulse on X0):")
        x_2ch = torch.zeros(1, 5, 2).to(device); x_2ch[0, 0, 0] = 3.0
        phi_t = torch.tensor([0.0, np.pi/2]).repeat(1, 1).to(device)
        h_buf = torch.zeros(1, 5, 5).to(device)
        for s in range(15):
            x_out, phi_t, _, _ = model(x_2ch, h_buf.view(1, -1), phi_t)
            v = x_out[0, :, 0].cpu().numpy()
            print(f"Step {s:2d} | {' '.join([f'X{i}:{v[i]:>6.2f}' for i in range(5)])}")
            if s == 0: x_2ch[0, 0, 0] = 0.0
            else: x_2ch[0, :, 0] = x_out[0, :, 0]
            x_2ch[0, :, 1] = x_out[0, :, 1]
            h_buf = torch.roll(h_buf, -1, dims=1); h_buf[0, 4, :] = torch.tensor(v).to(device)

if __name__ == "__main__":
    m = train_two_stage()
    evaluate(m)
