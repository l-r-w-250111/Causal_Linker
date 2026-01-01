import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. データ生成 ---
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
            c[2] -= 0.8 * p[4]  # 負の相互作用
            c[1] -= 0.6 * p[4]  # 負の相互作用
            c += 0.2 * p 
            data[b, t] = torch.clamp(c, -5, 5)
    return (data - data.mean()) / (data.std() + 1e-6)

# --- 2. 最終モデル定義 ---
class SignedSelectiveDampingModel(nn.Module):
    def __init__(self, n_vars=5, d_model=128):
        super().__init__()
        self.n_vars = n_vars
        self.K = 2
        # 指摘[2]: 符号付きS (初期値は小さく)
        self.S = nn.Parameter(torch.randn(n_vars, n_vars) * 0.1)
        
        # 指摘[1]: モード別の減衰パラメータ
        self.damping_params = nn.Parameter(torch.tensor([3.0, 3.0])) 
        
        self.mode_gen = nn.Sequential(
            nn.Linear(n_vars * 5, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, self.K * 2) 
        )
        self.path_weights = nn.Parameter(torch.randn(n_vars, n_vars, self.K))

    def forward(self, x_2ch, history, phi_prev):
        B = x_2ch.shape[0]
        params = self.mode_gen(history).view(B, self.K, 2)
        
        d_phi = torch.tanh(params[:, :, 0]) * (np.pi / 24) 
        phi_curr = phi_prev + d_phi
        r_mode = torch.sigmoid(params[:, :, 1])
        
        # モード別減衰率
        sigma_k = torch.sigmoid(self.damping_params) 
        
        w = torch.softmax(self.path_weights, dim=-1)
        theta_ij = torch.einsum('ijk,bk->bij', w, phi_curr)
        r_ij = torch.einsum('ijk,bk->bij', w, r_mode)
        
        # パスごとの減衰率をモード加重平均で算出
        damp_ij = torch.einsum('ijk,k->ij', w, sigma_k).unsqueeze(0)
        
        # 指摘[2]: Sの符号を活かしてゲイン計算
        A_raw = r_ij * self.S.unsqueeze(0)
        A_gain = damp_ij * (A_raw / (1.0 + A_raw.abs()))
        
        cos_t, sin_t = torch.cos(theta_ij), torch.sin(theta_ij)
        x_real, x_imag = x_2ch[:, :, 0].unsqueeze(1), x_2ch[:, :, 1].unsqueeze(1)
        
        next_real = torch.sum(A_gain * (cos_t * x_real - sin_t * x_imag), dim=2)
        next_imag = torch.sum(A_gain * (sin_t * x_real + cos_t * x_imag), dim=2)
        
        return torch.stack([next_real, next_imag], dim=-1), phi_curr

# --- 3. 学習プロセス ---
def train():
    n_vars = 5
    model = SignedSelectiveDampingModel(n_vars).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    mask = torch.zeros(n_vars, n_vars).to(device)
    for i in range(n_vars - 1): mask[i+1, i] = 1.0 
    mask[1, 4] = 1.0; mask[2, 4] = 1.0 
    mask += torch.eye(n_vars).to(device)

    print("Training phase starting (Signed & Selective Damping)...")
    for epoch in range(2501):
        model.train()
        optimizer.zero_grad()
        data = generate_5d_mechanical_data(32, 64)
        B, T, V = data.shape
        x_pad = torch.cat([torch.zeros(B, 4, V).to(device), data], dim=1)
        x_imag_latent = torch.zeros(B, V, 1).to(device)
        phi_latent = torch.tensor([0.0, np.pi/2]).repeat(B, 1).to(device)
        
        loss_mse = 0
        loss_phi_const = 0
        for t in range(T-1):
            x_in = torch.cat([data[:, t, :].unsqueeze(-1), x_imag_latent], dim=-1)
            x_out, phi_latent = model(x_in, x_pad[:, t:t+5, :].reshape(B, -1), phi_latent)
            x_imag_latent = x_out[:, :, 1].unsqueeze(-1)
            loss_mse += F.mse_loss(x_out[:, :, 0], data[:, t+1, :])
            if epoch < 500:
                loss_phi_const += 0.05 * torch.pow(phi_latent[:, 1] - np.pi/2, 2).mean()

        # スパース正則化
        loss_sparse = torch.mean(torch.abs(model.S) * (1 - mask))
        loss = (loss_mse / T) + loss_phi_const + 0.02 * loss_sparse
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            sig = torch.sigmoid(model.damping_params).detach().cpu().numpy()
            print(f"Epoch {epoch:4d} | MSE: {loss_mse.item()/T:.4f} | Dampings: {sig}")
    return model

# --- 4. 結果の解析 ---
def evaluate(model):
    model.eval()
    with torch.no_grad():
        s_matrix = (model.S / (1.0 + model.S.abs())).cpu().numpy()
        dampings = torch.sigmoid(model.damping_params).cpu().numpy()
        
        print(f"\n[Table 1] Learned Damping Factors per Mode:")
        print(f"Mode 0 (In-phase): {dampings[0]:.4f}, Mode 1 (Quadrature): {dampings[1]:.4f}")

        print("\n[Table 2] Signed Structural Matrix S (To \ From):")
        labels = [f"X{i}" for i in range(5)]
        print("      " + "".join([f"{l:<8}" for l in labels]))
        for i, row in enumerate(s_matrix):
            print(f"{labels[i]}: " + "".join([f"{v:>8.4f}" for v in row]))

        print("\n[Table 3] Rollout Dynamics (Impulse on X0):")
        x_2ch = torch.zeros(1, 5, 2).to(device); x_2ch[0, 0, 0] = 3.0
        phi_t = torch.tensor([0.0, np.pi/2]).repeat(1, 1).to(device)
        h_buf = torch.zeros(1, 5, 5).to(device)
        
        for s in range(15):
            x_out, phi_t = model(x_2ch, h_buf.view(1, -1), phi_t)
            v = x_out[0, :, 0].cpu().numpy()
            print(f"Step {s:2d} | {' '.join([f'X{i}:{v[i]:>6.2f}' for i in range(5)])}")
            if s == 0: x_2ch[0, 0, 0] = 0.0
            else: x_2ch[0, :, 0] = x_out[0, :, 0]
            x_2ch[0, :, 1] = x_out[0, :, 1]
            h_buf = torch.roll(h_buf, -1, dims=1); h_buf[0, 4, :] = torch.tensor(v).to(device)

if __name__ == "__main__":
    m = train()
    evaluate(m)
