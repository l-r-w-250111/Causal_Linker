import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. データ生成 (変更なし) ---
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

# --- 2. 安定化Shared2Modeモデル ---
class StabilizedShared2ModeModel(nn.Module):
    def __init__(self, n_vars=5, d_model=128):
        super().__init__()
        self.n_vars = n_vars
        self.K = 2 
        # Sの初期値を調整
        self.S = nn.Parameter(torch.rand(n_vars, n_vars) * 0.3 + 0.1)
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
        
        # [微調整2] 位相変化をさらに滑らかに (pi/16 -> pi/24)
        d_phi = torch.tanh(params[:, :, 0]) * (np.pi / 24) 
        phi_curr = phi_prev + d_phi
        
        # [微調整1] ゲインの最大値を抑制 (1.5 -> 1.0)
        r_mode = torch.sigmoid(params[:, :, 1]) * 1.0
        
        w = torch.softmax(self.path_weights, dim=-1)
        theta_ij = torch.einsum('ijk,bk->bij', w, phi_curr)
        r_ij = torch.einsum('ijk,bk->bij', w, r_mode)
        
        # [微調整1&3] ゲインの飽和と自己減衰的なソフトクランプ
        A_gain = torch.tanh(r_ij * F.relu(self.S).unsqueeze(0)) * 1.1
        
        cos_t, sin_t = torch.cos(theta_ij), torch.sin(theta_ij)
        x_real, x_imag = x_2ch[:, :, 0].unsqueeze(1), x_2ch[:, :, 1].unsqueeze(1)
        
        next_real = torch.sum(A_gain * (cos_t * x_real - sin_t * x_imag), dim=2)
        next_imag = torch.sum(A_gain * (sin_t * x_real + cos_t * x_imag), dim=2)
        
        return torch.stack([next_real, next_imag], dim=-1), phi_curr, d_phi

# --- 3. 学習プロセス ---
def train_stabilized():
    model = StabilizedShared2ModeModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Training Stabilized Model...")
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
            x_out, phi_latent, d_phi = model(x_in, x_pad[:, t:t+5, :].reshape(B, -1), phi_latent)
            x_imag_latent = x_out[:, :, 1].unsqueeze(-1)
            loss_mse += F.mse_loss(x_out[:, :, 0], data[:, t+1, :])
            # [微調整2] Phi1(直交モード)がpi/2から離れすぎないよう制約
            loss_phi_const += 0.01 * torch.pow(phi_latent[:, 1] - np.pi/2, 2).mean()

        # 正則化を少し強めに戻す (因果トポロジーの純化)
        loss = (loss_mse / T) + loss_phi_const + 0.01 * model.S.abs().mean()
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0: print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")
    return model

# --- 4. 結果の可視化 (表 & ヒートマップ) ---
def final_report(model):
    model.eval()
    with torch.no_grad():
        s_matrix = (torch.tanh(F.relu(model.S)) * 1.1).cpu().numpy()
        w = torch.softmax(model.path_weights, dim=-1).cpu().numpy()[:, :, 0]

        # [Table 1 & 2]
        print("\n=== [Final Analysis] Structural S and Mode 0 Dependency ===")
        header = "      " + "".join([f"X{i:<7}" for i in range(5)])
        print(header)
        for i in range(5):
            print(f"X{i} S: " + "".join([f"{v:>8.3f}" for v in s_matrix[i]]))
            print(f"X{i} M: " + "".join([f"{v:>8.3f}" for v in w[i]]))
            print("-" * 50)

        # ヒートマップ
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        im1 = ax1.imshow(s_matrix, cmap='YlGnBu')
        ax1.set_title("Connection Strength S")
        plt.colorbar(im1, ax=ax1)
        im2 = ax2.imshow(w, cmap='RdBu_r', vmin=0, vmax=1)
        ax2.set_title("Mode 0 Ratio (1.0=In-phase)")
        plt.colorbar(im2, ax=ax2)
        plt.show()

        # [Table 3] ロールアウト
        print("\n=== [Table 3] Stabilized Rollout (15 Steps) ===")
        x_2ch = torch.zeros(1, 5, 2).to(device)
        x_2ch[0, 0, 0] = 5.0
        phi_t = torch.tensor([[0.0, np.pi/2]]).to(device)
        h_buf = torch.zeros(1, 5, 5).to(device)
        
        print(f"{'Step':<4} | {'X0':>6} | {'X1':>6} | {'X2':>6} | {'X3':>6} | {'X4':>6} | {'Phi0/pi':>7} | {'Phi1/pi':>7}")
        for s in range(15):
            x_out, phi_t, _ = model(x_2ch, h_buf.view(1, -1), phi_t)
            v = x_out[0, :, 0].cpu().numpy()
            p = phi_t[0].cpu().numpy() / np.pi
            print(f"{s:<4d} | {v[0]:>6.2f} | {v[1]:>6.2f} | {v[2]:>6.2f} | {v[3]:>6.2f} | {v[4]:>6.2f} | {p[0]:>7.2f} | {p[1]:>7.2f}")
            if s == 0: x_2ch[0, 0, 0] = 0.0
            else: x_2ch[0, :, 0] = x_out[0, :, 0]
            x_2ch[0, :, 1] = x_out[0, :, 1]
            h_buf = torch.roll(h_buf, -1, dims=1)
            h_buf[0, 4, :] = torch.tensor(v).to(device)

if __name__ == "__main__":
    m = train_stabilized()
    final_report(m)
