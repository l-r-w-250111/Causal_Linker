import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sys

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. データ生成 (5次元・二次遅れ系) ---
def generate_5d_mechanical_data(batch_size=32, seq_len=100):
    n_vars = 5
    data = torch.zeros(batch_size, seq_len, n_vars).to(device)
    for b in range(batch_size):
        for t in range(1, seq_len):
            p = data[b, t-1]
            c = torch.randn(n_vars).to(device) * 0.03
            # 構造定義
            c[1] += 1.4 * p[0] 
            c[2] += 1.4 * p[1] 
            c[3] += 1.4 * p[2] 
            c[4] += 1.4 * p[3] 
            # 負帰還
            c[2] -= 0.8 * p[4] 
            c[1] -= 0.6 * p[4]
            c += 0.2 * p 
            data[b, t] = torch.clamp(c, -5, 5)
    return (data - data.mean()) / (data.std() + 1e-6)

# --- 2. ブラッシュアップ版モデル ---
class FinalRefinedModel(nn.Module):
    def __init__(self, n_vars=5, d_model=128):
        super().__init__()
        self.n_vars = n_vars
        self.K = 2
        # S: 指摘に基づき控えめに初期化
        self.S = nn.Parameter(torch.rand(n_vars, n_vars) * 0.1)
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
        
        # 位相変化
        d_phi = torch.tanh(params[:, :, 0]) * (np.pi / 24) 
        phi_curr = phi_prev + d_phi
        r_mode = torch.sigmoid(params[:, :, 1])
        
        w = torch.softmax(self.path_weights, dim=-1)
        theta_ij = torch.einsum('ijk,bk->bij', w, phi_curr)
        r_ij = torch.einsum('ijk,bk->bij', w, r_mode)
        
        # 指摘[1]: ゲインの飽和処理 (Soft-clamping)
        A_raw = r_ij * F.relu(self.S).unsqueeze(0)
        A_gain = A_raw / (1.0 + A_raw.abs())
        
        cos_t, sin_t = torch.cos(theta_ij), torch.sin(theta_ij)
        x_real, x_imag = x_2ch[:, :, 0].unsqueeze(1), x_2ch[:, :, 1].unsqueeze(1)
        
        next_real = torch.sum(A_gain * (cos_t * x_real - sin_t * x_imag), dim=2)
        next_imag = torch.sum(A_gain * (sin_t * x_real + cos_t * x_imag), dim=2)
        
        # 指摘[1]: 振幅減衰 (エネルギー保存・散逸の導入)
        damping = 0.95
        return torch.stack([next_real * damping, next_imag * damping], dim=-1), phi_curr

# --- 3. 学習ループ ---
def train_and_validate():
    n_vars = 5
    model = FinalRefinedModel(n_vars).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 指摘[2]: 因果マスク (順方向と特定の帰還路のみ1)
    mask = torch.zeros(n_vars, n_vars).to(device)
    for i in range(n_vars - 1): mask[i+1, i] = 1.0 
    mask[1, 4] = 1.0; mask[2, 4] = 1.0 # 負帰還パス
    mask += torch.eye(n_vars).to(device) # 自己保持

    print("Training phase starting...")
    for epoch in range(2001):
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
            
            # 指摘[3]: ウォームアップ (500 epochまでphi1を拘束)
            if epoch < 500:
                loss_phi_const += 0.05 * torch.pow(phi_latent[:, 1] - np.pi/2, 2).mean()

        # 指摘[2]: 因果マスク外へのL1ペナルティ
        loss_sparse = torch.mean(torch.abs(model.S) * (1 - mask))
        
        loss = (loss_mse / T) + loss_phi_const + 0.05 * loss_sparse
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:4d} | MSE: {loss_mse.item()/T:.6f} | Sparse: {loss_sparse.item():.6f}")

    return model, mask

# --- 4. 結果の表出力と可視化 ---
def plot_results(model, mask):
    model.eval()
    with torch.no_grad():
        # S行列の計算
        A_raw = F.relu(model.S)
        s_matrix = (A_raw / (1.0 + A_raw.abs())).cpu().numpy()
        w = torch.softmax(model.path_weights, dim=-1).cpu().numpy()[:, :, 0]

        print("\n=== [Table 1] Refined Structural Matrix S (Damped) ===")
        labels = [f"X{i}" for i in range(5)]
        print("      " + "".join([f"{l:<8}" for l in labels]))
        for i, row in enumerate(s_matrix):
            print(f"{labels[i]}: " + "".join([f"{v:>8.4f}" for v in row]))

        # 図1: ヒートマップ
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        im1 = ax1.imshow(s_matrix, cmap='Blues')
        ax1.set_title("Learned Structure S (Sparse & Damped)")
        ax1.set_xticks(range(5)); ax1.set_yticks(range(5))
        plt.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(w, cmap='RdBu_r', vmin=0, vmax=1)
        ax2.set_title("Mode 0 Ratio (1.0=In-phase, 0.0=Quadrature)")
        ax2.set_xticks(range(5)); ax2.set_yticks(range(5))
        plt.colorbar(im2, ax=ax2)
        plt.tight_layout()
        plt.show()

        # 図2: ロールアウト表
        print("\n=== [Table 2] Stabilized Rollout (Impulse on X0) ===")
        x_2ch = torch.zeros(1, 5, 2).to(device)
        x_2ch[0, 0, 0] = 3.0 # 入力パルス
        phi_t = torch.tensor([[0.0, np.pi/2]]).to(device)
        h_buf = torch.zeros(1, 5, 5).to(device)
        
        print(f"{'Step':<4} | {'X0':>6} | {'X1':>6} | {'X2':>6} | {'X3':>6} | {'X4':>6}")
        print("-" * 50)
        rollout_history = []
        for s in range(15):
            x_out, phi_t = model(x_2ch, h_buf.view(1, -1), phi_t)
            v = x_out[0, :, 0].cpu().numpy()
            rollout_history.append(v)
            print(f"{s:<4d} | {v[0]:>6.2f} | {v[1]:>6.2f} | {v[2]:>6.2f} | {v[3]:>6.2f} | {v[4]:>6.2f}")
            
            if s == 0: x_2ch[0, 0, 0] = 0.0
            else: x_2ch[0, :, 0] = x_out[0, :, 0]
            x_2ch[0, :, 1] = x_out[0, :, 1]
            h_buf = torch.roll(h_buf, -1, dims=1); h_buf[0, 4, :] = torch.tensor(v).to(device)

        # 図3: ロールアウトの時系列グラフ
        plt.figure(figsize=(10, 4))
        plt.plot(np.array(rollout_history))
        plt.title("Step Response (Dynamic Rollout)")
        plt.legend(labels)
        plt.xlabel("Step"); plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    trained_model, causal_mask = train_and_validate()
    plot_results(trained_model, causal_mask)
