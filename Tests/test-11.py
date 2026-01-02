import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. データ生成 (変更なし) ---
def generate_feedback_data(batch_size=32, seq_len=100):
    n_vars = 3
    data = torch.zeros(batch_size, seq_len, n_vars).to(device)
    for b in range(batch_size):
        for t in range(1, seq_len):
            p = data[b, t-1]
            c = torch.randn(n_vars).to(device) * 0.05
            c[1] += 1.4 * p[0]
            c[2] += 1.4 * p[1]
            c[1] -= 0.9 * p[2]
            c += 0.2 * p
            data[b, t] = torch.clamp(c, -10, 10)
    return (data - data.mean()) / (data.std() + 1e-6)

# --- 2. 疑似複素(2ch)因果モデル ---
class ComplexCausalModel(nn.Module):
    def __init__(self, n_vars=3, d_model=128):
        super().__init__()
        self.n_vars = n_vars
        self.S = nn.Parameter(torch.rand(n_vars, n_vars) * 0.5)
        self.theta_encoder = nn.Sequential(
            nn.Linear(n_vars * 5, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_vars * n_vars)
        )
        
    def forward(self, x_2ch, history):
        B = x_2ch.shape[0]
        theta = self.theta_encoder(history).view(B, self.n_vars, self.n_vars)
        S_act = F.relu(self.S).unsqueeze(0)
        
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        
        # インプレースを避けるためスライスで取得
        x_real = x_2ch[:, :, 0].unsqueeze(1) # [B, 1, V]
        x_imag = x_2ch[:, :, 1].unsqueeze(1) # [B, 1, V]
        
        # 複素数のかけ算（回転行列）の集計
        next_real = torch.sum(S_act * (cos_t * x_real - sin_t * x_imag), dim=2)
        next_imag = torch.sum(S_act * (sin_t * x_real + cos_t * x_imag), dim=2)
        
        return torch.stack([next_real, next_imag], dim=-1), theta

# --- 3. 学習プロセス (インプレースエラー回避版) ---
model = ComplexCausalModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training Complex Phase Model...")
for epoch in range(3001):
    model.train()
    optimizer.zero_grad()
    data = generate_feedback_data(32, 64)
    B, T, V = data.shape
    x_pad = torch.cat([torch.zeros(B, 4, V).to(device), data], dim=1)
    
    # 潜在変数(imag)の初期化
    x_imag_latent = torch.zeros(B, V, 1).to(device)
    loss_total = 0
    
    for t in range(T-1):
        history = x_pad[:, t:t+5, :].reshape(B, -1)
        # 観測データ(real)と潜在変数(imag)を結合して入力を生成
        # .clone() は不要だが、安全のために連結(cat)で新しいテンソルを作る
        curr_obs = data[:, t, :].unsqueeze(-1) # [B, V, 1]
        x_input = torch.cat([curr_obs, x_imag_latent], dim=-1) # [B, V, 2]
        
        x_out, _ = model(x_input, history)
        
        # 次のステップのための潜在変数(imag)を更新（計算グラフは維持）
        x_imag_latent = x_out[:, :, 1].unsqueeze(-1)
        
        # 実部(real)が観測値と一致するように学習
        loss_total += F.mse_loss(x_out[:, :, 0], data[:, t+1, :])
        
    loss_total = (loss_total / T) + 0.01 * model.S.abs().mean()
    loss_total.backward()
    optimizer.step()
    if epoch % 1000 == 0: print(f"Epoch {epoch} | Loss: {loss_total.item():.6f}")

# --- 4. 動的ロールアウト評価 ---
def run_complex_test(model):
    model.eval()
    with torch.no_grad():
        print("\n=== Complex Phase Rollout (2-Channel Dynamics) ===")
        x_2ch = torch.zeros(1, 3, 2).to(device)
        x_2ch[0, 0, 0] = 5.0 # X0実部にインパルス
        history_buf = torch.zeros(1, 5, 3).to(device)
        history_buf[0, 4, 0] = 5.0
        
        for s in range(12):
            history_flat = history_buf.view(1, -1)
            x_2ch, theta_t = model(x_2ch, history_flat)
            
            real_vals = x_2ch[0, :, 0].cpu().numpy()
            imag_vals = x_2ch[0, :, 1].cpu().numpy()
            ang_x2_x1 = (theta_t[0, 1, 2] / np.pi).cpu().numpy()
            
            print(f"Step {s:2d}: X1={real_vals[1]:>5.2f}(i:{imag_vals[1]:>5.2f}), X2={real_vals[2]:>5.2f} | Phase(X2->X1)={ang_x2_x1:>5.2f}π")
            
            # インパルスは最初だけ
            if s == 0: x_2ch[0, 0, 0] = 0.0
            
            # 履歴の更新
            history_buf = torch.roll(history_buf, -1, dims=1)
            history_buf[0, 4, :] = x_2ch[0, :, 0]

run_complex_test(model)
