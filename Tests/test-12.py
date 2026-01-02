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

# --- 2. 位相積分・振幅分離型モデル ---
class IntegralPhaseModel(nn.Module):
    def __init__(self, n_vars=3, d_model=128):
        super().__init__()
        self.n_vars = n_vars
        self.S = nn.Parameter(torch.rand(n_vars, n_vars) * 0.5)
        
        # Gain(r) と Delta_Theta(Δθ) を生成するエンコーダー
        self.dynamics_gen = nn.Sequential(
            nn.Linear(n_vars * 5, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_vars * n_vars * 2) # r と Δθ のため
        )
        
    def forward(self, x_2ch, history, theta_prev):
        B = x_2ch.shape[0]
        gen_out = self.dynamics_gen(history).view(B, 2, self.n_vars, self.n_vars)
        
        # [1] Gain(r) の分離: 0 ~ 2.0 程度に制限
        r = torch.sigmoid(gen_out[:, 0]) * 2.0
        
        # [2] 位相積分型 (Δθ): -pi/2 ~ pi/2 の範囲で位相を変化させる
        delta_theta = torch.tanh(gen_out[:, 1]) * (np.pi / 2)
        theta_curr = theta_prev + delta_theta
        
        # 基底構造 S と Gain r を統合
        A_gain = r * F.relu(self.S).unsqueeze(0)
        
        cos_t = torch.cos(theta_curr)
        sin_t = torch.sin(theta_curr)
        
        x_real = x_2ch[:, :, 0].unsqueeze(1)
        x_imag = x_2ch[:, :, 1].unsqueeze(1)
        
        # 複素回転演算
        next_real = torch.sum(A_gain * (cos_t * x_real - sin_t * x_imag), dim=2)
        next_imag = torch.sum(A_gain * (sin_t * x_real + cos_t * x_imag), dim=2)
        
        return torch.stack([next_real, next_imag], dim=-1), theta_curr

# --- 3. 学習プロセス (位相を状態として引き継ぐ) ---
model = IntegralPhaseModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training Integral Phase Model...")
for epoch in range(3001):
    model.train()
    optimizer.zero_grad()
    data = generate_feedback_data(32, 64)
    B, T, V = data.shape
    x_pad = torch.cat([torch.zeros(B, 4, V).to(device), data], dim=1)
    
    x_imag_latent = torch.zeros(B, V, 1).to(device)
    theta_latent = torch.zeros(B, V, V).to(device) # 位相状態の初期化
    loss_total = 0
    
    for t in range(T-1):
        history = x_pad[:, t:t+5, :].reshape(B, -1)
        x_input = torch.cat([data[:, t, :].unsqueeze(-1), x_imag_latent], dim=-1)
        
        x_out, theta_latent = model(x_input, history, theta_latent)
        x_imag_latent = x_out[:, :, 1].unsqueeze(-1)
        
        loss_total += F.mse_loss(x_out[:, :, 0], data[:, t+1, :])
        
    loss_total = (loss_total / T) + 0.01 * model.S.abs().mean()
    loss_total.backward()
    optimizer.step()
    if epoch % 1000 == 0: print(f"Epoch {epoch} | Loss: {loss_total.item():.6f}")

# --- 4. 動的ロールアウト評価 ---
def run_integral_test(model):
    model.eval()
    with torch.no_grad():
        print("\n=== Integral Phase Rollout (Stability Check) ===")
        x_2ch = torch.zeros(1, 3, 2).to(device)
        x_2ch[0, 0, 0] = 5.0
        theta_t = torch.zeros(1, 3, 3).to(device)
        history_buf = torch.zeros(1, 5, 3).to(device)
        
        for s in range(12):
            history_flat = history_buf.view(1, -1)
            x_2ch, theta_t = model(x_2ch, history_flat, theta_t)
            
            val = x_2ch[0, :, 0].cpu().numpy()
            ang = (theta_t[0, 1, 2] / np.pi).cpu().numpy() # X2 -> X1 位相
            
            print(f"Step {s:2d}: X1={val[1]:>5.2f}, X2={val[2]:>5.2f} | Phase(X2->X1)={ang:>6.2f}π")
            
            if s == 0: x_2ch[0, 0, 0] = 0.0
            history_buf = torch.roll(history_buf, -1, dims=1)
            history_buf[0, 4, :] = x_2ch[0, :, 0]

run_integral_test(model)
