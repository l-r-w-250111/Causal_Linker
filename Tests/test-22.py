import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. データ生成 (物理的連鎖を生成) ---
def generate_5d_mechanical_data(batch_size=32, seq_len=100):
    n_vars = 5
    total_len = seq_len + 5
    data = torch.zeros(batch_size, total_len, n_vars).to(device)
    for b in range(batch_size):
        for t in range(1, total_len):
            p = data[b, t-1]
            # 強力な物理連鎖を定義
            c = torch.zeros(n_vars).to(device)
            c[1] = 0.8 * p[0]; c[2] = 0.8 * p[1]
            c[3] = 0.8 * p[2]; c[4] = 0.8 * p[3]
            # 帰還
            c[1] -= 0.3 * p[4]
            # 自己保持とノイズ
            data[b, t] = torch.clamp(0.2 * p + c + torch.randn(n_vars).to(device) * 0.05, -3, 3)
    return (data - data.mean()) / (data.std() + 1e-6)

# --- 2. モデル定義 (強化版) ---
class HybridSharpModel(nn.Module):
    def __init__(self, n_vars=5, d_model=128):
        super().__init__()
        self.n_vars = n_vars
        self.K = 2
        # 初期値を大きく設定し、結合を見つけやすくする
        self.raw_S = nn.Parameter(torch.randn(n_vars, n_vars) * 0.5 + 0.5) 
        self.raw_phase = nn.Parameter(torch.randn(n_vars, n_vars) * 0.1)
        self.mode_gen = nn.Sequential(
            nn.Linear(25, d_model), nn.BatchNorm1d(d_model),
            nn.ReLU(), nn.Linear(d_model, self.K * 2) 
        )
        self.register_buffer("u", F.normalize(torch.randn(1, n_vars), dim=1))

    def _get_S_core(self, epoch):
        # 緩やかにスパース化を導入
        S = torch.tanh(self.raw_S)
        tau = max(0, (epoch - 1500) / 5000) if epoch > 1500 else 0
        mask = (torch.abs(S) > tau).float()
        S = S * mask
        
        diag_mask = torch.eye(self.n_vars).to(S.device)
        return S * (1 - diag_mask) + diag_mask * 0.95

    def forward(self, x_2ch, history_flat, phi_global, epoch=2000):
        B = x_2ch.shape[0]
        params = self.mode_gen(history_flat).view(B, self.K, 2)
        phi_base = torch.tensor([0.0, np.pi/2], device=device)
        phi_curr = phi_base.unsqueeze(0) + torch.tanh(params[:, :, 0]) * (np.pi / 12)
        r_mode = torch.stack([1.5 * torch.sigmoid(params[:, 0, 1]), 0.8 * torch.sigmoid(params[:, 1, 1])], dim=1)
        
        S = self._get_S_core(epoch)
        self_loop_mask = torch.eye(self.n_vars).to(S.device)
        x_real, x_imag = x_2ch[:, :, 0].unsqueeze(1), x_2ch[:, :, 1].unsqueeze(1)
        
        out_real, out_imag = 0, 0
        for k in range(self.K):
            w_k = (1.0 - self_loop_mask) if k == 0 else self_loop_mask
            theta = self.raw_phase.unsqueeze(0) + phi_curr[:, k].view(B, 1, 1)
            A = S.unsqueeze(0) * w_k * r_mode[:, k].view(B, 1, 1)
            out_real += torch.sum(A * (torch.cos(theta) * x_real - torch.sin(theta) * x_imag), dim=2)
            out_imag += torch.sum(A * (torch.sin(theta) * x_real + torch.cos(theta) * x_imag), dim=2)
        return torch.stack([out_real, out_imag], dim=-1), phi_curr

# --- 3. 学習プロセス ---
def train():
    model = HybridSharpModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    print("Training Started (Causal Bridge Activation)...")
    for epoch in range(2501):
        model.train(); optimizer.zero_grad()
        data_all = generate_5d_mechanical_data(16, 64)
        x_imag_latent = torch.zeros(16, 5, 1).to(device)
        phi_latent = torch.tensor([0.0, np.pi/2]).repeat(16, 1).to(device)
        loss_mse = 0
        for t in range(5, 68):
            h_in = data_all[:, t-5:t, :].reshape(16, -1)
            x_in = torch.cat([data_all[:, t, :].unsqueeze(-1), x_imag_latent], dim=-1)
            x_out, phi_latent = model(x_in, h_in, phi_latent, epoch)
            x_imag_latent = x_out[:, :, 1].unsqueeze(-1)
            loss_mse += F.mse_loss(x_out[:, :, 0], data_all[:, t+1, :])
        
        # スパース制約を最初は弱く、後半に効かせる
        S_reg = torch.norm(model._get_S_core(epoch), p=1) * 0.001
        loss = (loss_mse / 63) + S_reg
        loss.backward(); optimizer.step()
        if epoch % 500 == 0: print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f}")
    return model

# --- 4. 解析実行 ---
def run_final_analysis(model):
    model.eval()
    with torch.no_grad():
        steps = 40
        res = torch.zeros(steps, 5)
        x_2ch = torch.zeros(1, 5, 2).to(device)
        phi_t = torch.tensor([0.0, np.pi/2]).repeat(1, 1).to(device)
        h_buf = torch.zeros(1, 5, 5).to(device)
        
        for s in range(steps):
            if s < 10: x_2ch[0, 0, 0] = 1.0
            x_out, phi_t = model(x_2ch, h_buf.view(1, -1), phi_t, 2500)
            v = x_out[0, :, 0].cpu().numpy()
            if s < 10: v[0] = 1.0
            res[s] = torch.tensor(v)
            x_2ch[0, :, 0], x_2ch[0, :, 1] = torch.tensor(v).to(device), x_out[0, :, 1]
            h_buf = torch.roll(h_buf, -1, dims=1); h_buf[0, -1, :] = torch.tensor(v).to(device)

        # グラフ作成
        plt.figure(figsize=(10, 5))
        for i in range(5): plt.plot(res[:, i], label=f'X{i}', linewidth=2)
        plt.axvline(9.5, color='red', linestyle='--'); plt.legend(); plt.grid(True); plt.show()

        # 表出力
        print("\n### [Table 1] Time-Series Response")
        print(pd.DataFrame(res.numpy()[:31], columns=[f'X{i}' for i in range(5)]).to_string(float_format="%.2f"))

        print("\n### [Table 2] Dynamics Analysis")
        summary = []
        for i in range(5):
            sig = res[:, i].numpy()
            lag = np.where(np.abs(sig) > 0.1)[0][0] if np.any(np.abs(sig) > 0.1) else 0
            summary.append([f"X{i}", lag, "Active" if np.max(np.abs(sig[1:])) > 0.1 else "Damped"])
        print(pd.DataFrame(summary, columns=['Node', 'Lag', 'Status']).to_string(index=False))

if __name__ == "__main__":
    m = train(); run_final_analysis(m)
