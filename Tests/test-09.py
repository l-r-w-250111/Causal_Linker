import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. データ生成関数 ---
def generate_batch_data(batch_size=32, seq_len=100, context_type='forward'):
    n_vars = 3
    data = torch.zeros(batch_size, seq_len, n_vars).to(device)
    for b in range(batch_size):
        for t in range(1, seq_len):
            prev = data[b, t-1]
            curr = torch.randn(n_vars).to(device) * 0.1
            if context_type == 'forward':
                curr[1] += 1.5 * prev[0]  # X0 -> X1
                curr[2] += 1.5 * prev[1]  # X1 -> X2
            else:
                curr[1] += 1.5 * prev[2]  # X2 -> X1
                curr[0] += 1.5 * prev[1]  # X1 -> X0
            data[b, t] = curr
    return (data - data.mean()) / (data.std() + 1e-6)

# --- 2. 構造方程式モデルクラス ---
class PhysicalStructuralModel(nn.Module):
    def __init__(self, n_vars=3, d_model=128, history_len=5):
        super().__init__()
        self.n_vars = n_vars
        self.history_len = history_len
        self.context_encoder = nn.Sequential(
            nn.Linear(n_vars * history_len, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_vars * n_vars)
        )
        self.projector = nn.Sequential(nn.Linear(1, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)

    def get_A(self, x, history):
        B, T, V = x.shape
        gate = torch.sigmoid(self.context_encoder(history).view(B, T, V, V))
        feat = self.projector(x.reshape(-1, 1)).view(B, T, V, -1)
        scores = torch.matmul(self.Wq(feat), self.Wk(feat).transpose(-1, -2)) / (feat.shape[-1]**0.5)
        return torch.tanh(scores) * gate

    def predict_next(self, x, history):
        A = self.get_A(x, history)
        return torch.matmul(A, x.unsqueeze(-1)).squeeze(-1), A

# --- 3. DAG損失関数 ---
def strict_dag_loss(A):
    B, T, V, _ = A.shape
    A_sq = A * A
    res = torch.diagonal(torch.matrix_exp(A_sq.reshape(-1, V, V)), dim1=-2, dim2=-1).sum(-1) - V
    return res.mean()

# --- 4. 学習ループ ---
model = PhysicalStructuralModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Starting training...")
for epoch in range(2501):
    model.train()
    optimizer.zero_grad()
    
    loss_val = 0
    for c_type in ['forward', 'reverse']:
        data = generate_batch_data(16, 64, c_type)
        B, T, V = data.shape
        x_pad = torch.cat([torch.zeros(B, 4, V).to(device), data], dim=1)
        history = torch.stack([x_pad[:, i:i+5, :].reshape(B, -1) for i in range(T)], dim=1)[:, :-1, :]
        
        x_pred, A = model.predict_next(data[:, :-1, :], history)
        
        # 指摘に基づいた重み調整
        loss_mse = 10.0 * F.mse_loss(x_pred, data[:, 1:, :]) # MSEを最優先
        loss_dag = 5.0 * strict_dag_loss(A)
        loss_smooth = 0.1 * F.mse_loss(A[:, 1:], A[:, :-1]) # 平滑性
        loss_reg = 0.01 * A.abs().mean()
        
        loss_val += loss_mse + loss_dag + loss_smooth + loss_reg

    loss_val.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss_val.item():.4f}")

# --- 5. 厳格な反実仮想Do-Test ---
def final_strict_do_test(model):
    model.eval()
    with torch.no_grad():
        print("\n### 1. Learned Structural Matrix (A) ###")
        x_base = torch.zeros(1, 1, 3).to(device)
        h_base = x_base.repeat(1, 5, 1).reshape(1, 1, -1)
        A_fixed = model.get_A(x_base, h_base).squeeze(1)[0] # (3, 3)
        
        print(pd.DataFrame(A_fixed.cpu().numpy(), 
                           index=['To X0','To X1','To X2'], columns=['From X0','From X1','From X2']).round(3))

        print("\n### 2. Recursive Do-Test (Strict Propagation) ###")
        x_t = torch.zeros(3, 1).to(device)
        x_t[0, 0] = 5.0 # Step 0: Initial Impulse on X0
        print(f"Step 0 (Do X0=5.0): X0={x_t[0,0]:.1f}, X1={x_t[1,0]:.1f}, X2={x_t[2,0]:.1f}")
        
        for s in range(1, 5):
            x_t = torch.matmul(A_fixed, x_t) # 構造方程式の適用
            res = x_t.squeeze().cpu().numpy()
            print(f"Step {s}: X0={res[0]:.3f}, X1={res[1]:.3f}, X2={res[2]:.3f}")

final_strict_do_test(model)
