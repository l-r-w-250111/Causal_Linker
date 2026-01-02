import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. データ生成 (変更なし) ---
def generate_batch_data(batch_size=32, seq_len=100, context_type='forward'):
    n_vars = 3
    data = torch.zeros(batch_size, seq_len, n_vars).to(device)
    for b in range(batch_size):
        for t in range(1, seq_len):
            prev = data[b, t-1]
            curr = torch.randn(n_vars).to(device) * 0.1
            if context_type == 'forward':
                curr[1] += 1.5 * prev[0]
                curr[2] += 1.5 * prev[1]
            else:
                curr[1] += 1.5 * prev[2]
                curr[0] += 1.5 * prev[1]
            data[b, t] = curr
    return (data - data.mean()) / (data.std() + 1e-6)

# --- 2. DAG損失関数 ---
def strict_dag_loss(A):
    # A: (B, T, V, V) または (V, V)
    if A.dim() == 4:
        B, T, V, _ = A.shape
        A_sq = A * A
        res = torch.diagonal(torch.matrix_exp(A_sq.reshape(-1, V, V)), dim1=-2, dim2=-1).sum(-1) - V
        return res.mean()
    else:
        V = A.shape[-1]
        A_sq = A * A
        return torch.trace(torch.matrix_exp(A_sq)) - V

# --- 3. S ⊙ C(x, h) 分離型モデル ---
class HybridCausalModel(nn.Module):
    def __init__(self, n_vars=3, d_model=128):
        super().__init__()
        self.n_vars = n_vars
        # 固定構造 S (初期値を少し大きくして不感症を防ぐ)
        self.S = nn.Parameter(torch.randn(n_vars, n_vars) * 0.5)
        
        # 文脈係数 C (解像度を高める)
        self.context_encoder = nn.Sequential(
            nn.Linear(n_vars * 5, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_vars * n_vars)
        )
        
    def get_A(self, x, history):
        B, T, V = x.shape
        # C は構造の「導通」を制御。tanhにより極性反転も許容
        C = torch.tanh(self.context_encoder(history).view(B, T, V, V))
        A = self.S.unsqueeze(0).unsqueeze(0) * C
        return A

    def predict_next(self, x, history):
        A = self.get_A(x, history)
        return torch.matmul(A, x.unsqueeze(-1)).squeeze(-1), A

# --- 4. 学習プロセス ---
model = HybridCausalModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training with S ⊙ C and Sharpness focus...")
for epoch in range(2501):
    model.train()
    optimizer.zero_grad()
    
    total_loss = 0
    for c_type in ['forward', 'reverse']:
        data = generate_batch_data(16, 64, c_type)
        B, T, V = data.shape
        x_pad = torch.cat([torch.zeros(B, 4, V).to(device), data], dim=1)
        history = torch.stack([x_pad[:, i:i+5, :].reshape(B, -1) for i in range(T)], dim=1)[:, :-1, :]
        
        x_pred, A = model.predict_next(data[:, :-1, :], history)
        
        # 指摘に基づいた比率調整 MSE:DAG = 1:5
        loss_mse = F.mse_loss(x_pred, data[:, 1:, :])
        loss_dag = 5.0 * strict_dag_loss(A) # A全体に課して瞬時のDAGを担保
        loss_reg_S = 0.05 * model.S.abs().mean()
        
        # 因果の「鋭さ」を促すためのFlow正則化 (平均化による消滅を防ぐ)
        loss_flow = -0.02 * A.pow(2).mean() 
        
        total_loss += loss_mse + loss_dag + loss_reg_S + loss_flow

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | MSE: {loss_mse.item():.4f} | DAG: {loss_dag.item():.4f}")

# --- 5. 学習分布に基づく A_fixed での Do-Test ---
def final_hybrid_do_test(model):
    model.eval()
    with torch.no_grad():
        print("\n### 1. Fixed Base Structure (S) ###")
        print(pd.DataFrame(model.S.cpu().numpy(), 
                           index=['To X0','To X1','To X2'], columns=['From X0','From X1','From X2']).round(3))

        print("\n### 2. Average A from Forward Context ###")
        test_data = generate_batch_data(32, 100, 'forward')
        B, T, V = test_data.shape
        x_pad = torch.cat([torch.zeros(B, 4, V).to(device), test_data], dim=1)
        history = torch.stack([x_pad[:, i:i+5, :].reshape(B, -1) for i in range(T)], dim=1)[:, :-1, :]
        
        A_all = model.get_A(test_data[:, :-1, :], history)
        A_fixed = A_all.mean(dim=(0,1))

        print(pd.DataFrame(A_fixed.cpu().numpy(), 
                           index=['To X0','To X1','To X2'], columns=['From X0','From X1','From X2']).round(3))

        # Do-Test (介入パルス)
        x_t = torch.zeros(3, 1).to(device)
        x_t[0, 0] = 5.0
        print(f"\nStep 0 (Do X0=5.0): X0={x_t[0,0]:.1f}, X1={x_t[1,0]:.1f}, X2={x_t[2,0]:.1f}")
        
        for s in range(1, 4):
            x_t = torch.matmul(A_fixed, x_t)
            res = x_t.squeeze().cpu().numpy()
            print(f"Step {s}: X0={res[0]:.3f}, X1={res[1]:.3f}, X2={res[2]:.3f}")

final_hybrid_do_test(model)
