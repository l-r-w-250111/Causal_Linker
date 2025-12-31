import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. データ生成 (信号をさらに明瞭に) ---
def generate_batch_data(batch_size=32, seq_len=100, context_type='forward', p_int=0.3):
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

# --- 2. 構造アテンション (バランス調整) ---
class CounterfactualModel(nn.Module):
    def __init__(self, n_vars=3, d_model=128):
        super().__init__()
        self.n_vars = n_vars
        self.projector = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model)
        )
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        # ゲートの初期値を少し高めにして「最初は因果がある」前提から始める
        self.gate = nn.Parameter(torch.ones(n_vars, n_vars) * 0.5)
        
    def get_A(self, x_curr, x_prev):
        B, T, V, _ = x_curr.shape
        q_feat = self.projector(x_curr.reshape(-1, 1)).view(B, T, V, -1)
        k_feat = self.projector(x_prev.reshape(-1, 1)).view(B, T, V, -1)
        scores = torch.matmul(self.Wq(q_feat), self.Wk(k_feat).transpose(-1, -2)) / (q_feat.shape[-1]**0.5)
        
        # [改良] ReLU + Sigmoidゲート。Tanhによる過度な抑制を排除
        A = F.relu(scores) * torch.sigmoid(self.gate)
        return A

    def forward(self, x):
        x_prev = x[:, :-1, :].unsqueeze(-1)
        x_curr = x[:, 1:, :].unsqueeze(-1)
        A = self.get_A(x_curr, x_prev)
        x_pred = torch.matmul(A, x_prev).squeeze(-1)
        return x_pred, x[:, 1:, :], A

# --- 3. 学習 ---
model = CounterfactualModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

for epoch in range(2001):
    optimizer.zero_grad()
    df = generate_batch_data(16, 64, 'forward')
    dr = generate_batch_data(16, 64, 'reverse')
    
    pf, tf, Af = model(df)
    pr, tr, Ar = model(dr)
    
    loss_mse = F.mse_loss(pf, tf) + F.mse_loss(pr, tr)
    
    # DAG制約を「平均」と「インスタンス」のハイブリッドに
    def dag_loss(A):
        A_sq = A * A
        res = torch.diagonal(torch.matrix_exp(A_sq.mean(dim=(0,1))), dim1=-2, dim2=-1).sum(-1) - 3
        return res
    
    # 正則化を少し緩める（因果を消しすぎないように）
    loss_reg = 0.01 * (Af.mean() + Ar.mean()) 
    loss_dag = 2.0 * (dag_loss(Af) + dag_loss(Ar))
    
    loss = loss_mse + loss_dag + loss_reg
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | MSE: {loss_mse.item():.4f}")

# --- 4. 結果表示 & 反実仮想試験 ---
model.eval()
with torch.no_grad():
    test_data = generate_batch_data(1, 100, 'forward')
    _, _, A_matrix = model(test_data)
    A_avg = A_matrix.mean(dim=(0,1))

print("\n### Learned Structural Matrix (Forward) ###")
df_A = pd.DataFrame(A_avg.cpu().numpy(), index=['To X0','To X1','To X2'], columns=['From X0','From X1','From X2'])
print(df_A.round(3))

# --- [反実仮想試験] ---
print("\n### Counterfactual Test: 'What if X0 was increased by +5.0?' ###")
# 現状の X0 に介入して、X2 への波及をシミュレート
x_base = torch.zeros(1, 3).to(device) # 初期状態
x_cf = x_base.clone()
x_cf[0, 0] += 5.0 # X0 に介入

# 1ステップ先の予測
y_pred = torch.matmul(A_avg, x_cf.T).squeeze()
print(f"Intervention on X0: {x_cf[0].cpu().numpy()}")
print(f"Predicted Effect on [X0, X1, X2]: {y_pred.cpu().numpy().round(3)}")
