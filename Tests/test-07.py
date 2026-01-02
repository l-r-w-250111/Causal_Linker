import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. データ生成 (以前のものを維持) ---
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

# --- 2. Contextual Gate 搭載モデル ---
class ContextualStructuralModel(nn.Module):
    def __init__(self, n_vars=3, d_model=128, history_len=5):
        super().__init__()
        self.n_vars = n_vars
        self.history_len = history_len
        
        # [3] Contextual Gate: 履歴の統計から「今の因果構造」を決定する
        self.context_encoder = nn.Sequential(
            nn.Linear(n_vars * history_len, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_vars * n_vars)
        )
        
        self.projector = nn.Sequential(nn.Linear(1, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)

    def get_A(self, x_curr, x_prev, history):
        B, T, V, _ = x_curr.shape
        # history: (B, T, V * history_len)
        gate_logits = self.context_encoder(history).view(B, T, V, V)
        gate = torch.sigmoid(gate_logits) # [1] ゲートによるコントラスト
        
        q_feat = self.projector(x_curr.reshape(-1, 1)).view(B, T, V, -1)
        k_feat = self.projector(x_prev.reshape(-1, 1)).view(B, T, V, -1)
        scores = torch.matmul(self.Wq(q_feat), self.Wk(k_feat).transpose(-1, -2)) / (q_feat.shape[-1]**0.5)
        
        # [5] スケールの安定化のために Tanh を経由
        A = torch.tanh(scores).abs() * gate
        return A

    def forward(self, x):
        B, T, V = x.shape
        # 履歴バッファの作成
        x_pad = torch.cat([torch.zeros(B, self.history_len-1, V).to(device), x], dim=1)
        history = torch.stack([x_pad[:, i:i+self.history_len, :].reshape(B, -1) for i in range(T)], dim=1)
        history = history[:, :-1, :] # tとt-1の間の予測に使う履歴

        x_prev = x[:, :-1, :].unsqueeze(-1)
        x_curr = x[:, 1:, :].unsqueeze(-1)
        A = self.get_A(x_curr, x_prev, history)
        
        x_pred = torch.matmul(A, x_prev).squeeze(-1)
        return x_pred, x[:, 1:, :], A

# --- 3. 厳格なDAG制約 [2] ---
def strict_dag_loss(A):
    B, T, V, _ = A.shape
    A_sq = A * A
    # 全サンプル・全時刻で行列指数のトレースを計算
    # matrix_expは (..., V, V) を受け取れる
    res = torch.diagonal(torch.matrix_exp(A_sq.reshape(-1, V, V)), dim1=-2, dim2=-1).sum(-1) - V
    return res.mean()

# --- 4. 学習 ---
model = ContextualStructuralModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(2001):
    optimizer.zero_grad()
    df = generate_batch_data(16, 64, 'forward')
    dr = generate_batch_data(16, 64, 'reverse')
    
    pf, tf, Af = model(df)
    pr, tr, Ar = model(dr)
    
    loss_mse = F.mse_loss(pf, tf) + F.mse_loss(pr, tr)
    loss_dag = 5.0 * (strict_dag_loss(Af) + strict_dag_loss(Ar))
    loss_reg = 0.05 * (Af.mean() + Ar.mean())
    
    loss = loss_mse + loss_dag + loss_reg
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | MSE: {loss_mse.item():.4f} | DAG: {loss_dag.item():.4f}")

# --- 5. 再帰的反実仮想シミュレーション [1] ---
def run_recursive_do(model, intervention_val=5.0, steps=3):
    model.eval()
    with torch.no_grad():
        # 初期状態: Forward文脈のデータ末尾を起点にする
        base_data = generate_batch_data(1, 10, 'forward')
        curr_x = base_data[:, -1:, :] # (1, 1, 3)
        
        print(f"\n[Recursive Do-Test] Step 0 (Intervention): X0 = {intervention_val}")
        
        trajectory = []
        for s in range(steps):
            # 履歴の捏造（簡易的に現在のxを繰り返す）
            hist = curr_x.repeat(1, 5, 1).reshape(1, 1, -1)
            # Aの推定
            A = model.get_A(curr_x.unsqueeze(-1), curr_x.unsqueeze(-1), hist).squeeze(1) # (1, 3, 3)
            
            # do演算: X0を固定
            curr_x[0, 0, 0] = intervention_val
            
            # 次のステップを計算: x_t+1 = f(A, x_t)
            # ここでは線形結合 A * x を反復
            curr_x = torch.matmul(A, curr_x.transpose(-1, -2)).transpose(-1, -2)
            trajectory.append(curr_x.squeeze().cpu().numpy())
            print(f"Step {s+1} Predicted [X0, X1, X2]: {trajectory[-1].round(3)}")

# 結果表示
model.eval()
with torch.no_grad():
    test_f = generate_batch_data(1, 100, 'forward')
    _, _, Af_final = model(test_f)
    print("\n### Learned Structural Matrix (Forward) ###")
    print(pd.DataFrame(Af_final.mean(dim=(0,1)).cpu().numpy(), 
                       index=['To X0','To X1','To X2'], columns=['From X0','From X1','From X2']).round(3))

run_recursive_do(model)
