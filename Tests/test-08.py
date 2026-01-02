import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. データ生成 (共通) ---
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

# --- 2. モデル定義 (Contextual Gate + BottleNeck) ---
class RobustCausalModel(nn.Module):
    def __init__(self, n_vars=3, d_model=128):
        super().__init__()
        self.n_vars = n_vars
        self.context_encoder = nn.Sequential(
            nn.Linear(n_vars * 5, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_vars * n_vars)
        )
        self.projector = nn.Sequential(nn.Linear(1, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)

    def get_A(self, x, history):
        B, T, V = x.shape
        gate_logits = self.context_encoder(history).view(B, T, V, V)
        gate = torch.sigmoid(gate_logits)
        
        feat = self.projector(x.reshape(-1, 1)).view(B, T, V, -1)
        scores = torch.matmul(self.Wq(feat), self.Wk(feat).transpose(-1, -2)) / (feat.shape[-1]**0.5)
        
        # 構造係数 A (Tanhでスケール安定化 + ゲート)
        A = torch.tanh(scores).abs() * gate
        return A

    def predict_next(self, x, history):
        A = self.get_A(x, history)
        return torch.matmul(A, x.unsqueeze(-1)).squeeze(-1), A

# --- 3. 厳格なDAG損失 ---
def strict_dag_loss(A):
    B, T, V, _ = A.shape
    A_sq = A * A
    # 全ステップ・全サンプルでの非巡回制約
    res = torch.diagonal(torch.matrix_exp(A_sq.reshape(-1, V, V)), dim1=-2, dim2=-1).sum(-1) - V
    return res.mean()

# --- 4. 学習プロセス (再帰的一貫性の導入) ---
model = RobustCausalModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(2001):
    model.train()
    optimizer.zero_grad()
    
    total_loss = 0
    for c_type in ['forward', 'reverse']:
        data = generate_batch_data(16, 64, c_type)
        B, T, V = data.shape
        x_pad = torch.cat([torch.zeros(B, 4, V).to(device), data], dim=1)
        history = torch.stack([x_pad[:, i:i+5, :].reshape(B, -1) for i in range(T)], dim=1)[:, :-1, :]
        
        # [A] 1ステップ予測
        x_curr = data[:, :-1, :]
        x_next_gt = data[:, 1:, :]
        x_pred, A = model.predict_next(x_curr, history)
        loss_mse = F.mse_loss(x_pred, x_next_gt)
        
        # [B] ランダム・マルチステップ一貫性 (k=2〜3)
        # 媒介 X1 を飛ばして X2 を直接予測することを防ぐ
        k = random.randint(2, 3)
        x_multi = x_pred[:, :-k+1, :] 
        for _ in range(1, k):
            # 履歴は予測時点のものを使用
            curr_hist = history[:, :x_multi.shape[1], :]
            x_multi, _ = model.predict_next(x_multi, curr_hist)
        
        loss_multi = F.mse_loss(x_multi, data[:, k:, :])
        
        # [C] 構造制約
        loss_dag = 5.0 * strict_dag_loss(A)
        loss_reg = 0.05 * A.mean()
        
        total_loss += loss_mse + 0.5 * loss_multi + loss_dag + loss_reg

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | Loss: {total_loss.item():.4f}")

# --- 5. 反実仮想試験 (媒介分離の検証) ---
def final_do_test(model):
    model.eval()
    with torch.no_grad():
        print("\n### Learned Structural Matrix (Forward Average) ###")
        test_f = generate_batch_data(1, 100, 'forward')
        x_pad = torch.cat([torch.zeros(1, 4, 3).to(device), test_f], dim=1)
        hist = torch.stack([x_pad[:, i:i+5, :].reshape(1, -1) for i in range(100)], dim=1)[:, :-1, :]
        _, A_f = model.predict_next(test_f[:, :-1, :], hist)
        print(pd.DataFrame(A_f.mean(dim=(0,1)).cpu().numpy(), 
                           index=['To X0','To X1','To X2'], columns=['From X0','From X1','From X2']).round(3))

        # 再帰的なDo演算 (X0に5.0の衝撃を与え、その後の推移を見る)
        curr_x = torch.zeros(1, 1, 3).to(device)
        intervention_val = 5.0
        print(f"\n[Recursive Do-Test] X0 initial pulse: {intervention_val}")
        
        for s in range(4):
            # 簡易的な履歴生成
            h = curr_x.repeat(1, 5, 1).reshape(1, 1, -1)
            # Aの推定
            A_step = model.get_A(curr_x, h).squeeze(1)
            
            if s == 0:
                curr_x[0, 0, 0] = intervention_val # Step 0 のみ介入
            
            # 構造方程式による次状態の計算
            curr_x = torch.matmul(A_step, curr_x.transpose(-1, -2)).transpose(-1, -2)
            print(f"Step {s+1} [X0, X1, X2]: {curr_x.squeeze().cpu().numpy().round(3)}")

final_do_test(model)
