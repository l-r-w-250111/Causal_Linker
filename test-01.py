import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# --- 1. 定義 ---

def dag_constraint(A):
    """h(A) = tr(exp(A*A)) - d"""
    d = A.shape[0]
    A_sq = A * A
    E = torch.matrix_exp(A_sq)
    return torch.trace(E) - d

def generate_data(n, interventional_node=None, val=0.0):
    """Collider構造: X -> Z <- Y のデータ生成"""
    x = torch.randn(n, 1)
    y = torch.randn(n, 1)
    if interventional_node == 2: # Zに介入
        z = torch.full((n, 1), val)
    else:
        # 真の因果: Z = 0.8X + 0.6Y + noise
        z = 0.8 * x + 0.6 * y + torch.randn(n, 1) * 0.1
    return torch.cat([x, y, z], dim=1) # (n, 3)

class CausalAttentionModel(nn.Module):
    def __init__(self, num_vars=3):
        super().__init__()
        self.num_vars = num_vars
        # 因果行列Aの初期化（少しノイズを乗せる）
        self.A_logits = nn.Parameter(torch.randn(num_vars, num_vars) * 0.1)
        
    def get_A(self):
        # 自己ループ禁止 + シグモイドで0~1へ
        A = torch.sigmoid(self.A_logits)
        A = A * (1 - torch.eye(self.num_vars))
        return A

    def forward(self, x, intervention_node=None):
        A = self.get_A()
        if intervention_node is not None:
            # 介入ノードへの流入を遮断
            A_used = A.clone()
            A_used[intervention_node, :] = 0
        else:
            A_used = A
        
        # 簡易的な構造方程式による予測: hat_x = A_used @ x
        # 本来は (I-A)^-1 ですが、極小モデルの1ステップ予測として表現
        x_pred = torch.matmul(x, A_used.t())
        return x_pred, A

# --- 2. 学習関数 ---

def train_model(use_intervention=True):
    model = CausalAttentionModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    alpha, rho = 0.0, 1.0 # 拡張ラグランジュ用
    p_do = 0.3 if use_intervention else 0.0
    
    for epoch in range(1000):
        optimizer.zero_grad()
        
        # 介入判定
        is_int = (torch.rand(1) < p_do)
        int_node = 2 if is_int else None # Zに介入
        
        # データ生成と予測
        x_true = generate_data(64, interventional_node=int_node, val=2.0)
        x_pred, A = model(x_true, intervention_node=int_node)
        
        # 損失計算
        mse_loss = F.mse_loss(x_pred, x_true)
        h_a = dag_constraint(A)
        dag_loss = alpha * h_a + (rho / 2) * (h_a**2)
        l1_loss = 0.05 * torch.norm(A, 1)
        
        total_loss = mse_loss + dag_loss + l1_loss
        total_loss.backward()
        optimizer.step()
        
        # 外側ループ: 拡張ラグランジュ係数の更新（簡易版）
        if epoch % 100 == 0:
            with torch.no_grad():
                alpha += rho * h_a.item()
                if h_a.item() > 0.1: rho *= 1.2
                
    return model.get_A().detach().numpy()

# --- 3. 実行と視覚化 ---

A_obs = train_model(use_intervention=False)
A_int = train_model(use_intervention=True)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
vars = ['X', 'Y', 'Z']

for ax, data, title in zip(axes, [A_obs, A_int], ["Obs Only (Weak DAG)", "Obs + Intervention (True DAG)"]):
    im = ax.imshow(data, cmap='Blues', vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xticks(range(3)); ax.set_xticklabels(vars)
    ax.set_yticks(range(3)); ax.set_yticklabels(vars)
    ax.set_xlabel("Source (Cause)"); ax.set_ylabel("Target (Effect)")
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()
