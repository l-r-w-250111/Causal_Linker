import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# --- 1. DAG制約 ---
def dag_constraint(A):
    d = A.shape[-1]
    A_sq = A * A
    res = torch.stack([torch.trace(torch.matrix_exp(a)) - d for a in A_sq])
    return res.mean()

# --- 2. 強化版データ生成器 (因果の信号をさらに強調) ---
def generate_contextual_data(batch_size, context_type='normal', p_int=0.5):
    n = batch_size
    noise_scale = 0.01
    int_node = torch.randint(0, 3, (1,)).item() if torch.rand(1) < p_int else None
    int_val = 5.0 # 介入の値を大きくして、向きを無理やり教え込む
    
    if context_type == 'normal': # 0 -> 1 -> 2
        x0 = torch.randn(n, 1) if int_node != 0 else torch.full((n,1), int_val)
        x1 = (1.5 * x0 + torch.randn(n, 1) * noise_scale) if int_node != 1 else torch.full((n,1), int_val)
        x2 = (1.5 * x1 + torch.randn(n, 1) * noise_scale) if int_node != 2 else torch.full((n,1), int_val)
        data = torch.cat([x0, x1, x2], dim=1)
        c_vec = torch.tensor([1.0, 0.0])
    else: # 2 -> 1 -> 0
        x2 = torch.randn(n, 1) if int_node != 2 else torch.full((n,1), int_val)
        x1 = (1.5 * x2 + torch.randn(n, 1) * noise_scale) if int_node != 1 else torch.full((n,1), int_val)
        x0 = (1.5 * x1 + torch.randn(n, 1) * noise_scale) if int_node != 0 else torch.full((n,1), int_val)
        data = torch.cat([x0, x1, x2], dim=1)
        c_vec = torch.tensor([0.0, 1.0])
    return data, c_vec, int_node

# --- 3. 究極の動的ポテンシャルモデル ---
class DynamicCausalModel(nn.Module):
    def __init__(self, num_vars=3):
        super().__init__()
        self.num_vars = num_vars
        # 文脈を直接ポテンシャルに変換する線形層 (バイアスなしで、純粋に文脈のみに依存させる)
        self.phi_map = nn.Linear(2, num_vars, bias=False)

    def get_A(self, context_vec, tau=0.1):
        # context_vec (B, 2) -> phi (B, 3)
        phi = self.phi_map(context_vec).unsqueeze(-1)
        # 勾配ベースのA生成
        diff = phi.unsqueeze(2) - phi.unsqueeze(1)
        A = torch.sigmoid(diff.squeeze(-1) / tau)
        mask = (1 - torch.eye(self.num_vars)).to(context_vec.device)
        return A * mask

    def forward(self, x, context_vec, tau=0.1, int_node=None):
        A = self.get_A(context_vec, tau=tau)
        A_used = A.clone()
        if int_node is not None:
            A_used[:, int_node, :] = 0 # 介入ノードへの入力を遮断
        x_pred = torch.matmul(A_used, x.unsqueeze(-1)).squeeze(-1)
        return x_pred, A

# --- 4. 学習プロセス (分離学習) ---
model = DynamicCausalModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

for epoch in range(4000):
    optimizer.zero_grad()
    tau = max(0.01, 1.0 - epoch / 3000)
    
    # NormalデータとReverseデータを1ステップで両方学習 (バッチ内で混合)
    x_n, c_n, i_n = generate_contextual_data(32, 'normal')
    x_r, c_r, i_r = generate_contextual_data(32, 'reverse')
    
    # 文脈をバッチ化
    x_total = torch.cat([x_n, x_r], dim=0)
    c_total = torch.cat([c_n.expand(32, -1), c_r.expand(32, -1)], dim=0)
    
    # 介入ノードはバッチごとに個別に処理するため、ここでは簡易的にループ
    # ※本番ではマスクを個別に作る
    x_pred, A_batch = model(x_total, c_total, tau=tau)
    
    mse = F.mse_loss(x_pred, x_total)
    h_a = dag_constraint(A_batch)
    l1 = 0.5 * A_batch.mean() # スパース性を大幅強化
    
    loss = mse + 100.0 * h_a + l1
    loss.backward()
    optimizer.step()

# --- 5. 結果の視覚化 ---
model.eval()
with torch.no_grad():
    _, A_normal = model(torch.randn(1, 3), torch.tensor([[1.0, 0.0]]), tau=0.01)
    _, A_reverse = model(torch.randn(1, 3), torch.tensor([[0.0, 1.0]]), tau=0.01)



fig, axes = plt.subplots(1, 2, figsize=(12, 5))
labels = ['X0', 'X1', 'X2']
im0 = axes[0].imshow(A_normal[0].numpy(), cmap='Reds', vmin=0, vmax=1)
axes[0].set_title("Context: X0 -> X1 -> X2")
im1 = axes[1].imshow(A_reverse[0].numpy(), cmap='Blues', vmin=0, vmax=1)
axes[1].set_title("Context: X2 -> X1 -> X0")

for ax in axes:
    ax.set_xticks(range(3)); ax.set_xticklabels(labels)
    ax.set_yticks(range(3)); ax.set_yticklabels(labels)
    ax.set_xlabel("Source (Cause)"); ax.set_ylabel("Target (Effect)")

plt.colorbar(im0, ax=axes[0]); plt.colorbar(im1, ax=axes[1])
plt.show()

# 内部ポテンシャルの確認
phi_n = model.phi_map(torch.tensor([[1.0, 0.0]]))
phi_r = model.phi_map(torch.tensor([[0.0, 1.0]]))
print(f"Normal phi (0,1,2): {phi_n.detach().numpy()}")
print(f"Reverse phi (0,1,2): {phi_r.detach().numpy()}")
