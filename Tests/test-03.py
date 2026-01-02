import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# --- 1. 時系列データ生成器 ---
def generate_timeseries_causal_data(seq_len, context_type='forward', p_int=0.3):
    noise_scale = 0.02
    data = torch.zeros(seq_len, 3)
    data[0] = torch.randn(3) * 0.1
    
    # 介入の設定
    int_node = torch.randint(0, 3, (1,)).item() if torch.rand(1) < p_int else None
    int_val = 3.0

    for t in range(1, seq_len):
        prev = data[t-1]
        curr = torch.randn(3) * noise_scale
        
        if context_type == 'forward':
            # 自己回帰 + 0->1->2
            curr[0] += 0.9 * prev[0]
            curr[1] += 0.8 * prev[1] + 0.7 * prev[0]
            curr[2] += 0.8 * prev[2] + 0.7 * prev[1]
        else:
            # 自己回帰 + 2->1->0
            curr[2] += 0.9 * prev[2]
            curr[1] += 0.8 * prev[1] + 0.7 * prev[2]
            curr[0] += 0.8 * prev[0] + 0.7 * prev[1]
            
        # 介入の適用
        if int_node is not None:
            curr[int_node] = int_val
            
        data[t] = curr
        
    c_vec = torch.tensor([1.0, 0.0]) if context_type == 'forward' else torch.tensor([0.0, 1.0])
    return data, c_vec, int_node

# --- 2. 時系列動的因果モデル ---
class TimeSeriesCausalModel(nn.Module):
    def __init__(self, num_vars=3):
        super().__init__()
        self.num_vars = num_vars
        # 文脈から直接ポテンシャルを生成 (バイアスなし)
        self.phi_map = nn.Linear(2, num_vars, bias=False)

    def get_A(self, context_vec, tau=0.1):
        # context_vec: (B, 2) -> phi: (B, 3, 1)
        phi = self.phi_map(context_vec).unsqueeze(-1)
        # エントロピー勾配
        diff = phi.unsqueeze(2) - phi.unsqueeze(1)
        A = torch.sigmoid(diff.squeeze(-1) / tau)
        return A # 時系列なので mask (diag=0) はしない（自己回帰を許容）

    def forward(self, data, context_vec, tau=0.1, int_node=None):
        # A: (B, 3, 3)
        A = self.get_A(context_vec, tau=tau)
        
        # 介入ノードがある場合、そのノードへの過去からの流入を遮断
        if int_node is not None:
            A = A.clone()
            A[:, int_node, :] = 0
            # ただし自己回帰(対角)だけは介入によらず存在する可能性があるため、
            # 完全に遮断するかはタスクによりますが、ここでは流入全遮断とします。

        # data: (B, T, 3) に変換して一括計算
        x_prev = data[:, :-1, :] # (B, T-1, 3)
        x_curr_true = data[:, 1:, :] # (B, T-1, 3)
        
        # 行列演算: (B, T-1, 3) @ (B, 3, 3).transpose(1,2)
        # x_curr_pred_i = sum_j A_ij * x_prev_j
        x_curr_pred = torch.matmul(x_prev, A.transpose(1, 2)) 
        
        return x_curr_pred, x_curr_true, A

# --- 3. 学習 ---
model = TimeSeriesCausalModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

for epoch in range(4000):
    optimizer.zero_grad()
    tau = max(0.01, 1.0 - epoch / 3000)
    
    # データをバッチとして生成 (B=1として簡易化)
    d_f, c_f, i_f = generate_timeseries_causal_data(100, 'forward')
    d_b, c_b, i_b = generate_timeseries_causal_data(100, 'backward')
    
    # データの連結 (B=2)
    d_total = torch.stack([d_f, d_b], dim=0)
    c_total = torch.stack([c_f, c_b], dim=0)
    
    # 順伝播
    pred, true, A_batch = model(d_total, c_total, tau=tau)
    
    mse = F.mse_loss(pred, true)
    l1 = 0.2 * A_batch.mean() # スパース性
    
    loss = mse + l1
    loss.backward()
    optimizer.step()

# --- 4. 可視化 (書式維持) ---
model.eval()
with torch.no_grad():
    # ダミーデータを用いて行列を取得
    _, _, An = model(torch.zeros(1, 2, 3), torch.tensor([[1.0, 0.0]]), tau=0.01)
    _, _, Ar = model(torch.zeros(1, 2, 3), torch.tensor([[0.0, 1.0]]), tau=0.01)



fig, axes = plt.subplots(1, 2, figsize=(12, 5))
labels = ['X0', 'X1', 'X2']
im0 = axes[0].imshow(An[0].cpu().numpy(), cmap='Reds', vmin=0, vmax=1)
axes[0].set_title("Time Context: Normal (X0->X1->X2)")
im1 = axes[1].imshow(Ar[0].cpu().numpy(), cmap='Blues', vmin=0, vmax=1)
axes[1].set_title("Time Context: Reverse (X2->X1->X0)")

for ax in axes:
    ax.set_xticks(range(3)); ax.set_xticklabels(labels)
    ax.set_yticks(range(3)); ax.set_yticklabels(labels)
    ax.set_xlabel("Past State (t-1)"); ax.set_ylabel("Current State (t)")

plt.colorbar(im0, ax=axes[0]); plt.colorbar(im1, ax=axes[1])
plt.show()

print(f"Normal phi: {model.phi_map(torch.tensor([[1.0, 0.0]])).detach().numpy()}")
print(f"Reverse phi: {model.phi_map(torch.tensor([[0.0, 1.0]])).detach().numpy()}")
