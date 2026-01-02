import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. データ生成 (変更なし) ---
def generate_batch_data(batch_size=32, seq_len=100, context_type='forward', p_int=0.3):
    n_vars = 3
    data = torch.zeros(batch_size, seq_len, n_vars).to(device)
    int_nodes = []
    for b in range(batch_size):
        int_node = torch.randint(0, n_vars, (1,)).item() if torch.rand(1) < p_int else None
        int_nodes.append(int_node)
        for t in range(1, seq_len):
            prev = data[b, t-1]
            curr = torch.randn(n_vars).to(device) * 0.1
            if context_type == 'forward':
                curr[1] += 1.5 * prev[0] # 信号をより強化
                curr[2] += 1.5 * prev[1]
            else:
                curr[1] += 1.5 * prev[2]
                curr[0] += 1.5 * prev[1]
            if int_node is not None:
                curr[int_node] = 2.0
            data[b, t] = curr
    return (data - data.mean()) / (data.std() + 1e-6), int_nodes

# --- 2. 構造的アテンション (Softmax排除型) ---
class StructuralAttention(nn.Module):
    def __init__(self, n_vars=3, d_model=256):
        super().__init__()
        self.n_vars = n_vars
        self.projector = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model)
        )
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        
    def get_A(self, x_curr, x_prev):
        B, T, V, _ = x_curr.shape
        q_feat = self.projector(x_curr.reshape(-1, 1)).view(B, T, V, -1)
        k_feat = self.projector(x_prev.reshape(-1, 1)).view(B, T, V, -1)
        
        Q = self.Wq(q_feat)
        K = self.Wk(k_feat)
        
        # 内積スコア
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (Q.shape[-1]**0.5)
        
        # [改良] Softmaxを排除し、ReLUによるスパースな活性化を採用
        # これにより、負の相関や無関係なノイズを「0」に切り捨てる
        A = F.relu(scores) 
        
        # 行正規化 (因果強度の合計を1にする)
        A = A / (A.sum(-1, keepdim=True) + 1e-6)
        return A

    def forward(self, x, int_nodes=None):
        x_prev = x[:, :-1, :].unsqueeze(-1)
        x_curr = x[:, 1:, :].unsqueeze(-1)
        A = self.get_A(x_curr, x_prev)
        
        A_final = A.clone()
        if int_nodes is not None:
            for b, idx in enumerate(int_nodes):
                if idx is not None: A_final[b, :, idx, :] = 0
            
        x_pred = torch.matmul(A_final, x_prev).squeeze(-1)
        return x_pred, x[:, 1:, :], A

# --- 3. 学習 ---
model = StructuralAttention().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3001):
    optimizer.zero_grad()
    df, _ = generate_batch_data(16, 64, 'forward', p_int=0)
    dr, _ = generate_batch_data(16, 64, 'reverse', p_int=0)
    ddo, i_nodes = generate_batch_data(16, 64, 'forward', p_int=1.0)
    
    pf, tf, Af = model(df)
    pr, tr, Ar = model(dr)
    _, _, Ado = model(ddo, int_nodes=i_nodes)
    
    loss_mse = F.mse_loss(pf, tf) + F.mse_loss(pr, tr)
    
    # DAG制約: 指数トレースを用いるが、2乗ではなく絶対値でスパース性を支援
    def calculate_dag(A):
        A_m = A.mean(dim=(0, 1))
        return torch.trace(torch.matrix_exp(A_m)) - 3

    loss_dag = 5.0 * (calculate_dag(Af) + calculate_dag(Ar))
    
    # 差分介入損失 (不変性の担保)
    loss_delta = 0.0
    for b, idx in enumerate(i_nodes):
        loss_delta += torch.norm(Af[b, :, idx, :] - Ado[b, :, idx, :]) + torch.norm(Ado[b, :, idx, :])
    
    # L1正則化 (スパース性の強制)
    loss_l1 = 0.1 * (Af.mean() + Ar.mean())
    
    loss = loss_mse + loss_dag + (loss_delta / 16) + loss_l1
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | MSE: {loss_mse.item():.4f} | DAG: {loss_dag.item():.4f}")

# --- 4. 結果表示 ---
model.eval()
with torch.no_grad():
    df_f, _ = generate_batch_data(1, 200, 'forward', p_int=0)
    _, _, Af_final = model(df_f)
    df_r, _ = generate_batch_data(1, 200, 'reverse', p_int=0)
    _, _, Ar_final = model(df_r)

def print_df(A, title):
    avg = A.mean(dim=(0, 1)).cpu().numpy()
    print(f"\n### {title} ###")
    print(pd.DataFrame(avg, index=['To X0','To X1','To X2'], columns=['From X0','From X1','From X2']).round(3))

print_df(Af_final, "Structural Forward Matrix")
print_df(Ar_final, "Structural Reverse Matrix")
