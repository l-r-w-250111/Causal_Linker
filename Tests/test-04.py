import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. マルチバッチ・データ生成器 ---
def generate_batch_data(batch_size=32, seq_len=100, context_type='forward', p_int=0.3):
    n_vars = 3
    # (Batch, Time, Vars)
    data = torch.zeros(batch_size, seq_len, n_vars).to(device)
    int_nodes = []
    
    for b in range(batch_size):
        int_node = torch.randint(0, n_vars, (1,)).item() if torch.rand(1) < p_int else None
        int_nodes.append(int_node)
        for t in range(1, seq_len):
            prev = data[b, t-1]
            curr = torch.randn(n_vars).to(device) * 0.1
            if context_type == 'forward':
                curr[1] += 1.2 * prev[0]
                curr[2] += 1.2 * prev[1]
            else:
                curr[1] += 1.2 * prev[2]
                curr[0] += 1.2 * prev[1]
            
            if int_node is not None:
                curr[int_node] = 2.0
            data[b, t] = curr
            
    # 全体での標準化
    data = (data - data.mean()) / (data.std() + 1e-6)
    return data, int_nodes

# --- 2. 修正版モデル ---
class CausalAttentionModel(nn.Module):
    def __init__(self, n_vars=3, d_model=128):
        super().__init__()
        self.n_vars = n_vars
        # 特徴量次元をBatchNormの対象にする
        self.projector = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model)
        )
        self.bn = nn.BatchNorm1d(d_model) # 投影後の次元で正規化
        
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        
    def get_A(self, x_curr, x_prev, tau=0.01):
        B, T, V, _ = x_curr.shape
        # (B*T*V, 1) にフラットにして投影
        q_feat = self.projector(x_curr.reshape(-1, 1))
        k_feat = self.projector(x_prev.reshape(-1, 1))
        
        # 投影直後にBNを適用 (B*T*V, d_model)
        q_feat = self.bn(q_feat)
        k_feat = self.bn(k_feat)
        
        Q = self.Wq(q_feat).view(B, T, V, -1)
        K = self.Wk(k_feat).view(B, T, V, -1)
        
        # 内積スコア (B, T, V, V)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (Q.shape[-1]**0.5)
        A = torch.softmax(scores / tau, dim=-1)
        return A

    def forward(self, x, int_nodes=None, tau=0.01):
        # x: (B, T, V)
        x_prev = x[:, :-1, :].unsqueeze(-1) # (B, T-1, V, 1)
        x_curr = x[:, 1:, :].unsqueeze(-1)  # (B, T-1, V, 1)
        
        A = self.get_A(x_curr, x_prev, tau=tau)
        
        A_final = A.clone()
        if int_nodes is not None:
            for b, idx in enumerate(int_nodes):
                if idx is not None:
                    A_final[b, :, idx, :] = 0
            
        x_pred = torch.matmul(A_final, x_prev).squeeze(-1)
        return x_pred, x[:, 1:, :], A

# --- 3. 学習 ---
model = CausalAttentionModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

for epoch in range(2001):
    optimizer.zero_grad()
    
    # バッチサイズを増やして統計的な信号を安定させる
    df, _ = generate_batch_data(16, 64, 'forward', p_int=0)
    dr, _ = generate_batch_data(16, 64, 'reverse', p_int=0)
    ddo, i_nodes = generate_batch_data(16, 64, 'forward', p_int=1.0)
    
    tau = 0.05
    pf, tf, Af = model(df, tau=tau)
    pr, tr, Ar = model(dr, tau=tau)
    _, _, Ado = model(ddo, int_nodes=i_nodes, tau=tau)
    
    loss_mse = F.mse_loss(pf, tf) + F.mse_loss(pr, tr)
    
    # 介入ノードに対する不変性と遮断の損失
    # バッチごとの介入ノードに正しくアクセス
    delta_parts = []
    for b, idx in enumerate(i_nodes):
        delta_parts.append(torch.norm(Af[b, :, idx, :] - Ado[b, :, idx, :]) + torch.norm(Ado[b, :, idx, :]))
    loss_delta = 5.0 * torch.stack(delta_parts).mean()
    
    # DAG & L1 (0.333への収束を阻む)
    loss_l1 = 0.2 * (Af.abs().sum() + Ar.abs().sum()) / (Af.shape[0]*Af.shape[1])
    
    loss = loss_mse + loss_delta + loss_l1
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | MSE: {loss_mse.item():.4f} | Delta: {loss_delta.item():.4f}")

# --- 4. 結果表示 ---
model.eval()
with torch.no_grad():
    df_f, _ = generate_batch_data(1, 200, 'forward', p_int=0)
    _, _, Af_final = model(df_f, tau=0.05)
    df_r, _ = generate_batch_data(1, 200, 'reverse', p_int=0)
    _, _, Ar_final = model(df_r, tau=0.05)

def print_df(A, title):
    avg = A.mean(0).mean(0).cpu().numpy() # BatchとTimeで平均
    print(f"\n### {title} ###")
    print(pd.DataFrame(avg, index=['To X0','To X1','To X2'], columns=['From X0','From X1','From X2']).round(3))

print_df(Af_final, "Forward Matrix")
print_df(Ar_final, "Reverse Matrix")
