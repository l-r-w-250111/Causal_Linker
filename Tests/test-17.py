import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. データ生成 (5次元・二次遅れ系) ---
def generate_5d_mechanical_data(batch_size=32, seq_len=100):
    n_vars = 5
    data = torch.zeros(batch_size, seq_len, n_vars).to(device)
    for b in range(batch_size):
        for t in range(1, seq_len):
            p = data[b, t-1]
            c = torch.randn(n_vars).to(device) * 0.03
            c[1] += 1.4 * p[0] 
            c[2] += 1.4 * p[1] 
            c[3] += 1.4 * p[2] 
            c[4] += 1.4 * p[3] 
            c[2] -= 0.8 * p[4]  # 負の相互作用
            c[1] -= 0.6 * p[4]
            c += 0.2 * p 
            data[b, t] = torch.clamp(c, -5, 5)
    return (data - data.mean()) / (data.std() + 1e-6)

# --- 2. エッジ特定型モデル (Edge-Specific Complex Model) ---
class EdgeSpecificComplexModel(nn.Module):
    def __init__(self, n_vars=5, d_model=128):
        super().__init__()
        self.n_vars = n_vars
        self.K = 2
        
        # [指摘3] Sは符号付き構造行列 (正則化から分離)
        self.S = nn.Parameter(torch.randn(n_vars, n_vars) * 0.1)
        
        # [指摘1] エッジ×モード単位の減衰パラメータ
        self.gamma_ij_k = nn.Parameter(torch.randn(n_vars, n_vars, self.K) + 2.0)
        
        # [指摘2] エッジ単位の位相偏差
        self.delta_phi_ij_k = nn.Parameter(torch.randn(n_vars, n_vars, self.K) * 0.05)
        
        self.mode_gen = nn.Sequential(
            nn.Linear(n_vars * 5, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, self.K * 2) 
        )
        self.path_weights = nn.Parameter(torch.randn(n_vars, n_vars, self.K))

    def forward(self, x_2ch, history, phi_global):
        B = x_2ch.shape[0]
        params = self.mode_gen(history).view(B, self.K, 2)
        
        d_phi_global = torch.tanh(params[:, :, 0]) * (np.pi / 24) 
        phi_curr = phi_global + d_phi_global
        r_mode = torch.sigmoid(params[:, :, 1]) # モード単位のベースゲイン
        
        w = torch.softmax(self.path_weights, dim=-1)
        
        # [指摘1] エッジ・モード別減衰の合成
        sigma_ij_k = torch.sigmoid(self.gamma_ij_k)
        damp_ij = torch.einsum('ijk,ijk->ij', w, sigma_ij_k).unsqueeze(0)
        
        # [指摘2] エッジ偏差を含む合成位相
        theta_ij = torch.einsum('ijk,bk->bij', w, phi_curr) + \
                   torch.einsum('ijk,ijk->ij', w, self.delta_phi_ij_k).unsqueeze(0)
        
        # 伝達強度 r_ij
        r_ij = torch.einsum('ijk,bk->bij', w, r_mode)
        
        # 因果行列 A_gain の算出
        A_raw = r_ij * self.S.unsqueeze(0)
        A_gain = damp_ij * (A_raw / (1.0 + A_raw.abs()))
        
        cos_t, sin_t = torch.cos(theta_ij), torch.sin(theta_ij)
        x_real, x_imag = x_2ch[:, :, 0].unsqueeze(1), x_2ch[:, :, 1].unsqueeze(1)
        
        next_real = torch.sum(A_gain * (cos_t * x_real - sin_t * x_imag), dim=2)
        next_imag = torch.sum(A_gain * (sin_t * x_real + cos_t * x_imag), dim=2)
        
        return torch.stack([next_real, next_imag], dim=-1), phi_curr, A_gain

# --- 3. 学習プロセス ---
def train_edge_specific():
    n_vars = 5
    model = EdgeSpecificComplexModel(n_vars).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 因果マスク (事前知識としてのトポロジー)
    mask = torch.zeros(n_vars, n_vars).to(device)
    for i in range(n_vars - 1): mask[i+1, i] = 1.0 
    mask[1, 4] = 1.0; mask[2, 4] = 1.0 
    mask += torch.eye(n_vars).to(device)

    print("Training Edge-Specific Model...")
    for epoch in range(2501):
        model.train()
        optimizer.zero_grad()
        data = generate_5d_mechanical_data(32, 64)
        B, T, V = data.shape
        x_pad = torch.cat([torch.zeros(B, 4, V).to(device), data], dim=1)
        x_imag_latent = torch.zeros(B, V, 1).to(device)
        phi_latent = torch.tensor([0.0, np.pi/2]).repeat(B, 1).to(device)
        
        loss_mse = 0
        loss_phi_const = 0
        total_A_gain = 0
        
        for t in range(T-1):
            x_in = torch.cat([data[:, t, :].unsqueeze(-1), x_imag_latent], dim=-1)
            x_out, phi_latent, A_gain = model(x_in, x_pad[:, t:t+5, :].reshape(B, -1), phi_latent)
            x_imag_latent = x_out[:, :, 1].unsqueeze(-1)
            loss_mse += F.mse_loss(x_out[:, :, 0], data[:, t+1, :])
            total_A_gain += A_gain.abs().mean()
            
            if epoch < 500:
                loss_phi_const += 0.05 * torch.pow(phi_latent[:, 1] - np.pi/2, 2).mean()

        # [指摘3] スパース正則化を A_gain に対して適用 (マスク外の因果強度を削る)
        # A_gain はバッチ内の平均的な強度を使用
        A_final = A_gain[0] # 直近のバッチサンプル
        loss_sparse = torch.mean(torch.abs(A_final) * (1 - mask))
        
        loss = (loss_mse / T) + loss_phi_const + 0.05 * loss_sparse
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:4d} | MSE: {loss_mse.item()/T:.4f} | Sparse(A): {loss_sparse.item():.4f}")
    return model

# --- 4. 結果の解析 ---
def evaluate(model):
    model.eval()
    with torch.no_grad():
        # 代表的な A_gain を取得
        dummy_hist = torch.zeros(1, 25).to(device)
        dummy_x = torch.zeros(1, 5, 2).to(device)
        dummy_phi = torch.tensor([[0.0, np.pi/2]]).to(device)
        _, _, A_gain = model(dummy_x, dummy_hist, dummy_phi)
        
        a_matrix = A_gain[0].cpu().numpy()
        s_matrix = model.S.detach().cpu().numpy()
        
        print("\n[Table 1] Edge-Specific Causal Strength A (After Damping):")
        labels = [f"X{i}" for i in range(5)]
        print("      " + "".join([f"{l:<8}" for l in labels]))
        for i, row in enumerate(a_matrix):
            print(f"{labels[i]}: " + "".join([f"{v:>8.4f}" for v in row]))

        print("\n[Table 2] Structural Matrix S (Direction/Sign Only):")
        for i, row in enumerate(s_matrix):
            print(f"{labels[i]}: " + "".join([f"{v:>8.4f}" for v in row]))

        print("\n[Table 3] Rollout Dynamics (Impulse on X0):")
        x_2ch = torch.zeros(1, 5, 2).to(device); x_2ch[0, 0, 0] = 3.0
        phi_t = torch.tensor([0.0, np.pi/2]).repeat(1, 1).to(device)
        h_buf = torch.zeros(1, 5, 5).to(device)
        
        for s in range(15):
            x_out, phi_t, _ = model(x_2ch, h_buf.view(1, -1), phi_t)
            v = x_out[0, :, 0].cpu().numpy()
            print(f"Step {s:2d} | {' '.join([f'X{i}:{v[i]:>6.2f}' for i in range(5)])}")
            if s == 0: x_2ch[0, 0, 0] = 0.0
            else: x_2ch[0, :, 0] = x_out[0, :, 0]
            x_2ch[0, :, 1] = x_out[0, :, 1]
            h_buf = torch.roll(h_buf, -1, dims=1); h_buf[0, 4, :] = torch.tensor(v).to(device)

if __name__ == "__main__":
    m = train_edge_specific()
    evaluate(m)
