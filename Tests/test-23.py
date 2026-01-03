import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. 物理構造を持つ因果ダイナミクスモデル ---
class HybridCausalModel(nn.Module):
    def __init__(self, n_vars=5, d_model=128):
        super().__init__()
        self.n_vars = n_vars
        self.K = 2
        self.raw_S = nn.Parameter(torch.randn(n_vars, n_vars) * 0.5 + 0.5)
        self.raw_phase = nn.Parameter(torch.randn(n_vars, n_vars) * 0.1)
        self.mode_gen = nn.Sequential(
            nn.Linear(25, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, self.K * 2)
        )

    def _get_S_core(self):
        S = torch.tanh(self.raw_S)
        diag_mask = torch.eye(self.n_vars).to(S.device)
        return S * (1 - diag_mask) + diag_mask * 0.95

    def forward(self, x_2ch, history_flat, phi_global, intervention_S=None, jitter=0.0):
        B = x_2ch.shape[0]
        params = self.mode_gen(history_flat).view(B, self.K, 2)
        
        # エッジ位相にジッター成分を付与（CIIがこれを検知する）
        phi_t = torch.tensor([0.0, np.pi/2], device=device).unsqueeze(0) + \
                torch.tanh(params[:, :, 0]) * (np.pi / 12) + jitter
        
        r_mode = torch.stack([1.5 * torch.sigmoid(params[:, 0, 1]), 0.8 * torch.sigmoid(params[:, 1, 1])], dim=1)
        
        S = intervention_S if intervention_S is not None else self._get_S_core()
        self_loop_mask = torch.eye(self.n_vars).to(S.device)
        
        x_real, x_imag = x_2ch[:, :, 0].unsqueeze(1), x_2ch[:, :, 1].unsqueeze(1)
        out_real, out_imag = 0, 0
        for k in range(self.K):
            w_k = (1.0 - self_loop_mask) if k == 0 else self_loop_mask
            theta = self.raw_phase.unsqueeze(0) + phi_t[:, k].view(B, 1, 1)
            A = S.unsqueeze(0) * w_k * r_mode[:, k].view(B, 1, 1)
            out_real += torch.sum(A * (torch.cos(theta) * x_real - torch.sin(theta) * x_imag), dim=2)
            out_imag += torch.sum(A * (torch.sin(theta) * x_real + torch.cos(theta) * x_imag), dim=2)
            
        return torch.stack([out_real, out_imag], dim=-1), phi_t

# --- 2. データ生成 ---
def generate_5d_mechanical_data(batch_size=16, seq_len=64):
    n_vars = 5
    data = torch.zeros(batch_size, seq_len + 5, n_vars).to(device)
    for b in range(batch_size):
        for t in range(1, seq_len + 5):
            p = data[b, t-1]
            c = torch.zeros(n_vars).to(device)
            c[1] = 0.8 * p[0]; c[2] = 0.8 * p[1]; c[3] = 0.8 * p[2]; c[4] = 0.8 * p[3]
            c[1] -= 0.3 * p[4]
            data[b, t] = torch.clamp(0.2 * p + c + torch.randn(n_vars).to(device) * 0.05, -3, 3)
    return (data - data.mean()) / (data.std() + 1e-6)

# --- 3. 指標計算 ---
def calculate_cii(phi_log, T_do=10, delta_T=30):
    phi = phi_log[:, 0] # モード0（主因果伝播）の位相
    dphi = np.diff(phi) # 速度
    window = dphi[T_do:T_do + delta_T]
    return np.mean(np.abs(np.diff(window))) # 加速度（バタつき）の平均

def analyze_node_metrics(vals, phis, T_do=10, delta_T=30):
    metrics = []
    for i in range(vals.shape[1]):
        sig = vals[:, i]
        slope = np.max(np.diff(sig[:T_do+5])) if len(sig) > T_do else 0
        cum = np.cumsum(sig**2)
        t_half = np.where(cum > 0.5 * cum[-1])[0][0] if cum[-1] > 1e-6 else len(sig)
        csi = 0
        if i > 0:
            diff_phi = np.unwrap(phis[:, i]) - np.unwrap(phis[:, i-1])
            csi = np.var(diff_phi[T_do:T_do + delta_T])
        metrics.append({"Node": f"X{i}", "Slope": slope, "t_half": t_half, "CSI": csi})
    return metrics

# --- 4. メイン ---
def main():
    model = HybridCausalModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    print("Training...")
    for epoch in range(1201):
        model.train(); optimizer.zero_grad()
        data_all = generate_5d_mechanical_data()
        x_imag_latent = torch.zeros(16, 5, 1).to(device); phi_latent = torch.tensor([0.0, np.pi/2]).repeat(16, 1).to(device)
        loss = 0
        for t in range(5, 60):
            h_in = data_all[:, t-5:t, :].reshape(16, -1)
            x_in = torch.cat([data_all[:, t, :].unsqueeze(-1), x_imag_latent], dim=-1)
            x_out, phi_latent = model(x_in, h_in, phi_latent)
            x_imag_latent = x_out[:, :, 1].unsqueeze(-1)
            loss += F.mse_loss(x_out[:, :, 0], data_all[:, t+1, :])
        loss.backward(); optimizer.step()

    def run_sim(is_abnormal=False):
        model.eval()
        steps, T_do = 80, 10
        val_log, node_phi_log, edge_phi_log = [], [], []
        x_2ch = torch.zeros(1, 5, 2).to(device); phi_t = torch.tensor([0.0, np.pi/2]).repeat(1, 1).to(device); h_buf = torch.zeros(1, 5, 5).to(device)
        S_base = model._get_S_core().detach()

        with torch.no_grad():
            for s in range(steps):
                if s < T_do: x_2ch[0, 0, 0] = 1.0
                
                S_interv = S_base.clone()
                jitter_val = 0.0
                if is_abnormal and s >= 20:
                    # [1] 物理的切断 (X2 -> X3)
                    S_interv[2, 3] = 0.0
                    # [2] 位相崩壊ジッター (因果慣性の破壊をシミュレート)
                    jitter_val = np.random.normal(0, 0.3)
                
                x_out, phi_t = model(x_2ch, h_buf.view(1, -1), phi_t, intervention_S=S_interv, jitter=jitter_val)
                v = x_out[0, :, 0].clone()
                if s < T_do: v[0] = 1.0
                
                val_log.append(v.cpu().numpy())
                node_phi_log.append(torch.atan2(x_out[0, :, 1], x_out[0, :, 0]).cpu().numpy())
                edge_phi_log.append(phi_t.cpu().numpy()[0])
                
                x_2ch[0, :, 0], x_2ch[0, :, 1] = v, x_out[0, :, 1]
                h_buf = torch.roll(h_buf, -1, dims=1); h_buf[0, -1, :] = v
        return np.array(val_log), np.array(node_phi_log), np.array(edge_phi_log)

    print("Running diagnostics...")
    vn, pn, en = run_sim(False); va, pa, ea = run_sim(True)
    cii_n, cii_a = calculate_cii(en), calculate_cii(ea)
    node_n, node_a = analyze_node_metrics(vn, pn), analyze_node_metrics(va, pa)

    print("\n" + "="*85)
    print(f" [FINAL REPORT] CAUSAL SENSITIVITY   (CII Normal: {cii_n:.5f} / Abnormal: {cii_a:.5f})")
    print("="*85)
    report = []
    for i in range(5):
        report.append({
            "Node": f"X{i}", "Slope(A)": f"{node_a[i]['Slope']:.2f}",
            "t_half(N)": node_n[i]["t_half"], "t_half(A)": node_a[i]["t_half"],
            "Lag_Shift": node_a[i]["t_half"] - node_n[i]["t_half"],
            "CSI(N)": f"{node_n[i]['CSI']:.4f}", "CSI(A)": f"{node_a[i]['CSI']:.4f}",
            "CSI_Ratio": f"{(node_a[i]['CSI']/(node_n[i]['CSI']+1e-6)):.1f}"
        })
    print(pd.DataFrame(report).to_string(index=False))
    print(f"\n[Causal Inertia Index] CII Shift: {cii_a - cii_n:+.5f}")

    # CIIの挙動を可視化（加速度の差）
    plt.figure(figsize=(10, 4))
    plt.plot(np.abs(np.diff(np.diff(en[:, 0]))), label="Normal Edge Inertia (Smooth)")
    plt.plot(np.abs(np.diff(np.diff(ea[:, 0]))), label="Abnormal Edge Inertia (Jittery)", linestyle='--')
    plt.axvline(20, color='r', alpha=0.3, label="Causal Breakpoint")
    plt.title("2-2-3 Causal Inertia (Edge Phase Acceleration)"); plt.legend(); plt.grid(True); plt.show()

if __name__ == "__main__":
    main()
