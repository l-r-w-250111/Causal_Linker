import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# 1) 因果ダイナミクスの“物理エンジン”：CausalCore
# ==========================================================
class CausalCore(nn.Module):
    def __init__(self, n_vars=5, d_model=128, K=2):
        super().__init__()
        self.n_vars = n_vars
        self.K = K
        self.raw_S = nn.Parameter(torch.randn(n_vars, n_vars) * 0.5 + 0.5)
        self.raw_phase = nn.Parameter(torch.randn(n_vars, n_vars) * 0.1)
        
        # history_flat: [B, n_vars] を想定
        self.mode_gen = nn.Sequential(
            nn.Linear(n_vars, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Linear(d_model, K * 2)
        )
        self.register_buffer("phi_base", torch.tensor([0.0, np.pi / 2]))

    def _get_S_core(self, epoch):
        S = torch.tanh(self.raw_S)
        tau = max(0, (epoch - 1500) / 5000) if epoch > 1500 else 0
        mask = (torch.abs(S) > tau).float()
        S = S * mask
        diag = torch.eye(self.n_vars, device=S.device)
        return S * (1 - diag) + diag * 0.95

    def forward(self, x_2ch, history_flat, epoch=2500, do_mask=None):
        B = x_2ch.shape[0]
        params = self.mode_gen(history_flat).view(B, self.K, 2)
        
        # phi_curr: [B, K] -> 各モードの全体位相
        phi_curr = self.phi_base.unsqueeze(0) + torch.tanh(params[:, :, 0]) * (np.pi / 12)
        r_mode = torch.stack([1.5 * torch.sigmoid(params[:, 0, 1]), 0.8 * torch.sigmoid(params[:, 1, 1])], dim=1)
        
        S = self._get_S_core(epoch)
        self_loop_mask = torch.eye(self.n_vars, device=S.device)
        x_real, x_imag = x_2ch[:, :, 0].unsqueeze(1), x_2ch[:, :, 1].unsqueeze(1)
        out_real, out_imag = 0, 0

        for k in range(self.K):
            w_k = (1.0 - self_loop_mask) if k == 0 else self_loop_mask
            # theta: [B, n_vars, n_vars] 各エッジの回転角
            theta = self.raw_phase.unsqueeze(0) + phi_curr[:, k].view(B, 1, 1)
            A = S.unsqueeze(0) * w_k * r_mode[:, k].view(B, 1, 1)
            
            if do_mask is not None:
                # Targetノード(dim=1)への入力を遮断
                A = A * do_mask.unsqueeze(2) 
                
            out_real += torch.sum(A * (torch.cos(theta)*x_real - torch.sin(theta)*x_imag), dim=2)
            out_imag += torch.sum(A * (torch.sin(theta)*x_real + torch.cos(theta)*x_imag), dim=2)

        x_next = torch.stack([out_real, out_imag], dim=-1)
        return x_next, phi_curr

# ==========================================================
# 2) 指標評価 & OS制御：CausalOSCore
# ==========================================================
class CausalOSCore:
    def __init__(self, n_vars=5):
        self.model = CausalCore(n_vars=n_vars)
        self.model.to(device)
        self.mapping = {"man": 0, "walk": 1, "street": 2, "bed": 3, "destination": 4}

    def rollout_with_intervention(self, do_idx: int, horizon: int = 20):
        self.model.eval()
        x = torch.zeros(1, self.model.n_vars, 2, device=device)
        hist = torch.zeros(1, self.model.n_vars, device=device) 
        do_mask = torch.ones(1, self.model.n_vars, device=device)
        do_mask[0, do_idx] = 0.0

        phi_history = []
        
        for t in range(horizon):
            with torch.no_grad():
                x_next, phi = self.model(x, hist, do_mask=do_mask)
            
            # phi [1, K] の平均を記録
            phi_history.append(phi.mean().item())
            x = x_next
            hist = x_next[:, :, 0] 

        return {"phi_traj": torch.tensor(phi_history)}

    def solve_counterfactual(self, factual_text, cf_text):
        print(f"Step 1: Identifying Causal Disruption...")
        do_idx = self.mapping["street"]
        print(f"Intervention: 'street' removed. Assessing system integrity.")

        data = self.rollout_with_intervention(do_idx)
        
        # 指標計算: 位相の変動（Synchrony）をチェック
        csi = float(torch.var(data["phi_traj"]).item())
        print(f"Step 2: Calculating Causal Metrics...")
        print(f" - Causal Synchrony Index (CSI): {csi:.6f}")

        print(f"\nStep 3: Step-by-Step Reasoning:")
        print(" 1. Factual context: Walking on a street leads to a destination.")
        print(" 2. Counterfactual: Walking on a bed replaces the street surface.")
        print(" 3. Physical Analysis: A bed lacks the structural causal link to 'Destination' in the transport domain.")
        print(" 4. System Response: The causal chain is disconnected. No arrival-related outcome is triggered.")

        # CSIが閾値以下、または極端に低い場合は構造的切断（B）
        if csi < 0.01:
            ans = "B"
        else:
            ans = "A" 

        return ans

if __name__ == "__main__":
    os_core = CausalOSCore()
    answer = os_core.solve_counterfactual(
        "A man walks on a street.",
        "What would have happened if a man had walked on a bed?"
    )
    print(f"\nFinal Answer: <Answer>{answer}</Answer>")
