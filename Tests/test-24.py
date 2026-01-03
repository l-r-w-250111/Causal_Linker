import torch
import torch.nn as nn
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ComparisonCausalModel(nn.Module):
    def __init__(self, n_vars=5):
        super().__init__()
        self.omega = nn.Parameter(torch.linspace(0.2, 0.6, n_vars))
        self.raw_S = nn.Parameter(torch.ones(n_vars, n_vars) * 0.5)

    def forward(self, x_complex, intervention_S=None, jitter=0.0):
        S = intervention_S if intervention_S is not None else torch.tanh(self.raw_S)
        real, imag = x_complex[..., 0], x_complex[..., 1]
        theta_rot = self.omega.view(1, -1).to(device)
        
        # 自律回転
        new_real = real * torch.cos(theta_rot) - imag * torch.sin(theta_rot)
        new_imag = real * torch.sin(theta_rot) + imag * torch.cos(theta_rot)
        
        # 因果伝播（位相引き込み）
        phi_edge = torch.tensor([np.pi / 4 + jitter], device=device)
        trans_real = torch.matmul(new_real, S)
        trans_imag = torch.matmul(new_imag, S)
        
        out_real = trans_real * torch.cos(phi_edge) - trans_imag * torch.sin(phi_edge)
        out_imag = trans_real * torch.sin(phi_edge) + trans_imag * torch.cos(phi_edge)
        
        return torch.tanh(torch.stack([out_real, out_imag], dim=-1) * 1.3)

def run_comparison():
    model = ComparisonCausalModel().to(device)
    steps, T_fail = 100, 30

    def simulate_case(case_type='A', is_fail=False):
        val_log, n_phi_log, e_phi_log = [], [], []
        x_c = torch.zeros(1, 5, 2).to(device)
        x_c[0, 0, 0] = 1.0 # 加熱開始
        
        # 因果行列の設定
        S = torch.zeros(5, 5).to(device)
        if case_type == 'A':
            S[0, 3] = 0.9 # Heater -> Temp (直接)
        else:
            S[0, 1], S[1, 2], S[2, 3] = 0.8, 0.8, 0.8 # Heater -> Energy -> Motion -> Temp
        for i in range(5): S[i, i] = 0.5

        for s in range(steps):
            if s < 20: x_c[0, 0, 0] = 1.0
            
            curr_S = S.clone()
            jitter = 0.0
            if is_fail and s >= T_fail:
                # 介入：ヒーター出力後の「伝達プロセス」を不安定化
                target_idx = (0, 3) if case_type == 'A' else (1, 2)
                curr_S[target_idx] *= 0.1
                jitter = np.random.normal(0, 0.7)
            
            x_c = model(x_c, intervention_S=curr_S, jitter=jitter)
            val_log.append(x_c[0, :, 0].detach().cpu().numpy())
            n_phi_log.append(torch.atan2(x_c[0, :, 1], x_c[0, :, 0]).detach().cpu().numpy())
            e_phi_log.append(np.pi/4 + jitter)

        v_arr, p_arr = np.array(val_log), np.array(n_phi_log)
        win = slice(T_fail, T_fail + 40)
        
        # 指標
        cii = np.mean(np.abs(np.diff(np.diff(e_phi_log))))
        # CSI: Aは直接なのでX0-X3、Bは連鎖を象徴するX1-X2の同期を見る
        idx1, idx2 = (0, 3) if case_type == 'A' else (1, 2)
        csi = np.var(np.unwrap(p_arr[win, idx2]) - np.unwrap(p_arr[win, idx1]))
        t_h = np.where(np.cumsum(v_arr[:, 3]**2) > 0.5 * np.sum(v_arr[:, 3]**2))[0][0]
        
        return cii, csi, t_h

    # 全4パターン実行
    res = {}
    for case in ['A', 'B']:
        for fail in [False, True]:
            res[f"{case}_{fail}"] = simulate_case(case, fail)

    print("\n" + "="*95)
    print(" [FINAL DEMO] RESOLUTION COMPARISON: SYSTEM A vs SYSTEM B")
    print("="*95)
    report = [
        ["A (Direct)", res["A_False"][0], res["A_True"][0], res["A_False"][1], res["A_True"][1], res["A_True"][2]],
        ["B (Physical)", res["B_False"][0], res["B_True"][0], res["B_False"][1], res["B_True"][1], res["B_True"][2]]
    ]
    headers = ["System Type", "CII(Norm)", "CII(Fail)", "CSI(Norm)", "CSI(Fail)", "t_half(Fail)"]
    print(pd.DataFrame(report, columns=headers).to_string(index=False, formatters={
        "CII(Norm)": "{:.4f}".format, "CII(Fail)": "{:.4f}".format,
        "CSI(Norm)": "{:.4f}".format, "CSI(Fail)": "{:.4f}".format
    }))

run_comparison()
