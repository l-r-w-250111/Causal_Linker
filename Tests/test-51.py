import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re, json, os, itertools
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# 0) 便利関数：位相指標
# ==========================================================

def compute_CSI(phi_traj: torch.Tensor):
    # phi_traj: [T, N] (時刻 x ノード)
    if phi_traj.shape[1] < 2: return 0.0 # 比較対象がない場合
    mean_phi = phi_traj.mean(dim=1, keepdim=True)
    var_phi = ((phi_traj - mean_phi)**2).mean(dim=1)
    return var_phi.mean().item()

def compute_CII(phi: torch.Tensor):
    # phi: [T] 代表位相
    if len(phi) < 3: return 0.0
    d2 = phi[2:] - 2*phi[1:-1] + phi[:-2]
    return (d2**2).mean().item()

def compute_CII_prime(phi_edge, phi_node, alpha=0.5):
    def d2_abs(x):
        if len(x) < 3: return 0.0
        d2 = x[2:] - 2*x[1:-1] + x[:-2]
        return torch.abs(d2).mean().item()
    return alpha * d2_abs(phi_edge) + (1-alpha) * d2_abs(phi_node)


# ==========================================================
# 1) 言語→複素表現
# ==========================================================

class ComplexLanguageEncoder:
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct", dim=64):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # デバイスを明示的に指定
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto"
        )
        self.device = self.model.device
        self.dim = dim
        self.proj = self._init_structural_space()

    def _init_structural_space(self):
        V = self.tokenizer.vocab_size
        P = torch.randn((V, self.dim), device=self.device, dtype=torch.float16)
        return P / (torch.norm(P, dim=1, keepdim=True) + 1e-9)

    def embed(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        t = torch.tensor(ids, device=self.device)
        pos = torch.arange(len(ids), device=self.device).float()
        angles = pos * 0.5
        cos_t, sin_t = torch.cos(angles).unsqueeze(1), torch.sin(angles).unsqueeze(1)
        base = self.proj[t]
        real = (base * cos_t).sum(0)
        imag = (base * sin_t).sum(0)
        norm = torch.sqrt(real.pow(2).sum() + imag.pow(2).sum()) + 1e-9
        return real/norm, imag/norm


# ==========================================================
# 3) 複素因果ダイナミクス
# ==========================================================

class CausalCore(nn.Module):
    def __init__(self, n_vars=5, d_model=128, K=2):
        super().__init__()
        self.n_vars = n_vars
        self.K = K
        self.raw_S = nn.Parameter(torch.randn(n_vars, n_vars)*0.5 + 0.5)
        self.raw_phase = nn.Parameter(torch.randn(n_vars, n_vars)*0.1)

        self.mode_gen = nn.Sequential(
            nn.Linear(n_vars, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Linear(d_model, K*2)
        )
        # phi_baseを正しいデバイスに配置
        self.register_buffer("phi_base", torch.tensor([0.0, np.pi/2]))

    def _get_S_core(self, epoch):
        S = torch.tanh(self.raw_S)
        tau = max(0, (epoch-1500)/5000) if epoch>1500 else 0
        mask = (torch.abs(S)>tau).float()
        S = S*mask
        diag = torch.eye(self.n_vars, device=S.device)
        return S*(1-diag) + diag*0.95

    def forward(self, x_2ch, history_flat, epoch=2500, do_mask=None):
        B = x_2ch.shape[0]
        params = self.mode_gen(history_flat).view(B, self.K, 2)

        phi_curr = self.phi_base.unsqueeze(0) + torch.tanh(params[:,:,0])*(np.pi/12)
        r_mode = torch.stack([
            1.5*torch.sigmoid(params[:,0,1]),
            0.8*torch.sigmoid(params[:,1,1])
        ], dim=1)

        S = self._get_S_core(epoch)
        x_real = x_2ch[:,:,0].unsqueeze(1)
        x_imag = x_2ch[:,:,1].unsqueeze(1)

        out_r, out_i = 0,0
        for k in range(self.K):
            # thetaのデバイス合わせ
            theta = self.raw_phase.to(S.device).unsqueeze(0) + phi_curr[:,k].view(B,1,1)
            A = S.unsqueeze(0) * r_mode[:,k].view(B,1,1)

            if do_mask is not None:
                A = A * do_mask.unsqueeze(2)

            out_r += torch.sum(A*(torch.cos(theta)*x_real - torch.sin(theta)*x_imag), dim=2)
            out_i += torch.sum(A*(torch.sin(theta)*x_real + torch.cos(theta)*x_imag), dim=2)

        return torch.stack([out_r, out_i], dim=-1), phi_curr


# ==========================================================
# 4) 統合CausalOS
# ==========================================================

class CausalOS:
    def __init__(self):
        self.encoder = ComplexLanguageEncoder()
        self.model = CausalCore(n_vars=5).to(device)
        self.nodes = ["man","walk","street","bed","destination"]
        self.mapping = {n:i for i,n in enumerate(self.nodes)}

    def rollout(self, do_idx, horizon=20):
        # 推論時は必ず eval() にする（BatchNormのエラー回避）
        self.model.eval()
        
        x = torch.zeros(1,5,2,device=device)
        hist = torch.zeros(1,5,device=device)
        do_mask = torch.ones(1,5,device=device)
        do_mask[0,do_idx] = 0.0

        phi_node_list = []
        phi_edge_list = []

        with torch.no_grad():
            for t in range(horizon):
                x_next, phi = self.model(x, hist, do_mask=do_mask)
                phi_node_list.append(phi.mean().item())
                # モード間の位相差をエッジ活動とする
                phi_edge_list.append(phi.std().item() if phi.shape[1] > 1 else 0.0)
                x = x_next
                hist = x_next[:,:,0]

        return torch.tensor(phi_node_list), torch.tensor(phi_edge_list)

    def solve_counterfactual(self, factual, cf):
        print("=== CausalOS Inference ===")
        # street -> bed への介入
        do_idx = self.mapping["street"]
        phi_node, phi_edge = self.rollout(do_idx)

        # 指標計算
        CSI = compute_CSI(phi_node.unsqueeze(1))
        CII = compute_CII(phi_node)
        CIIp = compute_CII_prime(phi_edge, phi_node, alpha=0.5)

        print(f"CSI  = {CSI:.6f}")
        print(f"CII  = {CII:.6f}")
        print(f"CII' = {CIIp:.6f}")

        # 判定ロジック：CSI/CIIが極めて低い（因果連鎖の消失）場合はB
        if CSI < 0.001: 
            return "B"
        elif CIIp > 0.1: # 閾値はモデルの初期値に依存
            return "A"
        else:
            return "C"

if __name__ == "__main__":
    osys = CausalOS()
    ans = osys.solve_counterfactual(
        "A man walks on a street.",
        "What if he walked on a bed?"
    )
    print(f"\nFinal Answer: <Answer>{ans}</Answer>")
