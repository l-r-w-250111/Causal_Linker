import torch
import torch.nn.functional as F
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

class SoftCausalLogitProcessor(LogitsProcessor):
    def __init__(self, tokenizer, potency, alpha=30.0):
        # 複数の「Yes」「No」トークン（大文字小文字、スペース付き）を取得
        self.yes_ids = self._get_tokens(tokenizer, ["Yes", "yes", " Yes", " yes"])
        self.no_ids = self._get_tokens(tokenizer, ["No", "no", " No", " no"])
        self.potency = potency
        self.alpha = alpha

    def _get_tokens(self, tokenizer, words):
        ids = []
        for w in words:
            token = tokenizer.encode(w, add_special_tokens=False)
            if token: ids.append(token[-1])
        return list(set(ids))

    def __call__(self, input_ids, scores):
        # Potencyに基づき確率場を傾ける
        bias = self.alpha * self.potency
        for tid in self.yes_ids: scores[:, tid] += bias
        for tid in self.no_ids: scores[:, tid] -= bias
        return scores

class CausalOS_v32_1:
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct"):
        print("Initializing Causal OS v32.1 [Logit Sync Edition]...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        self.device = self.model.device
        self.dim = 64
        
        self.S_edges = [
            ("Tuberculosis", "Visit to Asia"), ("Lung Cancer", "Smoking"),
            ("Lung Cancer", "Bronchitis"), ("Visit to Asia", "Either"),
            ("Smoking", "Either"), ("Either", "X-ray"),
            ("Either", "Dyspnea"), ("Bronchitis", "Dyspnea")
        ]
        self.dist_map = self._build_distance_map(self.S_edges)
        self.proj = self._init_structural_space()

    def _build_distance_map(self, edges):
        nodes = list(set([u for u, v in edges] + [v for u, v in edges]))
        dist = {n1: {n2: float('inf') for n2 in nodes} for n1 in nodes}
        for n in nodes: dist[n][n] = 0
        for u, v in edges: dist[u][v] = 1
        for k in nodes:
            for i in nodes:
                for j in nodes:
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
        return dist

    def _init_structural_space(self):
        vocab_size = self.tokenizer.vocab_size
        proj = torch.randn((vocab_size, self.dim), device=self.device, dtype=torch.float16)
        return proj / (torch.norm(proj, dim=1, keepdim=True) + 1e-9)

    def get_sync(self, h, t):
        h_ids = torch.tensor(self.tokenizer.encode(h, add_special_tokens=False), device=self.device)
        t_ids = torch.tensor(self.tokenizer.encode(t, add_special_tokens=False), device=self.device)
        v_h = torch.mean(self.proj[h_ids], dim=0)
        v_t = torch.mean(self.proj[t_ids], dim=0)
        return F.cosine_similarity(v_h.unsqueeze(0), v_t.unsqueeze(0)).item()

    def run_benchmark(self, tests):
        bg = ", ".join([f"{u} causes {v}" for u, v in self.S_edges])
        
        print(f"\n{'='*140}")
        print(f"{'Path Candidate':<30} | {'Sync':<8} | {'L':<2} | {'Potent':<8} | {'Raw':<5} | {'Interlock'}")
        print(f"{'-'*140}")

        for h, t in tests:
            raw_sync = self.get_sync(h, t)
            L = self.dist_map.get(h, {}).get(t, float('inf'))
            is_valid = L < float('inf') and L > 0
            
            # (3) 対数減衰モデル（遠くても論理の灯を消さない）
            boost_base = 0.6
            if is_valid:
                # L=1: 0.60, L=2: 0.43, L=3: 0.35 ... 
                potent = raw_sync + (boost_base / math.log(math.e + L - 1))
            else:
                rev_L = self.dist_map.get(t, {}).get(h, float('inf'))
                is_reverse = rev_L < float('inf')
                # 逆方向または無関係
                penalty = (boost_base / math.log(math.e + rev_L - 1)) if is_reverse else boost_base
                potent = raw_sync - penalty

            prompt = f"Based on the following knowledge: {bg}.\nDoes {h} cause {t}?\nAnswer Yes or No.\nAnswer:"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                # 介入なし (1トークン限定)
                raw_ids = self.model.generate(**inputs, max_new_tokens=1, pad_token_id=self.tokenizer.eos_token_id)
                raw_ans = self.tokenizer.decode(raw_ids[0], skip_special_tokens=True).split("Answer:")[-1].strip()

                # (2) Potency連動介入
                proc = LogitsProcessorList([SoftCausalLogitProcessor(self.tokenizer, potency=potent)])
                int_ids = self.model.generate(**inputs, max_new_tokens=1, logits_processor=proc, pad_token_id=self.tokenizer.eos_token_id)
                final_ans = self.tokenizer.decode(int_ids[0], skip_special_tokens=True).split("Answer:")[-1].strip()

            print(f"{f'{h} -> {t}':<30} | {raw_sync:8.3f} | {int(L) if is_valid else '∞':<2} | {potent:8.3f} | {raw_ans:<5} | {final_ans}")

# --- 実行 ---
test_suite = [
    ("Tuberculosis", "Visit to Asia"), ("Lung Cancer", "Smoking"),
    ("Lung Cancer", "Bronchitis"), ("Visit to Asia", "Either"),
    ("Smoking", "Either"), ("Either", "X-ray"),
    ("Either", "Dyspnea"), ("Bronchitis", "Dyspnea"),
    ("Lung Cancer", "X-ray"), ("Smoking", "X-ray"),
    ("Visit to Asia", "X-ray"), ("Smoking", "Dyspnea"),
    ("Dyspnea", "Bronchitis")
]

if __name__ == "__main__":
    os_v32_1 = CausalOS_v32_1()
    os_v32_1.run_benchmark(test_suite)
