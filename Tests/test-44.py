import torch
import torch.nn.functional as F
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

class UncertainCausalLogitProcessor(LogitsProcessor):
    def __init__(self, tokenizer, potency, alpha=55.0, tau=0.1):
        self.yes_ids = self._get_tokens(tokenizer, ["Yes", "yes", " Yes"])
        self.no_ids = self._get_tokens(tokenizer, ["No", "no", " No"])
        self.potency = potency
        self.alpha = alpha
        self.tau = tau # 確信度の閾値

    def _get_tokens(self, tokenizer, words):
        ids = []
        for w in words:
            t = tokenizer.encode(w, add_special_tokens=False)
            if t: ids.append(t[-1])
        return list(set(ids))

    def __call__(self, input_ids, scores):
        # 改善(3): Potencyが閾値 tau 未満の場合は介入しない（LLMに委ねる）
        if abs(self.potency) < self.tau:
            return scores
            
        bias = self.alpha * self.potency
        for tid in self.yes_ids: scores[:, tid] += bias
        for tid in self.no_ids: scores[:, tid] -= bias
        return scores

class StableCausalAdapter:
    def __init__(self, edges, lr=0.5):
        # 改善(1): 正しい隣接リスト初期化
        self.strengths = {}
        self.frozen = {}
        for u, v in edges:
            self.strengths.setdefault(u, {})[v] = 0.6
            self.frozen.setdefault(u, {})[v] = False
        self.lr = lr

    def get_boost(self, u, v):
        raw_s = self.strengths.get(u, {}).get(v, 0.6)
        return 0.6 + 0.4 * math.tanh(raw_s - 0.6)

    def get_path_boost(self, path):
        if not path or len(path) < 2: return 0.6
        return sum(self.get_boost(path[i], path[i+1]) for i in range(len(path)-1)) / (len(path)-1)

    def learn_path(self, path, potency, target=0.8, epsilon=0.01):
        if not path or len(path) < 2: return
        error = target - potency
        if abs(error) < epsilon:
            for i in range(len(path)-1):
                self.frozen[path[i]][path[i+1]] = True
            return

        adjustment = self.lr * error
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            if not self.frozen.get(u, {}).get(v, False):
                self.strengths[u][v] += adjustment / (len(path) - 1)

class CausalOS_v34_2:
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        self.device = self.model.device
        
        self.S_edges = [
            ("Tuberculosis", "Visit to Asia"), ("Lung Cancer", "Smoking"),
            ("Lung Cancer", "Bronchitis"), ("Visit to Asia", "Either"),
            ("Smoking", "Either"), ("Either", "X-ray"),
            ("Either", "Dyspnea"), ("Bronchitis", "Dyspnea")
        ]
        self.adapter = StableCausalAdapter(self.S_edges)
        self.nodes = list(set([u for u, v in self.S_edges] + [v for u, v in self.S_edges]))
        self.dist_matrix, self.path_matrix = self._build_structure()
        self.proj = torch.randn((self.tokenizer.vocab_size, 64), device=self.device, dtype=torch.float16)
        self.proj /= (torch.norm(self.proj, dim=1, keepdim=True) + 1e-9)

    def _build_structure(self):
        dist = {n1: {n2: float('inf') for n2 in self.nodes} for n1 in self.nodes}
        next_n = {n1: {n2: None for n2 in self.nodes} for n1 in self.nodes}
        for n in self.nodes: dist[n][n] = 0
        for u, d in self.adapter.strengths.items():
            for v in d: dist[u][v], next_n[u][v] = 1, v
        for k in self.nodes:
            for i in self.nodes:
                for j in self.nodes:
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j], next_n[i][j] = dist[i][k] + dist[k][j], next_n[i][k]
        return dist, next_n

    def get_path(self, h, t):
        if h not in self.nodes or t not in self.nodes or self.dist_matrix[h][t] == float('inf'): return []
        path = [h]; curr = h
        while curr != t: curr = self.path_matrix[curr][t]; path.append(curr)
        return path

    def get_sync(self, h, t):
        h_ids = torch.tensor(self.tokenizer.encode(h, add_special_tokens=False), device=self.device)
        t_ids = torch.tensor(self.tokenizer.encode(t, add_special_tokens=False), device=self.device)
        v_h, v_t = torch.mean(self.proj[h_ids], dim=0), torch.mean(self.proj[t_ids], dim=0)
        return F.cosine_similarity(v_h.unsqueeze(0), v_t.unsqueeze(0)).item()

    def run_epoch(self, epoch, tests):
        print(f"\n--- v34.2 | EPOCH {epoch} | Intervention-Threshold tau=0.1 ---")
        for h, d in self.adapter.strengths.items():
            for t in d:
                s = self.get_sync(h, t); b = self.adapter.get_boost(h, t)
                self.adapter.learn_path([h, t], s + b)

        print(f"{'Path Candidate':<32} | {'Sync':<7} | {'Boost':<6} | {'Potent':<6} | {'Output'}")
        print("-" * 88)

        for h, t in tests:
            raw_sync = self.get_sync(h, t)
            path = self.get_path(h, t)
            L = len(path) - 1
            
            if L > 0:
                boost_val = self.adapter.get_path_boost(path)
                potent = (boost_val / math.log(math.e + L - 1))
                status = "Frozen" if all(self.adapter.frozen.get(path[i], {}).get(path[i+1], False) for i in range(len(path)-1)) else "Active"
            else:
                rev_path = self.get_path(t, h); rev_L = len(rev_path)-1 if rev_path else float('inf')
                # 因果の逆流がある場合は明確にペナルティ、そうでない場合は 0 に近づける
                penalty = (0.6 / math.log(math.e + rev_L - 1)) if rev_L != float('inf') else 0.0
                potent, boost_val, status = raw_sync - penalty, 0.6, "N/A"

            # 改善(3)の反映
            intervention = "Causal-Intervened" if abs(potent) >= 0.1 else "LLM-Native"
            
            prompt = f"Does {h} cause {t}?\nAnswer Yes or No:"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            proc = LogitsProcessorList([UncertainCausalLogitProcessor(self.tokenizer, potency=potent, tau=0.1)])
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=1, logits_processor=proc, pad_token_id=self.tokenizer.eos_token_id)
                ans = self.tokenizer.decode(out[0], skip_special_tokens=True).split("No:")[-1].strip()

            print(f"{f'{h}->{t}':<32} | {raw_sync:7.3f} | {boost_val:6.3f} | {potent:6.3f} | {ans} ({intervention})")
            if L > 0 and status == "Active":
                self.adapter.learn_path(path, potent)

# --- テスト ---
test_suite_13 = [
    ("Tuberculosis", "Visit to Asia"), ("Lung Cancer", "Smoking"),
    ("Lung Cancer", "Bronchitis"), ("Visit to Asia", "Either"),
    ("Smoking", "Either"), ("Either", "X-ray"),
    ("Either", "Dyspnea"), ("Bronchitis", "Dyspnea"),
    ("Lung Cancer", "X-ray"), ("Smoking", "X-ray"),
    ("Visit to Asia", "X-ray"), ("Smoking", "Dyspnea"),
    ("Dyspnea", "Bronchitis")
]

os_v34_2 = CausalOS_v34_2()
for e in range(1, 6):
    os_v34_2.run_epoch(e, test_suite_13)
