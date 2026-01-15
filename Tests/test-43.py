import torch
import torch.nn.functional as F
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

# --- (1) Logitバイアス制御 ---
class SoftCausalLogitProcessor(LogitsProcessor):
    def __init__(self, tokenizer, potency, alpha=50.0):
        self.yes_ids = self._get_tokens(tokenizer, ["Yes", "yes", " Yes"])
        self.no_ids = self._get_tokens(tokenizer, ["No", "no", " No"])
        self.potency = potency
        self.alpha = alpha

    def _get_tokens(self, tokenizer, words):
        ids = []
        for w in words:
            t = tokenizer.encode(w, add_special_tokens=False)
            if t: ids.append(t[-1])
        return list(set(ids))

    def __call__(self, input_ids, scores):
        bias = self.alpha * self.potency
        for tid in self.yes_ids: scores[:, tid] += bias
        for tid in self.no_ids: scores[:, tid] -= bias
        return scores

# --- (2) 多段パス対応・学習アダプター ---
class DynamicCausalAdapter:
    def __init__(self, edges, lr=0.5):
        self.strengths = {}
        for u, v in edges:
            if u not in self.strengths: self.strengths[u] = {}
            self.strengths[u][v] = 0.6
        self.lr = lr

    def get_path_boost(self, path):
        if not path or len(path) < 2: return 0.6
        s_vals = [self.strengths.get(path[i], {}).get(path[i+1], 0.6) for i in range(len(path)-1)]
        return sum(s_vals) / len(s_vals)

    def learn_path(self, path, potency, target_potency=0.8):
        if not path or len(path) < 2: return
        error = target_potency - potency
        adjustment = self.lr * error
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            if u in self.strengths and v in self.strengths[u]:
                self.strengths[u][v] += adjustment / (len(path) - 1)

# --- (3) 統合OSエンジン ---
class CausalOS_v33_Final:
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
        self.adapter = DynamicCausalAdapter(self.S_edges)
        self.nodes = list(set([u for u, v in self.S_edges] + [v for u, v in self.S_edges]))
        self.dist_matrix, self.path_matrix = self._build_path_structure()
        self.proj = torch.randn((self.tokenizer.vocab_size, 64), device=self.device, dtype=torch.float16)
        self.proj /= (torch.norm(self.proj, dim=1, keepdim=True) + 1e-9)

    def _build_path_structure(self):
        dist = {n1: {n2: float('inf') for n2 in self.nodes} for n1 in self.nodes}
        next_n = {n1: {n2: None for n2 in self.nodes} for n1 in self.nodes}
        for n in self.nodes: dist[n][n] = 0
        for u, v in self.S_edges: dist[u][v], next_n[u][v] = 1, v
        for k in self.nodes:
            for i in self.nodes:
                for j in self.nodes:
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j], next_n[i][j] = dist[i][k] + dist[k][j], next_n[i][k]
        return dist, next_n

    def get_path(self, u, v):
        if u not in self.nodes or v not in self.nodes: return []
        if self.dist_matrix[u][v] == float('inf'): return []
        path = [u]; curr = u
        while curr != v: curr = self.path_matrix[curr][v]; path.append(curr)
        return path

    def get_sync(self, h, t):
        h_ids = torch.tensor(self.tokenizer.encode(h, add_special_tokens=False), device=self.device)
        t_ids = torch.tensor(self.tokenizer.encode(t, add_special_tokens=False), device=self.device)
        v_h, v_t = torch.mean(self.proj[h_ids], dim=0), torch.mean(self.proj[t_ids], dim=0)
        return F.cosine_similarity(v_h.unsqueeze(0), v_t.unsqueeze(0)).item()

    def train_network(self):
        for h, t_dict in self.adapter.strengths.items():
            for t, boost in t_dict.items():
                sync = self.get_sync(h, t)
                self.adapter.learn_path([h, t], sync + boost)

    def run_epoch(self, epoch, tests):
        bg = ", ".join([f"{u} causes {v}" for u, v in self.S_edges])
        print(f"\n--- EPOCH {epoch} ---")
        self.train_network()
        print(f"{'Path Candidate':<32} | {'Sync':<7} | {'L':<1} | {'Boost':<6} | {'Potent':<6} | {'Output'}")
        print("-" * 85)

        for h, t in tests:
            raw_sync = self.get_sync(h, t)
            path = self.get_path(h, t)
            L = len(path) - 1 if path else float('inf')
            
            if 0 < L < float('inf'):
                boost_val = self.adapter.get_path_boost(path)
                potent = raw_sync + (boost_val / math.log(math.e + L - 1))
            else:
                rev_path = self.get_path(t, h); rev_L = len(rev_path) - 1 if rev_path else float('inf')
                penalty = (0.6 / math.log(math.e + rev_L - 1)) if rev_L != float('inf') else 0.6
                potent, boost_val = raw_sync - penalty, 0.6

            prompt = f"Background: {bg}.\nDoes {h} cause {t}?\nAnswer Yes or No:"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            proc = LogitsProcessorList([SoftCausalLogitProcessor(self.tokenizer, potency=potent)])
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=1, logits_processor=proc, pad_token_id=self.tokenizer.eos_token_id)
                ans = self.tokenizer.decode(out[0], skip_special_tokens=True).split("No:")[-1].strip()

            print(f"{f'{h}->{t}':<32} | {raw_sync:7.3f} | {int(L) if L!=float('inf') else '∞'} | {boost_val:6.3f} | {potent:6.3f} | {ans}")
            if 0 < L < float('inf'): self.adapter.learn_path(path, potent)

# --- ベンチマーク実行 ---
test_suite_13 = [
    ("Tuberculosis", "Visit to Asia"), ("Lung Cancer", "Smoking"),
    ("Lung Cancer", "Bronchitis"), ("Visit to Asia", "Either"),
    ("Smoking", "Either"), ("Either", "X-ray"),
    ("Either", "Dyspnea"), ("Bronchitis", "Dyspnea"),
    ("Lung Cancer", "X-ray"), ("Smoking", "X-ray"),
    ("Visit to Asia", "X-ray"), ("Smoking", "Dyspnea"),
    ("Dyspnea", "Bronchitis")
]

os_final = CausalOS_v33_Final()
for e in range(1, 6):
    os_final.run_epoch(e, test_suite_13)
