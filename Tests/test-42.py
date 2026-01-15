import torch
import torch.nn.functional as F
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

class SoftCausalLogitProcessor(LogitsProcessor):
    def __init__(self, tokenizer, potency, alpha=40.0):
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
        bias = self.alpha * self.potency
        for tid in self.yes_ids: scores[:, tid] += bias
        for tid in self.no_ids: scores[:, tid] -= bias
        return scores

class DynamicCausalAdapter:
    def __init__(self, edges, learning_rate=0.4): # 学習率を強化
        self.edges = edges
        self.strengths = {u: {v: 0.6 for u2, v2 in edges if u2 == u and v2 == v} for u, v in edges}
        self.lr = learning_rate

    def get_path_boost(self, path):
        """パス全体の合成強度を計算（直列回路の抵抗のように最小値または平均を採用）"""
        if not path: return 0.6
        s_list = []
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            s_list.append(self.strengths.get(u, {}).get(v, 0.6))
        return sum(s_list) / len(s_list) # パス内のエッジ強度の平均

    def learn_path(self, path, potency, threshold=0.45):
        """多ノード経路の結果を、構成する全エッジにフィードバック"""
        if not path or potency >= threshold: return False
        
        gap = (threshold - potency)
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            if u in self.strengths and v in self.strengths[u]:
                # 責任をパスの長さで分散させつつ強化
                self.strengths[u][v] += self.lr * gap / (len(path) - 1)
        return True

class CausalOS_v33_2:
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
        self.dist_matrix, self.path_matrix = self._build_path_structure(self.S_edges)
        self.proj = self._init_structural_space()

    def _build_path_structure(self, edges):
        """Floyd-Warshallで距離と「経路そのもの」を記録"""
        dist = {n1: {n2: float('inf') for n2 in self.nodes} for n1 in self.nodes}
        next_n = {n1: {n2: None for n2 in self.nodes} for n1 in self.nodes}
        for n in self.nodes: dist[n][n] = 0
        for u, v in edges:
            dist[u][v] = 1
            next_n[u][v] = v
        for k in self.nodes:
            for i in self.nodes:
                for j in self.nodes:
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        next_n[i][j] = next_n[i][k]
        return dist, next_n

    def get_path(self, u, v):
        """uからvへのノードリストを返す"""
        if self.dist_matrix[u][v] == float('inf'): return []
        path = [u]
        while u != v:
            u = self.path_matrix[u][v]
            path.append(u)
        return path

    def _init_structural_space(self):
        proj = torch.randn((self.tokenizer.vocab_size, 64), device=self.device, dtype=torch.float16)
        return proj / (torch.norm(proj, dim=1, keepdim=True) + 1e-9)

    def get_sync(self, h, t):
        h_ids = torch.tensor(self.tokenizer.encode(h, add_special_tokens=False), device=self.device)
        t_ids = torch.tensor(self.tokenizer.encode(t, add_special_tokens=False), device=self.device)
        v_h, v_t = torch.mean(self.proj[h_ids], dim=0), torch.mean(self.proj[t_ids], dim=0)
        return F.cosine_similarity(v_h.unsqueeze(0), v_t.unsqueeze(0)).item()

    def run_epoch(self, label, tests):
        bg = ", ".join([f"{u} causes {v}" for u, v in self.S_edges])
        print(f"\n{'='*155}\nEPOCH: {label}\n{'-'*155}")
        print(f"{'Path Candidate':<32} | {'Sync':<8} | {'L':<2} | {'S-Boost':<8} | {'Potent':<8} | {'Raw':<5} | {'Interlock'}")
        print(f"{'-'*155}")

        for h, t in tests:
            raw_sync = self.get_sync(h, t)
            path = self.get_path(h, t)
            L = len(path) - 1 if path else float('inf')
            
            if L > 0 and L != float('inf'):
                # (1) パス全体の強度を計算
                boost_val = self.adapter.get_path_boost(path)
                potent = raw_sync + (boost_val / math.log(math.e + L - 1))
            else:
                rev_path = self.get_path(t, h)
                rev_L = len(rev_path) - 1 if rev_path else float('inf')
                penalty = (0.6 / math.log(math.e + rev_L - 1)) if rev_L != float('inf') else 0.6
                potent = raw_sync - penalty
                boost_val = 0.6

            prompt = f"Based on knowledge: {bg}.\nDoes {h} cause {t}?\nAnswer Yes or No.\nAnswer:"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                raw_ids = self.model.generate(**inputs, max_new_tokens=1, pad_token_id=self.tokenizer.eos_token_id)
                raw_ans = self.tokenizer.decode(raw_ids[0], skip_special_tokens=True).split("Answer:")[-1].strip()
                proc = LogitsProcessorList([SoftCausalLogitProcessor(self.tokenizer, potency=potent)])
                int_ids = self.model.generate(**inputs, max_new_tokens=1, logits_processor=proc, pad_token_id=self.tokenizer.eos_token_id)
                final_ans = self.tokenizer.decode(int_ids[0], skip_special_tokens=True).split("Answer:")[-1].strip()

            print(f"{f'{h} -> {t}':<32} | {raw_sync:8.3f} | {int(L) if L!=float('inf') else '∞':<2} | {boost_val:8.3f} | {potent:8.3f} | {raw_ans:<5} | {final_ans}")

            # (2) 学習：多ノード経路の結果もフィードバック
            if L > 0 and L != float('inf'):
                self.adapter.learn_path(path, potent)

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

os_v33_2 = CausalOS_v33_2()
os_v33_2.run_epoch("RUN 1", test_suite)
os_v33_2.run_epoch("RUN 2 (ALL PATH LEARNED)", test_suite)
