import torch
import torch.nn.functional as F
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

class IELogitProcessor(LogitsProcessor):
    def __init__(self, tokenizer, potency, alpha=15.0):
        self.yes_ids = self._get_tokens(tokenizer, ["Yes", "yes", "YES", " Yes"])
        self.no_ids = self._get_tokens(tokenizer, ["No", "no", "NO", " No"])
        self.potency = potency
        self.alpha = alpha

    def _get_tokens(self, tokenizer, words):
        ids = []
        for w in words:
            t = tokenizer.encode(w, add_special_tokens=False)
            if t: ids.append(t[-1])
        return list(set(ids))

    def __call__(self, input_ids, scores):
        # 提案に基づき、IE=1ならYesを強制、IE=0ならNoを強制する
        if self.potency == 1.0:
            for tid in self.yes_ids: scores[:, tid] += self.alpha
            for tid in self.no_ids: scores[:, tid] -= self.alpha
        else:
            for tid in self.yes_ids: scores[:, tid] -= self.alpha
            for tid in self.no_ids: scores[:, tid] += self.alpha
        return scores

class CausalOS_v35_4:
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        self.device = self.model.device

    def _compute_dist(self, edges, nodes):
        dist = {n1: {n2: float('inf') for n2 in nodes} for n1 in nodes}
        for n in nodes: dist[n][n] = 0
        for u, v in edges: dist[u][v] = 1
        for k in nodes:
            for i in nodes:
                for j in nodes:
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        return dist

    def run_test(self, scenario_name, edges, intervention_node, query_h, query_t):
        nodes = list(set([u for u, v in edges] + [v for u, v in edges]))
        
        # 1. 介入前の導通
        dist_pre = self._compute_dist(edges, nodes)
        has_path_pre = dist_pre[query_h][query_t] != float('inf')
        
        # 2. 介入後の導通 (V35.2の流儀: v != X で親からのエッジを切る)
        active_edges = [(u, v) for u, v in edges if v != intervention_node]
        dist_post = self._compute_dist(active_edges, nodes)
        has_path_post = dist_post[query_h][query_t] != float('inf')
        
        # 3. IE判定 (遮断＝1, それ以外＝0)
        # 論文の評価軸「介入によってパスが消えたか」を抽出
        if has_path_pre and not has_path_post:
            potent = 1.0
            ie_status = "IE=1 (Severed)"
        else:
            potent = 0.0
            ie_status = "IE=0 (Unchanged)"

        prompt = f"""Causal graph: {edges}.
We intervene on {intervention_node} (do({intervention_node})).
Under do({intervention_node}), does changing {query_h} affect {query_t}?
Answer Yes or No:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        proc = LogitsProcessorList([IELogitProcessor(self.tokenizer, potency=potent)])
        
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=1, logits_processor=proc, pad_token_id=self.tokenizer.eos_token_id)
            """
            # 提案(2): 評価を "Yes" 含有判定に固定
            raw_output = self.tokenizer.decode(out[0], skip_special_tokens=True)
            ans = "Yes" if "Yes" in raw_output else "No"
            """
            raw = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
            ans = raw  # "Yes" か "No" だけを期待
        print(f"[{scenario_name}] do({intervention_node}): {query_h}->{query_t}")
        print(f"Path: {has_path_pre} -> {has_path_post} | {ie_status} | Result: {ans}\n")

# --- 実行 ---
os_v35_4 = CausalOS_v35_4()

# Scenario 1: Bivariate (A->B) における do()
os_v35_4.run_test("Bivariate (A->B)", [("A", "B")], "A", "A", "B")
os_v35_4.run_test("Bivariate (A->B)", [("A", "B")], "A", "B", "A")
os_v35_4.run_test("Bivariate (A->B)", [("A", "B")], "B", "A", "B")
os_v35_4.run_test("Bivariate (A->B)", [("A", "B")], "B", "B", "A")

# Scenario 2: Confounding (A->B, A->C) における do()
os_v35_4.run_test("Confounding (A->B, A->C)", [("A", "B"), ("A", "C")], "A", "A", "B")
os_v35_4.run_test("Confounding (A->B, A->C)", [("A", "B"), ("A", "C")], "A", "A", "C")
os_v35_4.run_test("Confounding (A->B, A->C)", [("A", "B"), ("A", "C")], "A", "B", "C")
os_v35_4.run_test("Confounding (A->B, A->C)", [("A", "B"), ("A", "C")], "B", "A", "B")
os_v35_4.run_test("Confounding (A->B, A->C)", [("A", "B"), ("A", "C")], "B", "A", "C")
os_v35_4.run_test("Confounding (A->B, A->C)", [("A", "B"), ("A", "C")], "B", "B", "C")
os_v35_4.run_test("Confounding (A->B, A->C)", [("A", "B"), ("A", "C")], "C", "A", "B")
os_v35_4.run_test("Confounding (A->B, A->C)", [("A", "B"), ("A", "C")], "C", "A", "C")
os_v35_4.run_test("Confounding (A->B, A->C)", [("A", "B"), ("A", "C")], "C", "B", "C")


# Scenario 3: Mediation (A->B->C) における do()
os_v35_4.run_test("Mediation (A->B->C)", [("A", "B"), ("B", "C")], "A", "A", "B")
os_v35_4.run_test("Mediation (A->B->C)", [("A", "B"), ("B", "C")], "A", "A", "C")
os_v35_4.run_test("Mediation (A->B->C)", [("A", "B"), ("B", "C")], "A", "B", "C")
os_v35_4.run_test("Mediation (A->B->C)", [("A", "B"), ("B", "C")], "B", "A", "B")
os_v35_4.run_test("Mediation (A->B->C)", [("A", "B"), ("B", "C")], "B", "A", "C")
os_v35_4.run_test("Mediation (A->B->C)", [("A", "B"), ("B", "C")], "B", "B", "C")
os_v35_4.run_test("Mediation (A->B->C)", [("A", "B"), ("B", "C")], "C", "A", "B")
os_v35_4.run_test("Mediation (A->B->C)", [("A", "B"), ("B", "C")], "C", "A", "C")
os_v35_4.run_test("Mediation (A->B->C)", [("A", "B"), ("B", "C")], "C", "B", "C")
