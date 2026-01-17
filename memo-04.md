# 一次まとめ v36.5


```python
import torch
import torch.nn.functional as F
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

class IELogitProcessor(LogitsProcessor):
    def __init__(self, tokenizer, potent, critical_point=0.40, alpha=15.0):
        self.yes_ids = self._get_tokens(tokenizer, ["Yes", "yes", "YES", " Yes"])
        self.no_ids = self._get_tokens(tokenizer, ["No", "no", "NO", " No"])
        self.is_yes = potent > critical_point
        self.alpha = alpha

    def _get_tokens(self, tokenizer, words):
        ids = []
        for w in words:
            t = tokenizer.encode(w, add_special_tokens=False)
            if t: ids.append(t[-1])
        return list(set(ids))

    def __call__(self, input_ids, scores):
        bias = self.alpha if self.is_yes else -self.alpha
        for tid in self.yes_ids: scores[:, tid] += bias
        for tid in self.no_ids: scores[:, tid] -= bias
        return scores

class CausalOS_v36_5:
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct"):
        self.memory_filename = "causal_s_graph_v36.json"
        print(f"Initializing Causal OS v36.5 [Pearl Rigidity Edition]...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        self.device = self.model.device
        
        self.dim = 64
        self.critical_point = 0.40
        self.S_edges = self._load_S_from_file()
        
        self.S_sub = {"ie_history": []}
        self.proj = self._init_structural_space()

    def _load_S_from_file(self):
        if os.path.exists(self.memory_filename):
            with open(self.memory_filename, "r", encoding="utf-8") as f:
                return [tuple(e) for e in json.load(f)]
        return [("Smoking", "Lung Cancer"), ("Lung Cancer", "Either")]

    def save_S_to_file(self):
        with open(self.memory_filename, "w", encoding="utf-8") as f:
            json.dump(self.S_edges, f, ensure_ascii=False, indent=2)

    def has_path_dfs(self, edges, start, end):
        if not edges: return False
        adj = {}
        for u, v in edges:
            adj.setdefault(u, []).append(v)
        stack = [start]
        visited = set()
        while stack:
            x = stack.pop()
            if x == end: return True
            if x in visited: continue
            visited.add(x)
            stack.extend(adj.get(x, []))
        return False

    def apply_intervention(self, edges, intervention_node):
        """Pearlのdo-演算: 介入を受けるノード(v)への流入エッジを全て削除"""
        if intervention_node is not None:
            return [(u, v) for u, v in edges if v != intervention_node]
        return edges

    # (1) IEスコアの拡張：起点介入の考慮
    def compute_ie_score(self, path_pre, path_post, head, intervention_node):
        """
        Pearlの定義に基づき、起点介入(head == intervention_node)はパスが切れないため、
        情報の遮断（IE=1）は発生していない(IE=0)とみなす。
        """
        if intervention_node == head:
            return 0.0      # 起点介入: IE=0 (Unchanged)
        
        if path_pre and not path_post:
            return 0.6      # 遮断確認: IE=1 (Severed)
        elif path_pre and path_post:
            return -0.1     # 導通継続: 因果はあるが介入の影響外
        else:
            return -0.4     # 因果なし

    def run_hybrid_audit(self, head, tail, intervention_node=None):
        # 意味論同期 (v29)
        v_h_r, _ = self.get_complex_state(head)
        v_t_r, _ = self.get_complex_state(tail)
        raw_sync = F.cosine_similarity(v_h_r.unsqueeze(0), v_t_r.unsqueeze(0)).item()

        # 構造的パス検定
        nodes_in_s = set([u for u, v in self.S_edges] + [v for u, v in self.S_edges])
        if head not in nodes_in_s or tail not in nodes_in_s:
            path_pre, path_post = False, False
        else:
            path_pre = self.has_path_dfs(self.S_edges, head, tail)
            active_edges = self.apply_intervention(self.S_edges, intervention_node)
            path_post = self.has_path_dfs(active_edges, head, tail)
        
        # (1) 修正されたIEスコア計算
        ie_boost = self.compute_ie_score(path_pre, path_post, head, intervention_node)
        final_potent = raw_sync + ie_boost

        self.S_sub["ie_history"].append({
            "head": head, "tail": tail, "intervention": intervention_node,
            "path_pre": path_pre, "path_post": path_post, "ie_boost": ie_boost, "potent": final_potent
        })
        return final_potent

    # (2) プロンプトにグラフ情報を復活
    def generate_answer(self, head, tail, potent, intervention_node=None):
        inter_text = f"do({intervention_node})" if intervention_node else "observation"
        
        prompt = f"""Causal graph: {self.S_edges}.
Under {inter_text}, does changing {head} affect {tail}?
Answer Yes or No:"""
            
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]
        
        proc = LogitsProcessorList([IELogitProcessor(self.tokenizer, potent, self.critical_point)])
        
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=2, 
                logits_processor=proc, 
                pad_token_id=self.tokenizer.eos_token_id
            )
            ans = self.tokenizer.decode(out[0, input_len:], skip_special_tokens=True).strip()
            
        return "Yes" if "Yes" in ans else ("No" if "No" in ans else ans)

    def _init_structural_space(self):
        vocab_size = self.tokenizer.vocab_size
        proj = torch.randn((vocab_size, self.dim), device=self.device, dtype=torch.float16)
        return proj / (torch.norm(proj, dim=1, keepdim=True) + 1e-9)

    def get_complex_state(self, text):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        t_ids = torch.tensor(tokens, device=self.device)
        pos = torch.arange(len(tokens), device=self.device).float()
        angles = pos * 0.5
        cos_t, sin_t = torch.cos(angles).unsqueeze(1), torch.sin(angles).unsqueeze(1)
        base_vecs = self.proj[t_ids]
        real_part = torch.sum(base_vecs * cos_t, dim=0)
        imag_part = torch.sum(base_vecs * sin_t, dim=0)
        norm = torch.sqrt(torch.sum(real_part**2) + torch.sum(imag_part**2)) + 1e-9
        return real_part / norm, imag_part / norm

    def run_discovery_step(self, head, tail, intervention_node=None):
        potent = self.run_hybrid_audit(head, tail, intervention_node)
        if potent > 0.7 and (head, tail) not in self.S_edges:
            self.S_edges.append((head, tail))
            self.save_S_to_file()
        return self.generate_answer(head, tail, potent, intervention_node)

# --- 実行 ---
if __name__ == "__main__":
    os_sys = CausalOS_v36_5()
    # Bivariate A->B における do(A) テスト (期待値: ie_boost=0.0 -> potentが維持され Yes)
    print(f"Test A->B under do(A): {os_sys.run_discovery_step('Smoking', 'Lung Cancer', intervention_node='Smoking')}")
```
