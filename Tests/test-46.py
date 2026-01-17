import torch
import torch.nn.functional as F
import re
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

# --- Logits Processor: potentがcritical_pointを超えている場合のみYesにバイアス ---
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

# --- Main OS Kernel ---
class CausalOS_v37_6:
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct"):
        print(f"Initializing Causal OS v37.6 [Final Logical Patch]...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        self.device = self.model.device
        
        self.dim = 64
        self.critical_point = 0.40
        self.S_edges = [] 
        self.memory_path = "S_memory.json"
        self.proj = self._init_structural_space()

    def _init_structural_space(self):
        vocab_size = self.tokenizer.vocab_size
        proj = torch.randn((vocab_size, self.dim), device=self.device, dtype=torch.float16)
        return proj / (torch.norm(proj, dim=1, keepdim=True) + 1e-9)

    def save_S(self):
        with open(self.memory_path, 'w') as f:
            json.dump(self.S_edges, f)

    def load_S(self):
        if os.path.exists(self.memory_path):
            with open(self.memory_path, 'r') as f:
                self.S_edges = json.load(f)

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

    def has_path_dfs(self, edges, start, end):
        if not edges: return False
        adj = {}
        for u, v in edges: adj.setdefault(u, []).append(v)
        stack = [start]
        visited = set()
        while stack:
            x = stack.pop()
            if x == end: return True
            if x in visited: continue
            visited.add(x)
            stack.extend(adj.get(x, []))
        return False

    def autonomous_exploration(self, variables):
        print(f"\n[Log] Phase 1: Exploration for {variables}")
        prompt = f"Identify 3-4 latent factors for {variables} and list causal edges as 'Node A -> Node B'.\nEdges:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=200)
            raw_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"--- LLM Raw Exploration Output ---\n{raw_response}\n-----------------------------")
        
        extracted_edges = []
        lines = raw_response.split("Edges:")[-1].split("\n")
        for line in lines:
            match = re.search(r"([\w\s]+)\s*->\s*([\w\s]+)", line)
            if match:
                u, v = match.group(1).strip(), match.group(2).strip()
                if u != v:
                    extracted_edges.append((u, v))
        return extracted_edges

    def generate_answer(self, head, tail, potent, intervention_node):
        # 修正③：介入ノード do(x) を明示的にプロンプトに反映
        if intervention_node:
            prompt = f"Causal graph: {self.S_edges}.\nUnder do({intervention_node}), does changing {head} affect {tail}?\nAnswer Yes or No:"
        else:
            prompt = f"Causal graph: {self.S_edges}.\nDoes changing {head} affect {tail}?\nAnswer Yes or No:"
            
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]
        proc = LogitsProcessorList([IELogitProcessor(self.tokenizer, potent, self.critical_point)])
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=2, logits_processor=proc, pad_token_id=self.tokenizer.eos_token_id)
            ans = self.tokenizer.decode(out[0, input_len:], skip_special_tokens=True).strip()
        return "Yes" if "Yes" in ans else "No"

    def run_discovery_test(self, variables):
        self.S_edges = [] 
        proposed = self.autonomous_exploration(variables)
        
        print(f"\n[Log] Phase 2: S-Memory Commitment")
        for u, v in proposed:
            if (u, v) not in self.S_edges:
                self.S_edges.append((u, v))
                print(f"  - Committing: {u} -> {v}")
        
        self.save_S()

        print(f"\n[Log] Phase 3: Structural Auditing")
        head, tail = variables[0], variables[1]

        # 修正②：逆方向の検定を独立して計算
        # ---------- 順方向 ----------
        v_h, _ = self.get_complex_state(head)
        v_t, _ = self.get_complex_state(tail)
        sync_fwd = F.cosine_similarity(v_h.unsqueeze(0), v_t.unsqueeze(0)).item()

        path_fwd = self.has_path_dfs(self.S_edges, head, tail)
        # 修正①：パスがあれば強い正ブースト (0.6)
        potent_fwd = sync_fwd + (0.6 if path_fwd else -0.4)
        ans_fwd = self.generate_answer(head, tail, potent_fwd, head)

        # ---------- 逆方向 ----------
        v_t_rev, _ = self.get_complex_state(tail)
        v_h_rev, _ = self.get_complex_state(head)
        sync_rev = F.cosine_similarity(v_t_rev.unsqueeze(0), v_h_rev.unsqueeze(0)).item()

        path_rev = self.has_path_dfs(self.S_edges, tail, head)
        potent_rev = sync_rev + (0.6 if path_rev else -0.4)
        ans_rev = self.generate_answer(tail, head, potent_rev, tail)

        print(f"\n[Final Results]")
        print(f"Direction {head} -> {tail}: {ans_fwd} (Potent: {potent_fwd:.4f})")
        print(f"Direction {tail} -> {head}: {ans_rev} (Potent: {potent_rev:.4f})")
        
        # 監査用ログ
        print("\n[Internal Audit Check]")
        print(f"S_edges: {self.S_edges}")
        print(f"Forward Path: {path_fwd}, Sync: {sync_fwd:.4f}")
        print(f"Backward Path: {path_rev}, Sync: {sync_rev:.4f}")

if __name__ == "__main__":
    tester = CausalOS_v37_6()
    tester.run_discovery_test(["Age", "Shell weight"])
