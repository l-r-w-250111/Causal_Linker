import torch
import torch.nn.functional as F
import re
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

class CausalOS_v37_8:
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct"):
        print(f"Initializing Causal OS v37.8 [Distance-Aware Decay]...")
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
        with open(self.memory_path, 'w') as f: json.dump(self.S_edges, f)

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

    def get_min_path_dist(self, edges, start, end):
        if not edges: return None
        adj = {}
        for u, v in edges: adj.setdefault(u, []).append(v)
        queue = [(start, 0)]
        visited = {start}
        while queue:
            curr, dist = queue.pop(0)
            if curr == end: return dist
            for neighbor in adj.get(curr, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return None

    def autonomous_exploration(self, variables):
        print(f"\n[Log] Phase 1: Exploration for {variables}")
        prompt = f"Identify all causal edges between {variables} and latent factors. Format: 'A -> B'.\nEdges:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=300)
            raw_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"--- LLM Raw Discovery Output ---\n{raw_response}\n-----------------------------")
        
        extracted_edges = []
        lines = raw_response.split("Edges:")[-1].split("\n")
        for line in lines:
            match = re.search(r"([\w\s]+)\s*->\s*([\w\s]+)", line)
            if match:
                u, v = match.group(1).strip(), match.group(2).strip()
                # クリーニング: 文頭の数字や指示語を排除
                u = re.sub(r'^\d+[\.\)]\s*', '', u).strip()
                if len(u) > 1 and len(v) > 1 and u.lower() not in ["a", "b", "node"]:
                    extracted_edges.append((u, v))
        return extracted_edges

    def generate_answer(self, head, tail, potent, intervention_node):
        prompt = f"Causal graph: {self.S_edges}.\nUnder do({intervention_node}), does changing {head} affect {tail}?\nAnswer Yes or No:"
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

        print(f"\n[Log] Phase 3: Structural Auditing (Distance-Aware)")
        head, tail = variables[0], variables[1]

        # 順方向
        v_h, _ = self.get_complex_state(head)
        v_t, _ = self.get_complex_state(tail)
        sync_fwd = F.cosine_similarity(v_h.unsqueeze(0), v_t.unsqueeze(0)).item()
        dist_fwd = self.get_min_path_dist(self.S_edges, head, tail)
        
        # 距離減衰ブースト: 1なら0.6, 2なら0.3, 3なら0.2...
        ie_boost_fwd = (0.6 / dist_fwd) if dist_fwd else -0.4
        potent_fwd = sync_fwd + ie_boost_fwd
        ans_fwd = self.generate_answer(head, tail, potent_fwd, head)

        # 逆方向
        v_t_rev, _ = self.get_complex_state(tail)
        v_h_rev, _ = self.get_complex_state(head)
        sync_rev = F.cosine_similarity(v_t_rev.unsqueeze(0), v_h_rev.unsqueeze(0)).item()
        dist_rev = self.get_min_path_dist(self.S_edges, tail, head)
        
        ie_boost_rev = (0.6 / dist_rev) if dist_rev else -0.4
        potent_rev = sync_rev + ie_boost_rev
        ans_rev = self.generate_answer(tail, head, potent_rev, tail)

        print(f"\n[Final Results]")
        print(f"Direction {head} -> {tail}: {ans_fwd} (Dist: {dist_fwd}, Potent: {potent_fwd:.4f})")
        print(f"Direction {tail} -> {head}: {ans_rev} (Dist: {dist_rev}, Potent: {potent_rev:.4f})")
        
        print(f"\n[Audit Log] S_edges length: {len(self.S_edges)}")

if __name__ == "__main__":
    tester = CausalOS_v37_8()
    tester.run_discovery_test(["Age", "Shell weight"])
