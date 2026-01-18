import torch
import torch.nn.functional as F
import re
import json
import os
import itertools
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

# ==========================================================
#  LOGITS PROCESSOR
# ==========================================================
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

# ==========================================================
#  MAIN SYSTEM: v37.21
# ==========================================================
class CausalOS_v37_21:
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct"):
        print("Initializing CausalOS v37.21 [Generalized Asymmetry]...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto"
        )
        self.device = self.model.device
        self.dim = 64
        self.critical_point = 0.40
        self.S_edges = []
        self.memory_path = "S_memory.json"
        self.proj = self._init_structural_space()
        self.load_S()

    def _init_structural_space(self):
        vocab_size = self.tokenizer.vocab_size
        proj = torch.randn((vocab_size, self.dim), device=self.device, dtype=torch.float16)
        return proj / (torch.norm(proj, dim=1, keepdim=True) + 1e-9)

    def save_S(self):
        serializable_edges = [list(e) for e in list(set(self.S_edges))]
        with open(self.memory_path, "w") as f:
            json.dump(serializable_edges, f, indent=2)
        print(f"[Log] S-Memory updated: {len(serializable_edges)} edges.")

    def load_S(self):
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "r") as f:
                    content = f.read().strip()
                    if content:
                        self.S_edges = [tuple(e) for e in json.loads(content)]
                        print(f"[Log] S-Memory loaded: {len(self.S_edges)} edges.")
            except: self.S_edges = []

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

    def embed_node(self, text):
        v, _ = self.get_complex_state(text)
        return v

    def merge_similar_nodes(self, nodes, threshold=0.78):
        if not nodes: return []
        embeddings = [self.embed_node(n) for n in nodes]
        clusters, used = [], set()
        for i, n in enumerate(nodes):
            if i in used: continue
            group = [n]
            used.add(i)
            for j, m in enumerate(nodes):
                if j in used: continue
                sim = F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)).item()
                if sim > threshold:
                    group.append(m)
                    used.add(j)
            clusters.append(group)
        return [g[0] for g in clusters]

    def get_min_path_dist(self, edges, start, end):
        adj = {}
        for u, v in edges: adj.setdefault(u, []).append(v)
        queue, visited = [(start, 0)], {start}
        while queue:
            curr, dist = queue.pop(0)
            if curr == end: return dist
            for nxt in adj.get(curr, []):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, dist + 1))
        return None

    def audit_direction(self, node_a, node_b):
        """包括的な因果の非対称性を判定"""
        prompt = f"""Compare two variables:
A: {node_a}
B: {node_b}

Criteria for 'A -> B':
1. Prerequisite: Does B require A as a necessary condition to exist?
2. Constraint: Does A set the limits for B's variation?
3. Flow: Is B a manifestation or outcome of the state of A?

Identify the dominant direction. Answer exactly: 'A -> B', 'B -> A', or 'None'.
Answer:"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out_ids = self.model.generate(**inputs, max_new_tokens=10)
        res = self.tokenizer.decode(out_ids[0], skip_special_tokens=True).split("Answer:")[-1].strip()
        if "A -> B" in res: return (node_a, node_b)
        if "B -> A" in res: return (node_b, node_a)
        return None

    def autonomous_exploration(self, variables):
        print("\n[Log] Phase 1: Constitutive Decomposition (Cleaned)")
        extracted_nodes = set(variables)
        
        for var in variables:
            prompt = f"""Break down '{var}' into 4 specific constituent parameters.
Output ONLY a comma-separated list of names. No explanation, no intro, no synonyms.
Output:"""
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out_ids = self.model.generate(**inputs, max_new_tokens=60)
            res = self.tokenizer.decode(out_ids[0], skip_special_tokens=True).split("Output:")[-1].strip()
            
            for v in res.split(","):
                # ゴミ取りパース
                cleaned = v.split("\n")[0].strip().strip(".")
                cleaned = re.sub(r'(?i)(Example|Note|Context|Format):.*', '', cleaned).strip()
                # ターゲット文字列との重複排除
                is_duplicate = any(p.lower() in cleaned.lower() for p in ["clay", "carbon", "soil"])
                if len(cleaned) > 2 and not is_duplicate:
                    extracted_nodes.add(cleaned)

        nodes = self.merge_similar_nodes(list(extracted_nodes), threshold=0.78)
        print(f"Nodes identified: {nodes}")

        print("\n[Log] Phase 1.5: Pairwise Directional Audit")
        new_edges = []
        for a, b in itertools.combinations(nodes, 2):
            # 不適切なノード名のスキップ
            if any(x in a.lower() or x in b.lower() for x in ["example", "context", "note"]): continue
            
            edge = self.audit_direction(a, b)
            if edge:
                print(f"  - System confirmed: {edge[0]} -> {edge[1]}")
                new_edges.append(edge)
        return new_edges

    def run_discovery_test(self, variables):
        new_edges = self.autonomous_exploration(variables)
        print("\n[Log] Phase 2: Building S-Memory")
        for u, v in new_edges:
            if (u, v) not in self.S_edges: self.S_edges.append((u, v))
        self.save_S()

        print("\n[Log] Phase 3: Path Auditing")
        head, tail = variables[0], variables[1]
        results = []
        for h, t in [(head, tail), (tail, head)]:
            dist = self.get_min_path_dist(self.S_edges, h, t)
            v_h, _ = self.get_complex_state(h)
            v_t, _ = self.get_complex_state(t)
            sync = F.cosine_similarity(v_h.unsqueeze(0), v_t.unsqueeze(0)).item()

            if dist is None:
                parents_h = {u for u, v in self.S_edges if v == h}
                parents_t = {u for u, v in self.S_edges if v == t}
                shared = parents_h & parents_t
                ie_boost = 0.2 if shared else -0.3
            else:
                ie_boost = 0.8 / (dist + 1) # パス依存の信頼度を強化

            potent = sync + ie_boost
            prompt = f"Structural Evidence: {self.S_edges}.\nUnder do({h}), does {t} change?\nAnswer Yes/No:"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            proc = LogitsProcessorList([IELogitProcessor(self.tokenizer, potent, self.critical_point)])
            
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=2, logits_processor=proc, pad_token_id=self.tokenizer.eos_token_id)
            ans = self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            results.append((h, t, dist, potent, "Yes" if "Yes" in ans else "No"))

        print("\n[Final Results (v37.21)]")
        for h, t, d, p, a in results:
            print(f"Direction {h} -> {t}: {a} (Dist: {d}, Potent: {p:.4f})")

if __name__ == "__main__":
    tester = CausalOS_v37_21()
    tester.run_discovery_test(["organic carbon in soil in forest", "clay content in soil in forest"])
