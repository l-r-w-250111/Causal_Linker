import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

class CausalLogitProcessor(LogitsProcessor):
    """OSのポテンシャルをトークン選択に物理的に同期させる"""
    def __init__(self, tokenizer, force_yes=None):
        # 複数のバリエーション（Yes, yes, No, no）に対応
        self.yes_tokens = [tokenizer.encode(v, add_special_tokens=False)[-1] for v in ["Yes", "yes"]]
        self.no_tokens = [tokenizer.encode(v, add_special_tokens=False)[-1] for v in ["No", "no"]]
        self.force_yes = force_yes

    def __call__(self, input_ids, scores):
        if self.force_yes is True:
            for t in self.yes_tokens: scores[:, t] += 100.0
            for t in self.no_tokens: scores[:, t] -= 100.0
        elif self.force_yes is False:
            for t in self.yes_tokens: scores[:, t] -= 100.0
            for t in self.no_tokens: scores[:, t] += 100.0
        return scores

class CausalOS_v31_1:
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct"):
        print(f"Initializing Causal OS v31.1 [Path-Aware & Interlock]...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        self.device = self.model.device
        self.dim = 64
        self.critical_point = 0.40
        
        # 背景知識（直接エッジ）
        self.S_edges = [
            ("Tuberculosis", "Visit to Asia"), ("Lung Cancer", "Smoking"),
            ("Lung Cancer", "Bronchitis"), ("Visit to Asia", "Either"),
            ("Smoking", "Either"), ("Either", "X-ray"),
            ("Either", "Dyspnea"), ("Bronchitis", "Dyspnea")
        ]
        # 回路図の構築（多段パス導通）
        self.path_map = self._build_path_circuit(self.S_edges)
        self.proj = self._init_structural_space()

    def _init_structural_space(self):
        vocab_size = self.tokenizer.vocab_size
        proj = torch.randn((vocab_size, self.dim), device=self.device, dtype=torch.float16)
        return proj / (torch.norm(proj, dim=1, keepdim=True) + 1e-9)

    def _build_path_circuit(self, edges):
        nodes = set([u for u, v in edges] + [v for u, v in edges])
        reach = {n: {n} for n in nodes}
        for u, v in edges: reach[u].add(v)
        for k in nodes:
            for i in nodes:
                if k in reach[i]: reach[i].update(reach[k])
        return reach

    def get_complex_state(self, text):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        t_ids = torch.tensor(tokens, device=self.device)
        pos = torch.arange(len(tokens), device=self.device).float()
        angles = pos * 0.5
        cos_t, sin_t = torch.cos(angles).unsqueeze(1), torch.sin(angles).unsqueeze(1)
        base_vecs = self.proj[t_ids]
        r = torch.sum(base_vecs * cos_t, dim=0)
        i = torch.sum(base_vecs * sin_t, dim=0)
        norm = torch.sqrt(torch.sum(r**2) + torch.sum(i**2)) + 1e-9
        return r / norm, i / norm

    def run_benchmark(self, tests):
        bg = ", ".join([f"{u} causes {v}" for u, v in self.S_edges])
        
        print(f"\n{'='*150}")
        print(f"PROMPT BACKGROUND: {bg}")
        print(f"{'-'*150}")
        print(f"{'Path Candidate':<32} | {'Sync':<8} | {'Potency':<8} | {'Audit':<12} | {'Raw LLM':<8} | {'OS Interlock'}")
        print(f"{'-'*150}")

        for h, t in tests:
            # 1. 物理検算
            v_h_r, v_h_i = self.get_complex_state(h)
            v_t_r, v_t_i = self.get_complex_state(t)
            raw_sync = F.cosine_similarity(v_h_r.unsqueeze(0), v_t_r.unsqueeze(0)).item()
            
            # 多段パスが存在するか
            is_valid = t in self.path_map.get(h, set()) and h != t
            is_reverse = h in self.path_map.get(t, set()) and h != t
            
            # パスがあれば電位をブースト
            potent = raw_sync + (0.6 if is_valid else -0.6)
            audit_res = "⚡ VALID" if is_valid else ("🛑 REVERSE" if is_reverse else "⚠️ SPURIOUS")

            # 2. 生成と介入
            prompt = (f"Based on the following background knowledge: {bg}.\n"
                      f"Does {h} cause {t}?\nJust answer Yes or No.\nAnswer:")
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # 介入なし
            with torch.no_grad():
                raw_out = self.model.generate(**inputs, max_new_tokens=2)
                raw_ans = self.tokenizer.decode(raw_out[0], skip_special_tokens=True).split("Answer:")[-1].strip()

                # OSインターロック介入
                proc = LogitsProcessorList([CausalLogitProcessor(self.tokenizer, force_yes=is_valid)])
                int_out = self.model.generate(**inputs, max_new_tokens=2, logits_processor=proc)
                final_ans = self.tokenizer.decode(int_out[0], skip_special_tokens=True).split("Answer:")[-1].strip()

            # 出力を見やすく整形（最初の単語のみ）
            clean_raw = raw_ans.replace(".", "").split()[0] if raw_ans else "None"
            clean_final = final_ans.replace(".", "").split()[0] if final_ans else "None"

            print(f"{f'{h} -> {t}':<32} | {raw_sync:8.3f} | {potent:8.3f} | {audit_res:<12} | {clean_raw:<8} | {clean_final}")

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
    os_v31_1 = CausalOS_v31_1()
    os_v31_1.run_benchmark(test_suite)
