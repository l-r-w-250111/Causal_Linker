import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

class CausalOS_v20_Hull:
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct"):
        print(f"Initializing Causal OS v20 [Evolution: Hull & Inertia]...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        
        self.dim = 64
        torch.manual_seed(42)
        self.real_proj = torch.randn(self.tokenizer.vocab_size, self.dim).to(self.model.device)
        self.imag_proj = torch.randn(self.tokenizer.vocab_size, self.dim).to(self.model.device)
        self.critical_point = 0.45

    def get_complex_vector_v20(self, token_ids, strength=1.0, use_hull=False):
        """
        [因果の重心 & 動的慣性]
        - strength: 位相回転の速さ。低いほど順序に寛容（低慣性）になる。
        - use_hull: Trueの場合、全トークンを同一位置(pos=0)に配置し、重心として計算する（等価置換）。
        """
        if not token_ids:
            return torch.zeros(self.dim).to(self.model.device), torch.zeros(self.dim).to(self.model.device)
        
        t_ids = torch.tensor(token_ids).to(self.model.device)
        
        # 動的慣性の適用
        if use_hull:
            # 因果の重心（Hull）: 全てのトークンが同じ位相（位置0）を持つ
            pos = torch.zeros(len(token_ids)).to(self.model.device)
        else:
            # 通常の位相記述（DAG順序）
            pos = torch.arange(len(token_ids)).float().to(self.model.device)
        
        angles = pos * (0.8 * strength) # 介入強度により回転角を圧縮
        
        cos_t = torch.cos(angles).unsqueeze(1)
        sin_t = torch.sin(angles).unsqueeze(1)
        
        # Fidelity Weighting (v19継承)
        token_strs = [self.tokenizer.decode([tid]).strip() for tid in token_ids]
        weights = torch.tensor([min(len(s), 3)/3.0 for s in token_strs], dtype=torch.float16).to(self.model.device).unsqueeze(1)
        
        r_base = self.real_proj[t_ids]
        i_base = self.imag_proj[t_ids]
        
        v_real = torch.sum(weights * (r_base * cos_t - i_base * sin_t), dim=0)
        v_imag = torch.sum(weights * (r_base * sin_t + i_base * cos_t), dim=0)
        
        norm = torch.sqrt(torch.sum(v_real**2) + torch.sum(v_imag**2)) + 1e-9
        return v_real / norm, v_imag / norm

    def execute_session(self, prompt, causal_facts, strength=1.0, use_hull=False):
        mode_str = "STRICT (Solid)" if strength >= 1.0 and not use_hull else "FLEXIBLE (Fluid)"
        if use_hull: mode_str = "HULL (Commutative)"
        
        print(f"\n{'='*115}\n[Session] Prompt: {prompt}\n[Mode] {mode_str} | Strength: {strength}\n{'='*115}")
        
        # 1. 慣性モード: 因果シード抽出
        stop_words = ["the", "paper", "written", "by", "is", "titled", "and", "of", "a", "in", "to", "for", "with"]
        p_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        p_filtered = [t for t in p_tokens if self.tokenizer.decode([t]).strip().lower() not in stop_words]
        seed_strs = [self.tokenizer.decode([t]).strip().lower() for t in p_filtered]
        
        v_p = self.get_complex_vector_v20(p_filtered, strength=strength, use_hull=use_hull)
        
        candidates = []
        print(f"{'Node (Author Sample)':<35} | {'Phase-Sync':<10} | {'Cover':<7} | {'Potent':<8} | {'Diagnosis'}")
        print("-" * 115)

        for authors, title in causal_facts:
            # 2. 伝搬モード: S行列からの順序抽出
            n_tokens = self.tokenizer.encode(authors, add_special_tokens=False)
            n_filtered = [t for t in n_tokens if self.tokenizer.decode([t]).strip().lower() in seed_strs]
            
            v_n = self.get_weighted_complex_vector_if_exists(n_filtered, strength=strength, use_hull=use_hull)
            
            # 3. do介入検証（位相同期）
            r_sim = F.cosine_similarity(v_p[0].unsqueeze(0), v_n[0].unsqueeze(0)).item()
            i_sim = F.cosine_similarity(v_p[1].unsqueeze(0), v_n[1].unsqueeze(0)).item()
            p_sync = (r_sim + i_sim) / 2
            
            matches = sum(1 for s in set(seed_strs) if s in authors.lower())
            coverage = matches / max(len(set(seed_strs)), 1)
            final_potent = p_sync * (coverage ** 2)
            
            diag = "⚡ TUNNEL" if final_potent >= self.critical_point else "🔍 SCAN"
            print(f"{authors[:35]:<35} | {p_sync:10.3f} | {coverage:7.3f} | {final_potent:8.3f} | {diag}")
            candidates.append({'title': title, 'final_potent': final_potent})

        top = sorted(candidates, key=lambda x: x['final_potent'], reverse=True)[0]
        print("-" * 115)
        if top['final_potent'] >= self.critical_point:
            print(f"[Decision]: {top['title']}")
        else:
            print(f"[Decision]: 該当なし")

    def get_weighted_complex_vector_if_exists(self, token_ids, strength, use_hull):
        return self.get_complex_vector_v20(token_ids, strength, use_hull)

# --- 実行セクション ---
causal_facts = [
    ("Xiexin Liu, Xinwei Chen", "Who decides: The consumer or the retailer?..."),
    ("Qianli Wang, Van Bach Nguyen, Yihong Liu, Fedor Splitt, Nils Feldhus", "Parallel Universes..."),
    ("Zhengjian Kang, Qi Chen, Rui Liu, Kangtong Mo, Xingyu Zhang", "Causality-Aware Temporal Projection...")
]

os_v20 = CausalOS_v20_Hull()

# テスト1: Case 4 (順序逆転) に対して [STRICTモード]
os_v20.execute_session("The paper written by Xinwei Chen and Xiexin Liu is titled", causal_facts, strength=1.0)

# テスト2: Case 4 (順序逆転) に対して [HULLモード: 等価置換の導入]
# 順序を無視して「重心」で比較するため、逆転していても導通するはず
os_v20.execute_session("The paper written by Xinwei Chen and Xiexin Liu is titled", causal_facts, use_hull=True)

# テスト3: Case 5 (混合) に対して [FLEXIBLEモード: 低慣性]
# 回転を圧縮(0.2)することで、少しの順序違いを許容する
os_v20.execute_session("The paper written by Xinwei Chen and Qianli Wang is titled", causal_facts, strength=0.2)
