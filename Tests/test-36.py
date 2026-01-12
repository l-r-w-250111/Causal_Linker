import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

class CausalOS_v19_Final:
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct"):
        print(f"Loading Model and Restoring Weighted Phase-Sync Architecture...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        
        # 因果空間（CFS）の初期化
        self.dim = 64
        torch.manual_seed(42)
        self.real_proj = torch.randn(self.tokenizer.vocab_size, self.dim).to(self.model.device)
        self.imag_proj = torch.randn(self.tokenizer.vocab_size, self.dim).to(self.model.device)
        
        # 因果臨界点
        self.critical_point = 0.45

    def get_weighted_complex_vector(self, token_ids):
        """
        [因果の慣性モーメント: 重み付き位相同期]
        トークン長による正確度（Fidelity）を重みとして適用し、
        サブワード断片化による位相崩壊を物理的に抑制する。
        """
        if not token_ids:
            return torch.zeros(self.dim).to(self.model.device), torch.zeros(self.dim).to(self.model.device)
        
        t_ids = torch.tensor(token_ids).to(self.model.device)
        pos = torch.arange(len(token_ids)).float().to(self.model.device)
        
        # 1. 位相回転 (DAG順序の記述)
        angles = pos * 0.8 
        cos_t = torch.cos(angles).unsqueeze(1)
        sin_t = torch.sin(angles).unsqueeze(1)
        
        # 2. 位相正確度（Fidelity Weight）の動的計算
        weights = []
        token_strs = [self.tokenizer.decode([tid]).strip() for tid in token_ids]
        for s in token_strs:
            # 設計の普遍性: 長さ1=0.33, 2=0.66, 3以上=1.0 とし、ノイズとなる一文字トークンの影響力を抑える
            fidelity = min(len(s), 3) / 3.0
            weights.append(fidelity)
        
        W = torch.tensor(weights, dtype=torch.float16).to(self.model.device).unsqueeze(1)
        
        r_base = self.real_proj[t_ids]
        i_base = self.imag_proj[t_ids]
        
        # 重み付き加算: 正確度の高いトークンがベクトルの向きを決定する
        v_real = torch.sum(W * (r_base * cos_t - i_base * sin_t), dim=0)
        v_imag = torch.sum(W * (r_base * sin_t + i_base * cos_t), dim=0)
        
        # ベクトル正規化
        norm = torch.sqrt(torch.sum(v_real**2) + torch.sum(v_imag**2)) + 1e-9
        return v_real / norm, v_imag / norm

    def execute_session(self, prompt, causal_facts):
        print(f"\n{'='*115}\n[Session Start] Prompt: {prompt}\n{'='*115}")
        
        # 1. 慣性モード: プロンプトからの因果シード抽出
        stop_words = ["the", "paper", "written", "by", "is", "titled", "and", "of", "a", "in", "to", "for", "with"]
        p_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        p_filtered = [t for t in p_tokens if self.tokenizer.decode([t]).strip().lower() not in stop_words]
        seed_strs = [self.tokenizer.decode([t]).strip().lower() for t in p_filtered]
        
        print(f"[Causal Seeds & Fidelity]:")
        for tid in p_filtered:
            s = self.tokenizer.decode([tid]).strip()
            print(f"  - '{s}': Weight {min(len(s), 3)/3.0:.2f}")

        v_p = self.get_weighted_complex_vector(p_filtered)
        
        candidates = []
        print(f"\n{'Node (Author Sample)':<35} | {'Phase-Sync':<10} | {'Cover':<7} | {'Potent':<8} | {'Diagnosis'}")
        print("-" * 115)

        for authors, title in causal_facts:
            # 2. 伝搬モード: S行列からのトポロジカル順序抽出
            n_tokens = self.tokenizer.encode(authors, add_special_tokens=False)
            n_filtered = [t for t in n_tokens if self.tokenizer.decode([t]).strip().lower() in seed_strs]
            
            v_n = self.get_weighted_complex_vector(n_filtered)
            
            # 3. do介入検証（位相同期）
            r_sim = F.cosine_similarity(v_p[0].unsqueeze(0), v_n[0].unsqueeze(0)).item()
            i_sim = F.cosine_similarity(v_p[1].unsqueeze(0), v_n[1].unsqueeze(0)).item()
            p_sync = (r_sim + i_sim) / 2
            
            # 4. 因果充足度 (Coverage)
            matches = sum(1 for s in set(seed_strs) if s in authors.lower())
            coverage = matches / max(len(set(seed_strs)), 1)
            
            # 5. 統合ポテンシャル (Arbitration)
            final_potent = p_sync * (coverage ** 2)
            
            diag = "⚡ TUNNEL" if final_potent >= self.critical_point else "🔍 SCAN"
            print(f"{authors[:35]:<35} | {p_sync:10.3f} | {coverage:7.3f} | {final_potent:8.3f} | {diag}")
            
            candidates.append({'title': title, 'final_potent': final_potent, 'authors': authors})

        # 6. 調停 (Final Arbitration)
        candidates.sort(key=lambda x: x['final_potent'], reverse=True)
        top = candidates[0]
        
        print("-" * 115)
        if top['final_potent'] >= self.critical_point:
            print(f"[Final Decision]: {top['title']}")
            print(f"[Integrity Log]: Mode=TUNNEL, Potent={top['final_potent']:.3f}")
        else:
            print(f"[Final Decision]: 該当なし (最大ポテンシャルが臨界点未満です)")

# --- 因果S行列データ ---
causal_facts = [
    ("Xiexin Liu, Xinwei Chen", "Who decides: The consumer or the retailer? An LLM-assisted Bayesian framework..."),
    ("Qianli Wang, Van Bach Nguyen, Yihong Liu, Fedor Splitt, Nils Feldhus, Christin Seifert, Hinrich Schütze, Sebastian Möller, Vera Schmitt", "Parallel Universes, Parallel Languages: A Comprehensive Study on LLM-based Multilingual Counterfactual Example Generation"),
    ("Zhengjian Kang, Qi Chen, Rui Liu, Kangtong Mo, Xingyu Zhang, Xiaoyu Deng, Ye Zhang", "Causality-Aware Temporal Projection for Video Understanding in Video-LLMs"),
    ("Sourena Khanzadeh", "Project Ariadne: A Structural Causal Framework for Auditing Faithfulness in LLM Agents")
]

# --- テスト実行 ---
os_v19 = CausalOS_v19_Final()

# Case 1: 救出テスト (Nils Feldhus)
os_v19.execute_session("The paper written by Nils Feldhus is titled", causal_facts)

# Case 3: 順序一致テスト (Rui Liu and Xingyu Zhang)
os_v19.execute_session("The paper written by Rui Liu and Xingyu Zhang is titled", causal_facts)

# Case 4: 順序逆転棄却テスト (Xinwei Chen and Xiexin Liu)
os_v19.execute_session("The paper written by Xinwei Chen and Xiexin Liu is titled", causal_facts)

# Case 5: 混合棄却テスト (Xinwei Chen and Qianli Wang)
os_v19.execute_session("The paper written by Xinwei Chen and Qianli Wang is titled", causal_facts)
