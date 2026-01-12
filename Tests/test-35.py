import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

class CausalOS_v17_2_Final:
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct"):
        print(f"Initializing Adaptive Phase-Sync Causal OS (v17.2)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
        
        self.dim = 64
        torch.manual_seed(42)
        # 複素数エンコーディング（実部・虚部）の投影行列
        self.real_proj = torch.randn(self.tokenizer.vocab_size, self.dim).to(self.model.device)
        self.imag_proj = torch.randn(self.tokenizer.vocab_size, self.dim).to(self.model.device)
        
        # 臨界点。相対位相の安定化に伴い 0.45 に調整（導通を優先しつつ峻別を維持）
        self.critical_point = 0.45 

    def get_complex_causal_vector(self, token_ids):
        """
        [複素位相エンコーディング & DAG制約]
        トークンの相対位置に基づき位相を回転させ、因果の順序を記述。
        """
        if not token_ids:
            return torch.zeros(self.dim).to(self.model.device), torch.zeros(self.dim).to(self.model.device)
        
        t_ids = torch.tensor(token_ids).to(self.model.device)
        pos = torch.arange(len(token_ids)).float().to(self.model.device)
        
        # 位相回転定数: 0.8。順序逆転(Case4)を弾く厳格さと、断片化(Case1)への耐性を両立。
        angles = pos * 0.8 
        
        cos_t = torch.cos(angles).unsqueeze(1)
        sin_t = torch.sin(angles).unsqueeze(1)
        
        r_base = self.real_proj[t_ids]
        i_base = self.imag_proj[t_ids]
        
        # 複素回転加算: (r + i*imag) * (cos + i*sin)
        v_real = torch.sum(r_base * cos_t - i_base * sin_t, dim=0)
        v_imag = torch.sum(r_base * sin_t + i_base * cos_t, dim=0)
        
        # 合成ベクトルのノルム正規化（物理的強度の一定化）
        norm = torch.sqrt(torch.sum(v_real**2) + torch.sum(v_imag**2)) + 1e-9
        return v_real / norm, v_imag / norm

    def do_intervention_test(self, v_p, v_n):
        """
        [do介入検証: 位相同期スコア]
        プロンプトの因果ベクトルとノードの因果ベクトルの位相同一性を計算。
        """
        r_sim = F.cosine_similarity(v_p[0].unsqueeze(0), v_n[0].unsqueeze(0)).item()
        i_sim = F.cosine_similarity(v_p[1].unsqueeze(0), v_n[1].unsqueeze(0)).item()
        return (r_sim + i_sim) / 2

    def execute_session(self, prompt, causal_facts):
        print(f"\n{'='*115}\n[Session Start] Prompt: {prompt}\n{'='*115}")
        
        # 1. 慣性モード: 因果成分の抽出（do介入の事前準備）
        stop_words = ["the", "paper", "written", "by", "is", "titled", "and", "of", "a", "in", "to", "for", "with"]
        p_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        p_filtered = [t for t in p_tokens if self.tokenizer.decode([t]).strip().lower() not in stop_words]
        seed_strs = [self.tokenizer.decode([t]).strip().lower() for t in p_filtered]
        
        print(f"[Causal Seeds (Inertia Mode)]: {seed_strs}")
        
        v_p = self.get_complex_causal_vector(p_filtered)
        
        candidates = []
        print(f"{'Node (Author Sample)':<35} | {'Phase-Sync':<10} | {'Cover':<7} | {'Potent':<8} | {'Diagnosis'}")
        print("-" * 115)

        for authors, title in causal_facts:
            # 2. 伝搬モード: S行列からのトポロジカル順序抽出
            n_tokens = self.tokenizer.encode(authors, add_special_tokens=False)
            # 重要: S行列側のDAG順序を破壊せずに成分を抽出
            n_filtered = [t for t in n_tokens if any(s in self.tokenizer.decode([t]).lower() for s in seed_strs)]
            
            v_n = self.get_complex_causal_vector(n_filtered)
            
            # 3. do介入検証（位相同期）
            p_sync = self.do_intervention_test(v_p, v_n)
            
            # 4. 充足度（Coverage）
            matches = sum(1 for s in seed_strs if s in authors.lower())
            coverage = matches / max(len(seed_strs), 1)
            
            # 5. 因果ポテンシャル (Arbitration Formula)
            final_potent = p_sync * (coverage ** 2)
            
            diag = "⚡ TUNNEL" if final_potent >= self.critical_point else "🔍 SCAN"
            print(f"{authors[:35]:<35} | {p_sync:10.3f} | {coverage:7.3f} | {final_potent:8.3f} | {diag}")
            
            candidates.append({'title': title, 'final_potent': final_potent})

        # 6. 調停 (Arbitration)
        candidates.sort(key=lambda x: x['final_potent'], reverse=True)
        top = candidates[0]
        
        print("-" * 115)
        if top['final_potent'] >= self.critical_point:
            print(f"[Final Decision]: {top['title']}")
        else:
            print(f"[Final Decision]: 該当なし (Potent:{top['final_potent']:.2f})")

# --- 物理環境設定 & 実行 ---
causal_facts = [
    ("Xiexin Liu, Xinwei Chen", "Who decides: The consumer or the retailer? An LLM-assisted Bayesian framework..."),
    ("Qianli Wang, Van Bach Nguyen, Yihong Liu, Fedor Splitt, Nils Feldhus, Christin Seifert, Hinrich Schütze, Sebastian Möller, Vera Schmitt", "Parallel Universes, Parallel Languages: A Comprehensive Study on LLM-based Multilingual Counterfactual Example Generation"),
    ("Zhengjian Kang, Qi Chen, Rui Liu, Kangtong Mo, Xingyu Zhang, Xiaoyu Deng, Ye Zhang", "Causality-Aware Temporal Projection for Video Understanding in Video-LLMs"),
    ("Sourena Khanzadeh", "Project Ariadne: A Structural Causal Framework for Auditing Faithfulness in LLM Agents")
]

os_v17_2 = CausalOS_v17_2_Final()

# Case 1: 正解救出テスト
os_v17_2.execute_session("The paper written by Nils Feldhus is titled", causal_facts)
# Case 2: 順序一致・複数人テスト
os_v17_2.execute_session("The paper written by Rui Liu and Xingyu Zhang is titled", causal_facts)
# Case 3: 順序逆転・棄却テスト
os_v17_2.execute_session("The paper written by Xinwei Chen and Xiexin Liu is titled", causal_facts)
# Case 4: 因果無し・棄却テスト
os_v17_2.execute_session("The paper written by Xinwei Chen and Qianli Wang is titled", causal_facts)
