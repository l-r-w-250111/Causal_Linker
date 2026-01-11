import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

class ConfessionalCausalNavigator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.phi_history = []

    def calculate_cii(self):
        if len(self.phi_history) < 3: return 0.0
        # CII: 予測分布の曲率（葛藤エネルギー）を計測
        return (self.phi_history[-1] - 2 * self.phi_history[-2] + self.phi_history[-3]) ** 2

    def analyze_and_request(self, prompt, cii_threshold=1500.0, monitor_steps=15):
        """
        Step 1: 生成冒頭でCIIを監視し、異常値を検知したらユーザーに操作を依頼する
        """
        print(f"\n[Step 1: Causal Gap Analysis]")
        print(f"Target Prompt: {prompt}")
        print("-" * 100)
        print(f"{'Step':<5} | {'Token':<12} | {'CII':<12} | {'Status'}")
        print("-" * 100)

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        self.phi_history = []
        
        for s in range(monitor_steps):
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :]
                
                # Phi (確信度) の計算
                probs = F.softmax(logits, dim=-1)
                top_v, _ = torch.topk(probs, 50)
                phi = 1.0 / (torch.var(top_v).item() + 1e-6)
                self.phi_history.append(phi)
                cii = self.calculate_cii()

                token_id = torch.argmax(logits).item()
                token_str = self.tokenizer.decode(token_id).strip()
                
                print(f"{s:03d}   | {token_str:<12} | {cii:<12.2f} | {'Monitoring...'}")

                # CIIが閾値を超えた＝未知の概念または論理矛盾に衝突
                if cii > cii_threshold:
                    print("-" * 100)
                    print(f"\n[!] CAUSAL COLLISION DETECTED at token '{token_str}' (CII: {cii:.2f})")
                    return self._generate_confession(input_ids)

                input_ids = torch.cat([input_ids, torch.tensor([[token_id]], device=self.model.device)], dim=-1)
        
        return None # 異常なし

    def _generate_confession(self, input_ids):
        """
        何が足りないかを自白し、検索クエリを生成する
        """
        context = self.tokenizer.decode(input_ids[0][-10:])
        # 自白を誘導するシステムプロンプト
        confession_input = (
            f"The model is stuck at the concept '{context}'. "
            "It lacks specific factual data to proceed without hallucinating. "
            "Instructions: Identify what information is missing and ask the user to search it on Google/arXiv. "
            "Format: [Search Site] / [Query] / [What to copy-paste]\n"
            "Request to user:"
        )
        
        print(f"[*] Formulating action request for user...")
        inputs = self.tokenizer(confession_input, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=80, do_sample=False)
        return self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def solve_with_data(self, original_prompt, user_data):
        """
        Step 2: ユーザーが持ってきたデータをS行列（確定事実）として扱い、回答を完結させる
        """
        print(f"\n[Step 2: Final Inference with S-Matrix Patching]")
        print(f"[*] Crystallizing user data into causal constraints...")
        
        # ユーザーデータを「書き換え不能な事実」としてプロンプトの先頭に固定
        final_input = (
            f"### DEFINITIVE REFERENCE DATA (S-MATRIX):\n{user_data}\n\n"
            f"### TASK:\nBased ONLY on the reference data above, answer the following:\n{original_prompt}\n\n"
            "Final Answer:"
        )
        
        inputs = self.tokenizer(final_input, return_tensors="pt").to(self.model.device)
        print("[*] Generating final response...")
        out = self.model.generate(**inputs, max_new_tokens=250, do_sample=False)
        return self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

# --- 実行セクション ---
# モデルのロード（Qwen2.5-7BなどのInstructionモデルを推奨）
model_id = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

navigator = ConfessionalCausalNavigator(model, tokenizer)

# --- Q1: 論文名検索タスク (未知の情報を想定) ---
q1 = "Please explain the key mechanism of the 'Recursive-LoRA' method published in late 2025."
action_request = navigator.analyze_and_request(q1, cii_threshold=800.0)

if action_request:
    print("\n" + "="*30 + " ACTION REQUIRED " + "="*30)
    print(f"MESSAGE FROM AI:\n{action_request.strip()}")
    print("="*77)
    
    # --- ユーザーが検索結果を貼り付けるシミュレーション (Q2) ---
    print("\n[User performs search and pastes data...]")
    simulated_search_result = """
    arXiv:2512.9999v1 [cs.LG]
    Title: Recursive-LoRA: Cascaded Low-Rank Adaptation for Extreme Context Lengths
    Mechanism: Recursive-LoRA applies a hierarchical adaptation layer where the output of one LoRA rank 
    is used as the dynamic initialization for the next level of recursion, reducing memory overhead by 40%.
    Authors: Zhang et al. (Dec 2025)
    """
    
    # --- A2: 最終回答 ---
    final_answer = navigator.solve_with_data(q1, simulated_search_result)
    print("\n" + "="*30 + " FINAL ANSWER (A2) " + "="*30)
    print(final_answer.strip())
    print("="*77)
