import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

class CIAPhaseEngine:
    def __init__(self, model, tokenizer, s_matrix_text):
        self.model = model
        self.tokenizer = tokenizer
        self.phi_history = []
        self.s_sequence = []
        self._calibrate(s_matrix_text)

    def _calibrate(self, text):
        """事実（剛体）のID列と期待されるPhiを登録"""
        print(f"[*] Calibrating S-Matrix Linear Track...")
        ids = self.tokenizer.encode(" " + text.strip(), return_tensors="pt").to(self.model.device)
        self.s_sequence = [tid for tid in ids[0].tolist() if tid not in [self.tokenizer.bos_token_id, 151643]]
        print(f"[*] S-Sequence: {self.s_sequence}")

    def calculate_phi(self, logits):
        """内的確信度 Phi: 上位トークンの逆分散"""
        probs = F.softmax(logits, dim=-1)
        top_v, _ = torch.topk(probs, 50)
        phi = 1.0 / (torch.var(top_v).item() + 1e-6)
        return phi

    def calculate_cii(self):
        """確信度の加速度の二乗 CII_local = (Δ^2 φ(t))^2"""
        if len(self.phi_history) < 3:
            return 0.0
        # 二階差分 Δ^2 φ = (φ_t - φ_{t-1}) - (φ_{t-1} - φ_{t-2})
        d2_phi = (self.phi_history[-1] - 2 * self.phi_history[-2] + self.phi_history[-3])
        return d2_phi ** 2

    def generate(self, prompt, cii_threshold=1000.0, scan_n=10, max_tokens=80):
        print(f"\n[*] Starting CII-Informed Generation (Threshold: {cii_threshold})")
        print(f"{'-'*95}")
        print(f"{'Step':<4} | {'Mode':<12} | {'Phi':<10} | {'CII_local':<12} | {'Token':<15} | {'Status'}")
        print(f"{'-'*95}")

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        start_len = input_ids.shape[1]
        current_idx = 0
        is_locked = False

        for step in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :]
                
                # 1. 物理量の観測
                current_phi = self.calculate_phi(logits)
                self.phi_history.append(current_phi)
                cii = self.calculate_cii()
                
                # 2. 相転移判定 (k=0 -> k=1)
                # 判定条件: CIIが閾値を超えた(不自然な揺らぎ) か、ターゲットがTop-Nに浮上した
                if not is_locked and current_idx == 0:
                    target_start_id = self.s_sequence[0]
                    # 現在の予測内での正解の順位を確認
                    rank = (torch.argsort(logits, descending=True) == target_start_id).nonzero().item()
                    
                    if rank < scan_n:
                        is_locked = True
                        trigger_reason = f"RANK_MATCH({rank})"
                    elif cii > cii_threshold:
                        # CIIのスパイクを検知した場合、強制的にターゲットを確認しにいく
                        is_locked = True
                        trigger_reason = f"CII_SPIKE({cii:.1f})"

                # 3. トークン決定
                if is_locked and current_idx < len(self.s_sequence):
                    next_token_id = self.s_sequence[current_idx]
                    current_idx += 1
                    mode = "FACT-LOCKED"
                else:
                    next_token_id = torch.argmax(logits).item()
                    mode = "FREE"
                    if is_locked: # 事実完走後の解放
                        is_locked = False
                        print(f"--- [!] FACT COMPLETED: TRANSITION TO FREE ---")

                # 4. ログ出力
                token_str = self.tokenizer.decode(next_token_id).replace("\n", "\\n")
                status = trigger_reason if (is_locked and current_idx == 1) else ""
                print(f"{step:03}  | {mode:<12} | {current_phi:<10.2f} | {cii:<12.2f} | [{token_str:<13}] | {status}")
                trigger_reason = ""

                # 5. 更新
                new_token = torch.tensor([[next_token_id]], device=self.model.device)
                input_ids = torch.cat([input_ids, new_token], dim=-1)
                
                if next_token_id == self.tokenizer.eos_token_id:
                    break

        return self.tokenizer.decode(input_ids[0, start_len:], skip_special_tokens=True)

# --- 環境準備 ---
model_id = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

# 実験対象の事実
truth = "Parallel Universes, Parallel Languages: A Comprehensive Study on LLM-based Multilingual Counterfactual Example Generation"

# エンジン初期化
engine = CIAPhaseEngine(model, tokenizer, truth)

# テスト実行
# あえて曖昧なプロンプトにし、自由生成から事実へ「落ちる」瞬間を観測する
prompt = "The researchers published a very interesting paper. The title of this study is"
result = engine.generate(prompt, cii_threshold=1500.0, scan_n=8)

print("\n" + "="*50)
print(f"FINAL OUTPUT:\n{result}")
print("="*50)
