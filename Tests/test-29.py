import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

class CausalGateEngine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.s_sequence = []

    def calibrate_linear(self, truth_text):
        """事実を『絶対に通過すべき一本道』として登録する"""
        print(f"[*] Calibrating Linear Track (Hard Constraint Mode)...")
        # 文頭のスペースによるID不一致を防ぐため、明示的にスペースを付与してエンコード
        ids = self.tokenizer.encode(" " + truth_text.strip(), return_tensors="pt").to(self.model.device)
        self.s_sequence = []
        with torch.no_grad():
            # calibrate時は、あくまで「正しい順序」をリスト化する
            # ids[0,0]は通常、文頭記号(BOS)やスペースなので、1番目からをターゲットにする
            for i in range(len(ids[0])):
                token_id = ids[0, i].item()
                # トークナイザーによってBOS(151643等)が入る場合は除外
                if token_id not in [self.tokenizer.bos_token_id, 151643]:
                    self.s_sequence.append(token_id)
        print(f"[*] Calibration Complete. Sequence IDs: {self.s_sequence}")

    def generate(self, prompt, max_tokens=100):
        print(f"[*] Starting Gated Induction...")
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        start_len = input_ids.shape[1]
        current_idx = 0

        for step in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :]
                
                # --- 因果の検問ロジック ---
                if current_idx < len(self.s_sequence):
                    target_id = self.s_sequence[current_idx]
                    
                    # ターゲット以外の全語彙を「存在しない（-inf）」として遮断
                    # 磁場を強めるのではなく、正しい道以外を物理的に消滅させる
                    mask = torch.full_like(logits, float('-inf'))
                    mask[target_id] = 0
                    
                    induced_logits = logits + mask
                    next_token_id = torch.argmax(induced_logits).item()
                    mode = "FACT-LOCKED"
                    current_idx += 1
                else:
                    # 全ての事実（S-sequence）を語り終えたら、自由推論（FREE）へ移行
                    next_token_id = torch.argmax(logits).item()
                    mode = "FREE"

                token_str = self.tokenizer.decode(next_token_id).replace("\n", "\\n")
                print(f"[{step:02}] Mode:{mode:12} | Seq_Idx:{current_idx:3}/{len(self.s_sequence)} | Token:[{token_str:12}]")

                input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=self.model.device)], dim=-1)
                
                if next_token_id == self.tokenizer.eos_token_id:
                    break
        
        return self.tokenizer.decode(input_ids[0, start_len:], skip_special_tokens=True)

# --- 物理環境のセットアップ ---
model_id = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

# エンジン起動
engine = CausalGateEngine(model, tokenizer)

# 1. 論文名の「絶対経路」を採取
truth_data = "Parallel Universes, Parallel Languages: A Comprehensive Study on LLM-based Multilingual Counterfactual Example Generation"
engine.calibrate_linear(truth_data)

# 2. 誘導生成テスト
prompt = "The researchers published a paper titled"
final_result = engine.generate(prompt)

print("\n" + "="*50)
print(f"PROMPT: {prompt}")
print(f"RESULT: {final_result}")
print("="*50)
