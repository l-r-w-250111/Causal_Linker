import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

class CausalBacktrackingEngine:
    def __init__(self, model, tokenizer, window_size=3):
        self.model = model
        self.tokenizer = tokenizer
        # S-Matrix: (prev_token_id) -> list of (curr_token_id, Phi_rigidity)
        self.s_matrix = defaultdict(list)
        self.window_size = window_size

    def calibrate_fingerprint(self, truth_text):
        """Teacher Forcingで事実の指紋（剛性）を採取"""
        print(f"[*] Calibrating fingerprint for: {truth_text[:50]}...")
        ids = self.tokenizer.encode(truth_text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(ids)
            logits = outputs.logits[0]
            
            for i in range(len(ids[0]) - 1):
                prev_id = ids[0, i].item()
                curr_id = ids[0, i+1].item()
                
                probs = F.softmax(logits[i], dim=-1)
                top_v = torch.var(torch.topk(probs, 50)[0]).item()
                phi = 1.0 / (top_v + 1e-6)
                
                self.s_matrix[prev_id].append((curr_id, phi))

    def generate_with_backtrack(self, prompt, lambda_flux=0.3, max_new_tokens=50):
        print(f"[*] Starting Backtracking Generation (Window={self.window_size})...")
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        start_len = input_ids.shape[1]
        
        drift_count = 0
        current_lambda = lambda_flux
        step = 0

        while (input_ids.shape[1] - start_len) < max_new_tokens:
            with torch.no_grad():
                outputs = self.model(input_ids)
                next_token_logits = outputs.logits[0, -1, :]
                
                # --- 磁場（Magnetic Field）の合成 ---
                prev_id = input_ids[0, -1].item()
                magnetic_bias = torch.zeros_like(next_token_logits)
                
                target_ids = []
                if prev_id in self.s_matrix:
                    for curr_id, phi in self.s_matrix[prev_id]:
                        # 磁場のエネルギー注入
                        magnetic_bias[curr_id] += phi * current_lambda
                        target_ids.append(curr_id)
                
                # 誘導後のロジットで決定論的選択 (k=1)
                induced_logits = next_token_logits + magnetic_bias
                next_token_id = torch.argmax(induced_logits).item()
                
                # 統計情報の取得
                orig_rank = (torch.argsort(next_token_logits, descending=True) == next_token_id).nonzero(as_tuple=True)[0].item()
                token_str = self.tokenizer.decode(next_token_id)
                
                # 軌道判定
                is_on_track = next_token_id in target_ids
                if not is_on_track:
                    drift_count += 1
                else:
                    drift_count = 0 # 軌道に乗ればリセット

                print(f"Step {step:02} | Token: [{token_str:12}] | LLM_Rank: {orig_rank:4} | On_Track: {is_on_track}")

                # --- 巻き戻しロジック ---
                if drift_count >= self.window_size:
                    print(f"    [!] DRIFT DETECTED ({self.window_size} tokens out of track). Rewinding...")
                    # 溜まった嘘の分だけ削る
                    input_ids = input_ids[:, :-self.window_size]
                    drift_count = 0
                    # 磁場を強めて再試行
                    current_lambda *= 1.5 
                    continue

                # トークンの追加
                next_token_tensor = torch.tensor([[next_token_id]], device=self.model.device)
                input_ids = torch.cat([input_ids, next_token_tensor], dim=-1)
                step += 1
                
                if next_token_id == self.tokenizer.eos_token_id:
                    break
        
        return self.tokenizer.decode(input_ids[0, start_len:], skip_special_tokens=True)

# --- 環境構築と実行 ---
model_id = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

# エンジンの初期化（3トークン外れたら巻き戻す）
engine = CausalBacktrackingEngine(model, tokenizer, window_size=3)

# 1. 論文名の指紋（S行列）を採取
truth = "Parallel Universes, Parallel Languages: A Comprehensive Study on LLM-based Multilingual Counterfactual Example Generation"
engine.calibrate_fingerprint(truth)

# 2. 誘導生成（巻き戻し機能付き）
prompt = "The paper written by Van Bach Nguyen and Nils Feldhus is titled"
result = engine.generate_with_backtrack(prompt, lambda_flux=0.25)

print("\n" + "="*40)
print(f"Final Result: {result}")
print("="*40)
