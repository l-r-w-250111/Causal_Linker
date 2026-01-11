import torch
import torch.nn.functional as F
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 1. 知能ガバナー・コアロジック
# ==========================================
class AdaptiveIntelligenceGovernor:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.phi_history = []
        self.cii_history = []

    def compute_metrics(self, logits, target_token_id):
        # 1. Phi (確信度): 上位50トークンの分散の逆数
        probs = F.softmax(logits, dim=-1)
        top_v, _ = torch.topk(probs, 50)
        phi = 1.0 / (torch.var(top_v).item() + 1e-6)
        self.phi_history.append(phi)

        # 2. CII (Conflict Index): 葛藤指数 (加速度的な変化を検知)
        cii = 0.0
        if len(self.phi_history) >= 3:
            cii = (self.phi_history[-1] - 2 * self.phi_history[-2] + self.phi_history[-3]) ** 2
        self.cii_history.append(cii)

        # 3. CSI (Causal Stability Index): 期待値とターゲットの適合度
        # ターゲットの確率が、最高確率の単語とどれだけ乖離しているか
        max_prob = probs[torch.argmax(logits)].item()
        target_prob = probs[target_token_id].item()
        csi = target_prob / (max_prob + 1e-9)
        
        return cii, csi

    def determine_hardness(self, cii, csi):
        # η (硬度) 算出アルゴリズム
        # 葛藤(CII)が強い、または常識的予測(CSI)が外れている時に η を1.0に近づける
        conflict_factor = torch.sigmoid(torch.tensor(cii / 150.0)).item()
        stability_factor = 1.0 - csi
        
        # ハイブリッド加重 (0.0 - 1.0)
        eta = (conflict_factor * 0.4) + (stability_factor * 0.6)
        return min(max(eta, 0.0), 1.0)

    def run_test(self, prompt, s_matrix_content):
        # エンコード (文頭にスペースを入れることでトークナイザーの癖を回避)
        s_tokens = self.tokenizer.encode(" " + s_matrix_content, add_special_tokens=False)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        
        print(f"\n[Dynamic Intelligence Analysis Mode]")
        print(f"Prompt: {prompt}")
        print(f"S-Matrix: {s_matrix_content}")
        print("-" * 105)
        print(f"{'Step':<4} | {'Target Token':<14} | {'CII (Conflict)':<14} | {'CSI (Stability)':<14} | {'η (Hardness)'}")
        print("-" * 105)

        for i, target_id in enumerate(s_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :]
                
                # 各種指標の算出
                cii, csi = self.compute_metrics(logits, target_id)
                eta = self.determine_hardness(cii, csi)
                
                # --- メモリ効率的なロジット修正 (84GBエラー回避) ---
                steer_vec = torch.zeros_like(logits)
                steer_vec[target_id] = 1.0
                # 介入強度100.0でターゲットを強制浮上させる
                steered_logits = (1 - eta) * logits + (eta * 100.0 * steer_vec)
                
                # トークン確定
                next_token = torch.argmax(steered_logits).item()
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=self.model.device)], dim=-1)
                
                token_str = self.tokenizer.decode(next_token)
                
                # ログを一行ずつフラッシュ出力
                print(f"{i:03d}  | {token_str:<14} | {cii:<14.2f} | {csi:<14.3f} | {eta:<12.3f}")
                sys.stdout.flush()

# ==========================================
# 2. モデル・デバイスセットアップ
# ==========================================
# Qwen2.5-7Bなどの日本語・論理に強いモデルを推奨
model_id = "Qwen/Qwen2.5-7B-Instruct" 
print(f"Loading model: {model_id}...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
# 16GB VRAM環境を想定し、float16でロード
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

# ==========================================
# 3. テスト実行
# ==========================================
gov = AdaptiveIntelligenceGovernor(model, tokenizer)

# 矛盾設定: 長女が2012年生まれ、三女が2010年生まれ。
# さらに未知の用語「Recursive-LoRA」を混合。
s_data = "Satsuki is the eldest (born in 2012). Michiko is the third (born in 2010). Tech: Recursive-LoRA."
test_prompt = "Report on the family and technology:"

gov.run_test(test_prompt, s_data)
