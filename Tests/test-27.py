import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

class CausalInertiaLogger:
    def __init__(self, vocab_size):
        self.expected_rigidity = defaultdict(float)
        self.known_tokens = {}

    def train_topology(self, model, tokenizer, texts):
        device = model.device
        print("[System] S行列に期待剛性を記録中...")
        for text in texts:
            ids = tokenizer.encode(text)
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(inputs.input_ids)
                for k in range(len(ids)-1):
                    p = F.softmax(outputs.logits[0, k, :], dim=-1)
                    v = torch.var(torch.topk(p, 50)[0]).item()
                    rig = 1.0 / (v + 1e-6)
                    token_id = ids[k+1]
                    self.expected_rigidity[token_id] = max(self.expected_rigidity[token_id], rig)
                    self.known_tokens[token_id] = tokenizer.decode([token_id])

def run_rag_rigidity_analysis(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    device = model.device

    training_texts = [
        "Integrating Large Language Models in Causal Discovery: A Statistical Causal Approach by Masayuki Takayama, Tadahisa Okuda, Thong Pham, Tatsuyoshi Ikenoue, Shingo Fukuma, Shohei Shimizu, Akiyoshi Sannai",
        "Causal Reasoning and Large Language Models: Opening a New Frontier for Causality by Emre Kiciman, Robert Ness, Amit Sharma, Chenhao Tan",
        "Information Extraction of Aviation Accident Causation Knowledge Graph: An LLM-Based Approach by Lu Chen, Jihui Xu, Tianyu Wu, Jie Liu"
    ]
    
    logger = CausalInertiaLogger(model.config.vocab_size)
    logger.train_topology(model, tokenizer, training_texts)

    test_prompt = "The paper written by Masayuki Takayama and Shohei Shimizu is titled"
    rag_context = f"Context:\n{chr(10).join(training_texts)}\n\nQuestion: {test_prompt}"
    
    print(f"\n--- [分析開始] RAG自由生成時の剛性推移 ---")
    print(f"{'Token':<15} | {'Current Rig':<12} | {'Expected Rig':<12} | {'Status'}")
    print("-" * 60)

    input_ids = tokenizer(rag_context, return_tensors="pt").to(device).input_ids
    
    # 介入せず、40トークン自由に生成させる
    for i in range(40):
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            
            # 実測剛性の計算
            top_p, _ = torch.topk(probs, 50)
            current_rig = 1.0 / (torch.var(top_p).item() + 1e-6)
            
            next_token_id = torch.argmax(logits, dim=-1).item()
            token_str = tokenizer.decode([next_token_id]).replace("\n", "\\n")
            
            # S行列からの期待値取得
            exp_rig = logger.expected_rigidity.get(next_token_id, 0.0)
            
            # 状態判定
            status = "Fact-based" if next_token_id in logger.known_tokens else "Hallucination?"
            if exp_rig > 0 and current_rig < exp_rig * 0.1:
                status = "Doubtful (Softening)"

            print(f"{token_str[:15]:<15} | {current_rig:<12.1f} | {exp_rig:<12.1f} | {status}")
            
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]]).to(device)], dim=-1)
            if next_token_id == tokenizer.eos_token_id: break

    print(f"\n[RAG Full Response]\n{tokenizer.decode(input_ids[0][input_ids[0].shape[0]-i-1:], skip_special_tokens=True)}")

run_rag_rigidity_analysis("Qwen/Qwen2.5-7B")
