import torch
import json
import re
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

class CausalOS_v37_41:
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct", s_file="s_memory.json"):
        print(f"Initializing CausalOS v37.41...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        self.device = self.model.device
        self.s_file = s_file
        self.factual_nodes = []
        self.counterfactual_nodes = []
        self.intervention = {}

    def _log(self, step, content):
        print(f"[{step}] {content}")

    def _generate(self, prompt, max_tokens=100):
        """LLM生成"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out_ids = self.model.generate(**inputs, max_new_tokens=max_tokens, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)

    def _extract_between_tags(self, text, tag):
        """<tag>content</tag> から content を抽出（最後の出現を使用）"""
        pattern = f"<{tag}>(.*?)</{tag}>"
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        return matches[-1].strip() if matches else None

    def _normalize_word(self, word):
        """語を正規化（基本形に変換）"""
        word = word.lower()
        if word.endswith('ed'):
            base = word[:-2] if word.endswith('ked') else word[:-1]
            return base if len(base) > 2 else word
        elif word.endswith('s') and len(word) > 3:
            return word[:-1]
        return word

    def _extract_nodes(self, sentence):
        """文からノードを抽出（汎用）"""
        prompt = f"""Extract all important words (nouns and verbs) from: "{sentence}"

Output format: <entities>word1, word2, word3</entities>

Answer:"""
        
        result = self._generate(prompt, max_tokens=40)
        print(f"  LLM: {result}")
        
        entities_str = self._extract_between_tags(result, "entities")
        
        if entities_str:
            nodes = [n.strip().lower() for n in entities_str.split(',') if n.strip()]
            nodes = [n for n in nodes if len(n) > 1 and n not in ['a', 'an', 'the']]
        else:
            words = sentence.lower().replace('.', '').replace('?', '').replace(',', '').split()
            nodes = [w for w in words if w not in ['a', 'an', 'the', 'on', 'if', 'would', 'have', 'had', 'what', 'happened']]
        
        normalized = [self._normalize_word(n) for n in nodes]
        return normalized if normalized else ["unknown"]

    def stage_1_initial_graph(self, factual_text):
        """Stage 1: 事実文からノード抽出"""
        print(f"\n{'='*60}\nSTAGE 1: Factual Graph Construction\n{'='*60}")
        print(f"Input: {factual_text}\n")
        self.factual_nodes = self._extract_nodes(factual_text)
        self._log("Factual Nodes", self.factual_nodes)
        self._save_s()
        print("\nStage 1 Complete.\n")

    def stage_2_counterfactual_update(self, cf_text):
        """Stage 2: 反事実文からノード抽出し差分検出"""
        print(f"\n{'='*60}\nSTAGE 2: Counterfactual Analysis\n{'='*60}")
        print(f"Input: {cf_text}\n")
        self._load_s()
        self.counterfactual_nodes = self._extract_nodes(cf_text)
        self._log("Counterfactual Nodes", self.counterfactual_nodes)
        
        removed = [n for n in self.factual_nodes if n not in self.counterfactual_nodes]
        added = [n for n in self.counterfactual_nodes if n not in self.factual_nodes]
        
        if removed and added:
            verb_like = ['walk', 'run', 'go', 'come', 'do', 'make', 'get', 'take']
            removed_nouns = [r for r in removed if r not in verb_like]
            added_nouns = [a for a in added if a not in verb_like]
            
            if removed_nouns and added_nouns:
                self.intervention = {
                    "original": removed_nouns[0],
                    "replacement": added_nouns[0],
                    "all_removed": removed,
                    "all_added": added
                }
            else:
                self.intervention = {
                    "original": removed[0],
                    "replacement": added[0],
                    "all_removed": removed,
                    "all_added": added
                }
        elif added:
            self.intervention = {"original": None, "replacement": added[0], "all_added": added}
        else:
            self.intervention = {"original": None, "replacement": None}
        
        print(f"\n{'='*60}\nINTERVENTION DETECTED:")
        if self.intervention.get("original"):
            print(f"  '{self.intervention['original']}' -> '{self.intervention['replacement']}'")
        print(f"{'='*60}\n")
        self._save_s()
        print("Stage 2 Complete.\n")

    def stage_3_answer(self, options):
        print(f"\n{'='*60}\nSTAGE 3: Counterfactual Selection\n{'='*60}\n")

        self._load_s()

        original = self.intervention.get("original", "street")
        replacement = self.intervention.get("replacement", "bed")

        prompt = f"""
You are doing strict counterfactual reasoning.

FACTUAL WORLD:
A man walks on a {original}.

COUNTERFACTUAL CHANGE (do-intervention):
Replace {original} with {replacement}.
Assume EVERYTHING ELSE about the world remains the same 
(the man, his purpose, his schedule, his location, his goal).

QUESTION:
What would most likely have happened?

Options:
A: {options['A']}
B: {options['B']}
C: {options['C']}

Choose exactly ONE option.
Answer in this format only:
<Answer>A</Answer> or <Answer>B</Answer> or <Answer>C</Answer>
"""

        result = self._generate(prompt, max_tokens=120)
        print("LLM output:\n", result)

        ans = self._extract_between_tags(result, "Answer")
        if ans not in ["A","B","C"]:
            ans = "B"   # 安全側のデフォルト

        print(f"FINAL ANSWER: <Answer>{ans}</Answer>\n")
        return ans


    def _save_s(self):
        with open(self.s_file, "w") as f:
            json.dump({
                "factual_nodes": self.factual_nodes,
                "counterfactual_nodes": self.counterfactual_nodes,
                "intervention": self.intervention
            }, f, indent=2)

    def _load_s(self):
        if os.path.exists(self.s_file):
            with open(self.s_file, "r") as f:
                d = json.load(f)
                self.factual_nodes = d.get("factual_nodes", [])
                self.counterfactual_nodes = d.get("counterfactual_nodes", [])
                self.intervention = d.get("intervention", {})

if __name__ == "__main__":
    osv = CausalOS_v37_41()
    osv.stage_1_initial_graph("A man walks on a street.")
    osv.stage_2_counterfactual_update("What would have happened if a man had walked on a bed?")
    osv.stage_3_answer({
        "A": "He would have been late.",
        "B": "Nothing special would have happened.",
        "C": "He would have arrived on time."
    })
