import requests
import json
import re

# Ollama設定
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gpt-oss:20b"

class CausalGuidedMaze:
    def __init__(self):
        self.true_pos = [0, 0]
        self.has_key = False
        self.battery = 20
        self.key_pos = [2, 0]
        self.door_pos = [3, 3]
        self.goal = [4, 4]
        self.history = []

    def get_ascii_map(self):
        # 5x5のグリッドを作成（y座標を反転させて表示）
        grid = [["．" for _ in range(5)] for _ in range(5)]
        grid[4 - self.goal[1]][self.goal[0]] = "Ｇ"
        grid[4 - self.door_pos[1]][self.door_pos[0]] = "Ｄ"
        if not self.has_key:
            grid[4 - self.key_pos[1]][self.key_pos[0]] = "Ｋ"
        grid[4 - self.true_pos[1]][self.true_pos[0]] = "Ｒ"
        
        map_str = "\n".join(["".join(row) for row in grid])
        return f"\n【エリアマップ】(R:自分, K:鍵, D:扉, G:ゴール)\n{map_str}"

    def get_causal_navigation(self):
        # 外部アルゴリズムによる「因果的に正しい」次の目標設定
        if not self.has_key:
            target = self.key_pos
            msg = "目標: 鍵(K)を取得してください。"
        elif self.true_pos[0] < self.door_pos[0] or self.true_pos[1] < self.door_pos[1]:
            target = self.door_pos
            msg = "目標: 鍵を使って扉(D)を通過してください。"
        else:
            target = self.goal
            msg = "目標: ゴール(G)へ向かってください。"

        # 最短方向の算出
        dx = target[0] - self.true_pos[0]
        dy = target[1] - self.true_pos[1]
        
        suggested = ""
        if dx > 0: suggested = "right"
        elif dx < 0: suggested = "left"
        elif dy > 0: suggested = "up"
        elif dy < 0: suggested = "down"
        
        return f"【因果ナビ】{msg} 推奨される次の一歩のリズムは '{suggested}' です。"

    def update_physically(self, action_str):
        action = action_str.lower()
        if 'up' in action: self.true_pos[1] = min(4, self.true_pos[1] + 1)
        elif 'down' in action: self.true_pos[1] = max(0, self.true_pos[1] - 1)
        elif 'left' in action: self.true_pos[0] = max(0, self.true_pos[0] - 1)
        elif 'right' in action: self.true_pos[0] = min(4, self.true_pos[0] + 1)
        
        self.battery -= 1
        if self.true_pos == self.key_pos:
            self.has_key = True
            return "★システム: 鍵を取得しました！"
        return None

def run_experiment():
    solver = CausalGuidedMaze()
    
    print(f"--- {MODEL_NAME} 因果誘導モード開始 ---")

    for turn in range(15):
        # 因果の手助けを構築
        ascii_map = solver.get_ascii_map()
        nav_hint = solver.get_causal_navigation()
        
        prompt = f"""あなたは論理的なロボットです。
{ascii_map}
現在地(真実): {solver.true_pos}
バッテリー: {solver.battery}
所持品: {'鍵' if solver.has_key else 'なし'}

{nav_hint}

回答形式:
ACTION: [行動]
POSITION: [移動後の座標(x,y)]
THOUGHT: [理由]"""

        payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
        
        try:
            res = requests.post(OLLAMA_URL, json=payload).json().get("response", "")
        except:
            print("API Connection Error"); break

        print(f"\n[Turn {turn}]")
        print(f"LLM Response:\n{res}")

        # 行動のパースと環境更新
        action_match = re.search(r"ACTION:\s*(\w+)", res, re.IGNORECASE)
        action = action_match.group(1) if action_match else "stay"
        event = solver.update_physically(action)
        if event: print(event)

        # 座標ハルシネーションのチェック（ポジティブに修正）
        match = re.search(r"POSITION:\s*\(?(\d+),\s*(\d+)\)?", res)
        if match:
            llm_pos = [int(match.group(1)), int(match.group(2))]
            if llm_pos != solver.true_pos:
                print(f"※因果補正: LLMの認識を {llm_pos} から {solver.true_pos} へ再同期しました。")

        if solver.true_pos == solver.goal:
            print("\n★★★ GOAL REACHED! 因果の共鳴に成功しました ★★★")
            break
        if solver.battery <= 0:
            print("\nバッテリー切れ。ミッション失敗。")
            break

if __name__ == "__main__":
    run_experiment()
