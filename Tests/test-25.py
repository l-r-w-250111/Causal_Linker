import torch
import torch.nn as nn
import torch.optim as optim

# データ: 3列(K=3)から4列(K=4)への無意味な1D列
stream = ["A1", "A2", "A3", "[VOID]", "B2", "B3", "C1", "C2", "C3", "C4", "[VOID]", "D2", "D3", "D4"]
n = len(stream)

class CausalResonanceModel(nn.Module):
    def __init__(self, n):
        super().__init__()
        # 最初の「S行列」の復活
        self.S_logits = nn.Parameter(torch.randn(n, n) * 0.05)

    def forward(self):
        mask = torch.tril(torch.ones(n, n), diagonal=-1)
        # 行ごとに因果ソースを決定する Softmax
        return torch.softmax(self.S_logits * mask - (1 - mask) * 1e9, dim=1)

def train_resonance():
    model = CausalResonanceModel(n)
    optimizer = optim.Adam(model.parameters(), lr=0.03)
    
    for epoch in range(1500):
        optimizer.zero_grad()
        S = model()
        
        loss = 0
        # 1. 因果の基本原則
        for i in range(n):
            if stream[i] == "[VOID]":
                # [VOID]は必ず過去のいずれかの「実体」と繋がらなければならない
                loss -= torch.log(S[i, :i].sum() + 1e-6)
            else:
                # 実体は「独立したRoot」である確率（対角成分への引力に近い自己保持）を優先
                # ただし、完全な孤立ではなく過去との繋がりも許容する
                loss += S[i, :i].sum() * 0.1 

        # 2. 【最重要】因果の共鳴（Resonance Loss）
        # S[i, j] の結合の「歩幅 (i-j)」が、
        # 他の地点での結合の「歩幅」と同じであることを報酬化する
        strides = torch.zeros(n)
        for i in range(1, n):
            # その地点での期待される「歩幅」を計算
            stride_probs = torch.zeros(n)
            for j in range(i):
                stride_probs[i-j] += S[i, j]
            
            # 前の地点の歩幅分布との「共鳴」を最大化
            if i > 1:
                # 過去の歩幅パターンと現在の歩幅パターンの内積
                # (特定の歩幅が維持されているとロスが減る)
                loss -= torch.sum(S[i, :i] * S[i-1, :i-1].mean()) * 0.5

        # 3. スパース性（因果を太い一本の線にする）
        loss += 0.05 * torch.norm(S, 1)
        
        loss.backward()
        optimizer.step()
    return model

# --- 解析とレポート出力 ---
model = train_resonance()
S_final = model().detach()

print("\n### 因果共鳴トポロジー・レポート（断崖検知） ###")
print("-" * 110)
print(f"{'Idx':<4} | {'Token':<8} | {'SourceIdx':<10} | {'Stride(K)':<10} | {'Rigidity':<10} | {'Boundary Sense'}")
print("-" * 110)



for i in range(n):
    weights = S_final[i]
    # 過去への依存の中で最大値をソースとする
    if i == 0:
        src_idx, stride, rigidity = 0, 0, 1.0
    else:
        src_idx = torch.argmax(weights[:i]).item()
        stride = i - src_idx
        entropy = -torch.sum(weights[:i] * torch.log(weights[:i] + 1e-9)).item()
        rigidity = 1.0 / (1.0 + entropy)
    
    # 境界検知：歩幅が急変したか、実体が強く独立を主張したか
    sense = ""
    if stream[i] == "[VOID]":
        sense = f"Flowing (K={stride})"
    elif i > 0 and src_idx == i:
        sense = "NEW ROOT (Wall)"
    elif i > 0:
        # 前回のVOID時のKと比較して変化があればSHIFT
        sense = "Structural Inertia"

    print(f"{i:<4} | {stream[i]:<8} | {src_idx:<10} | {stride:<10} | {rigidity:.4f} | {sense}")
