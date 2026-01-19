
### Test-51.py  

``` mermaid
flowchart LR

%% ===== 外部入力 =====
U1["自然言語<br>(factual文)"]
U2["自然言語<br>(counterfactual文)"]

%% ===== 言語表現層 =====
subgraph LLM_Layer["言語表現層（v37.21系）"]
    ENC["ComplexLanguageEncoder<br>(擬似複素埋め込み)"]
    NODE["ノード抽出<br>＋類似度マージ"]
    AUD["Pairwise Direction Audit<br>(A→B 判定)"]
    SBUILD["S行列構築<br>(因果グラフ)"]
end

%% ===== 構造→物理変換 =====
subgraph ADAPT["構造アダプタ"]
    SAD["SAdapter<br>(edge → 行列)"]
end

%% ===== 因果コア（物理側） =====
subgraph CORE["CausalCore（複素因果力学）"]
    ATT["複素Causal Attention"]
    PHASE["位相生成器<br>(phi)"]
    MODE["モード混合<br>(K=2)"]
    DO["do-mask<br>(介入=attention遮断)"]
end

%% ===== 時間発展 =====
subgraph ROLL["ロールアウト"]
    TRJ["時系列軌道<br>x(t) ∈ ℂ^N"]
end

%% ===== 評価層 =====
subgraph EVAL["反実評価層"]
    CSI["CSI<br>(空間同期)"]
    CII["CII<br>(時間慣性)"]
    CIIp["CII'<br>(エッジ＋ノード)"]
    DEC["最終判定<br>A/B/C"]
end

%% ===== 接続 =====
U1 --> ENC
U2 --> ENC

ENC --> NODE --> AUD --> SBUILD
SBUILD --> SAD
SAD --> CORE

CORE --> TRJ
TRJ --> CSI
TRJ --> CII
TRJ --> CIIp

CSI & CII & CIIp --> DEC
```
