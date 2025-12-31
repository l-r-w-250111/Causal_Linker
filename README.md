# Causal_Linker
Learning Structural-Equation-Like Information Flow in Transformers via DAG-Constrained Interventional Attention  
DAG制約付き Attention に do 介入を混入させ、Transformer の情報流を因果的に制御する  

### 因果誘導型 Transformer 拡張  
#### 設計思想
* Attention の柔軟性
    * Attention 機構を因果グラフで制御する
    * 因果構造の制御を表現制御として使用する
    * 学習過程にdo演算(情報流遮断介入)を混入する ∵因果 Attention Mask の形骸化の抑制
* NOTEARS 系の厳密性
    * DAG(: directed acyclic graph) の制約を課す
* LLM への実装容易性
    * Transformer の拡張で情報流の制御機構を実装する


#### 主要な式

$$A^{do(j)}\_{ik} = \begin{cases} 
0 & (\text{if } k = j \text{ and } i \neq j: \text{ノード } j \text{ から外への影響は維持}) \\
0 & (\text{if } i = j: \text{外部からノード } j \text{ への流入をすべて遮断}) \\
A\_{ik} & (\text{otherwise})
\end{cases}$$


$$\mathcal{L} = \underbrace{(1 - p\_{do})\mathcal{L}_{obs} + p\_{do}\mathcal{L}\_{int}}\_{\text{予測一貫性}} + \underbrace{\alpha h(A) + \frac{\rho}{2}|h(A)|^2}\_{\text{DAG制約}} + \lambda\|A\|\_1$$

$$h(A) = \text{tr}(e^{A \circ A}) - d$$ 


$$\text{Score}_{ij} = \frac{Q_i K_j^T}{\sqrt{d_k}} + \mathcal{M}(A_{ij})$$

$$ A_{ij} \in [0, 1] をマスク関数 \mathcal{M} で変換  $$

$$\mathcal{M}(A_{ij}) = 
\begin{cases} 
0 & (A_{ij} \to 1 \text{ のとき: 情報を流す}) \\
-\infty & (A_{ij} \to 0 \text{ のとき: 情報を遮断する}) 
\end{cases}$$

$$\mathcal{M}(A) = \log(A + \epsilon)$$ 

#### Future Work
1. 介入ノードの自動探索（Active Causal Learning） 
    反実仮想を、最も論理が崩れやすい箇所に自ら仕掛け、最も効率的に因果を解明できるようにする。学習された因果マスクの不確実性に基づいて介入対象を能動的に選択する。  
1. 動的な因果グラフ生成（Amortized Causal Discovery）
    $A$ を入力ごとに生成し、文脈に応じて、因果の「配管（Mask）」を動的に切り替える。  
1. 潜在空間での因果計算（Causal Representation Learning）
   「原因因子」と「結果因子」を分離（Disentanglement）し、その潜在ベクトル間で Causal Attention Layer を動かす。言語の表層的な並びではなく、潜在的な概念レベルでの「因果のドミノ倒し」を計算する。

