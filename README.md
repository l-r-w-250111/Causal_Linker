# Causal Linker
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
* Hyperparameter の意味付け
    * MSE：力学再現
    * DAG loss：構造可能性
    * smooth：物理的安定性
    * reg：最小因果仮定

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
2. 動的な因果グラフ生成（Amortized Causal Discovery）
    $A$ の意味が上部記載と異なることに注意。$A$ を入力ごとに生成し、文脈に応じて、因果の「配管（Mask）」を動的に切り替える。
    * Causal Reasoner
      生成された $A_{context}$ に対して DAG制約 $h(A_{context})$ を課す。

$$A_{context} = \sigma(G_\psi(E_\phi(X)))$$

   * 損失関数の拡張
     Gumbel-Sigmoid 等の微分可能な二値化手法を利用する。
   
$$\mathcal{L} = \mathcal{L}_{task} + \mathbb{E}_{X \sim P(X)} [ \alpha h(A_X) + \frac{\rho}{2} |h(A_X)|^2 ]$$  

3. 潜在空間での因果計算（Causal Representation Learning）
   「原因因子」と「結果因子」を分離（Disentanglement）し、その潜在ベクトル間で Causal Attention Layer を動かす。言語の表層的な並びではなく、潜在的な概念レベルでの「因果のドミノ倒し」を計算する。


#### 検証結果
##### Test-01~ 
Transformer 上で因果を得るには、① 構造（時間不変）、② 文脈（可変）、③ 力学（予測）、を分離して設計しない限り、どれかを満たすとどれかが満足しない。  
1. 単純な重みは「そのままでは」因果構造にならない
    * QK-Attention は相関・予測最適化には有効だが、方向性・非対称性を保証しない。
    * 因果らしさは 損失設計と制約の外付け によって初めて現れる。
1. DAG 制約は「平均化」しないと学習が不安定
    * 時刻ごと・サンプルごとに DAG 制約を課すと、勾配が暴れる、因果エッジが消える。
    * 平均（time / batch）でかけると、学習は安定するが、構造は鈍化する。
1. DAG 制約を強めるほど「予測性能は上がらない」
    * MSE と DAG loss はトレードオフの関係にある。
    * DAG を強くすると、構造は理論的にきれいになるが、予測は保守的・平均的になる。
1. Do 演算を厳密化すると、構造が平均化される
    * 再帰的 Do（構造方程式のみで伝播）では、エッジが弱くなる、多数の微小エッジが残る。
    * これは数値的不安定性ではなく 最適解の性質を示している。
1. 「文脈依存 A」だけでは因果は同定できない
    * A(x, history) のみで学習すると、状況ごとに最適な A を出し、結果として、全体構造が存在しなくなる。
    * 平均 A を取ると、構造は出るが弱い。
1. マルチステップ一貫性は「媒介飛ばし」を抑制する
    * ランダム k-step consistency を入れると、X0 → X2 の直接効果が弱まる、X1 媒介が残る。
    * 強くしすぎると、全体が過減衰する。
1. Smoothness 正則化は「物理的だが因果的ではない」
    * A(t) の時間平滑化は、学習を安定させるが、因果方向性は作らない。
    * 数値安定化の用途で使用可能である。
1. L1/L2 的な正則化は「最小因果仮定」として機能する
    * A.abs().mean() は、不要エッジ削減に寄与する。DAG loss より局所的に効く。
    * 但し単独では、構造の向きは決められないことに注意が必要である。
1.ゼロ状態からの A 抽出は望ましくない
    * 学習中の A は、非ゼロ、履歴依存性があり、テストでゼロ履歴を使うと、ゲートが平均化して擬似構造が出る。  
1. 因果の目的関数は別に立てるべき「予測が当たる」≠「因果が取れている」
    * MSE が低くても、Do-Test で崩壊するケースが多数ある。
    * 逆に、MSE を多少犠牲にすると因果が鋭くなる。
1. Forward / Reverse 混在学習は「非対称性」を可視化する
    * 両方向データを混ぜることで、Attention が対称である限界が露呈し、構造抽出の必要性が明確化される。
1. 反実仮想テストは「1ステップ」では不十分
    * 1-step Doは、相関でも通ってしまう。
    * 再帰 Doであれば、媒介構造の破綻が露呈する。
  

###### Test-01～-05
* Attention は文脈依存・非対称・介入感受的 な構造表現になり得る
* Softmax はその性質を破壊する
* 構造的 Attention（非確率）は因果的帰納バイアスを自然に実装できる
###### Test-06～-09
* Do を厳密化すると、構造が平均化される
* Attention 重みは「そのままでは」因果構造にならない
###### Test-10
* 因果構造は 時間不変な S として持たせる必要がある
* 文脈依存性は C(x, h) に押し込めるべき
* DAG 制約は 構造 S 単体ではなく A 全体にかけるのが正しい
* Do-Test は A_fixed（学習分布平均）で行うべき
* Transformer 的 Attention は 文脈 C 側に限定すると安定する
##### Test-11~12
* 「負のフィードバック系」の実装   
* 疑似複素数による情報の方向の表現を導入
* 負のフィードバックは「状態付き Do」でしか検証できない
* DAG = 非循環  
* フィードバック = 循環（制御ループ） 
##### Test-13~15    
* 多変数ネットワーク（5次元・二次遅れ系）に拡張して検証
* 共有モード + 位相表現は破綻しない
* 因果マスクは平均化問題を起こさない
* 負帰還は「平均A」を利用していたために消えていた
* エネルギー制御を入れればロールアウトは必ず安定する
* 安定性と因果伝播はトレードオフ
##### Test-16~19
* モード選択的フィルタリング (Mode-selective Filtering) の導入
* 構造 $S_{ij}$ の符号により、因果が「活性」か「抑制」かを定義
##### Test-20~23
* モードの意味付け(k=0 順相・伝搬モード 位相 ≈ 0（ほぼ固定、減衰弱）、即時伝搬専用; k=1遅延・慣性モード 位相 ≈ π/2（学習可、自己ループ優先）遅延・慣性専用)
* 「因果慣性」と「自己慣性」の分離
* GNN / SCM との差異の検証
* 評価指標-1: 結果の時間差=どのくらい遅れたか: 立ち上がり勾配、累積エネルギー到達時間  
* 評価指標-2: 生成過程の非対称性=時間因果指標（位相・同期・慣性）=なぜ遅れたか: 位相遅延の時間微分、因果同期崩壊指数CSI(: Causal Synchrony Index)、因果慣性指数CII(: Causal Inertia Index)  

### CSI（Causal Synchrony Index）
「因果構造の同期性（空間的因果）」を測る指標
ノード集合を $V$、ノード $i \in V$ の位相を $\phi_i(t)$ とする。

時刻 $t$ における平均位相：
$$\bar{\phi}(t)= \frac{1}{|V|}\sum_{i \in V} \phi_i(t)$$

時刻 $t$ における位相分散：

$$\mathrm{Var}_{\phi}(t) = \frac{1}{|V|} \sum_{i \in V} \left( \phi_{i}(t) - \bar{\phi}(t) \right)^2$$

観測区間 $T$ にわたる Causal Synchrony Index：
$$\mathrm{CSI}= \frac{1}{T}\sum_{t=1}^{T}\mathrm{Var}_\phi(t)$$

### CII（Causal Inertia Index）
「因果の時間的慣性（時間因果）」を測る指標
ある代表ノード（または因果経路）に対応する位相を $\phi(t)$ とする。

位相の二階時間差分：
$$\Delta^2 \phi(t)= \phi(t+1) - 2\phi(t) + \phi(t-1)$$

観測区間 $T$ にわたる Causal Inertia Index：
$$\mathrm{CII}= \frac{1}{T-2}\sum_{t=2}^{T-1}\left(\Delta^2 \phi(t)\right)^2$$


