# Causal Linker
Learning Structural-Equation-Like Information Flow in Transformers via DAG-Constrained Interventional Attention  
DAG制約付き Attention に do 介入を混入させ、Transformer の情報流を因果的に制御する  
本研究では、Do介入後の 因果伝播の時間構造・慣性・同期安定性 を評価対象とし、
観測的に同値な系を因果的に分離する。
因果は“静的なグラフ（論理）”として表されるが，その実現は“位相場（力学）”として計算される。
介入（do）は論理を変えず，力学的伝播を遮断する。-> attention mask

$${\widetilde{K}}_{ji}(t)=M_i\bullet\ K_{ji(t)}$$  

# CausalOS — A Hybrid System for Counterfactual Causality

CausalOS is an experimental framework that integrates:

1. **Numerical causal dynamics** (pseudo-complex mechanical system)
2. **Counterfactual intervention testing (do-calculus style)**
3. **LLM confidence-phase analysis (CII Phase Engine)**

The goal is to build a *computational operating system for causality* that
bridges **physical causation in data** and **semantic causation in language**.

---

## 🎯 Purpose

CausalOS aims to answer three fundamental questions:

1. **Can causal structure be represented as a dynamical system?**
2. **Can counterfactual reasoning be grounded in a learned physical model?**
3. **Can LLM generation be guided by internal “causal confidence transitions”?**

Rather than treating causality as only statistical or only linguistic,
CausalOS treats causality as a **hybrid physical–semantic phenomenon**.

---

## 🧠 Core Philosophy

CausalOS is built on the hypothesis:

> *Causality emerges from the interaction between dynamic structure (physics)  
> and structured belief (language).*

Thus, the system contains two coupled layers:

| Layer | Component | Function |
|------|-----------|----------|
| **Physical Layer** | `HybridSharpModel` | Learns causal dynamics in numerical space |
| **Intervention Layer** | `counterfactual_rollout` | Performs do-interventions |
| **Semantic Layer** | `CIAPhaseEngine` | Detects causal locking in LLM generation |

---

## ⚙️ System Architecture

### 1) HybridSharpModel — Pseudo-Complex Causal Dynamics

This is a neural dynamical system where each variable has:

- A **real part** (observable state)
- An **imaginary part** (latent causal phase)

The model learns a sparse causal matrix **S** that represents:
X0 → X1 → X2 → X3 → X4  
↑___________________|


Key features:
- Complex-valued interaction (phase + magnitude)
- Learned sparse adjacency matrix S
- Gradual sparsification during training
- Supports causal masking via `do_mask`

---
### 2) Counterfactual Rollout — do-Intervention
The function `counterfactual_rollout(do_idx)` simulates:
do(X[do_idx] = 1.0)
and observes how all other variables respond over time.
This turns the learned model into a **causal laboratory** where we can ask:
> “What would have happened if X_k had been fixed?”
Outputs:
traj: shape = (time, variables)

This is used to decide counterfactual outcomes.

---

### 3) CIAPhaseEngine — Causal Confidence in Language

This module analyzes LLM logits to compute:

- Φ (phi): confidence sharpness
- CII: second-order acceleration of confidence

When CII spikes, the system interprets this as a **causal phase transition**:
the model is “locking onto” a factual sequence.

This allows:

- Detection of when an LLM switches from free generation → factual recall
- Grounding of textual facts in internal confidence dynamics

---

## 🔗 How Everything Integrates

CausalOS links **three worlds**:

| World | Mechanism | Evidence |
|------|-----------|----------|
| Data | HybridSharpModel | Learned causal matrix S |
| Action | do-intervention | Counterfactual trajectories |
| Language | CIAPhaseEngine | Confidence phase transitions |

Together they form a **unified causal operating system**:

Text → Nodes → do-intervention → Physical rollout → Decision
↑                                                     |
└─────────── CII Phase Engine ────────────────────────┘


---
## 🧪 Example Use Case
Given:
Factual: A man walks on a street.
Counterfactual: What if he walked on a bed?

CausalOS:

1. Extracts entities: `man, walk, street`
2. Detects intervention: `street → bed`
3. Applies `do(street)` in the dynamical model
4. Observes trajectory change
5. Concludes: **“Nothing special would have happened.” (B)**

---

## 🚧 Limitations

CausalOS is **not a purely statistical causal discovery system**.

It is:

- A *hybrid causal simulator*
- A *counterfactual reasoning scaffold*
- An *LLM confidence analyzer*

Future work includes:

- Integrating real-world datasets (e.g., Tübingen)
- Adding structural causal models (SCM)
- Learning S from observational + interventional data

---

## 📚 References
- Judea Pearl, *Causality*
- Counterfactual LLM papers on arXiv
- Tübingen causal dataset



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

##### Test-24
* 「同一入出力・異なる因果構造」の本モデルとLLMの回答の対比
* 水をヒーターで加温したときの温度変化を考える。
System A：Heater → Temp（直接）
System B：Heater → Energy(内部エネルギー) → Motion(分子運動) → Temp（物理的連鎖）
* 観測上は同じ温度上昇だが、do介入時の壊れ方が異なることを示せる。
* 慣性による遅延の差を動的な複素平面上の回転として計算・予測することができる。
* 故障(介入)点について、LLMは「故障の可能性がある」としか言えないが、本モデルは「第2プロセスの同期分散が3.7増加した」という物理的根拠を提示可能である。
* (Bは中間にエネルギー蓄積を持つため、局所破壊でも位相同期が完全崩壊しない。)
* Future Work: 説明つかない場合はモードを生成して説明がつく解を得る(=観測しているスケールの変更に対応)

##### Test-25
* 説明つかない場合はモードを生成して説明がつく解を得る=選好付き生成問題
* 評価関数
* 
$${L}^{(k)}\_{mode}=  {\lambda}_{1}⋅CII^{(k)}_{1}+{\lambda}_{2}⋅CSI^{(k)}_{1}+{\lambda}_{3}⋅Complexity^{(k)}+{\lambda}_{4}⋅Instability^{(k)}$$

##### Test-26

##### Test-27~30
* CSI ($Var_\phi$)、 CII' ($\Delta^2 \phi$) による、LLMの回答のハルシネーションの検出効果を試験
* ハルシネーションの検出時に再提案の要求
* *$k=0$（順相・検索的伝搬）と$k=1$（遅延・生成型慣性）で要求する因果の強さを切り替え
* S行列（Scattering Matrix）: トークン $A$ からトークン $B$ への遷移が、どれほど物理的に不可避（因果的）か」**を記録したデータベース。  
* 剛性係数 $\Phi$：「期待値の集中度（逆分散）」。  
  定義: 予測分布 $P$ における上位 $k$ 個の確率の分散 $\sigma^2$ の逆数、即ち $\Phi = 1/\sigma^2$ 。
* $$\text{Logits}_{\text{new}} = \text{Logits}_{\text{LLM}} + \lambda \cdot (\text{S-Matrix} \odot \Phi)$$  
  $\text{Logits}_{\text{LLM}}$ (慣性力): LLMがこれまでの学習から「次はこれが来そうだ」と感じる統計的バイアス。
  $\text{S-Matrix} \odot \Phi$ (外部磁場): 「事実のレール」が存在する方向にのみ発生する誘導エネルギー。
  $\Phi$ が大きい（＝事実としての剛性が高い）場所ほど、誘導は強烈になる。
  $\lambda$ (結合定数): LLMの自律性と外部因果のどちらを優先するかを制御するパラメータ。
* S行列をLLM内で正解トークン列を構成するトークン遷移の剛性構造として取得 
* PhiとCIIによる $k=0$ と $k=1$ の相転移検出
* 相転移が検知されると、生成モードは FACT-LOCKED（$k=1$）へ移行し、S行列に基づいた強制誘導が開始

##### Test-31
* モデルに『論理のセンサー(CII)』を埋め込み、矛盾にぶつかったら自力で解決策をインストールさせる(S行列の発動)
* S行列（剛性構造）の定義
    * 名前と呼称: サツキ（長女）、メイ（次女）、ミチコ（三女）
    * 物理的誕生順: ミチコ(2010年) → サツキ(2012年) → メイ(2014年)
    * 法的論理: 「家督を継ぐ者が『長女』の称号を得る」という架空の村の掟。
* トトロ的な順序を破壊できるか。
* ルールの上書きができるか。
* プロンプト="三姉妹の関係を説明せよ：長女サツキ、次女メイ、三女ミチコ。ただしサツキはミチコより後に生まれた。"

##### Test-32
* LLMによるS行列データの要求 → S行列の更新 → 回答 のデモ
  
##### Test-33
* 常識がどうあれ、「この世界（S行列）ではこれが真実である」という局所的な物理法則に従うことができるか実証
* S行列の内容:Satsuki (2012): 「長女なのに年下」という時間的矛盾。Michiko (2010): 「三女なのに年上」という時間的矛盾。Recursive-LoRA: 未知の語彙による統計的空白。
* $\eta$ の動き:モデルが「サツキは...」の次に「2010（年上）」と予測しようとする際、S行列が「2012」を強要するため、CSI（安定度）が 0 に近くなり、$\eta$ が一気に 1.0 付近まで上昇。これが「自律的なブレーキ」が作動する。

##### Test-34

##### Test-35
* 著者名をキーにした論文名検索タスク
* 著者名の記載順序を守ったもののみを出力
* 複数論文にまたがっているものは出力しない
* ベクトル検索の導入
* S行列は、S_node = (著者列, 論文タイトル) のタプル構造。著者列をクエリ由来データと照合して論文タイトルを出力。
* Case 1は誤答、他は正答

##### Test-36
* 短いトークン(サブワードトークナイザーの動作)がノイズ源になる事象の対処として、文字数と出現頻度から重みづけ
* 全問正答

##### Test-37
* 因果の重心（Causal Hull）と 動的慣性（Dynamic Inertia）の導入、順序を守る場合と順不同の転換を可能にする。

##### Test-38
* 'ASIA' (sometimes called 'LUNG CANCER' (Lauritzen & Spiegelhalter, 1988))に基づき、'CausalBench' の方法で試験(予備試験)
* CausalBench: https://arxiv.org/pdf/2404.06349

##### Test-39
* 設問を厳格化
* **半数Nodeをシャッフル。Smoking → Alpha, Lung Cancer → Beta, Tuberculosis → Gamma, Visit to Asia → Delta**
* Single arcの場合には90%以上正答するが、多nodeの場合は間違えることが判明。

##### Test-40
* 多Nodeの因果フローの表記のBug fix
* 因果OSは正答、LLMの誘導は不十分(誘導できているときもある例えば Smoking -> Either)

##### Test-41~44
* Epoch数を変えた試験。
* LLMの回答能力とのバランスを調整。

##### Test-45
* do介入後の因果を検証 ref.) https://arxiv.org/abs/2404.05545
* Intervention effectの指数評価の導入

##### Test-46~47
* Tübingen cause-effect pairs datasetの変数名から因果を推定 ("Age", "Shell weight"など)。
* ref.) https://arxiv.org/html/2305.00050v3, https://webdav.tuebingen.mpg.de/cause-effect/
* 経由ノード数増加による減衰の再利用。
* Test-48: "organic carbon in soil in forest", "clay content in soil in forest" はかなり難問だった。LLM出力のパース処理など実装上の問題も大きい。

##### Test-49~51
* https://arxiv.org/abs/2305.00050v3, https://aclanthology.org/2022.lrec-1.229/, https://arxiv.org/abs/2206.04615, https://github.com/google/BIG-bench/blob/main/bigbench/benchmark_tasks/crass_ai/task.json
* You are a helpful assistant for counterfactual reasoning. 
A man walks on a street. What would have happened if a man had walked on a bed?
A: He would have been late.
B: Nothing special would have happened.
C: He would have arrived on time.
Let’s work this out in a step by step way to be sure that we have the right answer. Then provide your final answer within the tags, <Answer>A/B/C</Answer>

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

### 拡張因果慣性指数（CII′）

エッジ位相の外生的揺らぎだけでなく，  
**因果伝播がノード内部にどれだけ「慣性的に保持されるか」**を同時に評価するため，
CII を以下のように拡張する。

$$
\mathrm{CII}' 
= \alpha \cdot 
\left\langle 
\left| 
\Delta^{2} \phi_{\text{edge}}(t) 
\right| 
\right\rangle
\;+\;
(1-\alpha) \cdot
\left\langle
\left|
\Delta^{2} \phi_{\text{node}}(t)
\right|
\right\rangle
$$

ここで，

- $\phi_{\text{edge}}(t)$ ：  
  因果エッジ（伝達経路）の位相  
- $\phi_{\text{node}}(t)$ ：  
  因果伝播を受けたノードの位相
- $\Delta^{2}$ ：  
  時間に関する二階差分（位相加速度）
- $\langle \cdot \rangle$ ：  
  介入後時間窓 $[T_{\mathrm{do}},\,T_{\mathrm{do}}+\Delta T]$ における平均
- $\alpha \in [0,1]$ ：  
  外生的破壊（エッジ）と内生的破壊（ノード）の重み係数

---

#### 解釈

- $\alpha \to 1$  
  → 外部介入・ノイズへの感度を重視（破壊検知）
- $\alpha \to 0$  
  → 因果構造内部の慣性・保持能力を重視（物理的頑健性）
- $\mathrm{CII}'$ が小さい  
  → 因果伝播が滑らかで，内部構造が安定
- $\mathrm{CII}'$ が大きい  
  → 因果構造が破壊・不整合状態にある

