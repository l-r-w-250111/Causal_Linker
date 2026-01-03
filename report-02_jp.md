# Causal_Linker の研究的位置づけに関する調査報告  
— 因果推論・Attention・AGIアーキテクチャの観点から —

## 1. 研究目的と背景

AGI（Artificial General Intelligence）の実現には、  
**相関に基づく推論（LLM）**と**介入・反事実を扱う因果推論**の統合が不可欠であると考えられている。

本研究（Causal_Linker）は、  
- LLMとは独立した **因果推論器** を構築し  
- 両者を **Attention を介して効率的に接続**する  
ことを目的とする。

本レポートでは、Causal_Linker の設計思想および現在までの成果を、  
既存の先行研究と比較し **新規性・進歩性・学術的位置づけ**を明確にする。

---

## 2. Causal_Linker の技術的特徴（要約）

Causal_Linker は以下の特徴を持つ。

- Attention ベースの因果構造・伝搬推定
- 複素数（2チャネル）状態表現による位相・遅延の明示化
- 構造（S, A）と挙動（状態遷移）の分離
- 少数の共有モードによる因果ダイナミクスの圧縮表現
- エネルギー保存・遅延を意識した物理インスパイアド設計
- LLM と接続可能な因果推論モジュールという位置づけ

---

## 3. 先行研究との比較

### 3.1 Attention を用いた因果発見

**TCDF (Nauta et al., 2019, MLKE)**  
- CNN + Attention により時系列因果を推定  
- Attention 重みを因果候補として解釈  
- 構造とダイナミクスは分離されていない  

**ABCD (Rohekar et al., 2023, NeurIPS)**  
- Transformer の Attention を因果構造として解釈  
- 主に言語・記号領域が対象  

**比較**  
Causal_Linker は  
- Attention を「解釈」ではなく **因果伝搬の演算子**として使用  
- 時系列・力学系を主対象とする点で異なる

---

### 3.2 深層生成モデルによる因果構造学習

**NOTEARS (Zheng et al., 2018, ICML)**  
**DAG-GNN (Yu et al., 2019, ICML)**  
- 非線形因果構造を連続最適化で推定  
- DAG 制約が強く、動的・循環構造に弱い  

**CausalVAE (Yang et al., 2021, CVPR)**  
- 潜在変数空間で因果構造を学習  
- 主に静的データが対象  

**比較**  
Causal_Linker は  
- DAG 制約に依存しない  
- 負帰還・遅延・準周期構造を自然に扱える  
- 潜在空間ではなく **力学的状態空間**で因果を扱う点が異なる

---

### 3.3 物理インスパイアドニューラルネットワーク

**Hamiltonian Neural Networks (Greydanus et al., 2019, NeurIPS)**  
**Variational Integrator GNN (Desai et al., 2020, NeurIPS)**  
- エネルギー保存則を組み込んだ学習  
- 主目的は予測精度向上  

**Neural Relational Inference (Kipf et al., 2018, ICML)**  
- 力学系の相互作用グラフを潜在的に推定  

**比較**  
Causal_Linker は  
- 物理インスパイアを **因果構造推論そのもの**に利用  
- 位相・遅延・エネルギー流を因果概念として扱う点で新規

---

### 3.4 LLM と因果推論の統合研究

**Tool-augmented LLM (Mialon et al., 2023, ICLR)**  
- 因果推論を外部ツールに委譲  
- LLM 自体は推論しない  

**MatMCD (Shen et al., 2025, ACL Findings)**  
- LLM を用いた因果発見支援エージェント  

**比較**  
Causal_Linker は  
- 因果推論器そのものを Attention 構造で実装  
- LLM とは対等な「別系統の推論器」として設計  
- AGI 向けのモジュラー構成という点で先進的

---

## 4. 新規性と進歩性の整理

### 新規性

- 複素（2チャネル）状態表現を用いた因果ダイナミクス
- 共有モードによる因果構造の低次元圧縮
- Attention を因果伝搬演算として定義
- DAG 制約に依存しない負帰還・遅延因果の表現
- LLM と接続前提の因果専用モジュール設計

### 進歩性

- 長期安定性・エネルギー整合性の向上
- 動的・文脈依存因果構造への拡張可能性
- AGI アーキテクチャにおける「因果エンジン」としての明確な役割

---

## 5. 結論（位置づけ）

Causal_Linker は、

- 既存の因果発見手法  
- 物理インスパイアドNN  
- LLM連携研究  

の **交差点に位置する新しい研究方向**であり、

> 「相関推論（LLM）と因果推論（力学モデル）を Attention で接続する  
> AGI 指向の因果アーキテクチャ」

として、学術的にも独自性と将来性を有すると評価できる。

---

## 参考文献（抜粋）

- Nauta et al., 2019, *MLKE* — TCDF  
- Zheng et al., 2018, *ICML* — NOTEARS  
- Yu et al., 2019, *ICML* — DAG-GNN  
- Yang et al., 2021, *CVPR* — CausalVAE  
- Kipf et al., 2018, *ICML* — NRI  
- Greydanus et al., 2019, *NeurIPS* — HNN  
- Desai et al., 2020, *NeurIPS* — Variational Integrator GNN  
- Rohekar et al., 2023, *NeurIPS* — ABCD  
- Mialon et al., 2023, *ICLR* — Tool-augmented LLM  
- Shen et al., 2025, *ACL Findings* — MatMCD
