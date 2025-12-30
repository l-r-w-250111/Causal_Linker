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


