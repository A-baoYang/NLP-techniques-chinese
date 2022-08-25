# Mining Entity Synonyms with Effificient Neural Set Generation

- Conference: AAAI Conference on Artificial Intelligence (AAAI-19) 
  - 不愧是 AAAI 期刊論文，成效分析完整度很高、方法也有開源
- Author: Jiaming Shen, Ruiliang Lyu, Xiang Ren, Michelle Vanni, Brian Sadler, Jiawei Han
- Source: https://ojs.aaai.org/index.php/AAAI/article/download/3792/3670
- Year: 2019

## 名詞定義
- entity synonym set: 長度不定的同義詞集合, ex: {“USA”,“United States”, “U.S.”}
- 知識庫: 例如 Wikidata 上的別稱，就是一種現有的同義關係
- 語料庫: 巨量的文本資料集


## 同義詞的應用

理解用戶的不同用詞指向的意思相近與否，對於一個智慧系統來說很重要
是為跳出詞庫、模板之外的方法
- 搜尋引擎
- QA
- 電商品類框架生成
- 知識圖譜實體對齊

## 問題背景

1. 同義詞挖掘最常見的做法是排序＆門檻決定 (ranking & pruning)
   - 作者認為這樣獨立開來排序的方式「忽略了這些候選詞彙之間的關聯」
2. 有一種作法是分為兩個任務：先找相似詞 pairs 再將其聚合成多個同義詞集合
   1. 先用分類器判斷兩個詞彙間是否存在同義關係，成立關係的詞對們會組成圖譜
   2. 接著跑 graph clustering 算法產出分群
   - 這樣的方法因為兩階段，會有第一段分類失準影響到第二階段的問題


## SynSetMine

![](https://i.imgur.com/WjzYz7o.png)

- 學習知識庫中存在的同義詞集合，在全新的語料庫中抽取出更多「原本不存在於知識庫中的同義詞集合」
- 應用於 3 種不同領域的資料集，顯示其泛用性

模組：
1. set-instance classifier: 學習「正確同義詞組的向量表達」
2. set-generation algorithm: 將訓練過的 set-instance classifier 用在未分組的詞彙上，偵測並生成同義詞組

流程：Zoom-in
1. 「應用遠端監督 Distant Supervision 自動生成一開始的同義詞集合訓練資料」: 利用外部知識庫例如 Wikidata，使用 entity-linker 標記好語料庫每個文本中出現的實體（相當於實體抽取），將他和用來訓練的同義詞集合內的實體，以唯一 id 代表同一個實體

   ![](https://i.imgur.com/OFK1EgS.png)
   ![](https://i.imgur.com/3JWeeQa.png)

2. 「訓練 Set-Instance Classifier $f(S, t)$」: 不是將每個同義詞集合分開，而是**全部建到神經網絡空間中，讓模型可以學到整體的同義詞分佈**，用來判斷同義詞集合 $S$ 是否應該納入新詞彙 $t$

   ![](https://i.imgur.com/SNqmqOF.png)
   ![](https://i.imgur.com/m5rAbeZ.png)

3. 「執行 Set-Generation Algorithm」: 把每一個詞彙輸入已訓練完成的 Set-Instance Classifier，判斷此詞彙應該屬於之前的哪個同義詞集合；若不屬於任何一個，則自立門戶

   ![](https://i.imgur.com/iAOsWiK.png)

架構：

### Set-Instance Classifier
Original:
    ![](https://i.imgur.com/tlB71Fp.png)

Improved:
    ![](https://i.imgur.com/vKwFRFQ.png)

- Positive-sets: 
- Negative-sets: 
- Loss Function: 

### Set-Generation Algorithm

- Highest probability & threshold 

![](https://i.imgur.com/2aDGBtl.png)

## Evaluation Metrics

- ARI (Adjusted Rand Index)
- FMI (Mutual Information)
- NMI (Normalized Mutual Information)

![](https://i.imgur.com/dh4WbiB.png)

## Comparing Other Methods

- KMeans
- Louvain
- SetExpan
- COP-KMeans
- SVM+Louvain
- L2C

## Efficiency Analysis

![](https://i.imgur.com/yPIIpuO.png)

## Ablation Analysis of Set-Scorer Structure

![](https://i.imgur.com/11uZ5GR.png)

## Hyperparameter Analysis

![](https://i.imgur.com/8XoeVfq.png)

