# Text Summarization with Pretrained Encoders, 2019

- [中文導讀 link (FCU AI Research Cw)](https://fcuai.tw/2021/06/07/text-summarization-with-pretrained-encoders%ef%bc%9a%e5%a6%82%e4%bd%95%e5%b0%87-bert-%e6%87%89%e7%94%a8%e5%9c%a8-text-summarization-task-%e4%b8%8a/)

## Abstract
如何有效應用在 text summarization task 上，以及預訓練模型對 text summarization task 的影響，並且提出一個可以做 extractive 和 abstractive models 的通用框架


## Contribution
1. 分析 document encoding 在文本摘要上的重要性
2. 提出如何有效應用 pretrained models on text summarization task

---

### Background

- 複習 BERT
- 說明對 BERT 模型 input 的修改如何更好對應文本摘要任務的特性

### 複習 BERT
![](https://i.imgur.com/xOJ94YL.png)

文本會經過 tokenizer 處理後，被表示成多個 token 所組成的向量 $[w_1, w_2,...,w_n]$，並加上兩個特殊符號
- `[CLS]` : 在文本最開頭，代表整個文本的開始
- `[SEP]` : 在文本中「每個句子的尾端」，用來斷句

而每個字元會有三種 embeddings:
- token embeddings : 代表屬於 pre-trained model 中哪個 token
- segmentation embeddings : 代表此字元屬於哪個句子
- position embeddings : 代表此字元在句子中的位置
這三種 embeddings 組成了 BERT Model 的 "input" $X=[x_1,x_2,...,x_n]$

爾後輸入由幾層 Transformer Layers 疊起來組成的 Encoder，計算式如下：

![](https://i.imgur.com/hjAy7gG.png)

其中：
- $h$ : hidden states
- $h_0$ : the input of BERT model (X)
- $LN$ : Layer Normalization
- $MHAtt$ : Multi-Head Attention
- $FFN$ : Feedforward neural network
- $l$ : Transformer Layers 層數

最終 BERT 的 output 會是 $T = [t_1, t_2,..., t_n]$ ，每個 $t$ 是包含語意的 word embedding

---

## 文本摘要 

**Extractive Summarization** 抽取式摘要

藉由在文章裡找尋重要的句子，並把找出來的句子接在一起，當作這篇文章的 summary。在模型訓練上，則是當做 sentence classification 的任務，藉由分類每個句子是否為重要句子，來得到要做為 summary 的句子。

**Abstractive Summarization** 生成式摘要

將原始文章餵給模型吃，並直接吐出一段 summary 的方法，是一個 sequence to sequence 的問題。
需要能夠產生原本不存在於文章的詞彙或句子，並且要符合語言的規則

---

## Fine-tuning BERT for Summarization

BERT 直接用於文本摘要，會有什麼問題？

1. 文本摘要時使用的是 sentence embeddings，但 BERT 的訓練方式是 MLM（克漏字、填空），它對 sentence embeddings 是沒有經過充分學習的
2. 即使有 Next Sentence Prediction 任務預訓練，透過 segmentation embeddings 可以區分不同句子，但也是只有使用了 sentence pair 兩兩進行訓練；在文本摘要時通常需要2個以上多句子接在一起當作輸入

針對以上問題，論文修改了原始 BERT 的輸入方式，讓 BERT 可以更好的處理 summarization，並提出針對 summarization task 的 **BERTSUM** 

### BERTSUM - BERT Summarization Encoder
(句子的 encoder ，用於等下輸入 BERTSUMEXT 抽取式文本摘要模型)

![](https://i.imgur.com/trrRNfP.png)

改良點如下：
1. 文本中每個句子前都加入一個 `[CLS]`
   1. 為了凸顯個別句子
   2. 將這些 `[CLS]` 輸出作為 sentence embedding 用於後續的預測任務
2. segmentation embeddings 被修改為 **interval segment embeddings** 
   1. 更好的區分多句子
   2. $sent_i$ 會根據它是奇數句 or 偶數句來決定是 $E_A$ 或 $E_B$ 
      1. 舉例，文本若為 5 句子組成 $[sent_1, sent_2, sent_3, sent_4, sent_5]$ 則此文本的 `interval segment embeddings` = $[E_A, E_B, E_A, E_B, E_A]$
   3. 如此一來 BERTSUM 的句子向量可學到相鄰句子和整個文本的意思（？要再確認下原文的目的）

### BERTSUMEXT - BERTSUM of Extractive Summarization
簡單來說，是將每段句子的 sentence embedding 取出來並丟進分類器進行預測

論文在 **BERTSUM** 上，再加入了幾層 Transformer layers 構成 **BERTSUMEXT**

BERTSUM 就像是句子的 encoder，輸入是很多句子，其中：
- $d$ : 代表文本中的句子陣列 $[sent_1,sent_2,...,sent_m]$
  - 在文本中的第 $i$ 句向量以第 $i$ 個`[CLS]`符號輸出向量 $t_i$ 表示

將這些句向量輸入額外加入的 $l$ 層 Transformer layers ，這就是 BERTSUMEXT 模型，最終可獲取 document-level 的特徵

![](https://i.imgur.com/hjAy7gG.png)

其中：
- $h_0$ : $PosEmb(T)$ ($T$ 代表前面 BERTSUM 輸出的句子向量)
- 其他參數與 BERT Tokenizer 相同

不同的地方是，模型的最後會加上一個簡單的 sigmoid classifier 輸出句子是否加入摘要集合的標籤 : 

> $\hat{y_i}=\alpha(W_0h^L_i+b_o)$

其中
- $h^L_i$ : 是 Transformer layer 第 $L$ 層 $sent_i$ 的輸出

---

### BERTSUMABS, BERTSUMEXTABS - BERTSUM of Abstractive Summarization


