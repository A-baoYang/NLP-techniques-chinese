# NLP Techniques (Chinese)

###### [原站](https://www.idataagent.com) | [Medium]() | [Fanpage]() | [LinkedIn]() | Twitter | Cupoy | 方格子 | 知乎 | 微信公眾號 | 掘金

> 本 repo 旨在整理筆者在工作學習 NLP 的過程中曾使用的方法、參考資料及仿作案例，讓更多人瞭解 NLP 的博大精深（困難重重）。若有任何指教歡迎提交 issue 或來信 [martech.tw@gmail.com](mailto:martech.tw@gmail.com)

<!--
本 repo 主要介紹中文自然語言處理任務共分為 3 大類：

#### (I) **語意理解 Natural Language Understanding**
在語意理解中，又可以依照「字詞、句子」不同層級分成 3 種任務類型，分別包含以下任務：

1. 詞法分析 (Lexical Analysis)
    - 分詞 (Word Segmentation)
    - 詞性標註 (Part-of-speech Tagging)
2. 句法分析 (Sentence Analysis)
    - 句子邊界檢測 (Sentence Boundary Detection)
    - 依存句法分析 (Dependency Parsing)
    - 共指消解 (Coreference Resolution)
3. 語義分析 (Semantic Analysis)
    - 語義角色標註 (Semantic Role Labeling)
    - 詞義消歧 (Word Sense Disambiguation)：處理實體詞之間「一詞多義」的問題
    - 向量化表示 (Embeddings)

#### (II) **資訊抽取 Information Extraction**
利用模型對語義理解，從非結構化文本中抽取結構化資訊

1. 序列標註
   - 命名實體識別 (Named Entity Recognition)
2. 文本分類 (Classifications)

   利用文本向量化進行 常見機器學習任務(分類、分群、數值預測)
   - 情感分類/分數 (Sentiment Prediction)
   - 關係抽取 (Relation Extraction)：預測文本中兩實體之間的關係類型
   - 意圖識別 (Intent Detection)：分類問題背後的意圖
   - 主題分類 (Topic Modeling)：找出文本隱藏的主題
3. 綜合
   - 事件抽取 (Event Extraction)

#### (III) **語言生成 Natural Language Generation**
判斷輸入文本的語義後，經過抽取、重組、生成最佳的輸出結果

1. 機器翻譯 (Machine Translation)
2. 文本摘要 (Text Summarization)
3. 問答匹配 (Question-Answering System)：選出最適合回覆用戶提問的答案
4. 對話系統 (Dialogue System)
-->

首先由於 NLP 開源套件眾多，這邊先盤點功能較為完整的 NLP 開源平台：
- CKIP
  - ckiptagger
  - CKIP Transformer
- PaddleNLP
- LTP
- HanLP
- StanfordNLP
- SpaCy
- [JioNLP](https://github.com/dongrixinyu/JioNLP)

---

## 資料蒐集

### **爬蟲 (Web Crawling)**
應用 Python `Requests`, `Selenium`, `Scrapy` 等工具發送 HTTP Requests 來獲取公開資料

- 針對金融公開資料源，我將個人開發的一些爬蟲代碼都放在 [Crawlers](https://github.com/A-baoYang/Crawlers/tree/jupyter_gcp_cathayddt)

## 前處理 Preprocessing

### **文本清洗 (Data Cleansing)**
傳統會先進行的文本清洗、排除停用詞、標點等，再開始分析或建模。現因為 BERT 等預訓練模型的出現，為了保留完整語義，則不一定會每次都排除停用詞或標點。

停用詞列表：
- [繁體中文 zh-TW](preprocessing/stopwords_zhTW.txt)
- [簡體中文 zh-CN](preprocessing/stopwords_zhCN.txt)

### **正規化 (Data Normalization)**
其中，編碼、標點、空格、換行等等應統一格式，讓不論是 Embeddings 還是文字分析時，讓語義的一致性提高。例如：標點符號統一使用全形。

---

## (I) 語意理解 Natural Language Understanding

### **分詞 (Word Segmentation)**
從文本中定義各個詞彙的邊界。

#### 條件隨機域 (Condition Random Field, CRF)

#### 分詞器比較 
- Jieba/ckiptagger/CKIP Transformer/LTP/HanLP/StanfordNLP/SpaCy

#### [[案例] 關鍵字抽取](nlu/key_phrase_extraction/)
- 將分詞結果根據重要度再合併，形成更有意義的關鍵短語
- 使用到 `JioNLP`
- 計算關鍵詞頻次、佔比、隨時間的變化量，譬如可排序同一時間搜尋量增長最快關鍵詞

### **詞性標註 (Part-of-speech Tagging)**
根據語義分類文本中每個詞彙的詞性。

#### BIO Tagging

### **句子邊界檢測 (Sentence Boundary Detection)**
根據語義對文本進行合理斷句。

### **依存句法分析 (Dependency Parsing)**
根據語義分類文本間的詞彙的關係。

### **共指消解 (Coreference Resolution)**
挑出句子間意思相同的代名詞。

### **語義角色標註 (Semantic Role Labeling)**
根據語義分類句子中每個詞彙的語義角色。

### **詞義消歧 (Word Sense Disambiguation)**
處理實體詞之間「一詞多義」的問題。

### **向量化表示 (Embeddings)**
將詞彙、句子或文本段落轉換為能夠充分表達其語義的向量。

---

## (II) 資訊抽取 Information Extraction

### **命名實體識別 (Named Entity Recognition)**
判斷文本中實體所在的邊界及類型。

### **關係抽取 (Relation Extraction)**
預測文本中兩實體之間的關係類型。

### **情感分類/分數 (Sentiment Prediction)**
判斷文本的情緒為正面/負面/中立，或以分數形式表示、或分為不同類型（例如：驚訝、憂傷、憤怒...）。

### **意圖識別 (Intent Detection)**
分類問題背後的意圖。

### **主題分類 (Topic Modeling)：找出文本隱藏的主題**

### **事件抽取 (Event Extraction)**


---

## Knowledge Fusion 知識融合

### Entity / Relations Alignment 實體關係對齊

將相似的實體和關係之間合併，避免同義詞在圖譜內（就很像重複節點，會將本該整合的知識分散）

以中文為例：

- 筆畫比對
- 字符比對
- 詞向量距離(相似度)

<!--
---

## Data Storage 知識儲存

### Neo4j

---

## Visualization

### Neo4j Browser 

### Neo4j Bloom

### Graphlytic
-->
