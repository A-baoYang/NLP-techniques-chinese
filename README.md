# Knowledge Graph - NLP Techniques & Databases

## Text Data Collection

### Data Crawling

More info in my another repo: [Crawlers](https://github.com/A-baoYang/Crawlers/tree/jupyter_gcp_cathayddt)

### Data Cleansing

### Data Normalization 

資料風格正規化，例如標點符號統一為全形

---

## Knowledge Extraction 知識抽取

### Event Extraction 事件抽取

#### 語義角色標註

對文章中的每個分詞結果預測判斷其語義角色，例如是屬於本段落的主詞或是受詞

#### 依存句法分析

將文章中的各種語義角色根據中文句法規則連接起來，形成樹狀結構的語義關係。許多開源套件利用到 BERT Model 對語義的理解，增加正確判斷的機率

#### 指代消解

### Named Entity Recognition 命名實體識別

- CRF
- Token Classification

### Relation Extraction 關係抽取

- Long-text Classification

---

## Knowledge Fusion 知識融合

### Entity / Relations Alignment 實體關係對齊

將相似的實體和關係之間合併，避免同義詞在圖譜內（就很像重複節點，會將本該整合的知識分散）

以中文為例：

- 筆畫比對
- 字符比對
- 詞向量距離(相似度)

---

## Data Storage 知識儲存

### Neo4j

---

## Visualization

### Neo4j Browser 

### Neo4j Bloom

### Graphlytic
