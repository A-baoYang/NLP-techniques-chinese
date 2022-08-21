# 同義詞挖掘流程與問題

## 大綱
- 流程說明
- 待解決問題

---

### 流程說明

#### 單日

1. 依據實體分類區分詞典（ `ORG` 一類、`EVENT` 一類、動詞 `VERB` 一類...以此類推，找最相似詞的時候只從同一類型去尋找避免結果混亂
2. 將句子輸入中文全詞遮罩 BERT 推論出 Embeddings 後，切出實體詞/動詞向量
   1. 若有舊的向量空間，則加入要取得最接近詞的範圍中
3. 取最相似詞
   1. 跑 `fuzzychinese` 筆畫比對取得最相似實體/動詞
   2. 根據 Entity Embeddoings 取得空間中最接近的實體/動詞
4. 合併步驟 2 & 3 結果；篩選在 2 和 3 尋找到的最相似詞一致者，輸出為「此次合併實體對」
5. （是否移除被合併的實體向量 (TBD)）
6. 儲存所有實體的向量，當作下次搜尋最相似詞的範圍（會連同舊的實體向量一起更新到同個位址）
7. 合併到舊詞庫：搜尋今日詞庫是否已出現在舊詞庫，將其合併，其餘就加入新鍵值

  > #### Output
  - GCS bucket: https://console.cloud.google.com/storage/browser/dst-financial-knowledge-graph/similar_entity_dictionary
  ![](https://i.imgur.com/y57UEb4.png)


### 待更新
1. 動詞不能使用 fuzzychinese
2. 對英文實體的處理
   1. 更換為 Multi-Lingual BERT 因為常有混雜英文的實體詞
   2. fuzzychinese 只能適用中文、需找尋 by character 的英文模糊比對算法

### 新版
公司
1. 建公司辭典
2. 新的公司實體直接對公司辭典模糊比對 fuzzychinese
3. 沒有 match 到公司的，再彼此做相似詞搜尋；當詞庫越大，找到的相似詞會越合理
4. 測試分群

事件動詞
1. 直接用 BERT Embeddings 算相似度（Ray 分散式運行 測不同距離指標、測完看結果判斷要用哪種指標）
2. 測試分群

### References
儲存格式比較
- pickle / npy / npz / hdf
https://applenob.github.io/python/save/
- csv / json / pickle / bin / npy / png / mat...
https://stackoverflow.com/questions/9619199/best-way-to-preserve-numpy-arrays-on-disk

合併字典檔的寫法們
- https://favtutor.com/blogs/merge-dictionaries-python

