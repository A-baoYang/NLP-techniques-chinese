# 巨人肩膀s

同義詞挖掘的應用場景解釋
> 同义词挖掘主要应用搜索领域的非标准搜索关键词(实体)到标准实体的映射。在搜索场景下，用户搜索输入的查询query是五花八门的，经常会出现打错字，字符顺序错乱，关键词简写等问题。例如搜索电视剧“雪中悍刀行”可能会输入“电视剧悍刀雪中行”、“焊道雪中行”，“雪中行”等类似词语。而新聞的用詞也是一樣。这种非标准的实体需要通过同义词映射或者实体链指的方法来进行标准化映射，从而解析出用户真实的搜索结果。

### 經典論文
- **2017, Automatic Synonym Discovery with Knowledge Bases**

  ![](https://pic2.zhimg.com/80/v2-d0fc6d075e5269f45619b034a807ef7d_720w.jpg)

  - 基于已有知识库自动 发现新同义词，同时判定是否是同义词对的方法
  - 基于分布式语义的假设：根据语料的统计特征分析，通常具有相同上下文的词语，往往可能是同义词
  - 基于模板的挖掘思路： 通过局部上下文的特征，当同一个句子中同时提及两个实体，而且这两个实体的关系满足相同指代的意思 => 高精准、低召回率
    - ex: "The Uinted Stetes of Amearica" is commonly referred to as "America"

- **Mining Entity Synonyms with Effificient Neural Set Generation**

  ![](https://pic4.zhimg.com/v2-e09d418a275b81f6d859acf911492b9b_r.jpg)

  1. 训练一个同义词集合的分类器，通过表示一个同义词集合的整体分布和当前实体是否属于这个同义词集合。
  2. 设计一个高效的同义词生成算法，将上述学的的同义词集合判定分类器应用于词表生成所有可能的同义词集合。


### 實作方法
1. 借助已有知識庫、同義詞辭典
    - [用相似詞辭典 fine-tune Pre-trained model 後做分類，門檻值未滿則視為新詞](https://zhuanlan.zhihu.com/p/55981383)
        - 相似詞兩兩匹配為候選集，訓練出一個對相似詞敏感的向量空間；累積一定候選集之後，就可以丟 query set 進去轉向量，分類到候選集所在的各個群
        - 候選集來源
            - 開源辭典
                - `zh_CN`
                    - [相似詞詞林](https://github.com/fighting41love/funNLP/blob/master/data/%E5%90%8C%E4%B9%89%E8%AF%8D%E5%BA%93%E3%80%81%E5%8F%8D%E4%B9%89%E8%AF%8D%E5%BA%93%E3%80%81%E5%90%A6%E5%AE%9A%E8%AF%8D%E5%BA%93/%E5%90%8C%E4%B9%89%E8%AF%8D%E5%BA%93.txt)
                    - [Synonyms](https://github.com/chatopera/Synonyms)
                    - [KG][OwnThink](https://www.ownthink.com/docs/kg/) 從解釋中配對接近的詞彙
                    - [台股證券代號列表](https://isin.twse.com.tw/isin/C_public.jsp?strMode=1) + [wikidata](https://www.wikidata.org/wiki/Q713418?uselang=zh-tw)
                    - [PTT企業綽號列表](https://pttpedia.fandom.com/zh/wiki/PTT%E4%BC%81%E6%A5%AD%E7%B6%BD%E8%99%9F%E5%88%97%E8%A1%A8)
                - `en_US`
                    - [VerbNet](https://verbs.colorado.edu/verbnet/)
                    - [WordNet](https://wordnet.princeton.edu/download/standoff-files)
            - 爬蟲抓取特定模板的三元組，例如: "國立台灣大學" `-又稱->` "台大"
2. 上下文相似度 : 假如兩個詞的上下文 pattern 很接近，則兩詞為同義詞的機率較大
    - 常見做法: Word Embeddings 相似度為判斷是否是同義詞的標準
    - 缺點: 可能會變成同類詞或相關詞，但不一定同義，需要人工檢查
3. 詞面相似度 
    - 文本序列編輯距離
    - 筆畫比對


### 評估指標
**有答案(類別標記)的分群題目**，有三種常用的評估指標：
- ARI (Adjusted Rand Index) 
  - RI (Rand Index) : Measurement of the similarity between two data clusterings 反映兩類別劃分的重疊程度
  - 由於 Rand Index 無法保證隨機劃分的分群結果 RI 值接近 0，因此提出了 ARI，值域 [-1,1]，越大表示分群效果越好
- FMI (Fowlkes-Mallows Index)
  - geometric mean of the pairwise precision and recall
- NMI (Normalized [Mutual Information](https://www.cnblogs.com/emanlee/p/12492561.html)) 
  - 兩個隨機變量間的關聯程度
  - 由於分群的切分方式改變會影響到其他樣本的預測結果，所以會使用 mutual information 來衡量


## Reference
- [如何扩充知识图谱中的同义词](https://zhuanlan.zhihu.com/p/94726282)
- [中文同义词挖掘工作的踩坑之路 - 知乎](https://zhuanlan.zhihu.com/p/55981383)
- [同义词挖掘方法(一)](https://zhuanlan.zhihu.com/p/528310443)
- [同义词挖掘](https://parker2020.gitee.io/blogs/2021/04/13/%E5%90%8C%E4%B9%89%E8%AF%8D%E6%8C%96%E6%8E%98/)
- [同义词（近义词）算法总结 - CSDN](https://blog.csdn.net/u010960155/article/details/87285292)
