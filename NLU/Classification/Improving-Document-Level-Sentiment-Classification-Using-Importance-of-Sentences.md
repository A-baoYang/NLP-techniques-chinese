# Improving Document-Level Sentiment Classification Using Importance of Sentences

- Authors: Gihyeon Choi, Shinhyeok Oh and Harksoo Kim
- Publish Year Month: 2020.11
- Thesis: https://arxiv.org/pdf/2103.05167v1.pdf

## (Self Comment)
- 

## Context
- Abrstract
- Intro
- Document-Level Sentiment Analysis Model
- Data
- Experiments

## Document-Level Sentiment Analysis Model 

![](https://i.imgur.com/d61LthQ.png)

1. Sentence Embedding : By a sentence encoder using ALBERT
2. Sentence Embedding + Sentiment Class Embedding : Enrich the sentence embeddings by adding embedding of sentiment classes
    - 接一個維度是 sentiment label 數量的全連接層，將句子向量投影到情緒標籤數量維度的空間中
3. **Importance of Each Sentence** : Compute by **GRU gate functions** in document Encoder
    - 輸入到 Sigmoid function 中取得 importance degree
    - 然後將經過 Sigmoid function 後的 sentence embeddings 輸入到 GRU encoder 
4. Weighted Document Embedding : Weighted sum sentences according to computed importances
5. Weighted Document Embedding + Sentiment Class Embedding : Enrich the document embeddings by adding embedding of sentiment classes
    - 接一個維度是 sentiment label 數量的全連接層，將句子向量投影到情緒標籤數量維度的空間中
6. Sentiment Classes Probabilities : Through a linear classifier (in_dim=embeddings.shape[1], out_dim=num_classes)