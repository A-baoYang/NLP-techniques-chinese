# Text Classification / Multi-labeling

### Datasets

| Name | Field | Link | Source |
| ---- | ---- | ---- | ---- |
| CNYES (industry) | Finance | [cnyes_industry](https://drive.google.com/drive/folders/1H0M2h6g8NN3utUDPDgilciy6L-RhUjhZ?usp=sharing) | Crawled article & industry labels from news website [CNYES](https://news.cnyes.com/news/cat/tw_stock). 37 kinds of industries. |
| MONEYDJ (content) | Finance | [moneydj_content2](https://drive.google.com/drive/folders/1MzzdpNiVSI6FrYQlTlIO2b0UzL04lBX_?usp=sharing) | Crawled article & content categories from news website [MONEYDJ](https://www.moneydj.com/kmdj/common/listnewarticles.aspx?svc=NW&a=X0300001). 20 kinds of content categories.  |
| IFLYTEK (content) | General | [iflytek_public](https://drive.google.com/drive/folders/14VRSyjTCPkzQm-rWaUqqS6CSzLLbeZit?usp=sharing) | Released by [CLUE benchmark](https://github.com/CLUEbenchmark/CLUE). Around 17000 rows of long-text description about applications with 119 classes. |

### Guide
1. Download dataset and put under `../data/` folder
2. Set your custom params in `args.py`, ex: `task_name` (which is your data folder name)
3. Train/Predict: run `python run.py`

### Articles
If you need a clear walk through, read my article here: https://idataagent.com/2022/05/03/%E3%80%90%E8%87%AA%E7%84%B6%E8%AA%9E%E8%A8%80%E8%99%95%E7%90%86%E3%80%91longformer-%E4%B8%AD%E6%96%87%E9%95%B7%E6%96%87%E6%9C%AC-bert-%E6%A8%A1%E5%9E%8B-%E6%96%B0%E8%81%9E%E5%88%86%E9%A1%9E%E5%AF%A6/ (Chinese) 

---

### Credits
> Very big thanks to 
> - CNYES & MoneyDJ website for not blocking my crawlers :)
> - partial code reference from: 
>   - [《進擊的 BERT：NLP 界的巨人之力與遷移學習》](https://leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html)
>   - [Transfer Learning for NLP: Fine-Tuning BERT for Classification](https://www.analyticsvidhya.com/blog/2020/07/transfer-learning-for-nlp-fine-tuning-bert-for-text-classification/)
