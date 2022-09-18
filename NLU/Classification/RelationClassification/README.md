# Relation Classification

### Open Source Datasets

| Name | Field | Link |
| ---- | ---- | ---- |
| FinRE-v2 | Finance | [FinRE-v2](https://drive.google.com/drive/folders/17_X9cORCCJqqPYBZApQhd9chH9WN9q89?usp=sharing) |
| FinRE | Finance | [FinRE](https://drive.google.com/drive/folders/1YgDGjhwo9EDaNwjD40hc-87PQZTt_Fzl?usp=sharing) |
| people | Human Society Relations | [people](https://drive.google.com/drive/folders/10htEC8Xp-Q7oJuSV89m1TCP3CVyJROIZ?usp=sharing) |

### Some label work
I re-labeled the **FinRE** dataset as `FinRE-v2` with these rules below:
1. Extend `订单` to `订单`, `被下订单` relations so that can capture the characters between "provider" and "client"
2. Add relation `砍单`, `被砍单`: if `增持` and `减持` exist, there should have `订单` and `砍单`
3. Check and extend company relations in [`交易`, `签约`, `重组`]: more specific capture what kind of trading, eg. `买资`, `收购`, `持股`, `增持` or `减持`, etc.

### Guide
1. Download dataset and put under `datasets/` folder
2. Train: run `python demo_train.py`
3. Predict: run `python demo_predict.py`

---

### Credits
> Very big thanks to 
> - FinRE dataset comes from THUNLP: [https://github.com/thunlp/Chinese_NRE/tree/master/data/FinRE](https://github.com/thunlp/Chinese_NRE/tree/master/data/FinRE)
> - people dataset and code reference from: [https://github.com/Jacen789/relation-extraction](https://github.com/Jacen789/relation-extraction)

