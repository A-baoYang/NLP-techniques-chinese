# Named Entity Recognition

### Open Source Datasets
| Name | Field | Link |
| ---- | ---- | ---- |
| 中國人民日報新聞資料 | General | [people_daily](https://drive.google.com/drive/folders/1yITTNsiE5qTHi5aagyVpSvsIqLRycK2w?usp=sharing) |
| 微博貼文 | General | [weibo](https://drive.google.com/drive/folders/1Dj7QuUSqdi_tQXUjBPjVFFpVriSlnHJc?usp=sharing) |

**Folder Structure Example**

```
people_daily/
├── train/
    ├── seq.in
    └── seq.out
├── dev/
    ├── seq.in
    └── seq.out
├── test/
    ├── seq.in
    └── seq.out
└── slot_label.txt
```

### Guide
1. put the dataset folder under where you set as `args.data_dir`/`args.task`
    - ex: download the `weibo` dataset, put it under `data/` folder, set `args.data_dir`="data"
2. Adjust arguments in `args.py`
3. Run `python main.py` to stat train, evaluate, or predict

--- 

### Credits
> Very big thanks to 
> - [https://github.com/tyistyler/Bert-Chinese-NER](https://github.com/tyistyler/Bert-Chinese-NER)