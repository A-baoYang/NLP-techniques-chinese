import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report
import torch


class NewsDataset(torch.utils.data.Dataset):
    # 讀取資料、參數初始化
    def __init__(self, mode, tokenizer, max_length, df, label_dict):
        assert mode in ["train", "val", "test", "ood"]
        self.mode = mode
        self.df = df
        self.len = len(self.df)
        self.max_length = max_length
        self.label_map = label_dict
        self.tokenizer = tokenizer

    # 回傳一筆訓練/測試資料
    # @pysnooper.snoop() # 打印出轉換過程
    def __getitem__(self, idx):
        if self.mode == "ood":
            text = self.df.loc[idx, "title_content"]
            # label_tensor = None
        else:
            text, label = self.df.loc[idx, ["title_content", "category"]].values
            # label 文字轉換成索引數字
            label_ids = [self.label_map[name] for name in label]
            label_ids = [1 if i in label_ids else 0 for i in range(len(self.label_map))]
            label_tensor = torch.tensor(np.array(label_ids).astype(np.float32).tolist())

        tokens = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True)
        tokens_tensor = torch.tensor(tokens["input_ids"])
        segments_tensors = torch.tensor(tokens["token_type_ids"])
        masks_tensors = torch.tensor(tokens["attention_mask"])
        if self.mode == "ood":
            return (tokens_tensor, segments_tensors, masks_tensors)
        else:
            return (tokens_tensor, segments_tensors, masks_tensors, label_tensor)

    def __len__(self):
        return self.len


def multi_label_metrics(predictions, labels, threshold=0.5):
    # apply sigmoid on predictions which are of shape (batch_size, num_labels)
    probs = predictions.sigmoid().cpu()
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels.cpu()
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    roc_auc = roc_auc_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics
