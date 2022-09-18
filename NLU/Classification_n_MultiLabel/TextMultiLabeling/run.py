import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer, BertForSequenceClassification
import warnings
warnings.filterwarnings("ignore")
from args import params
from data_utils import NewsDataset
from train import model_train
from predict import get_predictions


if __name__ == "__main__":
    # load data
    np.random.seed(params.seed)
    tokenizer = BertTokenizer.from_pretrained(params.PRETRAINED_MODEL_NAME)
    df_train = pd.read_csv(f"../data/{params.task_name}/train.csv")
    df_val = pd.read_csv(f"../data/{params.task_name}/val.csv")
    df_test = pd.read_csv(f"../data/{params.task_name}/test.csv")
    for frame in [df_train, df_val, df_test]:
        frame["category"] = frame["category"].apply(lambda x: x.split(","))
    
    with open(
        f"../data/{params.task_name}/label2id.json", "r", encoding="utf-8"
    ) as f:
        label2id = json.load(f)
    
    with open(
        f"../data/{params.task_name}/id2label.json", "r", encoding="utf-8"
    ) as f:
        id2label = json.load(f)

    print(df_train.shape, df_val.shape, df_test.shape, len(label2id))
    trainset = NewsDataset(
        mode="train", tokenizer=tokenizer, max_length=params.max_len, 
        df=df_train, label_dict=label2id
    )
    valset = NewsDataset(
        mode="val", tokenizer=tokenizer, max_length=params.max_len, 
        df=df_val, label_dict=label2id
    )
    testset = NewsDataset(
        mode="test", tokenizer=tokenizer, max_length=params.max_len, 
        df=df_test, label_dict=label2id
    )
    print(len(trainset), len(valset), len(testset))


    # train
    params.NUM_LABELS = len(label2id)
    train_sampler = RandomSampler(trainset)
    val_sampler = RandomSampler(valset)
    test_sampler = RandomSampler(testset)
    trainloader = DataLoader(
        trainset, sampler=train_sampler, batch_size=params.BATCH_SIZE
    )
    valloader = DataLoader(
        valset, sampler=val_sampler, batch_size=params.BATCH_SIZE
    )
    testloader = DataLoader(
        testset, sampler=test_sampler, batch_size=params.BATCH_SIZE
    )
    params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(
        params.PRETRAINED_MODEL_NAME, num_labels=params.NUM_LABELS
    )
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model = model.to(params.device)
    model_train(model, trainloader, valloader, params)

    # predict on testset
    test_predictions, test_labels, test_metrics = get_predictions(
        model, testloader, params, compute_acc=True, compute_loss=False
    )
    for k, v in test_metrics.items():
        print(f"{k}: {np.mean(v):.3f}")
    
    params.threshold = 0.5
    _test_probs = test_predictions.sigmoid().cpu()
    _test_pred = np.zeros(_test_probs.shape)
    _test_pred[np.where(_test_probs >= params.threshold)] = 1
    print(classification_report(
        y_pred=_test_pred, y_true=test_labels.cpu(), 
        target_names=list(testset.label_map.keys())
    ))
