import json
import logging
import numpy as np
import os
import random
from seqeval.metrics import precision_score, recall_score, f1_score
import torch
from transformers import BertTokenizer, BertConfig

from args import Args
from model import JointBERT


args = Args()
MODEL_CLASSES = {"bert": (BertConfig, JointBERT, BertTokenizer)}


def set_seed():
    torch.manual_seed(args.seed)  # CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)  # GPU
        torch.cuda.manual_seed_all(args.seed)  # multi-GPU
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmard = True


def get_slot_labels(args):
    return [
        label.strip() 
        for label in open(
            os.path.join(args.data_dir, args.task, args.slot_label_file),
            "r",
            encoding="utf-8",
        )
    ]


def load_tokenizer():
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)


def get_slot_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {
        "slot_precision": precision_score(labels, preds),
        "slot_recall": recall_score(labels, preds),
        "slot_f1": f1_score(labels, preds),
    }


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    results = {}
    slot_result = get_slot_metrics(preds, labels)
    results.update(slot_result)
    return results


def read_prediction_text(args):
    return [
        text.strip()
        for text in open(
            os.path.join(args.pred_dir, args.pred_input_file), "r", encoding="utf-8"
        )
    ]


def init_logger(args):
    if not os.path.exists("logs"):
        os.mkdir("logs")

    logging.basicConfig(
        filename=f"logs/predict-{args.task}-{args.model_dir}.log",
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )