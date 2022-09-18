import os
import argparse

parser = argparse.ArgumentParser()

# model
parser.add_argument("--PRETRAINED_MODEL_NAME", type=str, default="longformerModel_full")
parser.add_argument("--task_name", type=str, default="cnyes_industry")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--seed", type=int, default=112)
parser.add_argument("--max_len", type=int, default=1024)
parser.add_argument("--BATCH_SIZE", type=int, default=12)
parser.add_argument("--EPOCHS", type=int, default=10)
parser.add_argument("--LR", type=float, default=1e-5)
parser.add_argument("--threshold", type=float, default=0.5)

parmas = parser.parse_args()
