import torch
from torch import nn
from tqdm import tqdm
from data_utils import multi_label_metrics


def get_predictions(model, dataloader, params, compute_acc=False, compute_loss=False):
    predictions, labelss = None, None
    # total, correct = 0, 0
    losses = 0.0
    metrics = {"f1": [], "roc_auc": [], "accuracy": []}
    loss_fn = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for data in tqdm(dataloader):
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda") for t in data if t is not None]
                if compute_acc:
                    tokens_tensors, segments_tensors, masks_tensors, labels = data
                else:
                    tokens_tensors, segments_tensors, masks_tensors = data

                outputs = model(
                    input_ids=tokens_tensors,
                    token_type_ids=segments_tensors,
                    attention_mask=masks_tensors
                )
                pred = outputs.logits
                if compute_loss:
                    loss = loss_fn(pred, labels)
                    losses += loss.mean().item()
                torch.cuda.empty_cache()

                # 計算分類成效
                if compute_acc:
                    metric = multi_label_metrics(predictions=pred, labels=labels, threshold=params.threshold)
                    for k in metrics.keys():
                        metrics[k].append(metric[k])
                    # total += labels.size(0)
                    # correct += (pred == labels).sum().item()

                # 紀錄當前 batch
                if predictions is None:
                    predictions = pred
                    labelss = labels
                else:
                    predictions = torch.cat((predictions, pred))
                    labelss = torch.cat((labelss, labels))
        
        if compute_acc and compute_loss:
            return predictions, labelss, metrics, losses
        if compute_acc:
            # acc = correct / total
            return predictions, labelss, metrics
        return predictions, losses

