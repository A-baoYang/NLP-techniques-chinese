import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data_utils import multi_label_metrics
from predict import get_predictions


def model_train(model, trainloader, valloader, params):

    writer = SummaryWriter(f'runs/multilabel-{params.task_name}-{params.PRETRAINED_MODEL_NAME}')

    print("Start training")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.LR)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(params.EPOCHS)):
        running_loss = 0.0
        metrics = {"f1": [], "roc_auc": [], "accuracy": []}

        for i, data in enumerate(tqdm(trainloader)):
            
            tokens_tensors, segments_tensors, masks_tensors, labels = [t.to(params.device) for t in data]
            optimizer.zero_grad()
            
            # forward pass
            outputs = model(input_ids=tokens_tensors,
                            token_type_ids=segments_tensors,
                            attention_mask=masks_tensors)
            loss = loss_fn(outputs.logits, labels)

            # backward
            loss.backward()  
            # https://discuss.pytorch.org/t/loss-backward-raises-error-grad-can-be-implicitly-created-only-for-scalar-outputs/12152/3
            optimizer.step()

            # 紀錄當前 batch loss
            running_loss += loss.mean().item()
            # 紀錄當前 metrics
            metric = multi_label_metrics(predictions=outputs.logits, labels=labels, threshold=params.threshold)
            for k in metrics.keys():
                metrics[k].append(metric[k])
            torch.cuda.empty_cache()

            if i % 5000 == 214:
                # 計算分類準確率
                # trainset
                print("[Epoch %s] Train Loss: %.3f \n" % (epoch, running_loss))
                # _, labelss, metrics = get_predictions(clf, trainloader, compute_acc=True, compute_loss=False)
                print("    Train F1: %.3f \n" % (np.mean(metrics["f1"])),
                      "    Train ROC AUC: %.3f \n " % (np.mean(metrics["roc_auc"])),
                      "    Train Accuracy: %.3f \n " % (np.mean(metrics["accuracy"])))
            
                # valset
                _, val_labelss, val_metrics, val_loss = get_predictions(
                    model, valloader, compute_acc=True, compute_loss=True
                )
                print("[Epoch %s] Val Loss: %.3f \n" % (epoch, val_loss))
                print("    Val F1: %.3f \n" % (np.mean(val_metrics["f1"])),
                      "    Val ROC AUC: %.3f \n " % (np.mean(val_metrics["roc_auc"])),
                      "    Val Accuracy: %.3f \n " % (np.mean(val_metrics["accuracy"])))

                writer.add_scalars(
                    "Training vs. Validation Loss",
                    {"Training": running_loss, "Validation": val_loss},
                    epoch * len(trainloader) + i)

                for name in ["f1", "roc_auc", "accuracy"]:
                    writer.add_scalars(
                        f"Training vs. Validation {name.upper()}",
                        {"Training": np.mean(metrics[name]), "Validation": np.mean(val_metrics[name])},
                        epoch * len(trainloader) + i)
            
                torch.save(model.state_dict(), "zh_longformer-multilabel-moneydj_content2.pth")
                writer.flush()
