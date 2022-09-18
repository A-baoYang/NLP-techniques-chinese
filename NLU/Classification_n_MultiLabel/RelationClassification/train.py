import os
import numpy as np
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics

from data_utils import SentenceREDataset, multi_label_metrics, get_idx2tag, load_checkpoint, save_checkpoint
from model import SentenceRE

here = os.path.dirname(os.path.abspath(__file__))


def train(hparams):
    device = hparams.device
    seed = hparams.seed
    torch.manual_seed(seed)

    pretrained_model_path = hparams.pretrained_model_path
    train_file = hparams.train_file
    validation_file = hparams.validation_file
    log_dir = hparams.log_dir
    output_dir = hparams.output_dir
    tagset_file = hparams.tagset_file
    model_file = hparams.model_file
    checkpoint_file = hparams.checkpoint_file

    max_len = hparams.max_len
    train_batch_size = hparams.train_batch_size
    validation_batch_size = hparams.validation_batch_size
    epochs = hparams.epochs

    learning_rate = hparams.learning_rate
    weight_decay = hparams.weight_decay

    # train_dataset
    train_dataset = SentenceREDataset(train_file, tagset_path=tagset_file,
                                      pretrained_model_path=pretrained_model_path,
                                      max_len=max_len)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    # model
    idx2tag = get_idx2tag(tagset_file)
    hparams.tagset_size = len(idx2tag)
    model = SentenceRE(hparams).to(device)

    # load checkpoint if one exists
    if os.path.exists(checkpoint_file):
        checkpoint_dict = load_checkpoint(checkpoint_file)
        best_f1 = checkpoint_dict['best_f1']
        epoch_offset = checkpoint_dict['best_epoch'] + 1
        model.load_state_dict(torch.load(model_file))
    else:
        checkpoint_dict = {}
        best_f1 = 0.0
        epoch_offset = 0
    # create model store path if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    running_loss = 0.0
    writer = SummaryWriter(os.path.join(log_dir, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))

    for epoch in range(epoch_offset, epochs):
        print("Epoch: {}".format(epoch))
        model.train()
        for i_batch, sample_batched in enumerate(tqdm(train_loader, desc='Training')):
            token_ids = sample_batched['token_ids'].to(device)
            token_type_ids = sample_batched['token_type_ids'].to(device)
            attention_mask = sample_batched['attention_mask'].to(device)
            e1_mask = sample_batched['e1_mask'].to(device)
            e2_mask = sample_batched['e2_mask'].to(device)
            tag_ids = sample_batched['tag_ids'].to(device)
            model.zero_grad()
            logits = model(token_ids, token_type_ids, attention_mask, e1_mask, e2_mask)
            loss = criterion(logits, tag_ids)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
            if i_batch % 10 == 9:
                writer.add_scalar('Training/training loss', running_loss / 10, epoch * len(train_loader) + i_batch)
                running_loss = 0.0

        if validation_file:
            validation_dataset = SentenceREDataset(validation_file, tagset_path=tagset_file,
                                                   pretrained_model_path=pretrained_model_path,
                                                   max_len=max_len)
            val_loader = DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=False)
            model.eval()
            with torch.no_grad():
                tags_true, tags_probs = None, None
                # tags_true, tags_pred = [], []
                metric_collect = {"f1": [],"precision": [],"recall": [],"roc_auc": [],"accuracy": []}
                for val_i_batch, val_sample_batched in enumerate(tqdm(val_loader, desc='Validation')):
                    token_ids = val_sample_batched['token_ids'].to(device)
                    token_type_ids = val_sample_batched['token_type_ids'].to(device)
                    attention_mask = val_sample_batched['attention_mask'].to(device)
                    e1_mask = val_sample_batched['e1_mask'].to(device)
                    e2_mask = val_sample_batched['e2_mask'].to(device)
                    tag_ids = val_sample_batched['tag_ids']
                    logits = model(token_ids, token_type_ids, attention_mask, e1_mask, e2_mask)
                    # pred_tag_ids = logits.argmax(1)
                    metric = multi_label_metrics(predictions=logits, labels=tag_ids, threshold=0.5)
                    for k in metric_collect.keys():
                        metric_collect[k].append(metric[k])
                    if tags_true is None:
                        tags_probs = logits
                        tags_true = tag_ids
                    else:
                        tags_probs = torch.cat((tags_probs, logits))
                        tags_true = torch.cat((tags_true, tag_ids))
                    # tags_true.extend(tag_ids.tolist())
                    # tags_pred.extend(logits.tolist())

                tags_probs = tags_probs.sigmoid().cpu()
                tags_pred = np.zeros(tags_probs.shape)
                tags_pred[np.where(tags_probs >= 0.5)] = 1
                print(metrics.classification_report(tags_true, tags_pred, labels=list(idx2tag.keys()), target_names=list(idx2tag.values())))
                # f1 = metrics.f1_score(tags_true, tags_pred, average='weighted')
                # precision = metrics.precision_score(tags_true, tags_pred, average='weighted')
                # recall = metrics.recall_score(tags_true, tags_pred, average='weighted')
                # accuracy = metrics.accuracy_score(tags_true, tags_pred)
                f1, precision, recall, accuracy = metric["f1"], metric["precision"], metric["recall"], metric["accuracy"]
                writer.add_scalar('Validation/f1', f1, epoch)
                writer.add_scalar('Validation/precision', precision, epoch)
                writer.add_scalar('Validation/recall', recall, epoch)
                writer.add_scalar('Validation/accuracy', accuracy, epoch)

                if checkpoint_dict.get('epoch_f1'):
                    checkpoint_dict['epoch_f1'][epoch] = f1
                else:
                    checkpoint_dict['epoch_f1'] = {epoch: f1}
                if f1 > best_f1:
                    best_f1 = f1
                    checkpoint_dict['best_f1'] = best_f1
                    checkpoint_dict['best_epoch'] = epoch
                    torch.save(model.state_dict(), model_file)
                save_checkpoint(checkpoint_dict, checkpoint_file)

    writer.close()
