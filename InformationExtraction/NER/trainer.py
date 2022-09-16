import json
import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.nn.utils import clip_grad_norm_
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
from args import Args
from utils import MODEL_CLASSES, get_slot_labels, load_tokenizer, compute_metrics


class NER_model(object):
    def __init__(self, train_dataset=None, val_dataset=None, test_dataset=None) -> None:
        self.args = Args()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.best_f = 0
        self.slot_label_list = get_slot_labels(self.args)
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = self.args.ignore_index
        # BertConfig, JointBERT, BertTokenizer
        self.config_class, self.model_class, _ = MODEL_CLASSES[self.args.model_type] 
        self.bert_config = self.config_class.from_pretrained(
            self.args.model_name_or_path, finetuning_task=self.args.task
        )
        self.tokenizer = load_tokenizer()
        # JointBERT
        self.model = self.model_class(
            self.bert_config, self.args, self.slot_label_list
        )
        self.device = (
            "cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu"
        )
        self.model = nn.DataParallel(self.model, device_ids=[int(i) for i in self.args.device_ids.split(",")])
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(
            self.train_dataset, 
            sampler=train_sampler, 
            batch_size=self.args.batch_size
        )
        # num_training_steps in scheduler
        t_total = (
            len(train_dataloader)
            // self.args.gradient_accumulation_steps
            * self.args.num_train_epochs
        )
        # optimizer & learning rate
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=t_total
        )

        # Train
        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()
        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            num_step = 0
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "slot_labels_ids": batch[3]
                }
                if self.args.model_type != "distilbert":
                    inputs["token_type_ids"] = batch[2]
                
                # Forward propagation
                outputs = self.model(**inputs)
                loss = outputs[0]

                # Gradient accumulation
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                loss.mean().backward()

                tr_loss += loss.mean().item()
                num_step += 1

                # Update gradient every n steps
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    # gradient clip
                    clip_grad_norm_(
                        self.model.parameters(), self.args.max_grad_norm
                    )
                    # Update optimizer
                    optimizer.step()
                    # Update learning rate
                    scheduler.step()
                    # Clean gradients
                    self.model.zero_grad()
                    global_step += 1
                
                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            # validation
            res_dict = dict()
            res = self.evaluate("val")
            # update metrics
            for k, v in res.items():
                if k not in res_dict:
                    res_dict.update({k: [v]})
                else:
                    res_dict[k].append(v)
            # update loss
            if "train_loss" not in res_dict:
                res_dict.update({"train_loss": tr_loss})
            else:
                res_dict["train_loss"].append(tr_loss)
            # update f1 & save model
            if self.best_f < res["slot_f1"]:
                self.best_f = res["slot_f1"]
                self.save_model()

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break
        
        # Save performance dict to logs/
        with open(
            f"logs/performance-{self.args.model_dir}.json",
            "w",
            encoding="utf-8",
        ) as f: 
            json.dump(res_dict, f, ensure_ascii=False, indent=4)
        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        if mode == "test":
            dataset = self.test_dataset
        elif mode == "val":
            dataset = self.val_dataset
        else:
            raise Exception("Only val and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(
            dataset, sampler=eval_sampler, batch_size=self.args.batch_size
        )
        eval_loss = 0.0
        nb_eval_steps = 0
        slot_preds, out_slot_labels_ids = None, None
        
        self.model.eval()
        
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "slot_labels_ids": batch[3]
                }
                if self.args.model_type != "distilbert":
                    inputs["token_type_ids"] = batch[2]

                outputs = self.model(**inputs)
                tmp_eval_loss, slot_logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Token classification prediction
            if slot_preds is None:
                if self.args.use_crf:
                    # torchcrf.decode() returns list with best index directly
                    slot_preds = np.array(self.model.module.crf.decode(slot_logits))
                else:
                    slot_preds = slot_logits.detach().cpu().numpy()
                out_slot_labels_ids = inputs["slot_labels_ids"].detach().cpu().numpy()
            
            else:
                if self.args.use_crf:
                    slot_preds = np.append(
                        slot_preds, 
                        np.array(self.model.module.crf.decode(slot_logits)), 
                        axis=0
                    )
                else:
                    slot_preds = np.append(
                        slot_preds, 
                        slot_logits.detach().cpu().numpy(),
                        axis=0
                    )
                out_slot_labels_ids = np.append(
                    out_slot_labels_ids,
                    inputs["slot_labels_ids"].detach().cpu().numpy(),
                    axis=0
                )
        eval_loss = eval_loss / nb_eval_steps
        results = {"loss": eval_loss}

        if not self.args.use_crf:
            slot_preds = np.argmax(slot_preds, axis=2)
        slot_label_map = {
            i: label for i, label in enumerate(self.slot_label_list)
        }
        out_slot_label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
        slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
        # compute metrics
        for i in range(out_slot_labels_ids.shape[0]):
            for j in range(out_slot_labels_ids.shape[1]):
                if out_slot_labels_ids[i, j] != self.pad_token_label_id:
                    # true label
                    out_slot_label_list[i].append(
                        slot_label_map[out_slot_labels_ids[i][j]]
                    )
                    # pred label
                    slot_preds_list[i].append(
                        slot_label_map[slot_preds[i][j]]
                    )
        total_result = compute_metrics(
            preds=slot_preds_list, labels=out_slot_label_list
        )
        results.update({
            "true_pred_list": [(out_slot_label_list, slot_preds_list)]
        })
        results.update(total_result)
        return results

    def save_model(self):
        """Save model checkpoint
        """
        output_dir = os.path.join(self.args.model_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        model_to_save.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_config.bin"))

    def load_model(self):
        try:
            self.bert_config = self.config_class.from_pretrained(self.args.model_dir)
            self.model = self.model_class.from_pretrained(
                self.args.model_dir, 
                config=self.bert_config,
                args=self.args,
                slot_label_list=self.slot_label_list
            )
            self.model.to(self.device)
        except:
            raise Exception("Some model files might be missing...")
        
    def _convert_texts_to_tensors(
        self,
        texts,
        cls_token_segment_id=0,
        pad_token_segment_id=0,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
    ):
        # Setting based on the current model type
        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token
        unk_token = self.tokenizer.unk_token
        pad_token_id = self.tokenizer.pad_token_id
        max_seq_len = self.args.max_seq_len
        input_ids_batch, attention_mask_batch, token_type_ids_batch, slot_label_mask_batch = [], [], [], []

        for text in texts:
            tokens, slot_label_mask = [], []
            for word in text.split():
                word_tokens = self.tokenizer.tokenize(word)
                if not word_tokens:
                    word_tokens = [unk_token]
                tokens.extend(word_tokens)
                slot_label_mask.extend(
                    [0] + [self.pad_token_label_id] * (len(word_tokens) - 1)
                )

            # Account for [CLS] & [SEP]
            special_tokens_count = 2
            if len(tokens) > (max_seq_len - special_tokens_count):
                tokens = tokens[:(max_seq_len - special_tokens_count)]
                slot_label_mask = slot_label_mask[:(max_seq_len - special_tokens_count)]
            # Add [SEP] token
            tokens += [sep_token]
            slot_label_mask += [self.pad_token_label_id]
            token_type_ids = [sequence_a_segment_id] * len(tokens)
            # Add [CLS] token
            tokens = [cls_token] + tokens
            slot_label_mask = [self.pad_token_label_id] + slot_label_mask
            token_type_ids = [cls_token_segment_id] + token_type_ids

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            padding_length = max_seq_len - len(input_ids)
            input_ids = input_ids + ([pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
            slot_label_mask = slot_label_mask + ([self.pad_token_label_id] * padding_length)

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            token_type_ids_batch.append(token_type_ids)
            slot_label_mask_batch.append(slot_label_mask)
            
        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long).to(self.device)
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long).to(self.device)
        token_type_ids_batch = torch.tensor(token_type_ids_batch, dtype=torch.long).to(self.device)
        slot_label_mask_batch = torch.tensor(slot_label_mask_batch, dtype=torch.long).to(self.device)
        print(input_ids_batch.size())

        dataset = TensorDataset(
            input_ids_batch,
            attention_mask_batch,
            token_type_ids_batch,
            slot_label_mask_batch,
        )
        return dataset

    def predict(self, texts):
        dataset = self._convert_texts_to_tensors(texts)
        sampler = SequentialSampler(dataset)
        data_loader = DataLoader(
            dataset, sampler=sampler, batch_size=self.args.batch_size
        )
        all_slot_label_mask, slot_preds = None, None

        for batch in tqdm(data_loader, desc="Predicting"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "slot_labels_ids": None,
                }
                if self.args.model_type != "distilbert":
                    inputs["token_type_ids"] = batch[2]
                outputs = self.model(**inputs)
                _, slot_logits = outputs[:2]

                if slot_preds is None:
                    if self.args.use_crf: 
                        slot_preds = np.array(self.model.crf.decode(slot_logits))
                    else:
                        slot_preds = slot_logits.detach().cpu().numpy()
                    all_slot_label_mask = batch[3].detach().cpu().numpy()
                else:
                    if self.args.use_crf:
                        slot_preds = np.append(
                            slot_preds,
                            np.array(self.model.crf.decode(slot_logits)),
                            axis=0
                        )
                    else:
                        slot_preds = np.append(
                            slot_preds, slot_logits.detach().cpu().numpy(),
                            axis=0
                        )
                    all_slot_label_mask = np.append(
                        all_slot_label_mask, batch[3].detach().cpu().numpy()
                    )
        if self.args.use_crf:
            slot_preds = slot_preds
        else:
            slot_preds = np.argmax(slot_preds, axis=2)
        slot_label_map = {i: label for i, label in enumerate(self.slot_label_list)}
        slot_preds_list = [[] for _ in range(slot_preds.shape[0])]

        for i in range(slot_preds.shape[0]):
            for j in range(slot_preds.shape[1]):
                if all_slot_label_mask[i, j] != self.pad_token_label_id:
                    slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])
        print("finish prediction!!!")
        return texts, slot_preds_list