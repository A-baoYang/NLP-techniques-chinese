# -*- coding: utf-8 -*-

import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel, DistilBertModel, AlbertModel
from torchcrf import CRF


PRETRAINED_MODEL_MAP = {
    'bert': BertModel,
    'distilbert': DistilBertModel,
    'albert': AlbertModel
}


class SlotClassifier(nn.Module):
    def __init__(self, input_dim, num_slot_labels, dropout_rate=0.):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class JointBERT(BertPreTrainedModel):
    def __init__(self, bert_config, args, slot_label_list):
        super(JointBERT, self).__init__(bert_config)
        self.args = args
        self.num_slot_labels = len(slot_label_list)
        self.dropout = nn.Dropout(0.3)
        if args.do_pred:
            self.bert = PRETRAINED_MODEL_MAP[args.model_type](config=bert_config)
        else:
            self.bert = PRETRAINED_MODEL_MAP[args.model_type].from_pretrained(
                args.model_name_or_path, config=bert_config, 
                from_tf=False, from_flax=False
            )
        self.slot_classifier = SlotClassifier(
            bert_config.hidden_size, self.num_slot_labels, args.dropout_rate
        )
        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)
        self.slot_pad_token_idx = slot_label_list.index(args.slot_pad_label)

    def forward(self, input_ids, attention_mask, token_type_ids, slot_labels_ids):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]
        slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0
        if slot_labels_ids is not None:
            if self.args.use_crf:
                padded_slot_labels_ids = slot_labels_ids.detach().clone()
                padded_slot_labels_ids[padded_slot_labels_ids == self.args.ignore_index] = self.slot_pad_token_idx
                slot_loss = self.crf(
                    slot_logits, padded_slot_labels_ids, 
                    mask=attention_mask.byte(), reduction="mean"
                )
                slot_loss = -1 * slot_loss
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(
                        slot_logits.view(-1, self.num_slot_labels), 
                        slot_labels_ids.view(-1)
                    )
            total_loss += self.args.slot_loss_coef * slot_loss
        outputs = (slot_logits,) + outputs[1:]
        outputs = (total_loss,) + outputs
        return outputs
