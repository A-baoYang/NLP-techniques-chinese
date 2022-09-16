# -*- coding: utf-8 -*-

import os


class Args(object):
    def __init__(self):
        self.task = "input"
        self.model_type = "bert"
        self.model_dir = "FiNER"  # Path to save & load model
        self.data_dir = "model_data"  # The input data path
        self.slot_label_file = "slot_label.txt"  # Slot label file
        self.model_name_or_path = "bert-base-chinese"
        self.seed = 1234  # Random seed for initialization
        self.device_ids = "0,1,2,3"
        self.batch_size = 16  # batch_size for training & evaluation
        self.max_seq_len = 200  # The maximum total input sequence length after tokenization
        self.learning_rate = 5e-5  # The initial learning rate for Adam
        self.num_train_epochs = 50  # Total number of training epochs
        self.weight_decay = 0.0
        self.gradient_accumulation_steps = 1  # Number of updates steps to accumulate before performing a backward/update pass
        self.adam_epsilon = 1e-8  # Epsilon for Adam optimizer
        self.max_grad_norm = 1.0  # Max gradient normalization
        self.max_steps = -1  # If > 0: set toal number of training strps to perform. Override num_train_epochs
        self.warmup_steps = 0  # Linear warmup over warmup_steps
        self.dropout_rate = 0.1  # Dropout for fully-connected layers
        self.logging_steps = 200  # Log every X updates steps
        self.save_steps = 200  # Save checkpoint every X updates steps
        self.do_train = True  # Whether to run training or not
        self.do_eval = True  # Whether to run evaluation on testset or not
        self.no_cuda = 0  # Avoid using CUDA when available
        self.ignore_index = -100  # Specifies a target value that is ignored and does not contribute to the input gradient
        self.slot_loss_coef = 1.0  # Coefficient for the slot loss
        # For prediction
        self.do_pred = True  # Whether to run prediction or not
        self.pred_input_dir = "./model_data/input"  # The prediction input dir
        self.pred_input_file = "ner_inputs.json"  # The input text file of lines for prediction
        self.pred_output_dir = "./model_data/output"  # The prediction input dir
        self.pred_output_file = "ner_outputs.json"  # The output file of prediction
        # CRF options
        self.use_crf = True  # Whether to use CRF or not
        self.slot_pad_label = "PAD"  # Pad token for slot label pad(to be ignore when cauculate loss)


