import copy
import json
import os
import torch
from torch.utils.data import TensorDataset
from args import Args
from utils import load_tokenizer, get_slot_labels


class InputExample(object):
    """A single training/test example for simple sequence classification
    Args:
        guid :str: Unique id for the example
        words :list: The words of the sequence
        slot_labels :list: The slot labels of the example
    """

    def __init__(self, guid, words, slot_labels=None):
        self.guid = guid
        self.words = words
        self.slot_labels = slot_labels

    def __repr__(self):  # terminal will print the message
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return (
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        )


class InputFeatures(object):
    """A single set of features of data
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, slot_labels_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.slot_labels_ids = slot_labels_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class JointProcessor(object):
    """ Processor for the JointBERT data set
    """
    def __init__(self):
        self.args = Args()
        self.slot_labels = get_slot_labels(self.args)
        self.input_text_file = "seq.in"
        self.slot_labels_file = "seq.out"
        self.tokenizer = load_tokenizer()

    @classmethod  # no need to instanilze
    def _read_file(cls, input_file, quotechar=None):
        """read file from local
        """
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, texts, slots, set_type):
        """create examples for train & validation sets
        """
        examples = []
        for i, (text, slot) in enumerate(zip(texts, slots)):
            guid = "%s-%s" % (set_type, i)
            # input_text
            words = text.split()
            # slot
            slot_labels = []
            for s in slot.split():
                slot_labels.append(
                    self.slot_labels.index(s)
                    if s in self.slot_labels
                    else self.slot_labels.index("UNK")
                )

            assert len(words) == len(slot_labels)  # 不满足条件，触发异常

            examples.append(
                InputExample(guid=guid, words=words, slot_labels=slot_labels)
            )
        return examples

    def get_examples(self, mode):
        """
        :param mode: {'train', 'val', 'test'}
        """
        data_path = os.path.join(self.args.data_dir, self.args.task, mode)
        return self._create_examples(
            texts=self._read_file(os.path.join(data_path, self.input_text_file)),
            slots=self._read_file(os.path.join(data_path, self.slot_labels_file)),
            set_type=mode,
        )

    def convert_examples_to_features(
        self, 
        examples, 
        cls_token_segment_id=0,
        pad_token_segment_id=0,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True
    ):
        # Setting based on the current model type.
        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token
        unk_token = self.tokenizer.unk_token
        pad_token_id = self.tokenizer.pad_token_id
        pad_token_label_id = self.args.ignore_index
        max_seq_len = self.args.max_seq_len

        # Tokenize word by word (for NER)
        features = []
        for (ex_index, example) in enumerate(examples):
            tokens, slot_labels_ids = [], []
            for word, slot_label in zip(example.words, example.slot_labels):
                word_tokens = self.tokenizer.tokenize(word)
                if not word_tokens:
                    word_tokens = [unk_token]
                tokens.extend(word_tokens)
                slot_labels_ids.extend(
                    [int(slot_label)] + [pad_token_label_id] * (len(word_tokens) - 1)
                )
            # Account for [CLS] and [SEP]
            special_tokens_count = 2
            # if surpass max_len then intercept
            if len(tokens) > (max_seq_len - special_tokens_count):
                tokens = tokens[:(max_seq_len - special_tokens_count)]
                slot_labels_ids = slot_labels_ids[:(max_seq_len - special_tokens_count)]
            # Add [SEP] token
            tokens += [sep_token]
            slot_labels_ids += [pad_token_label_id]
            token_type_ids = [sequence_a_segment_id] * len(tokens)
            # Add [CLS] token
            tokens = [cls_token] + tokens
            slot_labels_ids = [pad_token_label_id] + slot_labels_ids
            token_type_ids = [cls_token_segment_id] + token_type_ids
            # token indexes
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            # attention mask
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            # zero-padding up to the sequence length
            padding_length = max_seq_len - len(input_ids)
            input_ids = input_ids + ([pad_token_id] * padding_length)
            attention_mask = attention_mask + (
                [0 if mask_padding_with_zero else 1] * padding_length
            )
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
            slot_labels_ids = slot_labels_ids + ([pad_token_label_id] * padding_length)

            # check length
            assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(
                len(input_ids), max_seq_len
            )
            assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
                len(attention_mask), max_seq_len
            )
            assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(
                len(token_type_ids), max_seq_len
            )
            assert len(slot_labels_ids) == max_seq_len, "Error with slot labels length {} vs {}".format(
                len(slot_labels_ids), max_seq_len
            )

            features.append(
                InputFeatures(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    token_type_ids=token_type_ids, 
                    slot_labels_ids=slot_labels_ids,
                )
            )
        return features

    def load_and_cache_examples(self, mode):
        """
        :param mode: {'train', 'val', 'test'}
        """
        # Load data features form cache or datase file
        cached_features_file = os.path.join(
            self.args.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode,
                self.args.task,
                list(filter(None, self.args.model_name_or_path.split("/"))).pop(),
                self.args.max_seq_len,
            ),
        )
        # Load from cache if already exists
        if os.path.exists(cached_features_file):
            features = torch.load(cached_features_file)
        else:
            # process raw data using InputExample 
            if mode in ["train", "val", "test"]:
                examples = self.get_examples(mode)
            else:
                raise Exception("For mode, Only train,dev,test is available")
            features = self.convert_examples_to_features(examples)
            torch.save(features, cached_features_file)

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor(
            [f.input_ids for f in features], dtype=torch.long
        )
        all_attention_mask = torch.tensor(
            [f.attention_mask for f in features], dtype=torch.long
        )
        all_token_type_ids = torch.tensor(
            [f.token_type_ids for f in features], dtype=torch.long
        )
        all_slot_labels_ids = torch.tensor(
            [f.slot_labels_ids for f in features], dtype=torch.long
        )
        dataset = TensorDataset(
            all_input_ids, all_attention_mask, all_token_type_ids, all_slot_labels_ids
        )
        return dataset
