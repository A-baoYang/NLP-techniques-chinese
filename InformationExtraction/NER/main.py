import os
from args import Args
from processor import JointProcessor
from trainer import NER_model
from utils import set_seed, read_prediction_text


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    set_seed()
    args = Args()
    processor = JointProcessor()
    train_dataset = processor.load_and_cache_examples("train")
    val_dataset = processor.load_and_cache_examples("val")
    test_dataset = processor.load_and_cache_examples("test")

    ner = NER_model(train_dataset, val_dataset, test_dataset)

    if args.do_train:
        ner.train()
    elif args.do_eval:
        ner.load_model()
        ner.evaluate("test")
    elif args.do_pred:
        ner.load_model()
        texts = read_prediction_text(args)
        texts, slot_preds_list = ner.predict(texts)
