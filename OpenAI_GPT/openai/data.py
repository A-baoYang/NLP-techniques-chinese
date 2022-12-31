import gcsfs
import pandas as pd


fs = gcsfs.GCSFileSystem(project="dst-dev2021")
re_labels_filepath = (
    "gs://dst-financial-knowledge-graph/relation_classification/FinRE-v2/relation.txt"
)
re_train_filepath = (
    "gs://dst-financial-knowledge-graph/relation_classification/FinRE-v2/train.jsonl"
)
re_test_filepath = (
    "gs://dst-financial-knowledge-graph/relation_classification/FinRE-v2/test.jsonl"
)
article_filepath = "gs://dst-news-largitdata/domestic-extracted-backup/kgbuilder-output-AllNews-20221001.ndjson.gz"

with fs.open(re_labels_filepath, "r") as f:
    re_labels = f.read().split("\n")

re_labels = [item for item in re_labels if "被" not in item]
re_input = pd.read_json(re_train_filepath, lines=True)
re_input = pd.concat([re_input, pd.read_json(re_train_filepath, lines=True)])
news = pd.read_json(article_filepath, lines=True, orient="records", compression="gzip")
news["content"] = news["title"] + " " + news["content"]
text_list = news.content.unique().tolist()

print("FinRE labels: ", re_labels)
print("FinRE inputs amount: ", len(re_input))
print("News amount: ", len(text_list))
