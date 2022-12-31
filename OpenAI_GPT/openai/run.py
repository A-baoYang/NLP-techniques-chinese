from api import ChatGPTAPI
import json
from prompts import prompts
import pandas as pd
from tqdm import tqdm
from data import re_labels, re_input, news, text_list


def collect_response(prompt_templates, *prop_args):
    _res = ""
    for prop in prompt_templates:
        format_msg = prop % prop_args
        print(format_msg)

        try:
            res = chatgpt_api.get(format_msg)
            print(res)
        except Exception as e:
            print(e)
            res = ""

        _res += "\n" + res
    return _res


chatgpt_api = ChatGPTAPI()
type_list = ["ner", "ee", "sentiment", "aspect_bases_sentiment", "summarize"]
result_dict = {k: [] for k in type_list}
for type in type_list:
    _res_list = []
    for i, text in tqdm(enumerate(text_list)):
        _res = collect_response(prompts[type], text)
        _res_list.append(_res)

        if i > 10 and i % 10 == 1:
            result_dict[type] += _res_list
            pd.DataFrame(result_dict).to_json(
                "output/label-20221001.ndjson.gz", lines=True, orient="records"
            )


re_result_list = []
for i, row in tqdm(enumerate(re_input.values[:1000])):
    head, tail, text, rel = row
    head, tail = head["name"], tail["name"]
    _res = collect_response(prompts["re"], text, re_labels, head, tail, head, tail)
    re_result_list.append(_res)
    
    if i > 10 and i % 10 == 1:
        pd.DataFrame({"response": re_result_list}).to_json(
            "output/label-FinRE-20221001.ndjson.gz", lines=True, orient="records"
        )

# with open("output/20221001.json", "w", encoding="utf-8") as f:
#     json.dump(result_list, f, ensure_ascii=False, indent=4)
