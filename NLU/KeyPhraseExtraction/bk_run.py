#%%
import json
import os

from key_phrase_module import *

#%%
args = Args()
department = "CIST"
input_filepath = os.path.join(
    args.root_dir,
    args.data_dir,
    department,
    "chatbot_records_CSIT_20200803_20211101.csv",
)
#%%
xl = file_io(input_filepath)
df = xl.parse(xl.sheet_names[0]).iloc[:, :3]
date_col = "DATE"
df = df.rename(columns={date_col: "date"})
prod_kw = xl.parse(xl.sheet_names[1])
prod_kws = get_prod_keywords(prod_kw)

# 國泰專有產品字
word_weight = {key: 1 for key in prod_kws}

#%%
# 關鍵詞抽取
msg_col = "MSG_CONTENT"
df = gen_kw_phrase(df, msg_col, word_weight)
df = gen_time_durations(df, msg_col)
min_date, max_date = df.date.min(), df.date.max()
# filepath = os.path.join(
#     args.root_dir, args.output_dir,
#     f"afa_chatbot_records-{department}-{min_date}_{max_date}.csv")
# df.to_csv(filepath, index=False)

#%%
kw_col = "phrase"
drop_col = ["keyword"]
df_phrase_stacked = make_stacked(df, kw_col, drop_col, msg_col)

#%%
# 關鍵詞分析（頻次、佔比、成長率）
kw_col = "phrase"
df_melt_list = []
duration_cols = ["month", "weekOfYear", "date"]
for duration_col in tqdm(duration_cols):
    _df = gen_kw_stats_table(df_phrase_stacked, duration_col, kw_col)
    # growth_rate 將同樣數值的（關鍵詞新出現的那個時間點）改為空值輸出
    _df["growth_rate"] = np.where(
        _df["percentage"] == _df["growth_rate"], np.nan, _df["growth_rate"]
    )
    print(_df.head())
    df_melt_list.append(_df)

#%%
sheet_names = ["每月關鍵詞合併統計", "每週關鍵詞合併統計", "每日關鍵詞合併統計"]
filepath = os.path.join(
    args.root_dir,
    args.output_dir,
    f"afa_{department}-{kw_col}-stats-original-{min_date}_{max_date}.xlsx",
)
with pd.ExcelWriter(filepath) as writer:
    for i in tqdm(range(len(sheet_names))):
        df_melt_list[i].to_excel(writer, sheet_name=sheet_names[i])

#%%
all_temp = gen_kw_stats_only_new(df_phrase_stacked, duration_cols, kw_col)

# %%
sheet_names = ["每月新進關鍵詞合併統計", "每週新進關鍵詞合併統計", "每日新進關鍵詞合併統計"]
filepath = os.path.join(
    args.root_dir,
    args.output_dir,
    f"afa_{department}-{kw_col}-stats-new-{min_date}_{max_date}.xlsx",
)
with pd.ExcelWriter(filepath) as writer:
    for i in tqdm(range(len(sheet_names))):
        all_temp[duration_cols[i]].to_excel(writer, sheet_name=sheet_names[i])

# 關鍵詞共現統計
kw_col = "phrase"
afa_kw_corpus = collect_kw_corpus(df, kw_col)
kw_concurrency_sorted = count_concurrency(afa_kw_corpus)

#%%
filepath = os.path.join(
    args.root_dir, args.output_dir, f"sorted_cooccurrence-{kw_col}.json"
)
with open(filepath, "w", encoding="utf-8") as f:
    json.dump(kw_concurrency_sorted, f, ensure_ascii=False, indent=4)

# %%
