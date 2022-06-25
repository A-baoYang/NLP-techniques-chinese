from ckiptagger import construct_dictionary, WS
from collections import Counter
import datetime as dt
import jionlp as jio
from opencc import OpenCC
# import matplotlib
# matplotlib.rcParams['font.family'] = ['Heiti TC']
# import matplotlib.pyplot as plt
# from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 路徑設定
class Args:
    def __init__(self):
        self.root_dir = "/home/abao.yang@cathayholdings.com.tw"
        self.data_dir = "forResearch/data/gcs/afa_chatbot"
        self.output_dir = "forResearch/Afa_chatbot/external_output"

# report style 設定
# plt.style.use('ggplot')
# fontproperties = FontProperties(fname='font_ch.ttf')
# font = 'font_ch.ttf'

# Initialize drivers
# ws_driver = CkipWordSegmenter(level=3, device=0)
ws = WS('../data/gcs/ckiptagger_data')

# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer()

print('Loading Traditional Chinese stopwords...')
stopword_file_zhTW = 'stopwords_zhTW.txt'
stopwords_zhTW = list()
with open(stopword_file_zhTW, 'r', encoding='utf-8') as f:
    for l in tqdm(f.readlines()):
        if not len(l.strip()):
            continue
        stopwords_zhTW.append(l.strip())

stopwords_zhTW += ['想','請問','收到','剛剛']



# functions
def get_prod_keywords(prod_kw):
    prod_kw = prod_kw.fillna("")
    split_pattern = "\||\,"
    prod_kws = []
    prod_kw = [re.split(split_pattern, s) for s in prod_kw.values.reshape(-1) if s]
    for l in tqdm(prod_kw):
        for c in l:
            if c:
                prod_kws.append(c.strip())
    prod_kws = np.unique(prod_kws).tolist()
    return prod_kws


def word_segment(sent_list, word_weight, tool='ckiptagger', stopwords=stopwords_zhTW):
    ws_nostopwords = list()
    if tool == 'ckip-transformers':
        word_list = ws_driver([sent_list], use_delim=True)
    elif tool == 'ckiptagger':
        custom_dictionary = construct_dictionary(word_weight)
        word_list = ws(sent_list, coerce_dictionary=custom_dictionary)
    else:
        print('Please specify which word segmentation you want to use')

    for wl in tqdm(word_list):
        ws_nostopwords.append([w for w in wl if w not in stopwords])
    del word_list
    return ws_nostopwords
    
    
def phrase_extractor(sent, word_weight, stopwords=stopwords_zhTW):
    """

    """
    tw2sp = OpenCC('tw2sp')
    s2twp = OpenCC('s2twp')
    zhcn_sent = tw2sp.convert(sent)
    zhcn_word_weight = {tw2sp.convert(k): v for k, v in word_weight.items()}
    key_phrases = jio.keyphrase.extract_keyphrase(
        zhcn_sent,
        specified_words=zhcn_word_weight)
    key_phrases = [s2twp.convert(kw) for kw in key_phrases]
    key_phrases = [phrase for phrase in key_phrases if phrase not in stopwords]
    return key_phrases


def week_text(x, min_year):
    if (x > 53):
        year = min_year + 1
        x = x - 53
    else:
        year = min_year
    
    start = dt.datetime.strftime(dt.datetime.strptime(f"{year}-W{x}-1", "%Y-W%W-%w"), "%Y%m%d")
    end = dt.datetime.strftime(dt.datetime.strptime(f"{year}-W{x}-0", "%Y-W%W-%w"), "%Y%m%d")
        
    return f"{start}-{end}"


def gen_kw_phrase(df, msg_col, word_weight):
    """
    對 dataframe 生成 keyword & phrase 欄位
    """
    sent_list = df[msg_col].values.tolist()

    word_list = word_segment(
        sent_list=[str(sent) for sent in sent_list], 
        word_weight=word_weight, 
        tool='ckiptagger', stopwords=stopwords_zhTW)
    df['keyword'] = word_list
    df['keyword'] = df['keyword'].apply(lambda x: ','.join(x))
    
    phrase_list = list()
    for sent in tqdm(sent_list):
        phrase_list.append(phrase_extractor(
            sent=str(sent), 
            word_weight=word_weight,
            stopwords=stopwords_zhTW
        ))
    df['phrase'] = phrase_list
    df['phrase'] = df['phrase'].apply(lambda x: ','.join(x))
    # df['keyword_n_phrase'] = df['keyword'] + df['phrase']
    return df


def gen_time_durations(df, msg_col):
    """
    對 dataframe 生成 month & weekOfYear 欄位
    """
    df["date"] = df["date"].astype(str).apply(
        lambda x: str(x).replace("-",""))
    df['month'] = df["date"].apply(lambda x: str(x)[:6])
    min_year = df["date"].min()[:4]
    df['weekOfYear'] = df["date"].apply(
        lambda x: dt.datetime.strptime(str(x), '%Y%m%d').isocalendar()[1] + 
        53 * (dt.datetime.strptime(str(x), '%Y%m%d').isocalendar()[0] - int(min_year)))
    df['weekOfYear'] = df['weekOfYear'].apply(lambda x: week_text(x, min_year))
    select_cols = [msg_col,"date","month","weekOfYear","keyword","phrase"]
    df = df[df[msg_col].notnull()][select_cols].reset_index(drop=True)
    return df


def kw_cleasner(x):
    if not x:
        res = np.nan
    elif type(x) != str:
        res = x
    else:
        dirty_char = ['"','?',',','!']
        for char in tqdm(dirty_char):
            x = x.replace(char, '').strip()
            
        if len(np.unique(list(x)).tolist()) == 1:
            res = np.nan
        else:
            res = x
    return res


def make_stacked(df, kw_col, drop_col, msg_col):
    df_phrase_stacked = df.drop(kw_col, axis=1).join(df[kw_col]
                                               .str
                                               .split(',', expand=True)
                                               .stack()
                                               .reset_index(drop=True, level=1)
                                               .rename(kw_col))
    df_phrase_stacked = df_phrase_stacked.drop(drop_col + [msg_col], axis=1)
    df_phrase_stacked[kw_col] = df_phrase_stacked[kw_col].apply(lambda x: kw_cleasner(x))
    df_phrase_stacked = df_phrase_stacked[
        (df_phrase_stacked[kw_col].notnull()) 
        & (df_phrase_stacked[kw_col] != '')
    ].reset_index(drop=True)
    return df_phrase_stacked


def gen_kw_stats_table(df_phrase_stacked, duration_col, kw_col):
    df_duration = df_phrase_stacked[[duration_col, kw_col]]
    df_duration = df_duration.groupby(
        [duration_col, kw_col]).agg(
            {kw_col: "count"}).rename(
                columns={kw_col: "count"})
    # count
    df_duration_pivot = pd.pivot_table(
        df_duration, values="count", index=[duration_col], 
        columns=[kw_col], fill_value=0.0)
    # percentage
    df_duration_pivot_perc = df_duration_pivot.copy()
    for i in tqdm(range(df_duration_pivot_perc.shape[0])):
        df_duration_pivot_perc.iloc[i] = df_duration_pivot.iloc[i] / df_duration_pivot.iloc[i].sum() * 100
    df_duration_pivot_perc = df_duration_pivot_perc.round(3)
    # percentage change rate
    df_duration_pivot_perc_chgRate = df_duration_pivot_perc.diff()

    loop_set = [
        (df_duration_pivot.reset_index(), "count"), 
        (df_duration_pivot_perc.reset_index(), "percentage"),
        (df_duration_pivot_perc_chgRate.reset_index(), "growth_rate")
    ]
    for _set in loop_set:
        if _set[1] == "count":
            df_duration_melt = _set[0].melt(
                id_vars=[duration_col],
                var_name=kw_col,
                value_name=_set[1]
            )
        else:
            df_duration_melt[_set[1]] = _set[0].melt(
                id_vars=[duration_col], 
                var_name=kw_col,
                value_name=_set[1]
            )[_set[1]]
    df_duration_melt = df_duration_melt[df_duration_melt["count"] > 0].reset_index(drop=True)
    return df_duration_melt


def check(row, time_slot):
    no_show_months = list(row).count(0)
    res = time_slot[no_show_months]
    return res


def gen_kw_stats_only_new(df_phrase_stacked, duration_cols, kw_col):
    new_phrases = dict()
    for i in tqdm(range(len(duration_cols))):
        duration_col = duration_cols[i]
        new_phrases.update({duration_col: dict()})
        df_duration = df_phrase_stacked[[duration_col, kw_col]]
        df_duration = df_duration.groupby([duration_col, kw_col]).agg({kw_col: 'count'}).rename(columns={kw_col: 'count'})
        df_duration_pivot = pd.pivot_table(df_duration, values='count', index=[duration_col], columns=[kw_col], fill_value=0.0)
        df_duration_pivot_cumsum = df_duration_pivot.T.cumsum(axis=1)

        time_slot = df_duration_pivot_cumsum.columns.tolist()
        df_duration_pivot_cumsum['first_show'] = df_duration_pivot_cumsum.apply(lambda row: check(row, time_slot), axis=1)
        df_duration_pivot_cumsum = df_duration_pivot_cumsum.reset_index()

        for slot in tqdm(time_slot):
            new_phrases[duration_col].update({str(slot): list()})
            df_slot = df_duration_pivot_cumsum[df_duration_pivot_cumsum['first_show'] == slot]
            df_slot = df_slot[[kw_col, slot]].reset_index(drop=True)
            df_slot['pct'] = round(df_slot[slot] / df_slot[slot].sum(), 3)

            for row_id in range(df_slot.shape[0]):
                kw = df_slot['phrase'][row_id]
                val_count = int(df_slot[slot][row_id])
                val_pct = float(df_slot['pct'][row_id])
                new_phrases[duration_col][str(slot)].append([kw, val_count, val_pct])

    all_temp = dict()
    for key_col in new_phrases.keys():
        for time in tqdm(new_phrases[key_col].keys()):
            tmp = pd.DataFrame(new_phrases[key_col][time], columns=['關鍵詞','實際次數','佔比'])
            tmp[key_col] = time
            if time == list(new_phrases[key_col].keys())[0]:
                df_new = tmp.copy()
                all_temp.update({key_col: df_new})
            else:
                all_temp[key_col] = pd.concat([all_temp[key_col], tmp])
    return all_temp


def collect_kw_corpus(df, kw_col):
    afa_kw_corpus = list()
    for row in tqdm(df[kw_col].values.tolist()):
        try: 
            res = [item.strip() for item in row.split(',') if (item.strip() != '') and (len(item.strip()) > 1)]
            afa_kw_corpus.append(res)
        except:
            pass
    return afa_kw_corpus


def count_concurrency(afa_kw_corpus):
    """
    關鍵詞共現統計
    """
    afa_kw_corpus_expand = list()
    for row in tqdm(afa_kw_corpus):
        for item in row:
            afa_kw_corpus_expand.append(item)
    afa_kw_corpus_unique = np.unique(afa_kw_corpus_expand)

    kw_concurrency = dict()
    for kw in tqdm(afa_kw_corpus_unique):
        concurrent_words = list()
        for row in afa_kw_corpus:
            if kw in row:
                concurrent_words += row
            else:
                pass
            
        concurrent_words = [word for word in concurrent_words if word != kw]
        concurrent_words = dict(Counter(concurrent_words))
        kw_concurrency.update({kw: concurrent_words})
    
    afa_kw_corpus_unique_counts = dict(Counter(afa_kw_corpus_expand))
    afa_kw_corpus_unique_counts = dict(sorted(
        afa_kw_corpus_unique_counts.items(), 
        key=lambda k: k[1], reverse=True))

    kw_concurrency_sorted = dict()
    for kw in tqdm(afa_kw_corpus_unique_counts.keys()):
        sorted_temp_dict = dict(sorted(
            kw_concurrency[kw].items(), 
            key=lambda k: k[1], reverse=True))
        kw_concurrency_sorted.update({kw: sorted_temp_dict})
    return kw_concurrency_sorted

