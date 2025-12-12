import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from sentence_transformers import SentenceTransformer

# 设置路径
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOCIAL_FILE = "data/raw_social_data/reddit_data.xlsx"
OUTPUT_DIR = "data/processed"
STATES = ["NSW", "QLD", "SA", "TAS", "VIC"]

KEYWORDS = [
    "electricity", "power", "load", "energy", "generation", "consumption", "demand", "supply",
    "load shedding", "blackout", "outage", "peak load", "grid", "distribution", "transmission",
    "renewable", "solar", "wind", "hydro", "battery", "storage", "green energy",
    "Australia", "Australian", "New South Wales", "NSW", "QLD", "Queensland", "SA",
    "South Australia", "TAS", "Tasmania", "VIC", "Victoria"
]

EMBED_MODEL = "all-MiniLM-L6-v2"

# 每个州的关键词正则（忽略大小写）
STATE_REGEX = {
    "NSW": re.compile(r"\b(NSW|New South Wales)\b", re.IGNORECASE),
    "QLD": re.compile(r"\b(QLD|Queensland)\b", re.IGNORECASE),
    "SA":  re.compile(r"\b(SA|South Australia)\b", re.IGNORECASE),
    "TAS": re.compile(r"\b(TAS|Tasmania)\b", re.IGNORECASE),
    "VIC": re.compile(r"\b(VIC|Victoria)\b", re.IGNORECASE),
}

def load_social_data():
    df = pd.read_excel(SOCIAL_FILE)
    df = df.rename(columns={
        "时间": "Date",
        "内容": "Text",
        "搜索关键词": "Keyword",
        "搜索社区（用户）": "Community"
    })
    df = df.dropna(subset=["Date", "Text"])
    df["Date"] = pd.to_datetime(df["Date"], format="%y.%m.%d", errors="coerce").dt.date
    df = df.dropna(subset=["Date"])
    return df

def assign_states(row):
    """判断一条帖子适用于哪些州"""
    text = row["Text"]
    matched_states = []
    for state, pattern in STATE_REGEX.items():
        if pattern.search(text):
            matched_states.append(state)
    if not matched_states:
        # 无州名 → 全国通用
        matched_states = STATES
    return matched_states

def filter_by_keywords(texts, keywords):
    filtered = []
    for t in texts:
        text = t.lower()
        if any(k.lower() in text for k in keywords):
            filtered.append(t)
    return filtered

def group_and_embed(state_texts_dict, model):
    """按天嵌入"""
    date_to_vec = {}
    grouped = {}

    for date, text in state_texts_dict:
        if date not in grouped:
            grouped[date] = []
        grouped[date].append(text)

    for date, texts in tqdm(grouped.items(), desc="嵌入中"):
        filtered = filter_by_keywords(texts, KEYWORDS)
        if not filtered:
            continue
        embeddings = model.encode(filtered)
        daily_vec = np.mean(embeddings, axis=0)
        date_to_vec[date] = daily_vec

    return date_to_vec

def embed_and_save_social():
    df = load_social_data()

    # 每个州的帖子列表：[(date, text), ...]
    state_to_posts = {s: [] for s in STATES}

    for _, row in df.iterrows():
        date = row["Date"]
        text = row["Text"]
        applicable_states = assign_states(row)
        for state in applicable_states:
            state_to_posts[state].append((date, text))

    model = SentenceTransformer(EMBED_MODEL)

    for state in STATES:
        posts = state_to_posts[state]
        if not posts:
            print(f"⚠️ 跳过 {state}（无数据）")
            continue
        vecs = group_and_embed(posts, model)
        all_dates = sorted(vecs.keys())
        embed_matrix = np.array([vecs[d] for d in all_dates])
        np.save(os.path.join(OUTPUT_DIR, f"embed_reddit_{state}.npy"), embed_matrix)
        print(f"[{state}] 嵌入 shape = {embed_matrix.shape}")

if __name__ == "__main__":
    embed_and_save_social()
