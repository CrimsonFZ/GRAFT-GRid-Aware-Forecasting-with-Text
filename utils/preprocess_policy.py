import pandas as pd
import numpy as np
import os
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta

# ========== 参数 ==========
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATE_LIST = ["NSW", "QLD", "SA", "TAS", "VIC"]
DATA_PATH = "data/raw_policy_data/policy_data.xlsx"
SAVE_PATH = "data/processed"
MODEL_NAME = "all-MiniLM-L6-v2"

# ========== 初始化模型 ==========
model = SentenceTransformer(MODEL_NAME)

# ========== 加载数据 ==========
df = pd.read_excel(DATA_PATH)
df = df.rename(columns={"时间": "date", "文本": "content", "标题": "title"})

# 统一日期格式
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date", "content"])  # 删除无效行
df["date"] = df["date"].dt.strftime("%Y-%m-%d")  # 格式化为字符串

# ========== 提取州名 ==========
def extract_states(text):
    found = []
    for state in STATE_LIST:
        if re.search(rf"\b{state}\b", str(text), re.IGNORECASE):
            found.append(state)
    return found if found else ["ALL"]

df["states"] = df["content"].apply(extract_states)

# 合并标题和正文（可选）
df["full_text"] = df["title"].fillna("") + " " + df["content"]

# ========== 编码所有文本 ==========
print("正在编码政策文本 ...")
embeddings = model.encode(df["full_text"].tolist(), show_progress_bar=True)
df["embedding"] = list(embeddings)

# ========== 聚合嵌入（按州和日期） ==========
print("正在聚合每州每日嵌入 ...")

all_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
all_dates_str = all_dates.strftime("%Y-%m-%d").tolist()
state_date_embed = {state: {d: [] for d in all_dates_str} for state in STATE_LIST}

for _, row in df.iterrows():
    for state in row["states"]:
        targets = STATE_LIST if state == "ALL" else [state]
        for s in targets:
            state_date_embed[s][row["date"]].append(row["embedding"])

# 平均化 & 填补
for state in STATE_LIST:
    print(f"处理 {state} 州")
    final_embeds = []
    for d in all_dates_str:
        vecs = state_date_embed[state][d]
        if vecs:
            avg_vec = np.mean(vecs, axis=0)
        else:
            avg_vec = np.zeros(384)
        final_embeds.append(avg_vec)

    final_arr = np.stack(final_embeds)
    np.save(os.path.join(SAVE_PATH, f"embed_policy_{state}.npy"), final_arr)

print("✅ 所有政策嵌入保存完毕")
