import os
import re
import numpy as np
from tqdm import tqdm
from datetime import datetime
from sentence_transformers import SentenceTransformer

# 设置路径
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_NEWS_DIR = "data/raw_news_data"
OUTPUT_DIR = "data/processed"
STATES = ["NSW", "QLD", "SA", "TAS", "VIC"]

# 关键词（不区分大小写）
KEYWORDS = [
    # 电力系统相关
    "electricity", "power", "load", "energy", "generation", "consumption", "demand", "supply",
    "load shedding", "blackout", "outage", "peak load", "grid", "distribution", "transmission",

    # 可再生能源
    "renewable", "solar", "wind", "hydro", "battery", "storage", "green energy",

    # 与澳大利亚地区相关
    "Australia", "Australian", "New South Wales", "NSW", "QLD", "Queensland", "SA",
    "South Australia", "TAS", "Tasmania", "VIC", "Victoria"
]

# 嵌入模型
EMBED_MODEL = "all-MiniLM-L6-v2"

# 提取字段工具函数
def extract_field(pattern, text):
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""

# 加载新闻（正则方式解析字段）
def load_news():
    news_data = []
    block_pattern = re.compile(r'\{.*?\}', re.DOTALL)

    for fname in os.listdir(RAW_NEWS_DIR):
        if not fname.endswith(".txt"):
            continue
        file_path = os.path.join(RAW_NEWS_DIR, fname)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                matches = block_pattern.findall(content)
                print(f"[{fname}] 提取 {len(matches)} 条候选新闻")

                for i, block in enumerate(matches):
                    news = {
                        "title": extract_field(r'"title"\s*:\s*"([^"]+)"', block),
                        "category": extract_field(r'"category"\s*:\s*"([^"]+)"', block),
                        "summary": extract_field(r'"summary"\s*:\s*"([^"]+)"', block),
                        "link": extract_field(r'"link"\s*:\s*"([^"]+)"', block),
                        "publication_time": extract_field(r'"publication_time"\s*:\s*"([^"]+)"', block),
                        "full_article": extract_field(r'"full_article"\s*:\s*"([^"]+)"', block)
                    }
                    if news["title"] and news["publication_time"]:
                        news_data.append(news)
                    elif i < 3:
                        print(f"⚠️ 第{i+1}条提取失败")

        except Exception as e:
            print(f"❌ 读取失败: {fname}, 错误: {e}")
    print(f"✅ 共加载新闻 {len(news_data)} 条")
    return news_data

# 关键词筛选
def filter_by_keywords(news_list, keywords):
    filtered = []
    for item in news_list:
        text = (item.get("title", "") + " " + item.get("full_article", "")).lower()
        if any(k.lower() in text for k in keywords):
            filtered.append(item)
    print(f"关键词筛选后剩余 {len(filtered)} 条新闻")
    return filtered

# 按日期聚合
def group_by_date(news_list):
    daily_dict = {}
    for item in news_list:
        time_str = item.get("publication_time")
        if not time_str:
            continue
        try:
            date = datetime.fromisoformat(time_str).date()
        except ValueError:
            continue
        text = item.get("title", "") + " " + item.get("summary", "")
        if date not in daily_dict:
            daily_dict[date] = []
        daily_dict[date].append(text)
    return daily_dict

# 嵌入并保存
def embed_and_save_news():
    news_raw = load_news()
    news_filtered = filter_by_keywords(news_raw, KEYWORDS)
    daily_news = group_by_date(news_filtered)

    model = SentenceTransformer(EMBED_MODEL)
    date_to_vec = {}

    for date, texts in tqdm(daily_news.items(), desc="正在生成嵌入"):
        if not texts:
            continue
        embeddings = model.encode(texts)
        daily_vec = np.mean(embeddings, axis=0)  # 每天一个向量
        date_to_vec[date] = daily_vec

    # 为每个州生成一个向量序列
    for state in STATES:
        all_dates = sorted(date_to_vec.keys())
        embed_matrix = np.array([date_to_vec[date] for date in all_dates])
        np.save(os.path.join(OUTPUT_DIR, f"embed_news_{state}.npy"), embed_matrix)
        print(f"[{state}] 嵌入向量 shape: {embed_matrix.shape}")

# 主函数
if __name__ == "__main__":
    embed_and_save_news()
