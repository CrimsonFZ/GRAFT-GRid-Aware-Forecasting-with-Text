import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class ElectricityNewsDataset(Dataset):
    def __init__(self, state, in_len=336, out_len=48, split="train", sources="1"):
        """
        sources:
            "0" 或 ""  -> 不使用任何外部源（不返回 'news' 键）
            由 1/2/3 组成的字符串（如 "1", "12", "123"）
              1 = 新闻, 2 = Reddit, 3 = 政策
        """
        self.state = state
        self.in_len = in_len
        self.out_len = out_len
        self.split = split
        self.sources = (sources or "0").strip()

        # 基础路径
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # 1) 负荷+天气数据
        load_path = os.path.join(base_dir, f"data/processed/load_{state}.csv")
        df = pd.read_csv(load_path)
        df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"])
        df = df.sort_values("SETTLEMENTDATE")
        self.timestamps = df["SETTLEMENTDATE"].values
        self.data = df.drop(columns=["SETTLEMENTDATE"]).values.astype(np.float32)

        # 2) 外部嵌入（只有在 sources 不是 "0" 时才尝试加载）
        self.news_embeddings = None
        self.reddit_embeddings = None
        self.policy_embeddings = None

        if self.sources != "0":
            if "1" in self.sources:
                news_path = os.path.join(base_dir, f"data/processed/embed_news_{state}.npy")
                if os.path.exists(news_path):
                    self.news_embeddings = np.load(news_path).astype(np.float32)

            if "2" in self.sources:
                reddit_path = os.path.join(base_dir, f"data/processed/embed_reddit_{state}.npy")
                if os.path.exists(reddit_path):
                    self.reddit_embeddings = np.load(reddit_path).astype(np.float32)

            if "3" in self.sources:
                policy_path = os.path.join(base_dir, f"data/processed/embed_policy_{state}.npy")
                if os.path.exists(policy_path):
                    self.policy_embeddings = np.load(policy_path).astype(np.float32)

        # 3) 每日起始索引（每 48 个点为一天）
        self.daily_indices = [i for i in range(0, len(self.data), 48)]
        self.daily_map = {i: idx for idx, i in enumerate(self.daily_indices)}

        # 4) 构造训练/测试滑窗索引
        train_end_date = pd.to_datetime("2021-01-01")
        self.indices = []
        max_idx = len(self.data) - self.in_len - self.out_len
        for i in range(max_idx):
            cur_time = pd.to_datetime(self.timestamps[i + self.in_len - 1])
            if self.split == "train" and cur_time < train_end_date:
                self.indices.append(i)
            elif self.split == "test" and cur_time >= train_end_date:
                self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x_seq = self.data[i:i + self.in_len]
        y_seq = self.data[i + self.in_len:i + self.in_len + self.out_len]

        batch = {
            "x": torch.tensor(x_seq, dtype=torch.float32),
            "y": torch.tensor(y_seq, dtype=torch.float32),
        }

        # 如果不使用任何外部源，直接返回
        if self.sources == "0":
            return batch

        # 计算当前样本属于哪一天
        ref_time = pd.Timestamp(self.timestamps[i + self.in_len - 1])
        ref_day_idx = ref_time.hour * 2 + ref_time.minute // 30  # 当天的第几个 30min
        day_start = i + self.in_len - 1 - ref_day_idx
        day_idx = self.daily_map.get(day_start, 0)

        vectors = []
        # 只拼接“实际选择且存在文件”的源，不再用零向量占位
        if "1" in self.sources and self.news_embeddings is not None and len(self.news_embeddings) > 0:
            vectors.append(self.news_embeddings[min(day_idx, len(self.news_embeddings) - 1)])

        if "2" in self.sources and self.reddit_embeddings is not None and len(self.reddit_embeddings) > 0:
            vectors.append(self.reddit_embeddings[min(day_idx, len(self.reddit_embeddings) - 1)])

        if "3" in self.sources and self.policy_embeddings is not None and len(self.policy_embeddings) > 0:
            vectors.append(self.policy_embeddings[min(day_idx, len(self.policy_embeddings) - 1)])

        # 至少有一个外部源才返回 'news' 键
        if vectors:
            inter_vec = np.concatenate(vectors, axis=-1).astype(np.float32)
            batch["news"] = torch.tensor(inter_vec, dtype=torch.float32)

        return batch


def get_dataloader(state, in_len=336, out_len=48, batch_size=64, split="train", sources="1"):
    dataset = ElectricityNewsDataset(state, in_len=in_len, out_len=out_len, split=split, sources=sources)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))
