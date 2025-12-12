# make_source_time_heatmap.py
# -*- coding: utf-8 -*-
"""
生成“时间—来源”归因热图，并导出 γ_t = softmax(score_news, score_reddit, score_policy)
- 读取: embed_news_{STATE}.npy / embed_reddit_{STATE}.npy / embed_policy_{STATE}.npy
        load_{STATE}.csv（30min 分辨率负荷）
- 对齐: 以负荷的自然日索引为主索引，三类文本为日级向量（已按报告聚合/衰减）
- 训练: 2019-01-01~2020-09-30 为训练，2020-10-01~2020-12-31 验证（仅做监控），
        2021-01-01~2021-12-31 亦计算并输出热图（可自行调参）。
- 输出: ./heatmaps/heatmap_{STATE}.png 以及 ./heatmaps/gamma_{STATE}.csv
"""

from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# ==== 新增：莫兰迪配色 ====
from matplotlib.colors import LinearSegmentedColormap, Normalize

# Morandi（莫兰迪）低饱和顺序渐变：浅暖灰 → 砂石色 → 灰绿 → 石板绿
MORANDI_HEX = [
    "#F2F1ED",  # very light warm gray
    "#E5DFD3",  # sand
    "#D4C9BA",  # beige/khaki
    "#C3BDB1",  # greige
    "#B5C1BC",  # desaturated green-gray
    "#9FB1A9",  # sage green
    "#8A9B94",  # muted green-gray
    "#788A86",  # slate green
    "#677579"   # slate/stone
]
cmap_morandi = LinearSegmentedColormap.from_list("morandi", MORANDI_HEX)

STATES = ["NSW", "QLD", "SA", "TAS", "VIC"]

# --------- 配置可调 ----------
TRAIN_END = pd.Timestamp("2020-09-30")
VAL_END   = pd.Timestamp("2020-12-31")  # 验证区间：2020-10~2020-12
TEST_END  = pd.Timestamp("2021-12-31")  # 测试/展示区间
LOOKBACK_DAYS = 7                       # 用过去 7 天负荷构造查询特征
HEATMAP_DIR = "heatmaps"
# -----------------------------

def _find_dt_col(df: pd.DataFrame) -> str:
    """尽量鲁棒地识别时间列（AEMO常见: 'SETTLEMENTDATE' 或 'timestamp'）。"""
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            return c
    # 尝试把首列解析为时间
    for c in df.columns:
        try:
            df[c] = pd.to_datetime(df[c], utc=False, errors="raise")
            return c
        except Exception:
            continue
    raise ValueError("无法识别时间列，请检查 load_{STATE}.csv 的表头。")

def _find_load_col(df: pd.DataFrame) -> str:
    candidates = ["TOTALDEMAND","load","Load","demand","Demand","MW","value"]
    for c in candidates:
        if c in df.columns:
            return c
    # 若只有两列，默认第二列为负荷
    if df.shape[1] == 2:
        return df.columns[1]
    raise ValueError("无法识别负荷列，请检查 load_{STATE}.csv 的表头。")

def load_halfhour_load(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path)
    dt_col = _find_dt_col(df)
    load_col = _find_load_col(df)
    df = df[[dt_col, load_col]].dropna()
    df[dt_col] = pd.to_datetime(df[dt_col], utc=False)
    df = df.sort_values(dt_col)
    # 统一到 30 分钟频率（若有重复/缺失做轻量修补）
    s = df.set_index(dt_col)[load_col].asfreq("30T")
    s = s.interpolate(limit_direction="both")
    return s.rename("load")

def daily_index_from_load(half_hourly: pd.Series) -> pd.DatetimeIndex:
    """自然日索引（当地市场时间），保证每天 48 点。"""
    return half_hourly.resample("1D").max().index

def daily_curve(half_hourly: pd.Series) -> pd.DataFrame:
    """把 30min 曲线按‘天 × 48’展开，用于构造查询特征。"""
    df = half_hourly.to_frame()
    df["date"] = df.index.normalize()
    df["slot"] = ((df.index - df["date"]) / pd.Timedelta(minutes=30)).astype(int)
    pivot = df.pivot_table(index="date", columns="slot", values="load")
    # 轻量填补
    pivot = pivot.fillna(method="ffill").fillna(method="bfill")
    return pivot  # shape: [n_day, 48]

def build_query_feature(daily_48: pd.DataFrame, lookback=7) -> pd.DataFrame:
    """
    以过去 lookback 天的 48 点曲线形成查询特征 R̄_t：
      - 取过去 k 天的 48 点曲线均值与标准差
      - 拼接 [mean(48) || std(48)] 得到 96 维查询
    """
    mean48 = daily_48.rolling(lookback, min_periods=1).mean().shift(1)
    std48  = daily_48.rolling(lookback, min_periods=1).std(ddof=0).shift(1)
    feat = pd.concat([mean48.add_prefix("m_"), std48.add_prefix("s_")], axis=1)
    return feat

def load_text_embeddings(base: Path, state: str):
    """读取三类日级文本向量，返回 dict{source: DataFrame}，按日索引对齐。"""
    news = np.load(base / f"embed_news_{state}.npy", allow_pickle=True)
    rdt  = np.load(base / f"embed_reddit_{state}.npy", allow_pickle=True)
    pol  = np.load(base / f"embed_policy_{state}.npy", allow_pickle=True)
    # 推断天数与维度
    n_day_news, d_news = news.shape
    n_day_rdt,  d_rdt  = rdt.shape
    n_day_pol,  d_pol  = pol.shape
    # 用最短天数对齐，后续与负荷日索引再取交集
    n_day = min(n_day_news, n_day_rdt, n_day_pol)
    news = news[:n_day]; rdt = rdt[:n_day]; pol = pol[:n_day]
    # 临时以连续日序建立索引，稍后替换为负荷的真实自然日索引
    idx = pd.date_range("2019-01-01", periods=n_day, freq="1D")
    df_news = pd.DataFrame(news, index=idx).add_prefix("news_")
    df_rdt  = pd.DataFrame(rdt,  index=idx).add_prefix("rdt_")
    df_pol  = pd.DataFrame(pol,  index=idx).add_prefix("pol_")
    return df_news, df_rdt, df_pol

def softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    x = x / max(1e-8, temp)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)

def train_gate_and_make_gamma(state: str, data_dir: Path):
    # 1) 负荷：半小时→日内矩阵与“事件性”指标
    load_30m = load_halfhour_load(data_dir / f"load_{state}.csv")
    daily48   = daily_curve(load_30m)                       # [n_day, 48]
    R_query   = build_query_feature(daily48, LOOKBACK_DAYS) # [n_day, 96]

    # 事件性（用于学习门控的“监督信号”）：次日峰值相对过去 7 天中位数的偏离
    daily_peak = daily48.max(axis=1)
    ref_med    = daily_peak.rolling(LOOKBACK_DAYS, min_periods=1).median().shift(1)
    eventness  = (daily_peak - ref_med).fillna(0.0)

    # 2) 文本嵌入（已按日报告聚合/衰减），与负荷日索引对齐
    df_news, df_rdt, df_pol = load_text_embeddings(data_dir, state)
    # 用负荷真实日索引裁剪/对齐（以交集为准）
    common_idx = daily48.index.intersection(df_news.index).intersection(df_rdt.index).intersection(df_pol.index)
    R_query   = R_query.loc[common_idx]
    eventness = eventness.loc[common_idx]
    X_news    = df_news.loc[common_idx].values
    X_rdt     = df_rdt.loc[common_idx].values
    X_pol     = df_pol.loc[common_idx].values

    # 3) 分组特征与标准化
    X_concat = np.concatenate([X_news, X_rdt, X_pol], axis=1)
    scaler_x = StandardScaler(with_mean=True, with_std=True)
    X_std = scaler_x.fit_transform(X_concat)

    # 4) 分组岭回归：学习 f_t = w_news^T e_news_t + w_rdt^T e_rdt_t + w_pol^T e_pol_t 近似 eventness_t
    y = eventness.values
    # 时间切分
    tr_mask = common_idx <= TRAIN_END
    va_mask = (common_idx > TRAIN_END) & (common_idx <= VAL_END)

    model = Ridge(alpha=1.0, fit_intercept=True, random_state=0)
    model.fit(X_std[tr_mask], y[tr_mask])

    # 简要评估（可选）
    if va_mask.any():
        y_hat = model.predict(X_std[va_mask])
        r2 = r2_score(y[va_mask], y_hat)
        print(f"[{state}] Validation R^2 = {r2:.3f} on {va_mask.sum()} days")

    # 5) 计算“来源打分”与 γ_t（时间—来源归因）
    d_news = X_news.shape[1]; d_rdt = X_rdt.shape[1]; d_pol = X_pol.shape[1]
    w = model.coef_
    w_news = w[:d_news]
    w_rdt  = w[d_news:d_news+d_rdt]
    w_pol  = w[d_news+d_rdt:d_news+d_rdt+d_pol]

    # 反标准化到原始嵌入空间的线性函数：因为 X_std = (X - μ)/σ
    mu = scaler_x.mean_
    sig= scaler_x.scale_
    # 对每一组单独取出 μ,σ
    mu_news, sig_news = mu[:d_news], sig[:d_news]
    mu_rdt,  sig_rdt  = mu[d_news:d_news+d_rdt],  sig[d_news:d_news+d_rdt]
    mu_pol,  sig_pol  = mu[d_news+d_rdt:d_news+d_rdt+d_pol], sig[d_news+d_rdt:d_news+d_rdt+d_pol]

    # group-wise score: | ( (e - μ)/σ ) · w | ；为避免负号抵消取绝对值
    score_news = np.abs(((X_news - mu_news) / (sig_news + 1e-8)) @ w_news)
    score_rdt  = np.abs(((X_rdt  - mu_rdt ) / (sig_rdt  + 1e-8)) @ w_rdt)
    score_pol  = np.abs(((X_pol  - mu_pol ) / (sig_pol  + 1e-8)) @ w_pol)

    scores = np.stack([score_news, score_rdt, score_pol], axis=1)
    gamma  = softmax(scores, temp=0.5)  # 温度系数可调，越小越稀疏

    # 6) 输出 CSV 与热图（只画 2019~2021 段以便投稿图展示；可改）
    out_dir = Path(HEATMAP_DIR); out_dir.mkdir(exist_ok=True)
    gamma_df = pd.DataFrame(gamma, index=common_idx, columns=["News","Reddit","Policy"])
    gamma_df.loc[:TEST_END].to_csv(out_dir / f"gamma_{state}.csv", float_format="%.6f", encoding="utf-8-sig")

    # 叠一层“事件性强度”作为可选加权，凸显极端日（便于解释）
    evt_z = ((eventness - eventness.loc[:TEST_END].mean()) /
             (eventness.loc[:TEST_END].std(ddof=0) + 1e-8)).clip(-3, 3)
    weight = (evt_z.abs() / evt_z.abs().max()).fillna(0.0).to_numpy()[:, None]
    attrib = gamma * (0.5 + 0.5 * weight)  # 平滑放大极端日

    # 绘图：时间在 x 轴，来源在 y 轴（News/Reddit/Policy）
    show_df = pd.DataFrame(attrib, index=common_idx, columns=["News","Reddit","Policy"]).loc[:TEST_END]
    fig, ax = plt.subplots(figsize=(14, 3.4), dpi=160)

    # ==== 使用莫兰迪渐变，并显式归一化 ====
    vals = show_df.T.values
    norm = Normalize(vmin=float(np.nanmin(vals)), vmax=float(np.nanmax(vals)))
    im = ax.imshow(vals,
                   aspect="auto",
                   origin="lower",
                   extent=[0, show_df.shape[0], 0, 3],
                   cmap=cmap_morandi,
                   norm=norm)

    # x 轴标记为月份
    xticks = np.linspace(0, show_df.shape[0]-1, 12, dtype=int)
    ax.set_xticks(xticks)
    ax.set_xticklabels([show_df.index[i].strftime("%Y-%m") for i in xticks], rotation=45, ha="right")
    ax.set_yticks([0.5,1.5,2.5])
    ax.set_yticklabels(["News","Reddit","Policy"])
    ax.grid(False)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("attribution intensity")
    cbar.outline.set_visible(False)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(9)

    fig.tight_layout()
    fig.savefig(out_dir / f"heatmap_{state}.png", bbox_inches="tight")
    plt.close(fig)

    print(f"[{state}] saved: {out_dir / f'gamma_{state}.csv'} ; {out_dir / f'heatmap_{state}.png'}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=".", help="folder containing embed_* and load_*.csv")
    parser.add_argument("--state", type=str, default="ALL", help="NSW/QLD/SA/TAS/VIC or ALL")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.state.upper() == "ALL":
        for st in STATES:
            train_gate_and_make_gamma(st, data_dir)
    else:
        st = args.state.upper()
        assert st in STATES, f"state must be one of {STATES}"
        train_gate_and_make_gamma(st, data_dir)

if __name__ == "__main__":
    main()
