import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_FILE = os.path.join(BASE_DIR, "data/raw_time_series_data/weather_load_2019-2022.csv")

def denormalize(state, sources, result_path):
    """
    对预测结果进行反归一化
    :param state: 州名 (NSW, QLD, SA, TAS, VIC)
    :param sources: 外部信息源组合字符串（如 "1", "12", "0"）
    :param result_path: 预测结果文件路径
    """
    raw_df = pd.read_csv(RAW_FILE)
    raw_df["SETTLEMENTDATE"] = pd.to_datetime(raw_df["SETTLEMENTDATE"])

    # 获取原始数据
    state_df = raw_df[raw_df['State'] == state].copy()
    state_df = state_df.sort_values("SETTLEMENTDATE").reset_index(drop=True)
    state_df = state_df[state_df["SETTLEMENTDATE"].dt.year == 2021]

    # 取 TOTALDEMAND 列的 min/max
    target_col = state_df.select_dtypes(include='number').columns[0]
    min_val = state_df[target_col].min()
    max_val = state_df[target_col].max()

    # 读取预测结果
    result_df = pd.read_excel(result_path)
    pred_cols = [c for c in result_df.columns if c.startswith("pred_t")]
    preds = result_df[pred_cols].values
    preds = preds * (max_val - min_val) + min_val  # 反归一化

    # 构造时间戳
    start_time = pd.Timestamp("2021-01-01 00:00:00")
    timestamps = [start_time + pd.Timedelta(minutes=30 * i)
                  for i in range(preds.shape[0] * preds.shape[1])]

    preds_flat = preds.flatten()
    true_map = state_df.set_index("SETTLEMENTDATE")[target_col].to_dict()
    trues_flat = [true_map.get(ts, None) for ts in timestamps]

    rows = []
    for ts, true_val, pred_val in zip(timestamps, trues_flat, preds_flat):
        if ts.year == 2021:
            rows.append({
                "State": state,
                "Date": ts,
                "TRUE": true_val,
                "PRED": pred_val
            })

    denorm_df = pd.DataFrame(rows)
    denorm_df = denorm_df.groupby(["State", "Date"], as_index=False).mean()
    denorm_df = denorm_df.sort_values("Date")

    # 保存到相同 sources 目录
    save_path = os.path.join(BASE_DIR, f"output/{sources}/denorm_results_{state}_2021.csv")
    denorm_df.to_csv(save_path, index=False)
    print(f"✅ [{state}] 反归一化结果已保存到 {save_path}")

def denorm_results(state, sources):
    """主调用入口"""
    result_path = os.path.join(BASE_DIR, f"output/{sources}/results_{state}_2021.xlsx")
    if not os.path.exists(result_path):
        print(f"❌ 文件不存在：{result_path}")
        return
    denormalize(state, sources, result_path)
