import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

# 设置路径
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_FILE = "data/raw_time_series_data/weather_load_2019-2022.csv"
OUTPUT_DIR = "data/processed"
TARGET_STATES = ["NSW", "QLD", "SA", "TAS", "VIC"]

def process_time_series():
    # 读取原始数据
    df = pd.read_csv(RAW_FILE)
    print(f"原始数据共 {len(df)} 条")

    # 统一格式
    df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])
    df = df.sort_values(by=['State', 'SETTLEMENTDATE'])

    # 对每个州分别处理
    for state in TARGET_STATES:
        state_df = df[df['State'] == state].copy()
        state_df = state_df.reset_index(drop=True)

        # 填补缺失值：列均值填充
        state_df = state_df.fillna(state_df.mean(numeric_only=True))

        # 提取时间戳列备用
        timestamps = state_df[['SETTLEMENTDATE']]

        # 去除非数值列（State, SETTLEMENTDATE）
        numeric_cols = state_df.select_dtypes(include='number').columns
        state_numeric = state_df[numeric_cols]

        # 归一化（0-1）
        scaler = MinMaxScaler()
        state_scaled = scaler.fit_transform(state_numeric)
        state_scaled_df = pd.DataFrame(state_scaled, columns=numeric_cols)
        state_scaled_df.insert(0, 'SETTLEMENTDATE', timestamps.values)

        # 保存结果
        output_path = os.path.join(OUTPUT_DIR, f"load_{state}.csv")
        state_scaled_df.to_csv(output_path, index=False)
        print(f"[{state}] 保存归一化数据到 {output_path}")

if __name__ == "__main__":
    process_time_series()
