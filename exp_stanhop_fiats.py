import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.optim import Adam

from code_models.stanhop_fiats import STanHopFIATS
from utils.data_loader import get_dataloader
from utils.metrics import evaluate_all
from utils.denorm import denorm_results  # ✅ 自动反归一化

def train_and_predict(state="NSW",
                      in_len=336,
                      out_len=48,
                      batch_size=32,
                      epochs=20,
                      lr=1e-4,
                      device='cuda',
                      sources="1"):  # "0"=无外部源；否则如 "1" / "12" / "123"
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"\n================= 开始训练 {state} | 干预源: {sources if sources!='0' else '无'} =================")
    print(f"当前使用设备：{device}")
    print(f"CUDA 是否可用：{torch.cuda.is_available()}")
    print(f"当前 CUDA 设备：{torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无'}")

    # ===== 1. 加载数据 =====
    train_loader = get_dataloader(state, in_len, out_len, batch_size, split="train", sources=sources)
    test_loader  = get_dataloader(state, in_len, out_len, batch_size, split="test",  sources=sources)

    # ===== 2. 提取输入维度 =====
    sample_batch = next(iter(train_loader))
    x_sample = sample_batch["x"]
    data_dim = x_sample.shape[-1]

    # ===== 3. 外部干预源维度（仅在 sources != "0" 时计算） =====
    inter_dims = {}
    if sources != "0" and "news" in sample_batch:
        total_dim = int(sample_batch["news"].shape[-1])
        num_src = len(sources)
        if total_dim % num_src != 0:
            raise ValueError(f"外部嵌入维度 {total_dim} 不能被 sources 数 {num_src} 整除。")
        dim_per_source = total_dim // num_src
        for i, code in enumerate(sources):
            name = {"1": "news", "2": "reddit", "3": "policy"}[code]
            inter_dims[name] = dim_per_source

    # ===== 4. 初始化模型 =====
    model = STanHopFIATS(
        data_dim=data_dim,
        in_len=in_len,
        out_len=out_len,
        seg_len=12,
        inter_dims=inter_dims,          # 空字典 = 无外部源
        device=device,
        d_model=512,
        d_ff=1024,
        n_heads=8,
        e_layers=3,
        dropout=0.1,
        fusion_mode="crossattn"
    ).to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # ===== 5. 训练 =====
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{epochs}]"):
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            # 仅在 sources != "0" 时组装外部输入
            inter_inputs = {}
            if sources != "0" and "news" in batch:
                total_embed = batch["news"].to(device)  # 已拼接所有源
                dim_per_source = total_embed.shape[-1] // len(sources)
                for i, code in enumerate(sources):
                    name = {"1": "news", "2": "reddit", "3": "policy"}[code]
                    inter_inputs[name] = total_embed[:, i*dim_per_source:(i+1)*dim_per_source]

            pred = model(x, **inter_inputs) if inter_inputs else model(x)

            if torch.isnan(pred).any() or torch.isinf(pred).any():
                print("❌ 模型输出含 NaN/Inf，跳过该 batch")
                continue

            loss = loss_fn(pred, y)
            if torch.isnan(loss) or torch.isinf(loss):
                print("❌ Loss 为 NaN/Inf，跳过该 batch")
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        mean_loss = float(np.mean(losses)) if len(losses) > 0 else float("nan")
        print(f"Epoch {epoch+1}: Train MSE = {mean_loss:.6f}")

    # ===== 6. 测试 =====
    print(f"\n================= 📊 测试集评估 {state} =================")
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            inter_inputs = {}
            if sources != "0" and "news" in batch:
                total_embed = batch["news"].to(device)
                dim_per_source = total_embed.shape[-1] // len(sources)
                for i, code in enumerate(sources):
                    name = {"1": "news", "2": "reddit", "3": "policy"}[code]
                    inter_inputs[name] = total_embed[:, i*dim_per_source:(i+1)*dim_per_source]

            pred = model(x, **inter_inputs) if inter_inputs else model(x)
            all_preds.append(pred.cpu().numpy())
            all_trues.append(y.cpu().numpy())

    pred_arr = np.concatenate(all_preds, axis=0)  # [N, T_out, D]
    true_arr = np.concatenate(all_trues, axis=0)  # [N, T_out, D]

    # 仅对第一个目标维度（如 TOTALDEMAND）做评估
    pred_1d = pred_arr[:, :, 0].flatten()
    true_1d = true_arr[:, :, 0].flatten()
    metrics = evaluate_all(true_1d, pred_1d)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # ===== 7. 保存预测结果 =====
    save_path = save_predictions_to_excel(pred_arr, true_arr, state, sources)

    # ===== 8. 自动反归一化（与保存路径相同目录）=====
    try:
        denorm_results(state=state, sources=(sources if sources else "0"))
    except Exception as e:
        print(f"⚠️ 自动反归一化失败：{e}")


def save_predictions_to_excel(pred_arr, true_arr, state="NSW", sources="1"):
    # 当 sources="0" 时，输出目录为 output/0
    folder = sources if sources and sources != "0" else "0"
    save_dir = os.path.join("output", folder)
    os.makedirs(save_dir, exist_ok=True)

    df_pred = pd.DataFrame(pred_arr[:, :, 0])
    df_true = pd.DataFrame(true_arr[:, :, 0])
    df_pred.columns = [f"pred_t{i}" for i in range(df_pred.shape[1])]
    df_true.columns = [f"true_t{i}" for i in range(df_true.shape[1])]
    df = pd.concat([df_pred, df_true], axis=1)

    save_path = os.path.join(save_dir, f"results_{state}_2021.xlsx")
    df.to_excel(save_path, index=False)
    print(f"\n✅ 预测结果已保存：{save_path}")
    return save_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=str, default="NSW", help="州名，例如NSW、QLD等")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sources", type=str, default="1", help="使用哪些干预信息源，0=无外部源；1=新闻；12=新闻+社交；123=新闻+社交+政策")

    args = parser.parse_args()

    train_and_predict(
        state=args.state,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        sources=args.sources
    )
