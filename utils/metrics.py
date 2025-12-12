import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    mask = true != 0
    return np.mean(np.abs((pred[mask] - true[mask]) / true[mask])) * 100

def MSPE(pred, true):
    mask = true != 0
    return np.mean(np.square((pred[mask] - true[mask]) / true[mask]))

def evaluate_all(true, pred):
    """
    接口适配 exp_stanhop_fiats.py 的字典输出格式
    """
    return {
        "MAE": MAE(pred, true),
        "MSE": MSE(pred, true),
        "RMSE": RMSE(pred, true),
        "MAPE": MAPE(pred, true),
        "R2": r2_score(true, pred)
    }
