import numpy as np
import torch


# def RSE(pred, true):
#     return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


# def CORR(pred, true):
#     u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
#     d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
#     return (u / d).mean(-1)


# def MAE(pred, true):
#     return np.mean(np.abs(pred - true))


# def MSE(pred, true):
#     return np.mean((pred - true) ** 2)


# def RMSE(pred, true):
#     return np.sqrt(MSE(pred, true))


# def MAPE(pred, true):
#     return np.mean(np.abs((pred - true) / true))


# def MSPE(pred, true):
#     return np.mean(np.square((pred - true) / true))


# def metric(pred, true):
#     mae = MAE(pred, true)
#     mse = MSE(pred, true)
#     rmse = RMSE(pred, true)
#     mape = MAPE(pred, true)
#     mspe = MSPE(pred, true)

#     return mae, mse, rmse, mape, mspe

def RSE(pred, true, mask):     
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))



def CORR(pred, true, mask):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)



def MAE(pred, true, mask):
    if len(mask) == 0:
        return np.mean(np.abs(pred - true))
    else:
        return np.mean(np.abs(pred - true) * mask)


def MSE(pred, true, mask):
    if len(mask) == 0:
        return np.mean((pred - true) ** 2)
    else:
        return np.mean((pred - true) ** 2 * mask)


def RMSE(pred, true, mask):
    return np.sqrt(MSE(pred, true, mask))


def MAPE(pred, true, mask):
    if len(mask) == 0:
        return np.mean(np.abs((pred - true) / true))
    else:
        return np.mean(np.abs((pred - true) / true) * mask)


def MSPE(pred, true, mask):
    if len(mask) == 0:
        return np.mean(np.square((pred - true) / true))
    else:
        return np.mean(np.square((pred - true) / true) * mask) 

def metric(pred, true, mask=[]):
    mae = MAE(pred, true, mask)
    mse = MSE(pred, true, mask)
    rmse = RMSE(pred, true, mask)
    mape = MAPE(pred, true, mask)
    mspe = MSPE(pred, true, mask)

    return mae, mse, rmse, mape, mspe

# Mega-CRN
# DCRNN
def masked_mae_loss(y_pred, y_true, mask=[]):
    if len(mask) == 0:
        mask = (y_true != 0).float()
    # mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()

def masked_mape_loss(y_pred, y_true, mask=[]):
    if len(mask) == 0:
        mask = (y_true != 0).float()
    # mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(torch.div(y_true - y_pred, y_true))
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()

def masked_rmse_loss(y_pred, y_true, mask=[]):
    if len(mask) == 0:
        mask = (y_true != 0).float()
    # mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.pow(y_true - y_pred, 2)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return torch.sqrt(loss.mean())

def masked_mse_loss(y_pred, y_true, mask=[]):
    if len(mask) == 0:
        mask = (y_true != 0).float()
    # mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.pow(y_true - y_pred, 2)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()






