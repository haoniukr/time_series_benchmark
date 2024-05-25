import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
from lib.models.decode import XcodeYtimeDecoder, MuStdModel
from lib.models.revin import RevIN
from lib.models.metrics import pearsoncor
from lib.utils.checkpoint import EarlyStopping
from lib.utils.env import seed_all_rng

########################################
import time
import pandas as pd
import pickle
import re

class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
    
def inverse_scaler(scaler, y_pred, y_true):
    ns,_,nn = y_pred.shape
    outputs_ori = scaler.inverse_transform(y_pred.reshape(-1,nn)).reshape(ns,-1,nn)
    batch_y_ori = scaler.inverse_transform(y_true.reshape(-1,nn)).reshape(ns,-1,nn)
    return outputs_ori, batch_y_ori




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









def create_time_series_dataloader(datapath="/data", batchsize=8):
    dataloaders = {}
    for split in ["train", "val", "test"]:
        # original time series
        timex_file = os.path.join(datapath, "%s_x_original.npy" % split)

        timex = np.load(timex_file)
        timex = torch.from_numpy(timex).to(dtype=torch.float32)
        timey_file = os.path.join(datapath, "%s_y_original.npy" % split)
        timey = np.load(timey_file)
        timey = torch.from_numpy(timey).to(dtype=torch.float32)
        
        timey_mask_file = os.path.join(datapath, "%s_y_original_mask.npy" % split)
        timey_mask = np.load(timey_mask_file)
        timey_mask = torch.from_numpy(timey_mask).to(dtype=torch.float32)

        # x codes
        codex_file = os.path.join(datapath, "%s_x_codes.npy" % (split))
        codex = np.load(codex_file)
        codex = torch.from_numpy(codex).to(dtype=torch.int64)

        codey_oracle_file = os.path.join(datapath, "%s_y_codes_oracle.npy" % split)
        if not os.path.exists(codey_oracle_file):
            codey_oracle_file = os.path.join(datapath, "%s_y_codes.npy" % split)
        codey_oracle = np.load(codey_oracle_file)
        codey_oracle = torch.from_numpy(codey_oracle).to(dtype=torch.int64)

        print("[Dataset][%s] %d of examples" % (split, timex.shape[0]))

        dataset = torch.utils.data.TensorDataset(timex, timey, codex, codey_oracle, timey_mask)
        dataloaders[split] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batchsize,
            shuffle=True if split == "train" else False,
            num_workers=20,
            drop_last=True if split == "train" else False,
        )

    return dataloaders


def loss_fn(type, beta=1.0):
    if type == "mse":
        loss = nn.MSELoss()
    elif type == "smoothl1":
        loss = nn.SmoothL1Loss(beta=beta)
    else:
        raise ValueError("Invalid type")
    return loss


def train_one_epoch(
    dataloader,
    model_decode,
    model_mustd,
    codebook,
    compression,
    optimizer,
    scheduler,
    epoch,
    device,
    loss_type: str = "smoothl1",
    beta: float = 1.0,
    onehot: bool = False,
    scheme: int = 2,
    scaler=None
):
    running_loss, last_loss = 0.0, 0.0
    running_loss_mu, last_loss_mu = 0.0, 0.0
    log_every = max(len(dataloader) // 3, 3)

    code_dim = codebook.shape[1]

    lossfn = loss_fn(loss_type, beta=beta)
    for i, data in enumerate(dataloader):
        # ----- LOAD DATA ------ #
        x, y, codeids_x, codeids_y_labels, y_mask = data
        # x: (B, Tin, Sin)
        # y: (B, Tout, Sout)
        # codeids_x: (B, Tin / C, Sin)
        # codeids_y_labels: (B, Tout /C, Sout)
        x = x.to(device)
        y = y.to(device)
        codeids_x = codeids_x.to(device)
        codeids_y_labels = codeids_y_labels.to(device)
        y_mask = y_mask.to(device)

        # revin time series
        _ = model_mustd.revin_in(x, "norm")
        norm_y = model_mustd.revin_out(y, "norm")

        # prepare data
        B, TCin, Sin = codeids_x.shape
        B, TCout, Sout = codeids_y_labels.shape
        Tout = TCout * compression
        assert Tout == y.shape[1], "%d" % (TCout)

        # get codewords for input x
        code_ids = codeids_x.flatten()
        if onehot:
            xcodes = F.one_hot(code_ids, num_classes=codebook.shape[0])
        else:
            xcodes = codebook[code_ids]  # (B*Tin*Sin, D)
        xcodes = xcodes.reshape((B, TCin, Sin, xcodes.shape[-1]))  # (B, TCin, Sin, D)
        xcodes = torch.permute(xcodes, (0, 2, 1, 3))  # (B, Sin, TCin, D)
        xcodes = xcodes.reshape((B * Sin, TCin, xcodes.shape[-1]))  # (B * Sin, TCin, D)
        xcodes = torch.permute(xcodes, (1, 0, 2))  # (TCin, B * Sin, D)

        ytime = model_decode(xcodes)  # (B * Sout, Tout)

        # reshapes
        ytime = ytime.reshape((B, Sout, Tout))  # (B, Sout, Tout)
        ytime = torch.permute(ytime, (0, 2, 1))  # (B, Tout, Sout)

        # ----- MEAN & STD ----- #
        times = torch.permute(x, (0, 2, 1))  # (B, Sin, Tin * C)
        times = times.reshape((-1, times.shape[-1]))
        ymeanstd = model_mustd(times)

        # reshapes
        ymeanstd = ymeanstd.reshape((B, Sout, 2))  # (B, S, 2)
        ymeanstd = torch.permute(ymeanstd, (0, 2, 1))  # (B, 2, S)
        ymean = ymeanstd[:, 0, :].unsqueeze(1)  # (B, 1, S)
        ystd = ymeanstd[:, 1, :].unsqueeze(1)  # (B, 1, S)

        if scheme == 1:
            # losses
            loss_mu = lossfn(
                model_mustd.revin_out.mean - model_mustd.revin_in.mean, ymean
            )
            loss_std = lossfn(
                model_mustd.revin_out.stdev - model_mustd.revin_in.stdev, ystd
            )
            loss_decode = lossfn(ytime, norm_y)
            loss_all = lossfn(
                ytime * (ystd.detach() + model_mustd.revin_in.stdev)
                + (ymean.detach() + model_mustd.revin_in.mean),
                y,
            )

            loss = loss_decode + loss_mu + loss_std + loss_all
        elif scheme == 2:
            ytime = model_mustd.revin_in(ytime, "denorm")
            loss_decode = lossfn(ytime, y)
            loss_mu = loss_std = torch.zeros((1,), device=device)
            loss = loss_decode
            
        ################################
        elif scheme == 31:
            ytime = model_mustd.revin_in(ytime, "denorm")
            outputs, batch_y = inverse_scaler(scaler, ytime, y) 
            loss_decode = nn.L1Loss()(outputs, batch_y)
            loss_mu = loss_std = torch.zeros((1,), device=device)
            loss = loss_decode
        elif scheme == 32:
            ytime = model_mustd.revin_in(ytime, "denorm")
#             outputs, batch_y = inverse_scaler(scaler, ytime, y)   
            loss_decode = masked_mse_loss(ytime, y, y_mask)
            loss_mu = loss_std = torch.zeros((1,), device=device)
            loss = loss_decode
        elif scheme == 33:
            ytime = model_mustd.revin_in(ytime, "denorm")
            outputs, batch_y = inverse_scaler(scaler, ytime, y)   
            loss_decode = masked_mae_loss(outputs, batch_y, y_mask)
            loss_mu = loss_std = torch.zeros((1,), device=device)
            loss = loss_decode                          
        else:
            raise ValueError("Unknown prediction scheme %d" % scheme)

        # optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log
        running_loss += loss.item()
        running_loss_mu += loss_mu.item()
        if i % log_every == log_every - 1:
            last_loss = running_loss / log_every  # loss per batch
            last_loss_mu = running_loss_mu / log_every
            lr = optimizer.param_groups[0]["lr"]
            # lr = scheduler.get_last_lr()[0]
            print(
                f"| epoch {epoch:3d} | {i+1:5d}/{len(dataloader):5d} batches | "
                f"lr {lr:02.5f} | loss {last_loss:5.4f} | loss_mu {last_loss_mu:5.4f}"
            )
            running_loss = 0.0
            running_loss_mu = 0.0

        if scheduler is not None:
            scheduler.step()


def inference(
    data,
    model_decode,
    model_mustd,
    codebook,
    compression,
    device,
    onehot: bool = False,
    scheme: int = 2,
):
    """
    Returns:
        out: (B, Tout, Sout)
    """
    code_dim = codebook.shape[1]

    x, y, codeids_x, codeids_y_labels, y_mask = data
    x = x.to(device)
    codeids_x = codeids_x.to(device)

    # prepare data
    B, TCin, Sin = codeids_x.shape
    B, TCout, Sout = codeids_y_labels.shape
    Tout = TCout * compression
    del codeids_y_labels

    _ = model_mustd.revin_in(x, "norm")
    _ = model_mustd.revin_out(y, "norm")

    # get codewords for input x
    code_ids = codeids_x.flatten()
    if onehot:
        xcodes = F.one_hot(code_ids, num_classes=codebook.shape[0])
    else:
        xcodes = codebook[code_ids]  # (B*Tin*Sin, D)
    xcodes = xcodes.reshape((B, TCin, Sin, xcodes.shape[-1]))  # (B, TCin, Sin, D)
    xcodes = torch.permute(xcodes, (0, 2, 1, 3))  # (B, Sin, TCin, D)
    xcodes = xcodes.reshape((B * Sin, TCin, xcodes.shape[-1]))  # (B * Sin, TCin, D)
    xcodes = torch.permute(xcodes, (1, 0, 2))  # (TCin, B * Sin, D)

    ytime = model_decode(xcodes)  # (B * Sout, Tout)

    # reshapes
    ytime = ytime.reshape((B, Sout, Tout))  # (B, Sout, Tout)
    ytime = torch.permute(ytime, (0, 2, 1))  # (B, Tout, Sout)

    # ------ MEAN & STD ------ #
    times = torch.permute(x, (0, 2, 1))  # (B, Sin, Tin * C)
    times = times.reshape((-1, times.shape[-1]))
    ymeanstd = model_mustd(times)

    # reshapes
    ymeanstd = ymeanstd.reshape((B, Sout, 2))  # (B, S, 2)
    ymeanstd = torch.permute(ymeanstd, (0, 2, 1))  # (B, 2, S)
    ymean = ymeanstd[:, 0, :].unsqueeze(1)  # (B, 1, S)
    ystd = ymeanstd[:, 1, :].unsqueeze(1)  # (B, 1, S)

    if scheme == 1:
        ymean = ymean + model_mustd.revin_in.mean
        ystd = ystd + model_mustd.revin_in.stdev
        ytime = ytime * ystd + ymean
    elif scheme == 2:
        ytime = model_mustd.revin_in(ytime, "denorm")
            ################################
    elif scheme == 31:
        ytime = model_mustd.revin_in(ytime, "denorm")
    elif scheme == 32:
        ytime = model_mustd.revin_in(ytime, "denorm")    
    elif scheme == 33:
        ytime = model_mustd.revin_in(ytime, "denorm")          
    else:
        raise ValueError("Unknown prediction scheme %d" % scheme)

    return ytime


def train(args):
    if not os.path.exists(args.file_save_path):
        os.makedirs(args.file_save_path)
        
    print(args.data_missing, args.loss_type, args.loss_inverse)
    if (args.loss_type == "mae")&(args.loss_inverse == True):
        args.scheme = 31 # short term prediciton for the datasets without missing
    if (args.data_missing == True):
        args.scheme = 32 # long term prediciton for the datasets with missing
    if (args.data_missing == True)&(args.loss_type == "mae")&(args.loss_inverse == True):
        args.scheme = 33 # long term prediciton for the datasets with missing
    print("scheme:", args.scheme)
        
    device = torch.device("cuda:%d" % (args.cuda_id))
    torch.cuda.set_device(device)
        
    ################################
    raw_data_scalerv =pickle.load(open(args.file_save_path.replace("results", "data")+"scaler.pkl", 'rb'))
    print(raw_data_scalerv)
    scaler = StandardScaler(mean=torch.from_numpy(raw_data_scalerv[0]).to(device),
                                std=torch.from_numpy(np.sqrt(raw_data_scalerv[1])).to(device))
    scaler_np = StandardScaler(mean=raw_data_scalerv[0], std=np.sqrt(raw_data_scalerv[1]))

    save_name = str(args.data_type) + '_Tin' + str(args.Tin) + '_Tout' + str(args.Tout) + '_seed' + str(args.seed) + '.txt'
    save_file = open(args.file_save_path + save_name, 'w+')

    # -------- SET SEED ------- #
    seed_all_rng(None if args.seed < 0 else args.seed)

    # ------- PRED SCHEMES ------ #
    # Defined in args.scheme
    # if args.scheme is 1 predicts mu and std from input
    # if args.scheme is 2 unrevins the ytime prediction (like in PatchTST)
    # --------------------------- #

    # -------- PARAMS ------- #
    params = get_params(args.data_type, args.data_path)
    batchsize = params["batchsize"]
    dataroot = params["dataroot"]
    Sin, Sout = params["Sin"], params["Sout"]
    is_affine_revin = False
    compression = args.compression

    expname = "%s_CB%d_CF%d_Tin%d_Tout%d" % (
        args.data_type,
        args.codebook_size,
        args.compression,
        args.Tin,
        args.Tout,
    )
    datapath = dataroot

    # -------- CHECKPOINT ------- #
    if args.checkpoint:
        if not os.path.exists(args.checkpoint_path):
            os.makedirs(args.checkpoint_path)
    early_stopping = EarlyStopping(patience=args.patience, path=args.checkpoint_path)

    # -------- CODEBOOK ------- #
    codebook = np.load(os.path.join(datapath, "codebook.npy"), allow_pickle=True)
    codebook = torch.from_numpy(codebook).to(device=device, dtype=torch.float32)
    vocab_size, vocab_dim = codebook.shape

    assert vocab_size == args.codebook_size
    dim = vocab_size if args.onehot else vocab_dim

    # ------ DATA LOADERS ------- #
    dataloaders = create_time_series_dataloader(datapath=datapath, batchsize=batchsize)
    train_dataloader = dataloaders["train"]
    val_dataloader = dataloaders["val"]
    test_dataloader = dataloaders["test"]

    # ------- MODEL: XCODES TO YTIME -------- #
    model_decode = XcodeYtimeDecoder(
        d_in=dim,
        d_model=args.d_model,
        nhead=args.nhead,
        d_hid=args.d_hid,
        nlayers=args.nlayers,
        seq_in_len=args.Tin // compression,
        seq_out_len=args.Tout,
        dropout=0.0,
    )

    # ------- MODEL: MuStd ----------#
    model_mustd = MuStdModel(
        Tin=args.Tin,
        Tout=args.Tout,
        hidden_dims=[512, 512],
        dropout=0.2,
        is_mlp=True,
    )
    model_mustd.revin_in = RevIN(
        num_features=Sin, affine=is_affine_revin
    )  # expects as input (B, T, S)
    model_mustd.revin_out = RevIN(
        num_features=Sout, affine=is_affine_revin
    )  # expects as input (B, T, S)

    model_decode.to(device)
    model_mustd.to(device)

    # ------- OPTIMIZER -------- #
    num_iters = args.epochs * len(train_dataloader)
    step_lr_in_iters = args.steps * len(train_dataloader)
    model_params = list(model_decode.parameters()) + list(model_mustd.parameters())
    total_params = sum(param.numel() for param in model_decode.parameters()) + sum(param.numel() for param in model_mustd.parameters())
    
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model_params, lr=args.baselr, momentum=0.9)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model_params, lr=args.baselr)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model_params, lr=args.baselr)
    else:
        raise ValueError("Uknown optimizer type %s" % (args.optimizer))
    if args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_lr_in_iters, gamma=0.1
        )
    elif args.scheduler == "onecycle":
        # The learning rate will increate from max_lr / div_factor to max_lr in the first pct_start * total_steps steps,
        # and decrease smoothly to max_lr / final_div_factor then.
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=args.baselr,
            steps_per_epoch=len(train_dataloader),
            epochs=args.epochs,
            pct_start=0.2,
        )
    else:
        raise ValueError("Uknown scheduler type %s" % (args.scheduler))

    #######################################################
    start = time.time()
    val_loss_min_save = np.Inf
    # ------- TRAIN & EVAL -------- #
    for epoch in range(args.epochs):
        model_decode.train()
        model_mustd.train()
        train_one_epoch(
            train_dataloader,
            model_decode,
            model_mustd,
            codebook,
            args.compression,
            optimizer,
            scheduler,
            epoch,
            device,
            beta=args.beta,
            onehot=args.onehot,
            scheme=args.scheme,
            scaler=scaler
        )

        if val_dataloader is not None:
            model_decode.eval()
            model_mustd.eval()
            
            preds = []
            trues = []
            masks = []            
            
            running_mse, running_mae, running_cor = 0.0, 0.0, 0.0
            total_num, total_num_c = 0.0, 0.0
            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vdata in enumerate(val_dataloader):
                    pred_time = inference(
                        vdata,
                        model_decode,
                        model_mustd,
                        codebook,
                        args.compression,
                        device,
                        onehot=args.onehot,
                        scheme=args.scheme,
                    )
                    labels_code = vdata[-2]
                    labels_code = labels_code.to(device)
                    labels_time = vdata[1]
                    labels_time = labels_time.to(device)
                    
                    ##############################
                    outputs = pred_time.detach().cpu().numpy()
                    batch_y = labels_time.detach().cpu().numpy()
                    mask = vdata[-1].to(device).detach().cpu().numpy()
                    pred = outputs
                    true = batch_y
                    preds.extend(pred)
                    trues.extend(true)
                    masks.extend(mask)

                    running_mse += F.mse_loss(pred_time, labels_time, reduction="sum")
                    running_mae += (pred_time - labels_time).abs().sum()
                    running_cor += pearsoncor(pred_time, labels_time, reduction="sum")
                    total_num += labels_time.numel()  # B * S * T
                    total_num_c += labels_time.shape[0] * labels_time.shape[2]  # B * S
            running_mae = running_mae / total_num
            running_mse = running_mse / total_num
            running_cor = running_cor / total_num_c
            print(
                f"| [Val] mse {running_mse:5.4f} mae {running_mae:5.4f} corr {running_cor:5.4f}"
            )
            
            ##########################################
            preds = np.array(preds)
            trues = np.array(trues)
            masks = np.array(masks)
            print('test shape:', preds.shape, trues.shape, masks.shape)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            masks = masks.reshape(-1, trues.shape[-2], trues.shape[-1]) 
            
            
            preds_ori, trues_ori = inverse_scaler(scaler_np, preds, trues)
            print('test shape:', preds.shape, trues.shape, masks.shape)
            
            if args.scheme in [32,33]:
                mae, mse, rmse, mape, mspe = metric(preds, trues, masks)        
                mae_ori, mse_ori, rmse_ori, mape_ori, mspe_ori = metric(preds_ori, trues_ori, masks) 
            else:
                mae, mse, rmse, mape, mspe = metric(preds, trues)        
                mae_ori, mse_ori, rmse_ori, mape_ori, mspe_ori = metric(preds_ori, trues_ori) 
            
            if args.scheme in [31,33]:
                running_mse = mse_ori
                running_mae = mae_ori
            
            save_file.write(f"| [Val] mse {running_mse:5.4f} mae {running_mae:5.4f} corr {running_cor:5.4f}\n")
            early_stopping_counter = early_stopping(running_mse, running_mae, {"decode": model_decode, "mustd": model_mustd})
            
            if args.scheme in [31,33]:
                if running_mae < val_loss_min_save:
                    val_loss_min_save = running_mae                              
            else:
                if running_mse.detach().cpu().numpy() < val_loss_min_save:
                    val_loss_min_save = running_mse.detach().cpu().numpy()
                
            print("Min validataion loss:", val_loss_min_save)


        if test_dataloader is not None:
            model_decode.eval()
            model_mustd.eval()
            running_mse, running_mae, running_cor = 0.0, 0.0, 0.0
            total_num, total_num_c = 0.0, 0.0
            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():

                all_prediction_time = []

                for i, tdata in enumerate(test_dataloader):
                    pred_time = inference(
                        tdata,
                        model_decode,
                        model_mustd,
                        codebook,
                        args.compression,
                        device,
                        onehot=args.onehot,
                        scheme=args.scheme,
                    )
                    labels_code = tdata[-2]
                    labels_code = labels_code.to(device)
                    labels_time = tdata[1]
                    labels_time = labels_time.to(device)
                    
#                     print(pred_time.shape, labels_time.shape) #torch.Size([128, 96, 21]) torch.Size([128, 96, 21])

                    running_mse += F.mse_loss(pred_time, labels_time, reduction="sum")
                    running_mae += (pred_time - labels_time).abs().sum()
                    running_cor += pearsoncor(pred_time, labels_time, reduction="sum")
                    total_num += labels_time.numel()
                    total_num_c += labels_time.shape[0] * labels_time.shape[2]  # B * S

                    all_prediction_time.append(np.array(pred_time.detach().cpu()))

            running_mae = running_mae / total_num
            running_mse = running_mse / total_num
            running_cor = running_cor / total_num_c
            print(
                f"| [Test] mse {running_mse:5.4f} mae {running_mae:5.4f} corr {running_cor:5.4f}"
            )
            save_file.write(f"| [Test] mse {running_mse:5.4f} mae {running_mae:5.4f} corr {running_cor:5.4f}\n")
            save_file.write(f"Early stopping counter is: {early_stopping_counter}\n")

        if early_stopping.early_stop:
            print("Early stopping....")

            save_file.write("Early stopping....")
            save_file.write("Take the test values right before the last early stopping counter = 0")
            save_file.close()
            end = time.time()
            train_time = end -start
            break
        end = time.time()
        train_time = end -start
        
    start = time.time()
    model_decode.load_state_dict(torch.load(args.checkpoint_path+"/decode_checkpoint.pth"))
    model_mustd.load_state_dict(torch.load(args.checkpoint_path+"/mustd_checkpoint.pth"))

    if test_dataloader is not None:
        model_decode.eval()
        model_mustd.eval()
        running_mse, running_mae, running_cor = 0.0, 0.0, 0.0
        total_num, total_num_c = 0.0, 0.0
        # Disable gradient computation and reduce memory consumption.
        
        preds = []
        trues = []
        masks = []
        
        with torch.no_grad():

            all_prediction_time = []

            for i, tdata in enumerate(test_dataloader):
                pred_time = inference(
                    tdata,
                    model_decode,
                    model_mustd,
                    codebook,
                    args.compression,
                    device,
                    onehot=args.onehot,
                    scheme=args.scheme,
                )
                labels_code = tdata[-2]
                labels_code = labels_code.to(device)
                labels_time = tdata[1]
                labels_time = labels_time.to(device)
                
                ##############################
                outputs = pred_time.detach().cpu().numpy()
                batch_y = labels_time.detach().cpu().numpy()
                mask = tdata[-1].to(device).detach().cpu().numpy()
                pred = outputs
                true = batch_y
                preds.extend(pred)
                trues.extend(true)
                masks.extend(mask)
                
                running_mse += F.mse_loss(pred_time, labels_time, reduction="sum")
                running_mae += (pred_time - labels_time).abs().sum()
                running_cor += pearsoncor(pred_time, labels_time, reduction="sum")
                total_num += labels_time.numel()
                total_num_c += labels_time.shape[0] * labels_time.shape[2]  # B * S

                all_prediction_time.append(np.array(pred_time.detach().cpu()))

        running_mae = running_mae / total_num
        running_mse = running_mse / total_num
        running_cor = running_cor / total_num_c
        print(
            f"| [Test] mse {running_mse:5.4f} mae {running_mae:5.4f} corr {running_cor:5.4f}"
        )
        
        
        preds = np.array(preds)
        trues = np.array(trues)
        masks = np.array(masks)
        print('test shape:', preds.shape, trues.shape, masks.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        masks = masks.reshape(-1, trues.shape[-2], trues.shape[-1])
        
        preds_ori, trues_ori = inverse_scaler(scaler_np, preds, trues)
        
        print('test shape:', preds.shape, trues.shape, masks.shape)
        
        if args.scheme in [32,33]:
            mae, mse, rmse, mape, mspe = metric(preds, trues, masks)        
            mae_ori, mse_ori, rmse_ori, mape_ori, mspe_ori = metric(preds_ori, trues_ori, masks) 

            mae_perstep, mse_perstep, rmse_perstep, mape_perstep, mspe_perstep = [], [], [], [], []
            for stepi in range(preds.shape[1]):
                mae_stepi, mse_stepi, rmse_stepi, mape_stepi, mspe_stepi = metric(preds[:,stepi,:], trues[:,stepi,:], masks[:,stepi,:])
                mae_perstep.append(mae_stepi)
                mse_perstep.append(mse_stepi)
                rmse_perstep.append(rmse_stepi)
                mape_perstep.append(mape_stepi)
                mspe_perstep.append(mspe_stepi)  

            mae_perstep_ori, mse_perstep_ori, rmse_perstep_ori, mape_perstep_ori, mspe_perstep_ori = [], [], [], [], []
            for stepi in range(preds.shape[1]):
                mae_stepi_ori, mse_stepi_ori, rmse_stepi_ori, mape_stepi_ori, mspe_stepi_ori = metric(preds_ori[:,stepi,:], trues_ori[:,stepi,:], masks[:,stepi,:])
                mae_perstep_ori.append(mae_stepi_ori)
                mse_perstep_ori.append(mse_stepi_ori)
                rmse_perstep_ori.append(rmse_stepi_ori)
                mape_perstep_ori.append(mape_stepi_ori)
                mspe_perstep_ori.append(mspe_stepi_ori)

        else:
            mae, mse, rmse, mape, mspe = metric(preds, trues)        
            mae_ori, mse_ori, rmse_ori, mape_ori, mspe_ori = metric(preds_ori, trues_ori) 

            mae_perstep, mse_perstep, rmse_perstep, mape_perstep, mspe_perstep = [], [], [], [], []
            for stepi in range(preds.shape[1]):
                mae_stepi, mse_stepi, rmse_stepi, mape_stepi, mspe_stepi = metric(preds[:,stepi,:], trues[:,stepi,:])
                mae_perstep.append(mae_stepi)
                mse_perstep.append(mse_stepi)
                rmse_perstep.append(rmse_stepi)
                mape_perstep.append(mape_stepi)
                mspe_perstep.append(mspe_stepi)  

            mae_perstep_ori, mse_perstep_ori, rmse_perstep_ori, mape_perstep_ori, mspe_perstep_ori = [], [], [], [], []
            for stepi in range(preds.shape[1]):
                mae_stepi_ori, mse_stepi_ori, rmse_stepi_ori, mape_stepi_ori, mspe_stepi_ori = metric(preds_ori[:,stepi,:], trues_ori[:,stepi,:])
                mae_perstep_ori.append(mae_stepi_ori)
                mse_perstep_ori.append(mse_stepi_ori)
                rmse_perstep_ori.append(rmse_stepi_ori)
                mape_perstep_ori.append(mape_stepi_ori)
                mspe_perstep_ori.append(mspe_stepi_ori)        
        
        print('mse:{:.4f}, mae:{:.4f}, mape:{:.4f}'.format(mse, mae, mape))
        metric_save = [val_loss_min_save, mse, mae, mape]
        
        end = time.time()
        test_time = end -start
        
        result_columns = ["data", "model_id", "num_networks", "random_seed", "seq_len", "pred_len", "vail_loss", "mse", "mae", "mape",
                          "mse_ori", "mae_ori", "mape_ori", "mse_perstep", "mae_perstep", "mape_perstep", "mse_perstep_ori", "mae_perstep_ori", "mape_perstep_ori",
                          "train_epochs", "total_params", "train_time", "test_time", "args"]
        
        df = pd.DataFrame([[args.data_type,"TOTEM", 1, args.seed, args.Tin, args.Tout] + metric_save + [mse_ori, mae_ori, mape_ori, mse_perstep, mae_perstep, mape_perstep, mse_perstep_ori, mae_perstep_ori, mape_perstep_ori] + [epoch+1, total_params, train_time, test_time] + [args]], columns=result_columns)
        
        folder_path = "results/" + args.data_type + "/" + "forecasting" + "/"
        others_info = re.sub(r'[^a-zA-Z0-9]', '', args.date_split)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        save_path = folder_path + "TOTEM" + "_" + str(args.seed) + "_" + str(args.Tin) + "_" + str(args.Tout) + "_" + others_info + ".csv"
        df['others_info'] = others_info
        df.to_csv(save_path, index=False)
        

    return
    
    
def get_params(data_type, data_path):
    if data_type == "weather":
        batchsize = 128
        Sin = Sout = 21
        dataroot = data_path

    elif data_type == "traffic" :
        batchsize = 16
        Sin = Sout = 862
        dataroot = data_path

    elif data_type == "electricity":
        batchsize = 32
        Sin = Sout = 321
        dataroot = data_path

    elif data_type == "ETTh1":
        batchsize = 128
        Sin = Sout = 7
        dataroot = data_path

    elif data_type == "ETTh2":
        batchsize = 128
        Sin = Sout = 7
        dataroot = data_path

    elif data_type == "ETTm1":
        batchsize = 128
        Sin = Sout = 7
        dataroot = data_path

    elif data_type == "ETTm2":
        batchsize = 128
        Sin = Sout = 7
        dataroot = data_path

    elif data_type == "all":
        batchsize = 4096
        Sin = Sout = 1
        dataroot = data_path
        
    elif data_type == "NYTM-0919":
        batchsize = 96
        Sin = Sout = 67
        dataroot = data_path

    elif data_type == "2m-temperature-NA":
        batchsize = 64
        Sin = Sout = 126
        dataroot = data_path  
        
    elif data_type == "metr-la":
        batchsize = 32
        Sin = Sout = 207
        dataroot = data_path

    elif data_type == "pems-bay":
        batchsize = 32
        Sin = Sout = 325
        dataroot = data_path        
        
    else:
        raise ValueError("Unknown data type %s" % (args.data_type))
    return {"dataroot": dataroot, "batchsize": batchsize, "Sin": Sin, "Sout": Sout}


def default_argument_parser():
    """
    Create a parser.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="Code Prediction")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument("--cuda-id", default=0, type=int)
    # ---------- SEED ---------- #
    # Set seed to negative to fully randomize everything.
    # Set seed to positive to use a fixed seed. Note that a fixed seed increases
    # reproducibility but does not guarantee fully deterministic behavior.
    parser.add_argument("--seed", default=-1, type=int)
    # ---------- DATA ---------- #
    parser.add_argument("--data-type", default="weather", type=str)
    parser.add_argument("--codebook_size", default=256, type=int)
    parser.add_argument("--compression", default=4, type=int)
    parser.add_argument("--Tin", default=96, type=int)
    parser.add_argument("--Tout", default=96, type=int)
    parser.add_argument("--data_path", default='', type=str, help="path to data")

    # ----------- CHECKPOINT ------------ #
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument(
        "--checkpoint_path", default="/data/", type=str
    )
    parser.add_argument("--patience", default=3, type=int)
    # ---------- TRAINING ---------- #
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--scheduler", default="onecycle", type=str)
    parser.add_argument("--baselr", default=0.0001, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--steps", default=4, type=int, help="decay LR in epochs")

    parser.add_argument(
        "--beta", default=0.1, type=float, help="beta for smoothl1 loss"
    )
    # ----------- INPUT ------------ #
    parser.add_argument(
        "--onehot",
        action="store_true",
        help="use onehot representation if true, otherwise use codes",
    )
    # ----------- SCHEME ------------ #
    parser.add_argument(
        "--scheme",
        default=1,
        type=int,
        help="1 predicts mu/std, 2 urevins the prediction (as in PatchTST)",
    )
    # ----------- ENCODER PARAMS ------------ #
    parser.add_argument("--d-model", default=64, type=int)
    parser.add_argument("--d_hid", default=256, type=int)
    parser.add_argument("--nhead", default=4, type=int)
    parser.add_argument("--nlayers", default=4, type=int)

    parser.add_argument("--file_save_path", default='', type=str)
    
    parser.add_argument('--data_missing', action='store_true', help='for missing data', default=False)
    parser.add_argument('--date_split', type=str, default='None', help='data split based on given date')
    parser.add_argument('--loss_type', type=str, default='mse', help='mse or mae')
    parser.add_argument('--loss_inverse', action='store_true', help='inverse output data', default=False)


    return parser.parse_args()


if __name__ == "__main__":
    args = default_argument_parser()
    train(args)