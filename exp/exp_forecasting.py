from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, StandardScaler, inverse_scaler 
from utils.metrics import metric, masked_mae_loss, masked_mape_loss, masked_rmse_loss, masked_mse_loss
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


class Exp_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss_type == "mae":
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        #total_loss = []
        preds = []
        trues = []
        masks = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_y_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.model == 'MegaCRN':
                    batch_y = batch_y[:, -self.args.pred_len:, ...].to(self.device)
                    outputs_mix = self.model(batch_x, batch_y)
                    outputs = outputs_mix[0]
                    
                    if self.args.model == 'MegaCRN':
                        batch_y = batch_y[:, -self.args.pred_len:, :, 0]
#                         outputs = outputs[:, -self.args.pred_len:, :, 0]
                else:
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        
                        
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                batch_y_mask = batch_y_mask[:, -self.args.pred_len:, f_dim:]

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
               
                #loss = criterion(pred, true)
                #total_loss.append(loss)
                
                preds.extend(pred.numpy())
                trues.extend(true.numpy())
                masks.extend(batch_y_mask.numpy())
                
        #total_loss = np.average(total_loss)
        
        preds = np.array(preds)
        trues = np.array(trues)
        masks = np.array(masks)
        
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        masks = masks.reshape(-1, masks.shape[-2], masks.shape[-1])
        
        if self.args.loss_inverse:
            preds, trues = inverse_scaler(self.scaler_np, preds, trues)
            
#         _, total_loss, _, _, _ = metric(preds, trues)

        if self.args.data_missing == True:
            mae, mse, _, _, _ = metric(preds, trues, masks)
        else:
            mae, mse, _, _, _ = metric(preds, trues)

        self.model.train()
        
        if self.args.loss_type == "mae":
            return mae
        else:
            return mse

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        if self.args.model == 'MegaCRN':
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, eps=1e-3)
            lr_scheduler = optim.lr_scheduler.MultiStepLR(model_optim, milestones=[50, 100], gamma=0.1)
        else:
            model_optim = self._select_optimizer()
        
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        self.val_loss_min_save = np.Inf
        self.scaler = StandardScaler(mean=torch.from_numpy(train_data.scaler.mean_).to(self.device),
                                std=torch.from_numpy(np.sqrt(train_data.scaler.var_)).to(self.device))
        self.scaler_np = StandardScaler(mean=train_data.scaler.mean_, std=np.sqrt(train_data.scaler.var_))
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_y_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                batch_y_mask = batch_y_mask[:, -self.args.pred_len:, :].to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # encoder - decoder
                if self.args.model == 'MegaCRN':
                    batch_y = batch_y[:, -self.args.pred_len:, ...].to(self.device)
                    
                    if self.args.model == 'MegaCRN':
                        other_inputs = [iter_count-1, self.args, self.scaler, batch_y_mask]
                    outputs_mix = self.model(batch_x, batch_y, other_inputs)
                    
                    #outputs = outputs_mix[0]
                    loss = outputs_mix[1]
                    
                    train_loss.append(loss.item())
                    
                else:
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                            f_dim = -1 if self.args.features == 'MS' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            loss = criterion(outputs, batch_y)
                            train_loss.append(loss.item())
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        
                        if self.args.loss_inverse:
                            outputs, batch_y = inverse_scaler(self.scaler, outputs, batch_y)

                        if self.args.data_missing == True:
                            if self.args.loss_type == "mae":
                                loss = masked_mae_loss(outputs, batch_y, batch_y_mask)
                            else:
                                loss = masked_mse_loss(outputs, batch_y, batch_y_mask)
                        else:    
                            loss = criterion(outputs, batch_y)
                            
                        train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    if self.args.model == 'MegaCRN':
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if vali_loss < self.val_loss_min_save:
                self.val_loss_min_save = vali_loss
            print("Min validataion loss:", self.val_loss_min_save)
            
            if self.args.model == 'MegaCRN':
                lr_scheduler.step()
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        self.train_epochs = epoch

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            self.scaler_np = StandardScaler(mean=test_data.scaler.mean_, std=np.sqrt(test_data.scaler.var_))
            self.val_loss_min_save = "-"

        preds = []
        trues = []
        masks = []
#         folder_path = './test_results/' + setting + '/'
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_y_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.model == 'MegaCRN':
                    batch_y = batch_y[:, -self.args.pred_len:, ...].to(self.device)
                    outputs_mix = self.model(batch_x, batch_y)
                    outputs = outputs_mix[0]
                    
                    if self.args.model == 'MegaCRN':
                        batch_y = batch_y[:, -self.args.pred_len:, :, 0]
#                         outputs = outputs[:, -self.args.pred_len:, :, 0]
                else:
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
#                 if test_data.scale and self.args.inverse:
#                     shape = outputs.shape
#                     outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
#                     batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]
                batch_y_mask = batch_y_mask[:, -self.args.pred_len:, f_dim:]

                pred = outputs
                true = batch_y

#                 preds.append(pred)
#                 trues.append(true)
                preds.extend(pred)
                trues.extend(true)
                masks.extend(batch_y_mask.numpy())

#                 if i % 20 == 0:
#                     input = batch_x.detach().cpu().numpy()
#                     if test_data.scale and self.args.inverse:
#                         shape = input.shape
#                         input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
#                     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
#                     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
#                     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        masks = np.array(masks)
        print('test shape:', preds.shape, trues.shape, masks.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        masks = masks.reshape(-1, trues.shape[-2], trues.shape[-1])
        
        preds_ori, trues_ori = inverse_scaler(self.scaler_np, preds, trues)
        
        print('test shape:', preds.shape, trues.shape, masks.shape)

        # result save
        
        if self.args.data_missing == True:
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
                
#         mae_3, mse_3, rmse_3, mape_3, mspe_3 = metric(preds[:,2,:], trues[:,2,:])
#         mae_6, mse_6, rmse_6, mape_6, mspe_6 = metric(preds[:,5,:], trues[:,5,:])
#         mae_12, mse_12, rmse_12, mape_12, mspe_12 = metric(preds[:,11,:], trues[:,11,:])
        
        print('mse:{:.4f}, mae:{:.4f}, mape:{:.4f}'.format(mse, mae, mape))
        self.metric_save = [self.val_loss_min_save, mse, mae, mape]
        
        result_columns = ["data", "model_id", "num_networks", "random_seed", "seq_len", "pred_len", "vail_loss", "mse", "mae", "mape",
                          "mse_ori", "mae_ori", "mape_ori", "mse_perstep", "mae_perstep", "mape_perstep", "mse_perstep_ori", "mae_perstep_ori", "mape_perstep_ori",
                          "train_epochs", "total_params", "train_time", "test_time", "args"]
        
        if self.args.is_training:
            self.df = pd.DataFrame([[self.args.data_path[:-4],self.args.model_id, self.args.num_networks, self.args.random_seed, self.args.seq_len, self.args.pred_len] + self.metric_save + [mse_ori, mae_ori, mape_ori, mse_perstep, mae_perstep, mape_perstep, mse_perstep_ori, mae_perstep_ori, mape_perstep_ori] + [self.train_epochs+1, "-", "-", "-"] + [self.args]], columns=result_columns)
        else:
            self.df = pd.DataFrame([[self.args.data_path[:-4],self.args.model_id, self.args.num_networks, self.args.random_seed, self.args.seq_len, self.args.pred_len] + ['-'] + self.metric_save[1:] + [mse_ori, mae_ori, mape_ori, mse_perstep, mae_perstep, mape_perstep, mse_perstep_ori, mae_perstep_ori, mape_perstep_ori] + ["-", "-", "-", "-"] + [self.args]], columns=result_columns)

        
#         folder_path = './results/' + setting + '/'
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)

#         mae, mse, rmse, mape, mspe = metric(preds, trues)
#         print('mse:{}, mae:{}'.format(mse, mae))
#         f = open("result_long_term_forecast.txt", 'a')
#         f.write(setting + "  \n")
#         f.write('mse:{}, mae:{}'.format(mse, mae))
#         f.write('\n')
#         f.write('\n')
#         f.close()

#         np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
#         np.save(folder_path + 'pred.npy', preds)
#         np.save(folder_path + 'true.npy', trues)

        return
