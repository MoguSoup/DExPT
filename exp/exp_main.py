import math
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data._utils.collate import default_collate

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import DExPT
from utils.metrics import metric
from utils.ps_loss import PatchStructuralLoss
from utils.tools import EarlyStopping, adjust_learning_rate, visual

warnings.filterwarnings("ignore")


class IndexedDataLoader:
    def __init__(self, dataloader):
        self._dataloader = dataloader
        self.dataset = dataloader.dataset
        self.batch_sampler = dataloader.batch_sampler
        self.collate_fn = dataloader.collate_fn or default_collate

    def __iter__(self):
        for batch_indices in self.batch_sampler:
            indices = list(batch_indices)
            batch = [self.dataset[index] for index in indices]
            collated = self.collate_fn(batch)
            yield tuple(collated) + (torch.tensor(indices, dtype=torch.long),)

    def __len__(self):
        return len(self._dataloader)

    def __getattr__(self, name):
        return getattr(self._dataloader, name)


class SelectiveResidualMask:
    def __init__(self, num_samples, keep_ratio=0.8, warmup_epochs=1):
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("selective_ratio must be in (0, 1]")
        self.num_samples = num_samples
        self.keep_ratio = float(keep_ratio)
        self.warmup_epochs = int(warmup_epochs)
        self.completed_epochs = 0
        self.residual_sum = None
        self.residual_sq_sum = None
        self.residual_count = None
        self.uncertainty_mask = None

    def mask_for(self, indices, reference):
        if (
            indices is None
            or self.uncertainty_mask is None
            or self.completed_epochs < self.warmup_epochs
        ):
            return None

        _, output_len, _ = reference.shape
        idx = indices.to(dtype=torch.long, device=self.uncertainty_mask.device)
        offsets = idx.unsqueeze(-1) + torch.arange(output_len, device=idx.device).unsqueeze(0)
        return self.uncertainty_mask[offsets].to(reference.device, dtype=reference.dtype)

    def update(self, indices, residual):
        if indices is None:
            return

        residual = residual.detach().abs().cpu()
        output_len, num_features = residual.shape[1], residual.shape[2]
        if self.residual_sum is None:
            result_shape = (self.num_samples + output_len - 1, num_features)
            self.residual_sum = torch.zeros(result_shape, dtype=residual.dtype)
            self.residual_sq_sum = torch.zeros_like(self.residual_sum)
            self.residual_count = torch.zeros(result_shape[0], 1, dtype=residual.dtype)

        ids = (
            indices.cpu().to(dtype=torch.long)[:, None]
            + torch.arange(output_len, dtype=torch.long)[None, :]
        )
        flat_ids = ids.reshape(-1, 1)
        feature_ids = flat_ids.expand(-1, num_features)
        flat_residual = residual.reshape(-1, num_features)

        self.residual_sum.scatter_add_(0, feature_ids, flat_residual)
        self.residual_sq_sum.scatter_add_(0, feature_ids, flat_residual.pow(2))
        self.residual_count.scatter_add_(
            0,
            flat_ids,
            torch.ones_like(flat_ids, dtype=residual.dtype),
        )

    def on_epoch_end(self):
        if self.residual_sum is None:
            return

        count = self.residual_count.clamp_min(1.0)
        mean = self.residual_sum / count
        entropy = (self.residual_sq_sum / count) - mean.pow(2)
        threshold = torch.quantile(entropy, self.keep_ratio, dim=0, keepdim=True)
        self.uncertainty_mask = entropy <= threshold
        self.residual_sum.zero_()
        self.residual_sq_sum.zero_()
        self.residual_count.zero_()
        self.completed_epochs += 1


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        model = DExPT.Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss(), nn.L1Loss()

    def _weighted_mae(self, pred, true, criterion):
        ratio = np.array(
            [-math.atan(i + 1) + math.pi / 4 + 1 for i in range(self.args.pred_len)]
        )
        ratio = torch.tensor(ratio, dtype=true.dtype, device=self.device).unsqueeze(-1)
        return criterion(pred * ratio, true * ratio)

    def vali(self, vali_data, vali_loader, criterion, weighted=False):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark)
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:]

                if weighted:
                    loss = self._weighted_mae(outputs, batch_y, criterion)
                else:
                    loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())

        self.model.train()
        return np.average(total_loss)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        use_ps_loss = self.args.ps_lambda > 0.0
        use_selective_mask = use_ps_loss and self.args.selective_ratio < 1.0
        selective_ps_mask = None
        if use_selective_mask:
            train_loader = IndexedDataLoader(train_loader)
            selective_ps_mask = SelectiveResidualMask(
                len(train_data),
                self.args.selective_ratio,
                self.args.selective_warmup,
            )

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        mse_criterion, mae_criterion = self._select_criterion()
        ps_criterion = (
            PatchStructuralLoss(self.model, patch_len_threshold=self.args.patch_len_threshold)
            if use_ps_loss
            else None
        )

        time_now = time.time()
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, batch in enumerate(train_loader):
                if len(batch) == 5:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, batch_idx = batch
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                    batch_idx = None

                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark)
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:]

                loss = self._weighted_mae(outputs, batch_y, mae_criterion)
                if ps_criterion is not None:
                    step_mask = (
                        selective_ps_mask.mask_for(batch_idx, outputs)
                        if selective_ps_mask is not None
                        else None
                    )
                    loss = loss + self.args.ps_lambda * ps_criterion(batch_y, outputs, step_mask)
                if selective_ps_mask is not None:
                    selective_ps_mask.update(batch_idx, outputs - batch_y)

                train_loss.append(loss.item())
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print("\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            if selective_ps_mask is not None:
                selective_ps_mask.on_epoch_end()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, mae_criterion, weighted=True)
            test_loss = self.vali(test_data, test_loader, mse_criterion)
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1,
                    train_steps,
                    train_loss,
                    vali_loss,
                    test_loss,
                )
            )
            early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, "checkpoint.pth")
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        os.remove(best_model_path)
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")

        if test:
            print("loading model")
            checkpoint = os.path.join(self.args.checkpoints, setting, "checkpoint.pth")
            self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))

        preds = []
        trues = []
        folder_path = os.path.join("./test_results", setting)
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark)
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:]

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)

                if i % 20 == 0:
                    input_x = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input_x[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input_x[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + ".pdf"))

        preds = np.array(preds).reshape(-1, preds[0].shape[-2], preds[0].shape[-1])
        trues = np.array(trues).reshape(-1, trues[0].shape[-2], trues[0].shape[-1])

        mae, mse = metric(preds, trues)
        print("mse:{}, mae:{}".format(mse, mae))
        with open("result.txt", "a", encoding="utf-8") as f:
            f.write(setting + "  \n")
            f.write("mse:{}, mae:{}".format(mse, mae))
            f.write("\n\n")
