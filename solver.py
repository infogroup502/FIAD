import csv
import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from data_factory.data_loader import get_loader_segment
from metrics.combine_all_scores import combine_all_evaluation_scores
from metrics.metrics import *
from model.FIAD import FIAD
from model.RevIN import RevIN
warnings.filterwarnings('ignore')


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.time_size = self._normalize_window_size(self.time_size, self.win_size)
        self.spatial_size = self._normalize_window_size(self.spatial_size, self.win_size * self.input_c)
        self.freq_loss_weight = 0.5
        self.freq_score_weight = 0.5
        self.train_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size, win_size=self.win_size,
            mode='train', dataset=self.dataset, step=self.step)

        self.vali_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size, win_size=self.win_size,
            mode='val', dataset=self.dataset, step=self.step)

        self.test_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size, win_size=self.win_size,
            mode='test', dataset=self.dataset, step=self.step)

        self.thre_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size, win_size=self.win_size,
            mode='thre', dataset=self.dataset, step=self.step)

        self.build_model()
        self.model.to(self.device)

        if self.loss_fuc == 'MAE':
            self.criterion = nn.L1Loss()
            self.criterion_keep = nn.L1Loss(reduction='none')
        elif self.loss_fuc == 'MSE':
            self.criterion = nn.MSELoss()
            self.criterion_keep = nn.MSELoss(reduction='none')
        else:
            raise ValueError(f"Unsupported loss_fuc: {self.loss_fuc}")

        self.stage1_completed = False
        self.stage2_completed = False

    def _normalize_window_size(self, size_value, upper_bound):
        if isinstance(size_value, (list, tuple)):
            size_value = upper_bound if len(size_value) == 0 else int(size_value[0])
        else:
            size_value = int(size_value)

        return min(max(1, size_value), upper_bound)

    def build_model(self):
        self.model = FIAD(
            win_size=self.win_size,
            num_features=self.input_c,
            time_size=self.time_size,
            spatial_size=self.spatial_size,
            time_num_sin_waves=self.time_num_sin_waves,
            spatial_num_sin_waves=self.spatial_num_sin_waves,
            r=self.r,
            d_model=self.d_model
        )

        params_stage1 = list(self.model.sin_fitting.parameters())
        params_stage1 += list(self.model.spatial_sin_fitting.parameters())

        self.optimizer_stage1 = torch.optim.Adam(params_stage1, lr=self.stage1_lr)
        self.optimizer_stage2 = None

    def frequency_reconstruction_loss(self, pred, target, dim):
        pred_fft = torch.fft.rfft(pred, dim=dim, norm='ortho')
        target_fft = torch.fft.rfft(target, dim=dim, norm='ortho')
        return torch.mean(torch.abs(pred_fft - target_fft) ** 2)

    def frequency_error_map(self, pred, target, dim=-1):
        pred_fft = torch.fft.rfft(pred, dim=dim, norm='ortho')
        target_fft = torch.fft.rfft(target, dim=dim, norm='ortho')
        freq_error = torch.abs(pred_fft - target_fft) ** 2
        return torch.sum(freq_error, dim=dim)

    def train_stage1(self):
        self.model.set_training_phase(1)

        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0
            batch_count = 0
            start_time = time.time()

            for _, (input_data, _) in enumerate(self.train_loader):
                input = input_data.float().to(self.device)

                revin_layer = RevIN(num_features=self.input_c).to(self.device)
                x = revin_layer(input, 'norm')

                x_hat_time, time_windows, x_hat_spatial, spatial_windows = self.model(x)

                loss_time_rec = self.criterion(x_hat_time, time_windows)
                loss_spatial_rec = self.criterion(x_hat_spatial, spatial_windows)

                loss_time_freq = self.frequency_reconstruction_loss(x_hat_time, time_windows, dim=-1)
                loss_spatial_freq = self.frequency_reconstruction_loss(x_hat_spatial, spatial_windows, dim=-1)

                loss_time = (1 - self.freq_loss_weight) * loss_time_rec + self.freq_loss_weight * loss_time_freq
                loss_spatial = (
                    (1 - self.freq_loss_weight) * loss_spatial_rec
                    + self.freq_loss_weight * loss_spatial_freq
                )
                loss = self.r * loss_time + (1 - self.r) * loss_spatial

                self.optimizer_stage1.zero_grad()
                loss.backward()
                self.optimizer_stage1.step()

                train_loss += loss.item()
                batch_count += 1

            avg_loss = train_loss / batch_count if batch_count > 0 else 0.0
            epoch_time = time.time() - start_time

            print(f"Stage1 Epoch [{epoch + 1}/{self.num_epochs}] "
                  f"Loss: {avg_loss:.6f}, Time: {epoch_time:.2f}s")

            adjust_learning_rate(self.optimizer_stage1, epoch + 1, self.stage1_lr)

        self.stage1_completed = True

    def initialize_cnn_phase(self):
        self.model.initialize_cnn_phase()
        self.model.to(self.device)

        for p in self.model.sin_fitting.parameters():
            p.requires_grad = False

        for p in self.model.spatial_sin_fitting.parameters():
            p.requires_grad = False

        params = []

        if hasattr(self.model, 'time_convs') and self.model.time_convs is not None:
            params += list(self.model.time_convs.parameters())

        if hasattr(self.model, 'time_mlp') and self.model.time_mlp is not None:
            params += list(self.model.time_mlp.parameters())

        if hasattr(self.model, 'spatial_conv') and self.model.spatial_conv is not None:
            params += list(self.model.spatial_conv.parameters())

        if hasattr(self.model, 'spatial_mlp') and self.model.spatial_mlp is not None:
            params += list(self.model.spatial_mlp.parameters())

        self.optimizer_stage2 = torch.optim.Adam(params, lr=self.stage2_lr)

    def train_stage2(self):
        if not self.stage1_completed:
            print("Warning: Stage1 not completed")
            return

        self.model.set_training_phase(2)

        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0
            batch_count = 0
            start_time = time.time()

            for _, (input_data, _) in enumerate(self.train_loader):
                input = input_data.float().to(self.device)

                revin_layer = RevIN(num_features=self.input_c).to(self.device)
                x = revin_layer(input, 'norm')

                recon_time_from_time, time_target, recon_spatial_from_spatial, spatial_target = self.model(x)

                loss_time_rec = self.criterion(recon_time_from_time, time_target)
                loss_spatial_rec = self.criterion(recon_spatial_from_spatial, spatial_target)

                loss_time_freq = self.frequency_reconstruction_loss(recon_time_from_time, time_target, dim=-1)
                loss_spatial_freq = self.frequency_reconstruction_loss(recon_spatial_from_spatial, spatial_target, dim=-1)

                loss_time = (1 - self.freq_loss_weight) * loss_time_rec + self.freq_loss_weight * loss_time_freq
                loss_spatial = ((1 - self.freq_loss_weight) * loss_spatial_rec + self.freq_loss_weight * loss_spatial_freq)
                loss = self.r * loss_time + (1 - self.r) * loss_spatial

                self.optimizer_stage2.zero_grad()
                loss.backward()
                self.optimizer_stage2.step()

                train_loss += loss.item()
                batch_count += 1

            avg_loss = train_loss / batch_count if batch_count > 0 else 0.0
            epoch_time = time.time() - start_time

            print(f"Stage2 Epoch [{epoch + 1}/{self.num_epochs}] "
                  f"Loss: {avg_loss:.6f}, Time: {epoch_time:.2f}s")

            adjust_learning_rate(self.optimizer_stage2, epoch + 1, self.stage2_lr)

        self.stage2_completed = True

    def _compute_window_errors(self, x):
        recon_time_from_time, time_target, recon_spatial_from_spatial, spatial_target = self.model(x)

        # Combine time-domain and frequency-domain errors for each branch.
        errors_time_td = self.criterion_keep(recon_time_from_time, time_target)
        errors_time_td = torch.sum(errors_time_td, dim=-1)
        errors_time_fd = self.frequency_error_map(recon_time_from_time, time_target, dim=-1)

        errors_spatial_td = self.criterion_keep(recon_spatial_from_spatial, spatial_target)
        errors_spatial_td = torch.sum(errors_spatial_td, dim=-1)
        errors_spatial_fd = self.frequency_error_map(recon_spatial_from_spatial, spatial_target, dim=-1)

        errors_time = errors_time_td * (1 - self.freq_score_weight) + self.freq_score_weight * errors_time_fd
        errors_spatial = (errors_spatial_td * (1 - self.freq_score_weight) + self.freq_score_weight * errors_spatial_fd)

        return self.r * errors_time + (1 - self.r) * errors_spatial

    def _compute_energy(self, x):
        kk = min(self.topk_k, self.input_c)
        errors = self._compute_window_errors(x)
        topk_values, _ = torch.topk(errors, k=kk, dim=-1)
        time_step_scores = torch.mean(topk_values, dim=-1)
        return torch.softmax(time_step_scores, dim=-1)

    def run(self):
        print("Begin two-phase training")
        print("=" * 60)

        self.train_stage1()
        self.initialize_cnn_phase()
        self.train_stage2()

        print("Start testing")
        print("=" * 60)
        self.test()

    def test(self):
        if not self.stage2_completed:
            print("Warning: Stage2 not completed")

        self.topk_k = 3
        self.model.set_training_phase(2)

        # Build the training energy distribution.
        attens_energy = []
        self.model.eval()

        with torch.no_grad():
            for _, (input_data, _) in enumerate(self.train_loader):
                input = input_data.float().to(self.device)
                revin_layer = RevIN(num_features=self.input_c).to(self.device)
                x = revin_layer(input, 'norm')

                energy = self._compute_energy(x)
                attens_energy.append(energy.detach().cpu().numpy())

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # Build the threshold energy distribution.
        attens_energy = []

        with torch.no_grad():
            for _, (input_data, _) in enumerate(self.thre_loader):
                input = input_data.float().to(self.device)
                revin_layer = RevIN(num_features=self.input_c).to(self.device)
                x = revin_layer(input, 'norm')

                energy = self._compute_energy(x)
                attens_energy.append(energy.detach().cpu().numpy())

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)

        print(f"anormly_ratio: {self.anormly_ratio}")
        print(f"thresh: {thresh:.6f}")

        # Evaluate on the threshold loader labels.
        test_labels = []
        attens_energy = []

        with torch.no_grad():
            for _, (input_data, labels) in enumerate(self.thre_loader):
                input = input_data.float().to(self.device)
                revin_layer = RevIN(num_features=self.input_c).to(self.device)
                x = revin_layer(input, 'norm')

                energy = self._compute_energy(x)
                attens_energy.append(energy.detach().cpu().numpy())
                test_labels.append(labels.detach().cpu().numpy().reshape(-1))

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)

        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        print(f"test_energy.shape: {test_energy.shape}")
        print(f"test_labels.shape: {test_labels.shape}")

        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)

        print(f"pred.shape: {pred.shape}, gt.shape: {gt.shape}")

        matrix = [self.index]
        scores_simple = combine_all_evaluation_scores(pred, gt, test_energy)

        for key, value in scores_simple.items():
            matrix.append(value)
            print(f'{key:21} : {value:.4f}')

        anomaly_state = False

        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True

                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    if pred[j] == 0:
                        pred[j] = 1

                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    if pred[j] == 0:
                        pred[j] = 1

            elif gt[i] == 0:
                anomaly_state = False

            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, _ = precision_recall_fscore_support(gt, pred, average='binary')

        print("=" * 60)
        print("Final evaluation results")
        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1-score  : {f_score:.4f}")

        if self.data_path in ['UCR', 'UCR_AUG']:
            try:
                save_dir = 'all_resules/result_FIAD'

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                save_path = os.path.join(save_dir, self.data_path + '.csv')

                with open(save_path, 'a+', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(matrix)

                print(f"The results have been saved to {save_path}")

            except Exception as e:
                print(f"Error saving results: {e}")

        self.model.train()

        return accuracy, precision, recall, f_score

    def save_checkpoint(self, path, stage=1):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_stage1_state_dict': self.optimizer_stage1.state_dict(),
            'stage': stage,
            'stage1_completed': self.stage1_completed,
            'stage2_completed': self.stage2_completed,
        }

        if stage == 2 and self.optimizer_stage2 is not None:
            checkpoint['optimizer_stage2_state_dict'] = self.optimizer_stage2.state_dict()

        torch.save(checkpoint, path)
        print(f"Checkpoints have been saved to {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_stage1.load_state_dict(checkpoint['optimizer_stage1_state_dict'])

        stage = checkpoint.get('stage', 1)
        self.stage1_completed = checkpoint.get('stage1_completed', False)
        self.stage2_completed = checkpoint.get('stage2_completed', False)

        if stage == 2 and 'optimizer_stage2_state_dict' in checkpoint:
            if self.optimizer_stage2 is None:
                self.initialize_cnn_phase()

            self.optimizer_stage2.load_state_dict(checkpoint['optimizer_stage2_state_dict'])

        print(f"The checkpoint has been loaded from {path}")
        print(f"Recovery training phase: {stage}")

        return stage
