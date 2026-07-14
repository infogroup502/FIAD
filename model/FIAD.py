import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VariableSpecificSinFitting(nn.Module):
    def __init__(self, num_features=8, num_sin_waves=99, win_size=100):
        super().__init__()

        self.num_features = num_features
        self.num_sin_waves = num_sin_waves
        self.win_size = win_size

        self.A = nn.Parameter(torch.randn(num_features, num_sin_waves))
        self.w = nn.Parameter(torch.randn(num_features, num_sin_waves))
        self.b = nn.Parameter(torch.randn(num_features, num_sin_waves))

        t = torch.linspace(0, 2 * np.pi, win_size)
        self.register_buffer('t', t)

        self._init_parameters()

    def _init_parameters(self):
        nn.init.normal_(self.A, mean=1.0, std=0.1)
        nn.init.normal_(self.w, mean=1.0, std=0.1)
        nn.init.uniform_(self.b, -np.pi, np.pi)

    def forward(self, x, return_components=False):
        N, T, M = x.shape

        fitted_list = []
        component_list = []
        t_used = self.t[:T].to(x.device)

        for m in range(M):
            A_m = self.A[m]
            w_m = self.w[m]
            b_m = self.b[m]

            t_expanded = t_used.view(1, T, 1)
            A_expanded = A_m.view(1, 1, -1)
            w_expanded = w_m.view(1, 1, -1)
            b_expanded = b_m.view(1, 1, -1)

            sin_waves = A_expanded * torch.sin(w_expanded * t_expanded + b_expanded)
            sin_waves = sin_waves.expand(N, -1, -1)

            fitted_var = torch.sum(sin_waves, dim=-1)
            fitted_list.append(fitted_var.unsqueeze(2))

            if return_components:
                component_list.append(sin_waves.permute(0, 2, 1).unsqueeze(1))

        x_hat_sin = torch.cat(fitted_list, dim=2)

        if return_components:
            components = torch.cat(component_list, dim=1)
            return x_hat_sin, components

        return x_hat_sin

    def get_parameters(self):
        return {'A': self.A.detach(), 'w': self.w.detach(), 'b': self.b.detach()}


class SpatialSinFitting(nn.Module):
    def __init__(self, num_sin_waves, spatial_size):
        super().__init__()

        self.num_sin_waves = num_sin_waves
        self.spatial_size = spatial_size

        self.A = nn.Parameter(torch.randn(num_sin_waves))
        self.w = nn.Parameter(torch.randn(num_sin_waves))
        self.b = nn.Parameter(torch.randn(num_sin_waves))

        t = torch.linspace(0, 2 * np.pi, spatial_size)
        self.register_buffer('t', t)

        self._init_parameters()

    def _init_parameters(self):
        nn.init.normal_(self.A, mean=1.0, std=0.1)
        nn.init.normal_(self.w, mean=1.0, std=0.1)
        nn.init.uniform_(self.b, -np.pi, np.pi)

    def forward(self, x_spatial):
        N = x_spatial.shape[0]
        S = x_spatial.shape[-1]

        t_used = self.t[:S].to(x_spatial.device)

        t_exp = t_used.view(1, 1, -1)
        A_exp = self.A.view(1, self.num_sin_waves, 1)
        w_exp = self.w.view(1, self.num_sin_waves, 1)
        b_exp = self.b.view(1, self.num_sin_waves, 1)

        sin_waves = A_exp * torch.sin(w_exp * t_exp + b_exp)
        return sin_waves.expand(N, -1, -1)

    def get_parameters(self):
        return {'A': self.A.detach(), 'w': self.w.detach(), 'b': self.b.detach()}


class SharedSinLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.A = nn.Parameter(torch.randn(out_features, in_features))
        self.w = nn.Parameter(torch.randn(out_features, in_features))
        self.b = nn.Parameter(torch.randn(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        t_in = torch.linspace(0, 2 * np.pi, in_features)
        self.register_buffer('t_in', t_in)

        self._init_parameters()

    def _init_parameters(self):
        nn.init.normal_(self.A, mean=1.0, std=0.1)
        nn.init.normal_(self.w, mean=1.0, std=0.1)
        nn.init.uniform_(self.b, -np.pi, np.pi)

        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, x):
        t = self.t_in.to(x.device).view(1, -1)
        weight = self.A * torch.sin(self.w * t + self.b)
        out = torch.einsum('...i,oi->...o', x, weight)

        if self.bias is not None:
            out = out + self.bias

        return out


class SharedSinMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()

        self.first = SharedSinLinear(input_dim, hidden_dim)
        self.last = SharedSinLinear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.first(x)
        h = self.relu(h)
        h = self.dropout(h)
        return self.last(h)


class SinParamConv1d(nn.Module):
    def __init__(self, num_channels=1, kernel_size=100):
        super().__init__()

        self.C = num_channels
        self.L = kernel_size

        self.A = nn.Parameter(torch.randn(num_channels, kernel_size))
        self.w = nn.Parameter(torch.randn(num_channels, kernel_size))
        self.b = nn.Parameter(torch.randn(num_channels, kernel_size))
        self.log_sigma = nn.Parameter(torch.zeros(num_channels))

        t = torch.linspace(0, 2 * np.pi, kernel_size)
        self.register_buffer('t', t)

        self._init_parameters()

    def _init_parameters(self):
        nn.init.normal_(self.A, mean=1.0, std=0.1)
        nn.init.normal_(self.w, mean=1.0, std=0.1)
        nn.init.uniform_(self.b, -np.pi, np.pi)
        nn.init.constant_(self.log_sigma, 0.0)

    def forward(self, x):
        _, _, L = x.shape

        t_used = self.t[:L].to(x.device)

        t = t_used.unsqueeze(0).unsqueeze(0)
        A = self.A[:, :L].unsqueeze(-1)
        w = self.w[:, :L].unsqueeze(-1)
        b = self.b[:, :L].unsqueeze(-1)

        kernel_waveforms = A * torch.sin(w * t + b)

        dot = torch.einsum('bkl,cjl->bkcj', x, kernel_waveforms)
        x_norm = x.norm(dim=-1, keepdim=True)
        k_norm = kernel_waveforms.norm(dim=-1)

        eps = 1e-8
        denom = (x_norm.unsqueeze(2) * k_norm.view(1, 1, self.C, L)) + eps
        cos_sim = dot / denom

        temp = torch.exp(self.log_sigma).view(1, 1, self.C, 1).clamp(min=1e-4)
        sim = F.softmax(cos_sim / temp, dim=-1)

        weighted_phi = torch.einsum('bkcj,cjl->bkcl', sim, kernel_waveforms)
        out = x.unsqueeze(2) * weighted_phi
        return out.sum(dim=-1)


class FIAD(nn.Module):
    def __init__(self, win_size=100, num_features=8, time_size=10, spatial_size=8, time_num_sin_waves=50, spatial_num_sin_waves=50,
                 r=0.5, d_model=128):
        super().__init__()
        self.win_size = win_size
        self.num_features = num_features

        self.time_size = self._normalize_size(time_size, win_size)
        self.spatial_size = self._normalize_size(spatial_size, win_size * num_features)

        self.time_num_sin_waves = time_num_sin_waves
        self.time_cnn_channels = 1
        self.spatial_num_sin_waves = spatial_num_sin_waves
        self.spatial_cnn_channels = 1
        self.r = r
        self.d_model = d_model

        self.sin_fitting = VariableSpecificSinFitting(
            num_features=num_features,
            num_sin_waves=time_num_sin_waves,
            win_size=self.time_size
        )

        self.spatial_sin_fitting = SpatialSinFitting(
            num_sin_waves=spatial_num_sin_waves,
            spatial_size=self.spatial_size
        )

        self.time_convs = None
        self.spatial_conv = None
        self.time_mlp = None
        self.spatial_mlp = None
        self.training_phase = 1

    def _normalize_size(self, size_value, upper_bound):
        if isinstance(size_value, (list, tuple)):
            size_value = upper_bound if len(size_value) == 0 else int(size_value[0])
        else:
            size_value = int(size_value)

        return min(max(1, size_value), upper_bound)

    def _build_time_windows(self, x):
        B, L, M = x.shape

        left = self.time_size // 2
        right = self.time_size - 1 - left

        x_perm = x.permute(0, 2, 1).contiguous().view(B * M, 1, L)
        x_pad = F.pad(x_perm, (left, right), mode='replicate')
        windows = x_pad.unfold(dimension=-1, size=self.time_size, step=1)
        windows = windows.squeeze(1)
        windows = windows.contiguous().view(B, M, L, self.time_size)
        return windows.permute(0, 2, 1, 3).contiguous()

    def _build_spatial_windows(self, x):
        B, L, M = x.shape

        x_spatial = x.contiguous().view(B, L * M)

        left = self.spatial_size // 2
        right = self.spatial_size - 1 - left

        x_spatial_ = x_spatial.unsqueeze(1)
        x_spatial_pad = F.pad(x_spatial_, (left, right), mode='replicate')
        windows = x_spatial_pad.unfold(dimension=-1, size=self.spatial_size, step=1)
        windows = windows.squeeze(1).contiguous()

        return x_spatial, windows

    def initialize_cnn_phase(self):
        self.time_convs = nn.ModuleList([
            SinParamConv1d(num_channels=self.time_cnn_channels, kernel_size=self.time_size)
            for _ in range(self.num_features)
        ])

        self.spatial_conv = SinParamConv1d(
            num_channels=self.spatial_cnn_channels,
            kernel_size=self.spatial_size
        )

        time_feat_dim = self.time_num_sin_waves * self.time_cnn_channels
        spatial_feat_dim = self.spatial_num_sin_waves * self.spatial_cnn_channels
        mlp_input_dim = time_feat_dim + spatial_feat_dim

        self.time_mlp = SharedSinMLP(
            input_dim=mlp_input_dim,
            hidden_dim=self.d_model,
            output_dim=self.time_size,
            dropout=0.1
        )

        self.spatial_mlp = SharedSinMLP(
            input_dim=mlp_input_dim,
            hidden_dim=self.d_model,
            output_dim=self.spatial_size,
            dropout=0.1
        )

        device = next(self.parameters()).device

        self.time_convs = self.time_convs.to(device)
        self.spatial_conv = self.spatial_conv.to(device)
        self.time_mlp = self.time_mlp.to(device)
        self.spatial_mlp = self.spatial_mlp.to(device)

    def forward(self, x):
        B, L, M = x.shape

        if self.training_phase == 1:
            time_windows = self._build_time_windows(x)

            time_input = time_windows.permute(0, 1, 3, 2).contiguous()
            time_input = time_input.view(B * L, self.time_size, M)

            x_hat_time = self.sin_fitting(time_input)
            x_hat_time = x_hat_time.view(B, L, self.time_size, M)
            x_hat_time = x_hat_time.permute(0, 1, 3, 2).contiguous()

            _, spatial_windows = self._build_spatial_windows(x)
            spatial_input = spatial_windows.contiguous().view(B * L * M, self.spatial_size)

            sin_spatial = self.spatial_sin_fitting(spatial_input)
            x_hat_spatial = sin_spatial.sum(dim=1)
            x_hat_spatial = x_hat_spatial.view(B, L * M, self.spatial_size)

            return x_hat_time, time_windows, x_hat_spatial, spatial_windows

        if self.time_convs is None:
            raise ValueError("initialize_cnn_phase() must be called before phase 2.")

        time_windows = self._build_time_windows(x)

        time_input = time_windows.permute(0, 1, 3, 2).contiguous()
        time_input = time_input.view(B * L, self.time_size, M)

        _, time_components = self.sin_fitting(time_input, return_components=True)

        time_feat_list = []

        for m in range(M):
            sin_m = time_components[:, m, :, :]
            feat_m = self.time_convs[m](sin_m)
            feat_m = feat_m.contiguous().view(B, L, -1)
            time_feat_list.append(feat_m.unsqueeze(2))

        time_feats = torch.cat(time_feat_list, dim=2)

        _, spatial_windows = self._build_spatial_windows(x)

        spatial_target = spatial_windows.contiguous().view(B, L, M, self.spatial_size)
        spatial_input = spatial_windows.contiguous().view(B * L * M, self.spatial_size)

        sin_spatial = self.spatial_sin_fitting(spatial_input)
        feat_spatial = self.spatial_conv(sin_spatial)
        feat_spatial = feat_spatial.contiguous().view(B, L, M, -1)

        combined = torch.cat([time_feats, feat_spatial], dim=-1)

        recon_time_from_time = self.time_mlp(combined)
        recon_spatial_from_spatial = self.spatial_mlp(combined)

        return (
            recon_time_from_time,
            time_windows,
            recon_spatial_from_spatial,
            spatial_target
        )

    def set_training_phase(self, phase):
        self.training_phase = phase
