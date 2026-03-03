import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pdb
from .srm import all_normalized_hpf_list
from .freq_stem import ConvNet

def DCT_mat(size):
    m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m

def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j < start else 1. for j in range(size)] for i in range(size)]

def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.

class Filter(nn.Module):
    def __init__(self, size, band_start, band_end, use_learnable=False, norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))), requires_grad=False)


    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y

class DCT_base_Rec_Module(nn.Module):
    """_summary_

    Args:
        x: [C, H, W] -> [C*level, output, output]
    """
    def __init__(self, window_size=32, stride=16, output=224, grade_N=6, level_fliter=[0]):
        super().__init__()
        
        assert output % window_size == 0
        assert len(level_fliter) > 0
        
        self.window_size = window_size
        self.grade_N = grade_N
        self.level_N = len(level_fliter)
        self.N = (output // window_size) * (output // window_size)
        
        self._DCT_patch = nn.Parameter(torch.tensor(DCT_mat(window_size)).float(), requires_grad=False)
        self._DCT_patch_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(window_size)).float(), 0, 1), requires_grad=False)
        
        self.unfold = nn.Unfold(
            kernel_size=(window_size, window_size), stride=stride
        )
        self.fold0 = nn.Fold(
            output_size=(window_size, window_size), 
            kernel_size=(window_size, window_size), 
            stride=window_size
        )
        
        lm, mh = 2.82, 2
        level_f = [
            Filter(window_size, 0, window_size * 2)
        ]
        
        self.level_filters = nn.ModuleList([level_f[i] for i in level_fliter])
        self.grade_filters = nn.ModuleList([Filter(window_size, window_size * 2. / grade_N * i, window_size * 2. / grade_N * (i+1), norm=True) for i in range(grade_N)])
        
        
    def forward(self, x):
        
        N = self.N
        grade_N = self.grade_N
        level_N = self.level_N
        window_size = self.window_size
        B, C, W, H = x.shape
        x_unfold = self.unfold(x)        

        _, _, L = x_unfold.shape
        x_unfold = x_unfold.transpose(1, 2).reshape(B, L, C, window_size, window_size) 

        x_dct = self._DCT_patch @ x_unfold @ self._DCT_patch_T
        
        y_list = []
        for i in range(self.level_N):
            x_pass = self.level_filters[i](x_dct)
            y = self._DCT_patch_T @ x_pass @ self._DCT_patch
            y_list.append(y)
        level_x_unfold = torch.cat(y_list, dim=2)
        
        grade = torch.zeros(B, L).to(x.device)
        w, k = 1, 2
        for _ in range(grade_N):
            _x = torch.abs(x_dct)
            _x = torch.log(_x + 1)
            _x = self.grade_filters[_](_x)
            _x = torch.sum(_x, dim=[2,3,4])
            grade += w * _x            
            w *= k
        
        _, idx = torch.sort(grade, dim= -1)
        max_idx = torch.flip(idx, dims=[1])[:, :N]

        maxmax_idx = max_idx[:, 0:1]
        if len(max_idx) == 1:
            maxmax_idx1 = max_idx[:, 0:1]
        else:
            maxmax_idx1 = max_idx[:, 1:2]

        min_idx = idx[:, :N]
        minmin_idx = idx[:, 0:1]
        if len(min_idx) == 1:
            minmin_idx1 = idx[:, 0:1]
        else:
            minmin_idx1 = idx[:, 1:2]

        x_minmin = torch.gather(level_x_unfold, dim=1, index=minmin_idx.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1, -1, C, window_size, window_size))
        x_maxmax = torch.gather(level_x_unfold, dim=1, index=maxmax_idx.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1, -1, C, window_size, window_size))
        x_minmin1 = torch.gather(level_x_unfold, dim=1, index=minmin_idx1.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1, -1, C, window_size, window_size))
        x_maxmax1 = torch.gather(level_x_unfold, dim=1, index=maxmax_idx1.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1, -1, C, window_size, window_size))

        x_minmin = x_minmin.reshape(B, 1, level_N*C*window_size* window_size).transpose(1, 2)
        x_maxmax = x_maxmax.reshape(B, 1, level_N*C*window_size* window_size).transpose(1, 2)
        x_minmin1 = x_minmin1.reshape(B, 1, level_N*C*window_size* window_size).transpose(1, 2)
        x_maxmax1 = x_maxmax1.reshape(B, 1, level_N*C*window_size* window_size).transpose(1, 2)

        x_minmin = self.fold0(x_minmin)
        x_maxmax = self.fold0(x_maxmax)
        x_minmin1 = self.fold0(x_minmin1)
        x_maxmax1 = self.fold0(x_maxmax1)

        return x_minmin, x_maxmax, x_minmin1, x_maxmax1

class DCT_base_Rec_index(nn.Module):
    """_summary_

    Args:
        x: [C, H, W] -> [C*level, output, output]
    """
    def __init__(self, window_size=14, stride=14, output=224, grade_N=6, level_fliter=[0]):
        super().__init__()
        
        assert output % window_size == 0
        assert len(level_fliter) > 0
        
        self.window_size = window_size
        self.grade_N = grade_N
        self.level_N = len(level_fliter)
        self.N = (output // window_size) * (output // window_size)
        
        self._DCT_patch = nn.Parameter(torch.tensor(DCT_mat(window_size)).float(), requires_grad=False)
        self._DCT_patch_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(window_size)).float(), 0, 1), requires_grad=False)
        
        self.unfold = nn.Unfold(
            kernel_size=(window_size, window_size), stride=stride
        )
        self.fold0 = nn.Fold(
            output_size=(window_size, window_size), 
            kernel_size=(window_size, window_size), 
            stride=window_size
        )
        
        lm, mh = 2.82, 2
        level_f = [
            Filter(window_size, 0, window_size * 2)
        ]
        
        self.level_filters = nn.ModuleList([level_f[i] for i in level_fliter])
        self.grade_filters = nn.ModuleList([Filter(window_size, window_size * 2. / grade_N * i, window_size * 2. / grade_N * (i+1), norm=True) for i in range(grade_N)])
        
        
    def forward(self, x):
        
        N = self.N
        grade_N = self.grade_N
        level_N = self.level_N
        window_size = self.window_size
        B, C, W, H = x.shape
        x_unfold = self.unfold(x)        

        _, _, L = x_unfold.shape
        x_unfold = x_unfold.transpose(1, 2).reshape(B, L, C, window_size, window_size) 

        x_dct = self._DCT_patch @ x_unfold @ self._DCT_patch_T
        
        y_list = []
        for i in range(self.level_N):
            x_pass = self.level_filters[i](x_dct)
            y = self._DCT_patch_T @ x_pass @ self._DCT_patch
            y_list.append(y)
        level_x_unfold = torch.cat(y_list, dim=2)
        
        grade = torch.zeros(B, L).to(x.device)
        w, k = 1, 2
        for _ in range(grade_N):
            _x = torch.abs(x_dct)
            _x = torch.log(_x + 1)
            _x = self.grade_filters[_](_x)
            _x = torch.sum(_x, dim=[2,3,4])
            grade += w * _x            
            w *= k
        
        _, idx = torch.sort(grade, dim= -1)
        max_idx = torch.flip(idx, dims=[1])[:, :N]

        maxmax_idx = max_idx[:, 0:1]

        min_idx = idx[:, :N]
        minmin_idx = idx[:, 0:1]
        return maxmax_idx, minmin_idx

class HPF(nn.Module):
    def __init__(self):
        super(HPF, self).__init__()
        
        num_filters = 30
        filter_size = 5
        hpf_array = np.empty((num_filters, filter_size, filter_size), dtype=np.float32)
        
        for i, hpf_item in enumerate(all_normalized_hpf_list):
            if hpf_item.shape[0] == 3:
                hpf_padded = np.pad(
                    hpf_item, 
                    pad_width=((1, 1), (1, 1)),
                    mode='constant'
                )
                hpf_array[i] = hpf_padded
            else:
                hpf_array[i] = hpf_item
        
        hpf_weight = torch.from_numpy(hpf_array)
        hpf_weight = hpf_weight.view(num_filters, 1, filter_size, filter_size).contiguous()
        hpf_weight = hpf_weight.repeat(1, 3, 1, 1).contiguous()
        
        self.hpf = nn.Conv2d(
            in_channels=3,
            out_channels=num_filters,
            kernel_size=filter_size,
            padding=2,
            bias=False
        )
        self.hpf.weight = nn.Parameter(hpf_weight, requires_grad=False)
        
    def forward(self, input):
        return self.hpf(input)
    
class DCT_Condition_Module(nn.Module):
    """_summary_

    Args:
        x: [C, H, W] -> [C*level, output, output]
    """
    def __init__(self, window_size=32, stride=16, output=224, grade_N=6, level_fliter=[0]):
        super().__init__()
        
        assert output % window_size == 0
        assert len(level_fliter) > 0
        
        self.window_size = window_size
        self.grade_N = grade_N
        self.level_N = len(level_fliter)
        self.N = (output // window_size) * (output // window_size)
        
        self._DCT_patch = nn.Parameter(torch.tensor(DCT_mat(window_size)).float(), requires_grad=False)
        self._DCT_patch_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(window_size)).float(), 0, 1), requires_grad=False)
        
        self.unfold = nn.Unfold(
            kernel_size=(window_size, window_size), stride=stride
        )
        self.fold0 = nn.Fold(
            output_size=(window_size, window_size), 
            kernel_size=(window_size, window_size), 
            stride=window_size
        )
        
        lm, mh = 2.82, 2 
        level_f = [
            Filter(window_size, 0, window_size * 2)
        ]
        
        self.level_filters = nn.ModuleList([level_f[i] for i in level_fliter])
        self.grade_filters = nn.ModuleList([Filter(window_size, window_size * 2. / grade_N * i, window_size * 2. / grade_N * (i+1), norm=True) for i in range(grade_N)])
        
        # feat extract
        self.hpf = HPF()
        self.model_min = ConvNet(30, 128)
        self.model_max = ConvNet(30, 128)

        # # gated
        self.gamma = nn.Parameter(torch.ones(1024) * 1e-6)
        self.gamma_2 = nn.Parameter(torch.ones(1024) * 1e-6)

        # sup
        self.sup = nn.Linear(1024, 1)

    def forward(self, x):
        
        N = self.N
        grade_N = self.grade_N
        level_N = self.level_N
        window_size = self.window_size
        B, C, W, H = x.shape
        x_unfold = self.unfold(x)        

        _, _, L = x_unfold.shape
        x_unfold = x_unfold.transpose(1, 2).reshape(B, L, C, window_size, window_size) 

        x_dct = self._DCT_patch @ x_unfold @ self._DCT_patch_T
        
        y_list = []
        for i in range(self.level_N):
            x_pass = self.level_filters[i](x_dct)
            y = self._DCT_patch_T @ x_pass @ self._DCT_patch
            y_list.append(y)
        level_x_unfold = torch.cat(y_list, dim=2)
        
        grade = torch.zeros(B, L).to(x.device)
        w, k = 1, 2
        for _ in range(grade_N):
            _x = torch.abs(x_dct)
            _x = torch.log(_x + 1)
            _x = self.grade_filters[_](_x)
            _x = torch.sum(_x, dim=[2,3,4])
            grade += w * _x            
            w *= k
        
        _, idx = torch.sort(grade, dim= -1)
        max_idx = torch.flip(idx, dims=[1])[:, :N]
        
        minmin_idx = max_idx[:, 0:1] # max和min都是一样的
        maxmax_idx = max_idx[:, 0:1]
        
        x_minmin = torch.gather(level_x_unfold, dim=1, index=minmin_idx.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1, -1, C, window_size, window_size))
        x_maxmax = torch.gather(level_x_unfold, dim=1, index=maxmax_idx.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1, -1, C, window_size, window_size))

        x_minmin = x_minmin.reshape(B, 1, level_N*C*window_size* window_size).transpose(1, 2)
        x_maxmax = x_maxmax.reshape(B, 1, level_N*C*window_size* window_size).transpose(1, 2)

        with torch.cuda.amp.autocast(enabled=False):
            x_minmin = x_minmin.to(torch.float32)
            x_maxmax = x_maxmax.to(torch.float32)
            
            x_minmin = self.fold0(x_minmin)
            x_maxmax = self.fold0(x_maxmax)
            
            x_minmin = x_minmin.to(x.dtype)
            x_maxmax = x_maxmax.to(x.dtype)

        x_minmin = self.hpf(x_minmin)
        x_maxmax = self.hpf(x_maxmax)
        bias_min = self.model_min(x_minmin)
        bias_max = self.model_max(x_maxmax)

        # sup  
        pred_bias = self.sup(bias_max)
        
        bias_min = bias_min * self.gamma
        bias_max = bias_max * self.gamma_2

        bias = torch.cat([bias_min.unsqueeze(1), bias_max.unsqueeze(1)], dim=1)

        return bias, pred_bias