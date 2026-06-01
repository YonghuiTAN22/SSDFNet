import torch
import torch.nn as nn
import torch.nn.functional as F

import mmcv
from mmcv.cnn import ConvModule


class SELayer(nn.Module):
    """Squeeze-and-Excitation Module.

    Args:
        channels (int): The input (and output) channels of the SE layer.
        ratio (int): Squeeze ratio in SELayer, the intermediate channel will be
            ``int(channels/ratio)``. Default: 16.
        conv_cfg (None or dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        act_cfg (dict or Sequence[dict]): Config dict for activation layer.
            If act_cfg is a dict, two activation layers will be configured
            by this dict. If act_cfg is a sequence of dicts, the first
            activation layer will be configured by the first dict and the
            second activation layer will be configured by the second dict.
            Default: (dict(type='ReLU'), dict(type='HSigmoid', bias=3.0,
            divisor=6.0)).
    """

    def __init__(self,
                 channels,
                 ratio=16,
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'),
                        #   dict(type='HSigmoid', bias=3.0, divisor=6.0),
                          dict(type='Sigmoid')
                          )
                          ):
        super(SELayer, self).__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert mmcv.is_tuple_of(act_cfg, dict)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(
            in_channels=channels,
            # out_channels=make_divisible(channels // ratio, 8),
            out_channels=channels // ratio,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            # in_channels=make_divisible(channels // ratio, 8),
            in_channels=channels // ratio,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return out
    

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        # return x * y.expand_as(x)
        return y.expand_as(x)
    

class FDEFM(nn.Module):
    """Optimized Frequency Mixing module for building extraction decoder.

    Args:
        in_channels (int): Number of input channels.
        k_list (list): List of kernel sizes for low-pass filters. Default: [3, 5].
        compress_ratio (int): Channel reduction ratio for attention. Default: 16.
        spatial_group (int): Number of groups for spatial convolution. Default: 1.
        act_type (str): Activation type ('sigmoid' or 'softmax'). Default: 'sigmoid'.
        channel_res (bool): Whether to add residual connection. Default: True.
    """
    def __init__(self, 
                 in_channels,
                 k_list=[3, 5],
                 compress_ratio=32,
                 spatial_group=1,
                 act_type='sigmoid',
                 channel_res=True):
        super(FDEFM, self).__init__()
        self.in_channels = in_channels
        self.k_list = sorted(k_list)
        self.compress_ratio = compress_ratio
        self.spatial_group = min(spatial_group, in_channels)
        self.act_type = act_type
        self.channel_res = channel_res
        self.freq_thres = 0.25  # Adaptive frequency threshold, default to 0.25 (25% of the frequency spectrum)

        # Spatial convolution for frequency weight
        self.freq_weight_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=(len(k_list) + 1) * self.spatial_group,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=self.spatial_group,
            bias=True
        )

        # Low-pass filter modules
        self.lp_list = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad2d(padding=k // 2),
                nn.AvgPool2d(kernel_size=k, stride=1, padding=0)
            ) for k in k_list
        ])

        # Efficient channel attention (single SE-like module)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // compress_ratio, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // compress_ratio, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Compute frequency weights
        freq_weight = self.freq_weight_conv(x)
        if self.act_type == 'sigmoid':
            freq_weight = F.sigmoid(freq_weight)
        elif self.act_type == 'softmax':
            freq_weight = F.softmax(freq_weight, dim=1) * freq_weight.shape[1]
        else:
            raise ValueError(f"Unsupported activation type: {self.act_type}")

        # Frequency decomposition using FFT
        x_fft = torch.fft.fftshift(torch.fft.fft2(x))
        _, _, h, w = x.shape
        low_mask = torch.zeros_like(x_fft, device=x.device)
        high_mask = torch.ones_like(x_fft, device=x.device)
        low_mask[:, :, 
                 round(h/2 - h * self.freq_thres):round(h/2 + h * self.freq_thres), 
                 round(w/2 - w * self.freq_thres):round(w/2 + w * self.freq_thres)] = 1.0
        high_mask[:, :, 
                  round(h/2 - h * self.freq_thres):round(h/2 + h * self.freq_thres), 
                  round(w/2 - w * self.freq_thres):round(w/2 + w * self.freq_thres)] = 0.0

        # Low and high frequency components
        low_x_fft = x_fft * low_mask
        high_x_fft = x_fft * high_mask
        low_part = torch.fft.ifft2(torch.fft.ifftshift(low_x_fft)).real
        high_part = x - low_part

        # Apply channel attention
        low_c_att = self.channel_att(low_x_fft.abs())
        high_c_att = self.channel_att(high_x_fft.abs())

        # Combine frequency components with weights
        low_part = low_part * freq_weight[:, 0:1, :, :] * low_c_att
        high_part = high_part * freq_weight[:, 1:2, :, :] * high_c_att

        # low_part = low_part * freq_weight[:, 0:1,] * self.channel_att_low((x_fft * low_mask).abs()) 
        # high_part = high_part * freq_weight[:, 1:2,] * self.channel_att_high((x_fft * high_mask).abs())

        # Residual connection
        out = low_part + high_part
        if self.channel_res:
            out += x

        return out
    
    def count_parameters(self):
        total_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                total_params += param.numel()
        return total_params / 1_000_000  # Convert to millions (M)
    