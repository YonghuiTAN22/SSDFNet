import sys
sys.path.append('/public/home/tanyh_25/project/Mamba/SSDFNet')

import torch
import torch.nn as nn
import torch.nn.functional as F
from classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
import math

from buildingextraction.models.main_module.sdfm import *
from buildingextraction.models.main_module.FDConv import FDConv
from buildingextraction.models.main_module.fdefm import *
from buildingextraction.models.main_module.DSFM import DSFM


class ConvBlock(nn.Module):
    """简单 Conv + BN + GELU 模块"""
    def __init__(self, in_channels, out_channels, k=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class EdgeSupervisionModule(nn.Module):
    """
    多尺度边界监督模块
    输入：多尺度特征 E1, E2, E3
    输出：edge_map (B, 1, H, W)
    """
    def __init__(self, in_channels_list, mid_channels=32):
        """
        in_channels_list: [E1_channels, E2_channels, E3_channels]
        mid_channels: ConvBlock 输出通道数
        """
        super().__init__()
        E1_ch, E2_ch, E3_ch = in_channels_list

        # 上采样 + ConvBlock / 1x1 Conv 调整通道
        self.E1_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(E1_ch, mid_channels)
        )
        self.E2_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(E2_ch, mid_channels, kernel_size=1, bias=False)
        )
        self.E3_up = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),
            nn.Conv2d(E3_ch, mid_channels, kernel_size=1, bias=False)
        )

        # 拼接后的 ConvBlock
        self.fusion_conv = ConvBlock(mid_channels*3, mid_channels)

        # 1x1 Conv + Sigmoid 输出边界图
        self.out_conv = nn.Conv2d(mid_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, E1, E2, E3):
        x1 = self.E1_up(E1)
        x2 = self.E2_up(E2)
        x3 = self.E3_up(E3)

        # 拼接
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.fusion_conv(x)

        # edge_map
        # edge_map = self.sigmoid(self.out_conv(x))
        return x


class SemanticDecoderFusionEnhanceFreOPTSAR(nn.Module):
    def __init__(self, encoder_dims, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs):
        super(SemanticDecoderFusionEnhanceFreOPTSAR, self).__init__()

        ## ---------------------------------------- ##
        self.dsf1 = DSFM(in_channels_opt=encoder_dims[-4], in_channels_sar=encoder_dims[-4], out_channels=encoder_dims[-4], kernel_size=3, num_heads=4)
        self.dsf2 = DSFM(in_channels_opt=encoder_dims[-3], in_channels_sar=encoder_dims[-3], out_channels=encoder_dims[-3], kernel_size=3, num_heads=4)
        self.dsf3 = DSFM(in_channels_opt=encoder_dims[-2], in_channels_sar=encoder_dims[-2], out_channels=encoder_dims[-2], kernel_size=3, num_heads=4)
        self.dsf4 = DSFM(in_channels_opt=encoder_dims[-1], in_channels_sar=encoder_dims[-1], out_channels=encoder_dims[-1], kernel_size=3, num_heads=4)
        ## ---------------------------------------- ##

        ## ---------------------------------------- ##
        self.sdfm_1 = SDFM(in_channels=encoder_dims[-4]+encoder_dims[-3]+encoder_dims[-2], out_channels=encoder_dims[-4])
        self.sdfm_2 = SDFM(in_channels=encoder_dims[-4]+encoder_dims[-3]+encoder_dims[-2], out_channels=encoder_dims[-3])
        self.sdfm_3 = SDFM(in_channels=encoder_dims[-4]+encoder_dims[-3]+encoder_dims[-2], out_channels=encoder_dims[-2])
        ## ---------------------------------------- ##

        self.edge_super = EdgeSupervisionModule(in_channels_list=[encoder_dims[-4], encoder_dims[-3], encoder_dims[-2]], mid_channels=32)

        self.fre_enhance = FDEFM(in_channels=128)

        # Define the VSS Block for Spatio-temporal relationship modelling
        self.st_block_4_semantic = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-1], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_3_semantic = nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_2_semantic = nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_1_semantic = nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )           

        self.trans_layer_3 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-2], out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())
        self.trans_layer_2 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-3], out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())
        self.trans_layer_1 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-4], out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())


        # Smooth layer
        self.smooth_layer_3_semantic = ResBlock(in_channels=128, out_channels=128, stride=1) 
        self.smooth_layer_2_semantic = ResBlock(in_channels=128, out_channels=128, stride=1) 
        self.smooth_layer_1_semantic = ResBlock(in_channels=128, out_channels=128, stride=1) 
        self.smooth_layer_0_semantic = ResBlock(in_channels=128, out_channels=128, stride=1) 
    
    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, features_opt, features_sar):
        feat_o_1, feat_o_2, feat_o_3, feat_o_4 = features_opt
        feat_s_1, feat_s_2, feat_s_3, feat_s_4 = features_sar

        feat_1 = self.dsf1(feat_o_1, feat_s_1)
        feat_2 = self.dsf2(feat_o_2, feat_s_2)
        feat_3 = self.dsf3(feat_o_3, feat_s_3)
        feat_4 = self.dsf4(feat_o_4, feat_s_4)

        ## ---------------------------------------- ##
        feat_2_1 = F.interpolate(feat_2, scale_factor=2.0, mode='bilinear', align_corners=True)
        feat_3_1 = F.interpolate(feat_3, scale_factor=4.0, mode='bilinear', align_corners=True)

        feat_1_2 = F.interpolate(feat_1, scale_factor=0.5, mode='bilinear', align_corners=True)
        feat_3_2 = F.interpolate(feat_3, scale_factor=2.0, mode='bilinear', align_corners=True)

        feat_1_3 = F.interpolate(feat_1, scale_factor=0.25, mode='bilinear', align_corners=True)
        feat_2_3 = F.interpolate(feat_2, scale_factor=0.5, mode='bilinear', align_corners=True)

        feat_1 = self.sdfm_1(feat_1, feat_2_1, feat_3_1)
        feat_2 = self.sdfm_2(feat_1_2, feat_2, feat_3_2)
        feat_3 = self.sdfm_3(feat_1_3, feat_2_3, feat_3)
        ## ---------------------------------------- ##

        feat_edge = self.edge_super(feat_1, feat_2, feat_3)

        '''
            Stage I
        '''
        p4 = self.st_block_4_semantic(feat_4)
        p4 = self.fre_enhance(p4)
       
        '''
            Stage II
        '''
        p3 = self.trans_layer_3(feat_3)
        p3 = self._upsample_add(p4, p3)
        p3 = self.smooth_layer_3_semantic(p3)
        p3 = self.st_block_3_semantic(p3)
        p3 = self.fre_enhance(p3)

        '''
            Stage III
        '''
        p2 = self.trans_layer_2(feat_2)
        p2 = self._upsample_add(p3, p2)
        p2 = self.smooth_layer_2_semantic(p2)
        p2 = self.st_block_2_semantic(p2)
        p2 = self.fre_enhance(p2)

        '''
            Stage IV
        '''
        p1 = self.trans_layer_1(feat_1)
        p1 = self._upsample_add(p2, p1)
        p1 = self.smooth_layer_1_semantic(p1)
        p1 = self.st_block_1_semantic(p1)
        p1 = self.fre_enhance(p1)
        p1 = self.smooth_layer_0_semantic(p1)

        return p1, feat_edge

   
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
