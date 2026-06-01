import sys
sys.path.append('/public/home/tanyh_25/project/Mamba/SSDFNet')

import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
from buildingextraction.models.Mamba_backbone import Backbone_VSSM, Backbone_ResNet50, Backbone_EfficientNetB5
from classification.models.vmamba import LayerNorm2d

import torch
import torch.nn as nn
import torch.nn.functional as F
from buildingextraction.models.SSDFNet_Dec import SemanticDecoderFusionEnhanceFreOPTSAR
    

class SSDFNet(nn.Module):
    def __init__(self, output_building, pretrained, **kwargs):
        super(SSDFNet, self).__init__()
        
        # self.encoder_opt_0 = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)
        # self.encoder_opt = Backbone_ResNet50(out_indices=(0, 1, 2, 3), 
        #                                      out_channels=[128, 256, 512, 1024], 
        #                                      norm_layer='bn',
        #                                      pretrained='imagenet'
        # )
        self.encoder_opt = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)
        # self.encoder_opt = Backbone_EfficientNetB5(out_indices=(0, 1, 2, 3),
        #                                            pretrained="imagenet",
        #                                            norm_layer="ln2d",
        #                                            out_channels=[128, 256, 512, 1024],
        #                                            in_channels=3,
        #                                            frozen_stages=1,
        # )

        # self.encoder_sar = Backbone_EfficientNetB5(out_indices=(0,1,2,3),
        #                                            pretrained="imagenet",
        #                                            norm_layer="ln2d",
        #                                            out_channels=[128, 256, 512, 1024],
        #                                            in_channels=3,
        #                                            frozen_stages=1,
        # )

        # self.encoder_sar = Backbone_ResNet50(out_indices=(0, 1, 2, 3), 
        #                                      out_channels=[128, 256, 512, 1024], 
        #                                      norm_layer='bn', 
        #                                      pretrained='imagenet'
        # )

        self.encoder_sar = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        
        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        self.channel_first = self.encoder_opt.channel_first

        print(self.channel_first)

        norm_layer: nn.Module = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)        
        ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)

        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}

        self.decoder_building = SemanticDecoderFusionEnhanceFreOPTSAR(
            encoder_dims=self.encoder_sar.dims,
            channel_first=self.encoder_opt.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.aux_clf = nn.Conv2d(in_channels=128, out_channels=output_building, kernel_size=1)
        self.aux_clf_edge = nn.Conv2d(in_channels=32, out_channels=output_building, kernel_size=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, input_opt, input_sar):
        # Encoder processing
        opt_features = self.encoder_opt(input_opt)
        sar_features = self.encoder_sar(input_sar)

        # Decoder processing - passing encoder outputs to the decoder
        output_building, feat_edge = self.decoder_building(sar_features, opt_features)
        # output_building = self.decoder_building(mid_features)
        
        output_building = self.aux_clf(output_building)
        feat_edge = self.aux_clf_edge(feat_edge)

        output_building = F.interpolate(output_building, size=input_opt.size()[-2:], mode='bilinear')
        feat_edge = F.interpolate(feat_edge, size=input_opt.size()[-2:], mode='bilinear')
       
        return output_building, feat_edge


class SSDFNet_Visualization(nn.Module):
    def __init__(self, output_building, pretrained, **kwargs):
        super(SSDFNet_Visualization, self).__init__()
        
        # self.encoder_opt_0 = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)
        # self.encoder_opt = Backbone_ResNet50(out_indices=(0, 1, 2, 3), 
        #                                      out_channels=[128, 256, 512, 1024], 
        #                                      norm_layer='bn',
        #                                      pretrained='imagenet'
        # )
        self.encoder_opt = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)
        # self.encoder_opt = Backbone_EfficientNetB5(out_indices=(0, 1, 2, 3),
        #                                            pretrained="imagenet",
        #                                            norm_layer="ln2d",
        #                                            out_channels=[128, 256, 512, 1024],
        #                                            in_channels=3,
        #                                            frozen_stages=1,
        # )

        # self.encoder_sar = Backbone_EfficientNetB5(out_indices=(0,1,2,3),
        #                                            pretrained="imagenet",
        #                                            norm_layer="ln2d",
        #                                            out_channels=[128, 256, 512, 1024],
        #                                            in_channels=3,
        #                                            frozen_stages=1,
        # )

        # self.encoder_sar = Backbone_ResNet50(out_indices=(0, 1, 2, 3), 
        #                                      out_channels=[128, 256, 512, 1024], 
        #                                      norm_layer='bn', 
        #                                      pretrained='imagenet'
        # )

        self.encoder_sar = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        
        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        self.channel_first = self.encoder_opt.channel_first

        print(self.channel_first)

        norm_layer: nn.Module = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)        
        ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)

        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}

        self.decoder_building = SemanticDecoderFusionEnhanceFreOPTSAR(
            encoder_dims=self.encoder_sar.dims,
            channel_first=self.encoder_opt.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.aux_clf = nn.Conv2d(in_channels=128, out_channels=output_building, kernel_size=1)
        self.aux_clf_edge = nn.Conv2d(in_channels=32, out_channels=output_building, kernel_size=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, input_opt, input_sar):
        # Encoder processing
        opt_features = self.encoder_opt(input_opt)
        sar_features = self.encoder_sar(input_sar)

        opt_features_1, opt_features_2, opt_features_3, opt_features_4 = opt_features
        sar_features_1, sar_features_2, sar_features_3, sar_features_4 = sar_features

        # Decoder processing - passing encoder outputs to the decoder
        opt_sar_features_1, opt_sar_features_2, opt_sar_features_3, opt_sar_features_4, \
        fusion_feat_1, fusion_feat_2, fusion_feat_3,\
        p1_feat_enhance, p2_feat_enhance, p3_feat_enhance, p4_feat_enhance, \
        p1_feat, p2_feat, p3_feat, p4_feat, \
        output_building, feat_edge = self.decoder_building(sar_features, opt_features)
        # output_building = self.decoder_building(mid_features)
        
        output_building = self.aux_clf(output_building)
        feat_edge = self.aux_clf_edge(feat_edge)

        output_building = F.interpolate(output_building, size=input_opt.size()[-2:], mode='bilinear')
        feat_edge = F.interpolate(feat_edge, size=input_opt.size()[-2:], mode='bilinear')
       
        return opt_features_1, opt_features_2, opt_features_3, opt_features_4, \
            sar_features_1, sar_features_2, sar_features_3, sar_features_4, \
            opt_sar_features_1, opt_sar_features_2, opt_sar_features_3, opt_sar_features_4, \
            fusion_feat_1, fusion_feat_2, fusion_feat_3,\
            p1_feat_enhance, p2_feat_enhance, p3_feat_enhance, p4_feat_enhance, \
            p1_feat, p2_feat, p3_feat, p4_feat, \
            output_building, feat_edge