import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

class CrossAttention(nn.Module):
    """
    互跨注意力模块，用于光学和SAR特征的交互融合。
    
    主要功能：
    - x1_out = softmax(Q1 @ K2^T / scale) @ V2  (光学关注SAR)
    - x2_out = softmax(Q2 @ K1^T / scale) @ V1  (SAR关注光学)
    
    参数：
    - dim: 输入/输出维度
    - num_heads: 注意力头数
    - qkv_bias: 是否使用偏置
    - qk_scale: 缩放因子
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        # Q投影
        self.q_proj1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_proj2 = nn.Linear(dim, dim, bias=qkv_bias)
        
        # KV投影
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x1, x2):
        B, N, C = x1.shape  # 假设x2.shape == [B, N, C]
        
        # Q1, Q2: [B, N, C] -> [B, h, N, d]
        q1 = self.q_proj1(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q2 = self.q_proj2(x2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # K1, V1 from x1
        kv1 = self.kv1(x1).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k1, v1 = kv1[0], kv1[1]
        
        # K2, V2 from x2
        kv2 = self.kv2(x2).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k2, v2 = kv2[0], kv2[1]
        
        # 互跨注意力
        attn12 = (q1 @ k2.transpose(-2, -1)) * self.scale
        attn12 = attn12.softmax(dim=-1)
        
        attn21 = (q2 @ k1.transpose(-2, -1)) * self.scale
        attn21 = attn21.softmax(dim=-1)
        
        # 输出
        x1_out = (attn12 @ v2).transpose(1, 2).reshape(B, N, C).contiguous()
        x2_out = (attn21 @ v1).transpose(1, 2).reshape(B, N, C).contiguous()
        
        return x1_out, x2_out

class DSFM(nn.Module):
    """
    完整的融合光学影像和SAR影像在下采样过程中的特征模块。
    
    整合了CrossAttention：
    - 提取特征后，下采样。
    - 展平下采样特征为序列，应用CrossAttention交互。
    - 重塑回2D，融合（加法 + 1x1 conv）。
    
    考虑差异：
    - SAR: Log变换去噪。
    - 下采样: 光学AvgPool, SAR MaxPool。
    
    输入：optical [B, 3, H, W], sar [B, 1, H, W]
    输出：[B, out_channels, H//2, W//2]
    """
    def __init__(self, in_channels_opt=3, in_channels_sar=1, out_channels=64, 
                 kernel_size=3, stride=1, padding=1, downsample_factor=2, num_heads=8):
        super(DSFM, self).__init__()
        
        feat_channels = out_channels // 2
        
        # 光学分支
        self.opt_conv = nn.Conv2d(in_channels_opt, feat_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.opt_bn = nn.BatchNorm2d(feat_channels)
        
        # SAR分支
        self.sar_conv = nn.Conv2d(in_channels_sar, feat_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.sar_bn = nn.BatchNorm2d(feat_channels)
        
        # 模态特定下采样
        self.opt_down = nn.AvgPool2d(kernel_size=downsample_factor, stride=downsample_factor)
        self.sar_down = nn.MaxPool2d(kernel_size=downsample_factor, stride=downsample_factor)
        
        # CrossAttention (dim = feat_channels)
        self.cross_attn = CrossAttention(dim=feat_channels, num_heads=num_heads)
        
        # 融合: 加法后1x1 conv
        self.fusion_conv = nn.Conv2d(feat_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.fusion_bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, optical, sar):
        B, _, H, W = optical.shape
        down_H, down_W = H // 2, W // 2  # 假设downsample_factor=2
        
        # 光学特征
        opt_feat = F.relu(self.opt_bn(self.opt_conv(optical)))  # [B, 32, H, W]
        opt_down = self.opt_down(opt_feat)  # [B, 32, H//2, W//2]
        
        # SAR特征 + Log
        sar_feat = F.relu(self.sar_bn(self.sar_conv(sar)))  # [B, 32, H, W]
        sar_feat = torch.log(1 + sar_feat.clamp(min=0))  # 去噪
        sar_down = self.sar_down(sar_feat)  # [B, 32, H//2, W//2]
        
        # 展平为序列 [B, down_H*down_W, feat_channels]
        N = down_H * down_W
        opt_flat = opt_down.flatten(2).transpose(1, 2)  # [B, N, 32]
        sar_flat = sar_down.flatten(2).transpose(1, 2)  # [B, N, 32]
        
        # CrossAttention
        # opt_enh_flat, sar_enh_flat = self.cross_attn(opt_flat, sar_flat)
        
        # 重塑回2D [B, 32, H//2, W//2]
        opt_enh = opt_flat.transpose(1, 2).reshape(B, -1, down_H, down_W)
        sar_enh = sar_flat.transpose(1, 2).reshape(B, -1, down_H, down_W)
        
        # 融合: 加法
        fused = opt_enh + sar_enh  # [B, 32, H//2, W//2]
        
        # 1x1 conv 扩展通道
        fused = F.relu(self.fusion_bn(self.fusion_conv(fused)))  # 临时扩展以匹配out_channels
        
        return fused
    

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def profile_one_stage(name, in_c, h, w, out_c=64, num_heads=8, batch_size=1, device="cpu"):
    model = DSFM(
        in_channels_opt=in_c,
        in_channels_sar=in_c,
        out_channels=out_c,
        num_heads=num_heads
    ).to(device).eval()

    x1 = torch.randn(batch_size, in_c, h, w).to(device)
    x2 = torch.randn(batch_size, in_c, h, w).to(device)

    macs, params = profile(model, inputs=(x1, x2), verbose=False)
    flops = macs * 2  # thop 返回 MACs，通常 FLOPs = 2 * MACs

    print(f"{name}:")
    print(f"  input shape each = [{batch_size}, {in_c}, {h}, {w}]")
    print(f"  Params = {params:,} ({params / 1e6:.6f} M)")
    print(f"  MACs   = {macs:,} ({macs / 1e9:.6f} GMac)")
    print(f"  FLOPs  = {flops:,} ({flops / 1e9:.6f} G)")
    print()

    return params, flops


if __name__ == "__main__":
    # ===== 这里改成你的 VMamba-Base 四个 stage 特征尺寸 =====
    # 例子而已，请按你自己的 backbone 输出修改
    stage_shapes = [
        ("stage1", 128, 128, 128),
        ("stage2", 256, 64, 64),
        ("stage3", 512, 32, 32),
        ("stage4", 1024, 16, 16),
    ]

    total_params = 0
    total_flops = 0

    for name, c, h, w in stage_shapes:
        params, flops = profile_one_stage(name, c, h, w, out_c=64, num_heads=8)
        total_params += params
        total_flops += flops

    print("Total over 4 stages:")
    print(f"Total Params = {total_params:,} ({total_params / 1e6:.6f} M)")
    print(f"Total FLOPs  = {total_flops:,} ({total_flops / 1e9:.6f} G)")
    print(f"  GFLOPs = {total_flops / 1e9:.6f}")