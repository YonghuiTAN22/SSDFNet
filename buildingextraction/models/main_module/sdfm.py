import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# 基础注意力
# ---------------------------
class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        red = max(1, in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, red, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(red, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# ---------------------------
# 深度可分离卷积 + BN + GELU
# ---------------------------
class DSConv(nn.Module):
    def __init__(self, dim, k, padding):
        super().__init__()
        self.depthwise = nn.Conv2d(dim, dim, kernel_size=k, padding=padding, groups=dim, bias=False)
        self.pointwise = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


# ---------------------------
# 极轻量 edge_map 生成模块
# ---------------------------
class EdgeMapLite(nn.Module):
    def __init__(self, in_channels, hidden_channels=4):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # Sobel-like depthwise 卷积
        sobel_kernel_x = torch.tensor([[[-1,0,1],
                                        [-2,0,2],
                                        [-1,0,1]]], dtype=torch.float32)
        sobel_kernel_y = torch.tensor([[[-1,-2,-1],
                                        [0,0,0],
                                        [1,2,1]]], dtype=torch.float32)

        self.dw_conv_x = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, groups=hidden_channels, bias=False)
        self.dw_conv_y = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, groups=hidden_channels, bias=False)

        self.dw_conv_x.weight.data.copy_(sobel_kernel_x.unsqueeze(0).repeat(hidden_channels,1,1,1))
        self.dw_conv_y.weight.data.copy_(sobel_kernel_y.unsqueeze(0).repeat(hidden_channels,1,1,1))

    def forward(self, x):
        x = self.relu(self.reduce(x))
        gx = self.dw_conv_x(x)
        gy = self.dw_conv_y(x)
        # 绝对值幅值近似
        edge_map = torch.mean(torch.abs(gx) + torch.abs(gy), dim=1, keepdim=True)
        att = torch.sigmoid(edge_map)
        return att


class EdgeAttentionLite(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.edge_map_gen = EdgeMapLite(dim)

    def forward(self, feat):
        att = self.edge_map_gen(feat)
        feat = feat * (1.0 + att)
        return feat, att


# ---------------------------
# 融合模块（轻量多尺度 + 学习权重 + 通道/空间/边界注意力）
# ---------------------------
class FusionConv(nn.Module):
    def __init__(self, in_channels, out_channels, factor=4.0):
        super().__init__()
        dim = max(8, int(out_channels // factor))
        self.down = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1, bias=False)
        self.bn_in = nn.BatchNorm2d(dim)

        # 多尺度深度可分离卷积
        self.conv_3x3 = DSConv(dim, 3, 1)
        self.conv_5x5 = DSConv(dim, 5, 2)
        self.conv_7x7 = DSConv(dim, 7, 3)

        # 学习分支权重
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.w3 = nn.Conv2d(dim, 1, 1, bias=True)
        self.w5 = nn.Conv2d(dim, 1, 1, bias=True)
        self.w7 = nn.Conv2d(dim, 1, 1, bias=True)

        self.ca = ChannelAttentionModule(dim)
        self.sa = SpatialAttentionModule()
        self.edge_att = EdgeAttentionLite(dim)

        self.up = nn.Conv2d(dim, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn_out = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

        self.proj_identity = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x1, x2, x4, return_edge=False):
        x = torch.cat([x1, x2, x4], dim=1)
        identity = self.proj_identity(x)

        x = self.down(x)
        x = self.bn_in(x)

        x3 = self.conv_3x3(x)
        x5 = self.conv_5x5(x)
        x7 = self.conv_7x7(x)

        g3 = self.pool(x3); g5 = self.pool(x5); g7 = self.pool(x7)
        w = torch.cat([self.w3(g3), self.w5(g5), self.w7(g7)], dim=1)
        w = torch.softmax(w, dim=1)
        xs = w[:,0:1]*x3 + w[:,1:2]*x5 + w[:,2:3]*x7

        # 通道注意力 + 边界注意力 + 空间注意力
        xc = xs * self.ca(xs)
        xe, edge_map = self.edge_att(xc)
        xs = xe * (1.0 + self.sa(xe))

        out = self.up(xs)
        out = self.bn_out(out)
        out = self.act(out + identity)

        if return_edge:
            return out, edge_map
        return out


# ---------------------------
# MSAA 主模块
# ---------------------------
class SDFM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fusion_conv = FusionConv(in_channels, out_channels)
        self.final_edge_att = EdgeAttentionLite(out_channels)

    def forward(self, x1, x2, x4, last=False):
        x_fused = self.fusion_conv(x1, x2, x4, return_edge=False)
        if last:
            x_fused, edge_map = self.final_edge_att(x_fused)
            edge_map = edge_map.expand(-1, x_fused.shape[1], -1, -1)
            return x_fused, edge_map
        else:
            return x_fused