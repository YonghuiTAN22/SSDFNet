import sys
sys.path.append('/public/home/tanyh_25/project/Mamba/ChangeMamba')

from classification.models.vmamba import VSSM, LayerNorm2d

import torch
import torch.nn as nn


class Backbone_VSSM(VSSM):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer='ln2d', **kwargs):
        # norm_layer='ln'
        kwargs.update(norm_layer=norm_layer)
        super().__init__(**kwargs)
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)        
        
        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.classifier
        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return
        
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)        
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, x):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y

        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x) # (B, H, W, C)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                if not self.channel_first:
                    out = out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        if len(self.out_indices) == 0:
            return x
        
        return outs


try:
    # 若你已有这个类，直接复用（与 VSSM 工程一致的接口/行为）
    from classification.models.vmamba import LayerNorm2d
except Exception:
    # 兜底实现：在 NCHW 上做 LayerNorm（对 C 归一化）
    class LayerNorm2d(nn.Module):
        def __init__(self, num_channels, eps=1e-6, affine=True):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(num_channels)) if affine else None
            self.bias = nn.Parameter(torch.zeros(num_channels)) if affine else None

        def forward(self, x):  # x: [N,C,H,W]
            mean = x.mean(dim=(2, 3), keepdim=True)
            var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
            x_hat = (x - mean) / torch.sqrt(var + self.eps)
            if self.weight is not None:
                x_hat = x_hat * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
            return x_hat
        

class Backbone_ResNet50(nn.Module):
    """
    一个与 Backbone_VSSM 风格匹配的 ResNet-50 骨干：
    - out_indices: 选择输出的 stage（0/1/2/3 -> C2/C3/C4/C5）
    - norm_layer: 'bn' | 'ln2d' | 'ln'（仅用于输出特征归一化，不改变 ResNet 内部 BN）
    - out_channels: None / int / [c2, c3, c4, c5]，用于对齐每个输出的通道数（1x1 Conv）
    - pretrained: None / "imagenet" / "torchvision" / <ckpt_path>
    - replace_stride_with_dilation: 与 torchvision.resnet 一致，用于控制输出步幅（支持空洞）
    """
    def __init__(
        self,
        out_indices=(0, 1, 2, 3),
        pretrained=None,
        norm_layer='bn',
        out_channels=None,
        in_channels=3,
        replace_stride_with_dilation=(False, False, False),  # 对应 layer2/3/4
        frozen_stages=-1,  # >=0 冻结到指定 stage（含），-1 不冻结
    ):
        super().__init__()
        self.out_indices = tuple(out_indices)
        self.norm_kind = norm_layer.lower()
        self.channel_first = (self.norm_kind in ["bn", "ln2d"])

        # ------------- 构建 ResNet-50 主干 -------------
        try:
            from torchvision.models import resnet50, ResNet50_Weights
            # 内部仍使用 BN；输出归一化由我们单独控制
            if pretrained in ("imagenet", "torchvision", True):
                self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2, replace_stride_with_dilation=replace_stride_with_dilation)
            else:
                self.backbone = resnet50(weights=None, replace_stride_with_dilation=replace_stride_with_dilation)
        except Exception:
            raise RuntimeError(
                "需要 torchvision 才能直接实例化 ResNet-50。"
                "若你的环境不便安装，请改为载入你已有的 ResNet 定义，并把 layer1~4 接到下方 forward。"
            )

        # 替换输入通道（若不等于3）
        if in_channels != 3:
            old_conv1 = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(in_channels, old_conv1.out_channels,
                                            kernel_size=old_conv1.kernel_size,
                                            stride=old_conv1.stride,
                                            padding=old_conv1.padding,
                                            bias=old_conv1.bias is not None)

        # 删除分类头
        self.backbone.fc = nn.Identity()
        self.backbone.avgpool = nn.Identity()

        # ResNet-50 四个阶段输出通道（C2~C5）
        self.stage_dims = [256, 512, 1024, 2048]

        # ------------- 处理通道对齐配置 -------------
        # 规范化 out_channels 为长度4的列表（对应 C2~C5）
        if out_channels is None:
            self.out_dims = self.stage_dims[:]  # 不做对齐
        elif isinstance(out_channels, int):
            self.out_dims = [out_channels] * 4
        else:
            assert len(out_channels) == 4, "out_channels 若为序列，长度必须为4（对应 C2~C5）"
            self.out_dims = list(out_channels)

        # >>> 关键：对外暴露 VSSM 风格的 dims 属性（用于下游模块读取通道数）
        self.dims = tuple(self.out_dims)

        # 对每个 stage，如需要则加 1x1 conv 做通道转换
        for i in range(4):
            in_c = self.stage_dims[i]
            out_c = self.out_dims[i]
            if in_c != out_c:
                proj = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
                nn.init.kaiming_normal_(proj.weight, mode='fan_out', nonlinearity='relu')
                self.add_module(f'proj{i}', proj)
            else:
                self.add_module(f'proj{i}', nn.Identity())

        # ------------- 输出归一化层（与 VSSM 风格一致）-------------
        _NORMS = dict(
            bn=nn.BatchNorm2d,
            ln2d=LayerNorm2d,   # 我们的 LN2d 是 NCHW 的 LayerNorm(C)
            ln=nn.LayerNorm,    # 注意：nn.LayerNorm 期望最后一个维度为 C，所以 forward 里会临时转为 NHWC
        )
        _norm_ctor = _NORMS.get(self.norm_kind, None)
        if _norm_ctor is None:
            raise ValueError(f"Unsupported norm_layer: {norm_layer}. Use 'bn'|'ln2d'|'ln'.")

        for i in self.out_indices:
            dim = self.out_dims[i]
            layer = _norm_ctor(dim)
            self.add_module(f'outnorm{i}', layer)

        # ------------- 预训练权重载入 -------------
        self.load_pretrained(pretrained)

        # ------------- 冻结到指定 stage -------------
        if frozen_stages >= 0:
            self._freeze_stages(frozen_stages)

    # 与 VSSM 风格相仿：支持从文件 ckpt 里读
    def load_pretrained(self, pretrained=None, key="model"):
        if pretrained is None or pretrained in ("imagenet", "torchvision", True):
            # 已在构造时用 torchvision 的 weights 载入；此处无需重复
            return
        try:
            # 允许直接传本地 ckpt 路径：结构应与 torchvision.resnet50.state_dict() 一致，或包含 key="model"
            sd = torch.load(pretrained, map_location='cpu')
            if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
                sd = sd[key]
            incompatible = self.backbone.load_state_dict(sd, strict=False)
            print(f"[Backbone_ResNet50] Loaded ckpt from {pretrained}")
            print(incompatible)
        except Exception as e:
            print(f"[Backbone_ResNet50] Failed to load ckpt {pretrained}: {e}")

    def _freeze_stages(self, frozen_stages:int):
        # -1: 不冻结；0: 冻结 stem；1: 冻结 stem+layer1；... 4: 冻结到 layer4
        # 冻结 stem
        if frozen_stages >= 0:
            for m in [self.backbone.conv1, self.backbone.bn1]:
                for p in m.parameters():
                    p.requires_grad = False
        # 冻结 layer1~4
        for k in range(1, min(frozen_stages, 4) + 1):
            layer = getattr(self.backbone, f'layer{k}')
            for p in layer.parameters():
                p.requires_grad = False

    def _apply_outnorm(self, x, idx:int):
        # x: NCHW
        norm = getattr(self, f'outnorm{idx}')
        if self.norm_kind == 'ln':
            # nn.LayerNorm 期望最后一维为 C -> 临时转 NHWC
            x = x.permute(0, 2, 3, 1).contiguous()
            x = norm(x)  # 归一化通道
            x = x.permute(0, 3, 1, 2).contiguous()
            return x
        else:
            return norm(x)

    def forward(self, x):
        # ---- stem ----
        x = self.backbone.conv1(x)   # /2
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x) # /4

        outs = []
        feats = []

        # ---- layer1..4 -> C2..C5 ----
        c2 = self.backbone.layer1(x)  # stride 不变: /4, C=256
        c3 = self.backbone.layer2(c2) # /8,  C=512 （若 replace_stride_with_dilation[0]=True -> 空洞保持 /4）
        c4 = self.backbone.layer3(c3) # /16, C=1024（同理可能 /8）
        c5 = self.backbone.layer4(c4) # /32, C=2048（同理可能 /16）

        feats = [c2, c3, c4, c5]  # 索引 0..3

        for i, f in enumerate(feats):
            if i in self.out_indices:
                # 1x1 对齐通道
                proj = getattr(self, f'proj{i}')
                f = proj(f)
                # 输出归一化（与 VSSM 风格一致）
                f = self._apply_outnorm(f, i)
                # 与 VSSM 一样，最终统一输出 BCHW
                outs.append(f)

        if len(self.out_indices) == 0:
            # 若用户不想要任何中间输出，则返回最后一个特征
            return c5

        return outs
    

try:
    from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights
except Exception:
    efficientnet_b5, EfficientNet_B5_Weights = None, None


class Backbone_EfficientNetB5(nn.Module):
    """
    与 Backbone_ResNet50 风格匹配的 EfficientNet-B5 骨干：
    - out_indices: 选择输出的 stage（0/1/2/3 -> C2/C3/C4/C5）
    - norm_layer: 'bn' | 'ln2d' | 'ln'（仅用于输出特征归一化，不改变主干内部 BN）
    - out_channels: None / int / [c2, c3, c4, c5]，用于对齐每个输出的通道数（1x1 Conv）
    - pretrained: None / "imagenet" / "torchvision" / <ckpt_path>
    - in_channels: 输入通道数，≠3 时会重建 stem conv
    - frozen_stages: >=0 冻结到指定 stage（含），-1 不冻结；0: stem；1..4: C2..C5
    """
    def __init__(
        self,
        out_indices=(0, 1, 2, 3),
        pretrained=None,
        norm_layer='bn',
        out_channels=None,
        in_channels=3,
        frozen_stages=-1,
    ):
        super().__init__()
        if efficientnet_b5 is None:
            raise RuntimeError(
                "需要 torchvision 才能实例化 EfficientNet-B5。"
                "请安装 torchvision>=0.13 并确保含 efficientnet_b5 接口。"
            )

        self.out_indices = tuple(out_indices)
        self.norm_kind = norm_layer.lower()
        self.channel_first = (self.norm_kind in ["bn", "ln2d"])

        # ---------- 构建 EfficientNet-B5 主干 ----------
        if pretrained in ("imagenet", "torchvision", True):
            self.backbone = efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
        else:
            self.backbone = efficientnet_b5(weights=None)

        # 替换分类头为 Identity（与 ResNet 包装保持一致）
        self.backbone.classifier = nn.Identity()

        # 替换输入通道（stem）
        if in_channels != 3:
            stem = self.backbone.features[0]      # ConvNormActivation
            old_conv = stem[0]                    # nn.Conv2d
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                dilation=old_conv.dilation,
                groups=old_conv.groups,
                bias=(old_conv.bias is not None),
                padding_mode=old_conv.padding_mode,
            )
            self.backbone.features[0][0] = new_conv

        # ---------- 自动推断 C2..C5 的原生通道数 & stage 边界 ----------
        native_stage_dims, stage_end_idx = self._infer_stage_info(in_channels)
        self._stage_end_idx = stage_end_idx  # 用于冻结

        # ---------- 处理通道对齐配置 ----------
        if out_channels is None:
            self.out_dims = list(native_stage_dims)
        elif isinstance(out_channels, int):
            self.out_dims = [out_channels] * 4
        else:
            assert len(out_channels) == 4, "out_channels 若为序列，长度必须为4（对应 C2~C5）"
            self.out_dims = list(out_channels)

        # 对外暴露 VSSM 风格的 dims
        self.dims = tuple(self.out_dims)

        # 为每个 stage 构建 1x1 投影
        for i in range(4):
            in_c = native_stage_dims[i]
            out_c = self.out_dims[i]
            if in_c != out_c:
                proj = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
                nn.init.kaiming_normal_(proj.weight, mode='fan_out', nonlinearity='relu')
            else:
                proj = nn.Identity()
            self.add_module(f'proj{i}', proj)

        # ---------- 输出归一化（与 VSSM 风格一致） ----------
        _NORMS = dict(
            bn=nn.BatchNorm2d,
            ln2d=LayerNorm2d,   # 下方提供兜底实现
            ln=nn.LayerNorm,    # 注意：nn.LayerNorm 作用于最后一维，forward 里会临时转 NHWC
        )
        _norm_ctor = _NORMS.get(self.norm_kind, None)
        if _norm_ctor is None:
            raise ValueError(f"Unsupported norm_layer: {norm_layer}. Use 'bn'|'ln2d'|'ln'.")

        for i in self.out_indices:
            dim = self.out_dims[i]
            self.add_module(f'outnorm{i}', _norm_ctor(dim))

        # ---------- 额外 ckpt 载入 ----------
        self._load_pretrained_file(pretrained)

        # ---------- 冻结到指定 stage ----------
        if frozen_stages >= 0:
            self._freeze_stages(frozen_stages)

    # ========== 辅助：仅加载外部 ckpt ==========
    def _load_pretrained_file(self, pretrained=None, key="model"):
        if pretrained is None or pretrained in ("imagenet", "torchvision", True):
            return
        try:
            sd = torch.load(pretrained, map_location='cpu')
            if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
                sd = sd[key]
            incompatible = self.backbone.load_state_dict(sd, strict=False)
            print(f"[Backbone_EfficientNetB5] Loaded ckpt from {pretrained}")
            print(incompatible)
        except Exception as e:
            print(f"[Backbone_EfficientNetB5] Failed to load ckpt {pretrained}: {e}")

    # ========== 辅助：在 init 时推断 stage 通道与边界 ==========
    @torch.no_grad()
    def _infer_stage_info(self, in_channels, h=256, w=256):
        """
        返回：
          - native_stage_dims: [c2, c3, c4, c5]
          - stage_end_idx: features 中每个 stage 结束所在的模块索引（用于冻结）
        """
        was_training = self.backbone.training
        self.backbone.eval()

        device = next(self.backbone.parameters()).device
        x = torch.zeros(1, in_channels, h, w, device=device)

        feats = self._forward_stages_raw(x)  # [C2..C5], raw（未投影/未归一化）
        native_stage_dims = [f.shape[1] for f in feats]

        # 同步得到 stage 边界索引（C2..C5 结束模块索引）
        stage_end_idx = self._get_stage_end_indices(h, w, in_channels)

        if was_training:
            self.backbone.train()
        return native_stage_dims, stage_end_idx

    @torch.no_grad()
    def _get_stage_end_indices(self, h, w, in_channels):
        """
        通过一次遍历 features，记录分辨率变化点，从而得到 C2..C5 的结束位置索引。
        """
        was_training = self.backbone.training
        self.backbone.eval()

        device = next(self.backbone.parameters()).device
        x = torch.zeros(1, in_channels, h, w, device=device)

        features = self.backbone.features
        x = features[0](x)  # stem: /2
        prev = x
        factor = 2  # 当前下采样倍数（相对输入）

        end_idx = []
        last = x
        # 遍历主干余下模块，定位在 factor==4/8/16 的“离开该分辨率”时刻的上一个输出
        for i in range(1, len(features)):
            y = features[i](prev)
            # 分辨率变化 -> 说明上一段结束
            if y.shape[-2:] != prev.shape[-2:]:
                if factor in (4, 8, 16):
                    end_idx.append(i - 1)  # 上一个模块是该 stage 的最后一个
                factor *= 2
            last = y
            prev = y
        # 末段（/32）以最后一个模块结束
        end_idx.append(len(features) - 1)

        if was_training:
            self.backbone.train()
        return end_idx  # 长度为4

    @torch.no_grad()
    def _forward_stages_raw(self, x):
        """
        仅用于推断/工具：返回模型内部原生的 C2..C5（未投影/未归一化）。
        逻辑：跟踪分辨率变化，在 factor==4/8/16 的“下一次下采样发生瞬间”收集上一分辨率的最后特征；
             结束时收集 /32 的最后特征。
        """
        features = self.backbone.features

        x = features[0](x)  # stem: /2
        prev = x
        last = x
        factor = 2
        outs = []  # 将得到 [C2, C3, C4]，最后补上 C5

        for i in range(1, len(features)):
            y = features[i](prev)
            if y.shape[-2:] != prev.shape[-2:]:
                # 刚刚离开 factor 分辨率
                if factor in (4, 8, 16):
                    outs.append(last)  # 收集 C2/C3/C4
                factor *= 2
            last = y
            prev = y

        outs.append(last)  # C5 (/32)
        assert len(outs) == 4, f"期望得到4个stage输出，实际得到{len(outs)}。"
        return outs

    # ========== 冻结 ==========
    def _freeze_stages(self, frozen_stages: int):
        # 0: 冻结 stem
        if frozen_stages >= 0:
            stem = self.backbone.features[0]
            for p in stem.parameters():
                p.requires_grad = False

        # 1..4: 冻结到 C2..C5
        if frozen_stages > 0:
            end_indices = self._stage_end_idx  # [C2_end, C3_end, C4_end, C5_end]
            # 冻结 features[1 .. end_indices[k-1]]（k=1..4）
            end_idx = end_indices[min(frozen_stages, 4) - 1]
            for i in range(1, end_idx + 1):
                for p in self.backbone.features[i].parameters():
                    p.requires_grad = False

    # ========== 归一化辅助 ==========
    def _apply_outnorm(self, x, idx: int):
        # x: NCHW
        norm = getattr(self, f'outnorm{idx}')
        if self.norm_kind == 'ln':
            x = x.permute(0, 2, 3, 1).contiguous()
            x = norm(x)
            x = x.permute(0, 3, 1, 2).contiguous()
            return x
        else:
            return norm(x)

    # ========== 正式前向 ==========
    def forward(self, x):
        """
        返回与 VSSM 风格一致的输出：
        - 若 len(out_indices)>0 -> List[Tensor]，按索引顺序返回指定 stage 的 BCHW 特征
        - 若 len(out_indices)==0 -> 返回最后一个特征（C5）
        """
        # 先拿到原生 C2..C5
        raw_feats = self._forward_stages_raw(x)  # [C2, C3, C4, C5] (N,C,H,W)

        outs = []
        for i, f in enumerate(raw_feats):
            if i in self.out_indices:
                proj = getattr(self, f'proj{i}')
                f = proj(f)                 # 1x1 对齐通道
                f = self._apply_outnorm(f, i)  # 输出归一化
                outs.append(f)

        if len(self.out_indices) == 0:
            # 返回最后一个特征（不做投影/归一化）
            return raw_feats[-1]
        return outs


# 兜底实现：在 NCHW 上做 LayerNorm（对 C 归一化）
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels)) if affine else None
        self.bias = nn.Parameter(torch.zeros(num_channels)) if affine else None

    def forward(self, x):  # x: [N,C,H,W]
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        if self.weight is not None:
            x_hat = x_hat * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x_hat