from collections import OrderedDict
from functools import partial
from typing import Callable, Optional

import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F


# def bdcn_crop(data1, data2):
#     _, _, h1, w1 = data1.size()
#     _, _, h2, w2 = data2.size()
#     assert(h2 <= h1 and w2 <= w1)
#     data = data1[:, :, (h1-h2):h1, (w1-w2):w1]
#     return data


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    This function is taken from the rwightman.
    It can be seen here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ConvBNAct(nn.Module):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        super(ConvBNAct, self).__init__()

        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU  # alias Swish  (torch>=1.7)

        self.conv = nn.Conv2d(in_channels=in_planes,
                              out_channels=out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=False)

        self.bn = norm_layer(out_planes)
        self.act = activation_layer()

    def forward(self, x):
        result = self.conv(x)
        result = self.bn(result)
        result = self.act(result)

        return result


class SqueezeExcite(nn.Module):
    def __init__(self,
                 input_c: int,   # block input channel
                 expand_c: int,  # block expand channel
                 se_ratio: float = 0.25):
        super(SqueezeExcite, self).__init__()
        squeeze_c = int(input_c * se_ratio)
        self.conv_reduce = nn.Conv2d(expand_c, squeeze_c, 1)
        self.act1 = nn.SiLU()  # alias Swish
        self.conv_expand = nn.Conv2d(squeeze_c, expand_c, 1)
        self.act2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = x.mean((2, 3), keepdim=True)
        scale = self.conv_reduce(scale)
        scale = self.act1(scale)
        scale = self.conv_expand(scale)
        scale = self.act2(scale)
        return scale * x


class MBConv(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 input_c: int,
                 out_c: int,
                 expand_ratio: int,
                 stride: int,
                 se_ratio: float,
                 drop_rate: float,
                 norm_layer: Callable[..., nn.Module]):
        super(MBConv, self).__init__()

        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.has_shortcut = (stride == 1 and input_c == out_c)

        activation_layer = nn.SiLU  # alias Swish
        expanded_c = input_c * expand_ratio

        assert expand_ratio != 1
        # Point-wise expansion
        self.expand_conv = ConvBNAct(input_c,
                                     expanded_c,
                                     kernel_size=1,
                                     norm_layer=norm_layer,
                                     activation_layer=activation_layer)

        # Depth-wise convolution
        self.dwconv = ConvBNAct(expanded_c,
                                expanded_c,
                                kernel_size=kernel_size,
                                stride=stride,
                                groups=expanded_c,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer)

        self.se = SqueezeExcite(input_c, expanded_c, se_ratio) if se_ratio > 0 else nn.Identity()

        # Point-wise linear projection
        self.project_conv = ConvBNAct(expanded_c,
                                      out_planes=out_c,
                                      kernel_size=1,
                                      norm_layer=norm_layer,
                                      activation_layer=nn.Identity)  # 注意这里没有激活函数，所有传入Identity

        self.out_channels = out_c

        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        result = self.expand_conv(x)
        result = self.dwconv(result)
        result = self.se(result)
        result = self.project_conv(result)

        if self.has_shortcut:
            if self.drop_rate > 0:
                result = self.dropout(result)
            result += x

        return result


class FusedMBConv(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 input_c: int,
                 out_c: int,
                 expand_ratio: int,
                 stride: int,
                 se_ratio: float,
                 drop_rate: float,
                 norm_layer: Callable[..., nn.Module]):
        super(FusedMBConv, self).__init__()

        assert stride in [1, 2]
        assert se_ratio == 0

        self.has_shortcut = stride == 1 and input_c == out_c
        self.drop_rate = drop_rate

        self.has_expansion = expand_ratio != 1

        activation_layer = nn.SiLU  # alias Swish
        expanded_c = input_c * expand_ratio

        if self.has_expansion:
            # Expansion convolution
            self.expand_conv = ConvBNAct(input_c,
                                         expanded_c,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         norm_layer=norm_layer,
                                         activation_layer=activation_layer)

            self.project_conv = ConvBNAct(expanded_c,
                                          out_c,
                                          kernel_size=1,
                                          norm_layer=norm_layer,
                                          activation_layer=nn.Identity)  # 注意没有激活函数
        else:
            # 当只有project_conv时的情况
            self.project_conv = ConvBNAct(input_c,
                                          out_c,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          norm_layer=norm_layer,
                                          activation_layer=activation_layer)  # 注意有激活函数

        self.out_channels = out_c

        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        if self.has_expansion:
            result = self.expand_conv(x)
            result = self.project_conv(result)
        else:
            result = self.project_conv(x)

        if self.has_shortcut:
            if self.drop_rate > 0:
                result = self.dropout(result)

            result += x

        return result


class EfficientNetV2(nn.Module):
    def __init__(self,
                 model_cnf: list,
                 drop_connect_rate: float = 0.2):
        super(EfficientNetV2, self).__init__()

        for cnf in model_cnf:
            assert len(cnf) == 8

        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        stem_filter_num = model_cnf[0][4]

        self.stem = ConvBNAct(3,
                              stem_filter_num,
                              kernel_size=3,
                              stride=2,
                              norm_layer=norm_layer)

        total_blocks = sum([i[0] for i in model_cnf])
        block_id = 0
        blocks = []
        for cnf in model_cnf:
            repeats = cnf[0]
            op = FusedMBConv if cnf[-2] == 0 else MBConv
            for i in range(repeats):
                blocks.append(op(kernel_size=cnf[1],
                                 input_c=cnf[4] if i == 0 else cnf[5],
                                 out_c=cnf[5],
                                 expand_ratio=cnf[3],
                                 stride=cnf[2] if i == 0 else 1,
                                 se_ratio=cnf[-1],
                                 drop_rate=drop_connect_rate * block_id / total_blocks,
                                 norm_layer=norm_layer))
                block_id += 1
        self.blocks = nn.Sequential(*blocks)

        self.cat_channel = 24
        self.fusion = fusion()

        # upsample-->(320, 480)
        self.o1_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.o1_up_conv = nn.Conv2d(6 * 24, 1, 1)


        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x_block = self.stem(x)
        feature_maps = []

        # last_x = None
        for idx, block in enumerate(self.blocks):
            x_block = block(x_block)

            if idx == 1 or idx == 5 or idx == 9 or idx == 15 or idx == 24 or idx == 39:
                feature_maps.append(x_block)
        o1, o2, o3, o4, o5, o6 = self.fusion(feature_maps)

        o1 = self.o1_up_conv(self.o1_up(o1))

        return torch.sigmoid(o1)

class fusion(nn.Module):
    def __init__(self, cat_channel=24):
        super(fusion, self).__init__()
        # cat_channel = 24

        # 6
        self.f6_6 = ConvBNAct(256, cat_channel, kernel_size=1)

        self.f5_6_down = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.f5_6_conv = ConvBNAct(160, cat_channel, kernel_size=3)

        self.f4_6_down = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.f4_6_conv = ConvBNAct(128, cat_channel, kernel_size=3)

        self.f3_6_down = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.f3_6_conv = ConvBNAct(64, cat_channel, kernel_size=3)

        self.f2_6_down = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.f2_6_conv = ConvBNAct(48, cat_channel, 3)

        self.f1_6_down = nn.MaxPool2d(16, 16, ceil_mode=True)
        self.f1_6_conv = ConvBNAct(24, cat_channel, kernel_size=3)

        self.o6 = ConvBNAct(6 * cat_channel, 6*cat_channel, kernel_size=3)

        # 5
        # self.o6_5_up = nn.ConvTranspose2d(6 * cat_channel, 6 * cat_channel, kernel_size=4, stride=2, padding=1)
        self.o6_5_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.o6_5_conv = ConvBNAct(6*cat_channel, cat_channel, kernel_size=3)

        self.f5_5_conv = ConvBNAct(160, cat_channel, kernel_size=3)

        self.f4_5_conv = ConvBNAct(128, cat_channel, 3)

        self.f3_5_down = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.f3_5_conv = ConvBNAct(64, cat_channel, 3)

        self.f2_5_down = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.f2_5_conv = ConvBNAct(48, cat_channel, 3)

        self.f1_5_down = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.f1_5_conv = ConvBNAct(24, cat_channel, 3)

        self.o5 = ConvBNAct(6*cat_channel, 6*cat_channel, 3)

        # 4
        # self.o6_4_up = nn.ConvTranspose2d(6 * cat_channel, 6 * cat_channel, kernel_size=4, stride=2, padding=1)
        self.o6_4_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.o6_4_conv = ConvBNAct(6 * cat_channel, cat_channel, kernel_size=3)

        self.o5_4_conv = ConvBNAct(6*cat_channel, cat_channel, 3)

        self.f4_4_conv = ConvBNAct(128, cat_channel, 3)

        self.f3_4_down = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.f3_4_conv = ConvBNAct(64, cat_channel, 3)

        self.f2_4_down = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.f2_4_conv = ConvBNAct(48, cat_channel, 3)

        self.f1_4_down = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.f1_4_conv = ConvBNAct(24, cat_channel, 3)

        self.o4 = ConvBNAct(6*cat_channel, 6*cat_channel, 3)

        # 3
        # self.o6_3_up = nn.ConvTranspose2d(6*cat_channel, 6*cat_channel, kernel_size=8, stride=4, padding=2)
        self.o6_3_up = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.o6_3_conv = ConvBNAct(6*cat_channel, cat_channel, 3)

        # self.o5_3_up = nn.ConvTranspose2d(6*cat_channel, 6*cat_channel, kernel_size=4, stride=2, padding=1)
        self.o5_3_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.o5_3_conv = ConvBNAct(6*cat_channel, cat_channel, 3)

        # self.o4_3_up = nn.ConvTranspose2d(6*cat_channel, 6*cat_channel, kernel_size=4, stride=2, padding=1)
        self.o4_3_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.o4_3_conv = ConvBNAct(6*cat_channel, cat_channel, 3)

        self.f3_3_conv = ConvBNAct(64, cat_channel, 3)

        self.f2_3_down = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.f2_3_conv = ConvBNAct(48, cat_channel, 3)

        self.f1_3_down = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.f1_3_conv = ConvBNAct(24, cat_channel, 3)

        self.o3 = ConvBNAct(6*cat_channel, 6*cat_channel, 3)

        # 2
        # self.o6_2_up = nn.ConvTranspose2d(6 * cat_channel, 6 * cat_channel, kernel_size=16, stride=8, padding=4)
        self.o6_2_up = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False)
        self.o6_2_conv = ConvBNAct(6 * cat_channel, cat_channel, 3)

        # self.o5_2_up = nn.ConvTranspose2d(6 * cat_channel, 6 * cat_channel, kernel_size=8, stride=4, padding=2)
        self.o5_2_up = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.o5_2_conv = ConvBNAct(6 * cat_channel, cat_channel, 3)

        # self.o4_2_up = nn.ConvTranspose2d(6 * cat_channel, 6 * cat_channel, kernel_size=8, stride=4, padding=2)
        self.o4_2_up = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.o4_2_conv = ConvBNAct(6 * cat_channel, cat_channel, 3)

        # self.o3_2_up = nn.ConvTranspose2d(6*cat_channel, 6*cat_channel, kernel_size=4, stride=2, padding=1)
        self.o3_2_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.o3_2_conv = ConvBNAct(6*cat_channel, cat_channel, 3)

        self.f2_2_conv = ConvBNAct(48, cat_channel, 3)

        self.f1_2_down = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.f1_2_conv = ConvBNAct(24, cat_channel, 3)

        self.o2 = ConvBNAct(6 * cat_channel, 6 * cat_channel, 3)

        # 1
        # self.o6_1_up = nn.ConvTranspose2d(6 * cat_channel, 6 * cat_channel, kernel_size=16, stride=16)
        self.o6_1_up = nn.Upsample(scale_factor=16, mode="bilinear", align_corners=False)
        self.o6_1_conv = ConvBNAct(6 * cat_channel, cat_channel, 3)

        # self.o5_1_up = nn.ConvTranspose2d(6 * cat_channel, 6 * cat_channel, kernel_size=16, stride=8, padding=4)
        self.o5_1_up = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False)
        self.o5_1_conv = ConvBNAct(6 * cat_channel, cat_channel, 3)

        # self.o4_1_up = nn.ConvTranspose2d(6 * cat_channel, 6 * cat_channel, kernel_size=16, stride=8, padding=4)
        self.o4_1_up = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False)
        self.o4_1_conv = ConvBNAct(6 * cat_channel, cat_channel, 3)

        # self.o3_1_up = nn.ConvTranspose2d(6 * cat_channel, 6 * cat_channel, kernel_size=8, stride=4, padding=2)
        self.o3_1_up = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.o3_1_conv = ConvBNAct(6 * cat_channel, cat_channel, 3)

        # self.o2_1_up = nn.ConvTranspose2d(6*cat_channel, 6*cat_channel, kernel_size=4, stride=2, padding=1)
        self.o2_1_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.o2_1_conv = ConvBNAct(6*cat_channel, cat_channel, 3)

        self.f1_1_conv = ConvBNAct(24, cat_channel, 3)

        self.o1 = ConvBNAct(6 * cat_channel, 6 * cat_channel, 3)

    def forward(self, features_map):
        p1, p2, p3, p4, p5, p6 = features_map
        # 6
        f6_6 = self.f6_6(p6)
        f5_6 = self.f5_6_conv(self.f5_6_down(p5))
        f4_6 = self.f4_6_conv(self.f4_6_down(p4))
        f3_6 = self.f3_6_conv(self.f3_6_down(p3))
        f2_6 = self.f2_6_conv(self.f2_6_down(p2))
        f1_6 = self.f1_6_conv(self.f1_6_down(p1))
        o6 = self.o6(torch.cat([f6_6, f5_6, f4_6, f3_6, f2_6, f1_6], dim=1))

        # 5
        f6_5 = self.o6_5_conv(self.o6_5_up(o6))
        f5_5 = self.f5_5_conv(p5)
        f4_5 = self.f4_5_conv(p4)
        f3_5 = self.f3_5_conv(self.f3_5_down(p3))
        f2_5 = self.f2_5_conv(self.f2_5_down(p2))
        f1_5 = self.f1_5_conv(self.f1_5_down(p1))
        o5 = self.o5(torch.cat([f6_5, f5_5, f4_5, f3_5, f2_5, f1_5], dim=1))

        # 4

        f6_4 = self.o6_4_conv(self.o6_4_up(o6))
        f5_4 = self.o5_4_conv(o5)
        f4_4 = self.f4_4_conv(p4)
        f3_4 = self.f3_4_conv(self.f3_4_down(p3))
        f2_4 = self.f2_4_conv(self.f2_4_down(p2))
        f1_4 = self.f1_4_conv(self.f1_4_down(p1))
        o4 = self.o4(torch.cat([f6_4, f5_4, f4_4, f3_4, f2_4, f1_4], dim=1))

        # 3
        f6_3 = self.o6_3_conv(self.o6_3_up(o6))
        f5_3 = self.o5_3_conv(self.o5_3_up(o5))
        f4_3 = self.o4_3_conv(self.o4_3_up(o4))
        f3_3 = self.f3_3_conv(p3)
        f2_3 = self.f2_3_conv(self.f2_3_down(p2))
        f1_3 = self.f1_3_conv(self.f1_3_down(p1))
        o3 = self.o3(torch.cat([f6_3, f5_3, f4_3, f3_3, f2_3, f1_3], dim=1))

        # 2
        f6_2 = self.o6_2_conv(self.o6_2_up(o6))
        f5_2 = self.o5_2_conv(self.o5_2_up(o5))
        f4_2 = self.o4_2_conv(self.o4_2_up(o4))
        f3_2 = self.o3_2_conv(self.o3_2_up(o3))
        f2_2 = self.f2_2_conv(p2)
        f1_2 = self.f1_2_conv(self.f1_2_down(p1))
        o2 = self.o2(torch.cat([f6_2, f5_2, f4_2, f3_2, f2_2, f1_2], dim=1))

        # 1
        f6_1 = self.o6_2_conv(self.o6_1_up(o6))
        f5_1 = self.o5_1_conv(self.o5_1_up(o5))
        f4_1 = self.o4_1_conv(self.o4_1_up(o4))
        f3_1 = self.o3_1_conv(self.o3_1_up(o3))
        f2_1 = self.o2_1_conv(self.o2_1_up(o2))
        f1_1 = self.f1_1_conv(p1)
        o1 = self.o3(torch.cat([f6_1, f5_1, f4_1, f3_1, f2_1, f1_1], dim=1))

        return [o1, o2, o3, o4, o5, o6]


def efficientnetv2_s(num_classes: int = 1000):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 300, eval_size: 384

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[2, 3, 1, 1, 24, 24, 0, 0],
                    [4, 3, 2, 4, 24, 48, 0, 0],
                    [4, 3, 2, 4, 48, 64, 0, 0],
                    [6, 3, 2, 4, 64, 128, 1, 0.25],
                    [9, 3, 1, 6, 128, 160, 1, 0.25],
                    [15, 3, 2, 6, 160, 256, 1, 0.25]]

    model = EfficientNetV2(model_cnf=model_config,
                           )
    return model


def efficientnetv2_m(num_classes: int = 1000):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 384, eval_size: 480

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[3, 3, 1, 1, 24, 24, 0, 0],
                    [5, 3, 2, 4, 24, 48, 0, 0],
                    [5, 3, 2, 4, 48, 80, 0, 0],
                    [7, 3, 2, 4, 80, 160, 1, 0.25],
                    [14, 3, 1, 6, 160, 176, 1, 0.25],
                    [18, 3, 2, 6, 176, 304, 1, 0.25],
                    [5, 3, 1, 6, 304, 512, 1, 0.25]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.3)
    return model


def efficientnetv2_l(num_classes: int = 1000):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 384, eval_size: 480

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[4, 3, 1, 1, 32, 32, 0, 0],
                    [7, 3, 2, 4, 32, 64, 0, 0],
                    [7, 3, 2, 4, 64, 96, 0, 0],
                    [10, 3, 2, 4, 96, 192, 1, 0.25],
                    [19, 3, 1, 6, 192, 224, 1, 0.25],
                    [25, 3, 2, 6, 224, 384, 1, 0.25],
                    [7, 3, 1, 6, 384, 640, 1, 0.25]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.4)
    return model

if __name__ == '__main__':
    batch_size = 1
    img_height = 320
    img_width = 480

    device = "cuda:5" if torch.cuda.is_available() else "cpu"
    input = torch.rand(batch_size, 3, img_height, img_width).to(device)
    # target = torch.rand(batch_size, 1, img_height, img_width).to(device)
    print(f"input shape: {input.shape}")
    model = efficientnetv2_s().to(device)

    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model, (3, 320, 480), as_strings=True, print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    params = model.state_dict()

    output = model(input)

    print(f"output shape: {output.shape}")