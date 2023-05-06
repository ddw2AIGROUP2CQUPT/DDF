import torch
import torch.nn as nn
import torchvision.models as models
from typing import Callable, Optional



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
            activation_layer = nn.ReLU  # alias Swish  (torch>=1.7)

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


class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()

        # pretrained_model = models.vgg16_bn(pretrained=True)
        pretrained_model = models.vgg16_bn(pretrained=False)
        pre = torch.load(r'../vgg16_bn-6c64b313.pth')
        pretrained_model.load_state_dict(pre)
        # hard code copying weights from vgg16_bn pretrained model

        self.feature1 = nn.Sequential(
            *list(pretrained_model.features.children())[:6])
        self.feature2 = nn.Sequential(
            *list(pretrained_model.features.children())[6:13])
        self.feature3 = nn.Sequential(
            *list(pretrained_model.features.children())[13:23])
        self.feature4 = nn.Sequential(
            *list(pretrained_model.features.children())[23:33])
        self.feature5 = nn.Sequential(
            *list(pretrained_model.features.children())[33:43])


        # # self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up_5 = nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1)
        # self.up_4 = nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1)
        # self.up_3 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1)
        # self.up_2 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)
        # self.bn = nn.BatchNorm2d(1)
        self.fusion = fusion()
        self.down_channel = nn.Conv2d(5 * 64, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.gen_final_feat = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)
        # self.gen_final_feat.bias.data.fill_(2)

    def _make_layer(self, block, planes, block_nums):
        layers = []

        for i in range(0, block_nums):
            layers.append(block(planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # get the feats from vgg pretrained model
        x1 = self.feature1(x)
        x2 = self.feature2(x1)
        x3 = self.feature3(x2)
        x4 = self.feature4(x3)
        x5 = self.feature5(x4)

        feature_maps = [x1, x2, x3, x4, x5]

        o = self.fusion(feature_maps)

        final_feat = self.gen_final_feat(self.down_channel(o))

        return torch.sigmoid(final_feat)


class fusion(nn.Module):
    def __init__(self, cat_channel=64):
        super(fusion, self).__init__()
        # cat_channel = 24

        # 5
        # self.o6_5_up = nn.ConvTranspose2d(6 * cat_channel, 6 * cat_channel, kernel_size=4, stride=2, padding=1)
        # self.o6_5_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        # self.o6_5_conv = ConvBNAct(6*cat_channel, cat_channel, kernel_size=3)

        self.f5_5_conv = ConvBNAct(512, cat_channel, kernel_size=3)

        self.f4_5_down = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.f4_5_conv = ConvBNAct(512, cat_channel, 3)

        self.f3_5_down = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.f3_5_conv = ConvBNAct(256, cat_channel, 3)

        self.f2_5_down = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.f2_5_conv = ConvBNAct(128, cat_channel, 3)

        self.f1_5_down = nn.MaxPool2d(16, 16, ceil_mode=True)
        self.f1_5_conv = ConvBNAct(64, cat_channel, 3)

        self.o5 = ConvBNAct(5*cat_channel, 5*cat_channel, 3)

        # 4
        # self.o6_4_up = nn.ConvTranspose2d(6 * cat_channel, 6 * cat_channel, kernel_size=4, stride=2, padding=1)
        self.o5_4_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.o5_4_conv = ConvBNAct(5*cat_channel, cat_channel, 3)

        self.f4_4_conv = ConvBNAct(512, cat_channel, 3)

        self.f3_4_down = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.f3_4_conv = ConvBNAct(256, cat_channel, 3)

        self.f2_4_down = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.f2_4_conv = ConvBNAct(128, cat_channel, 3)

        self.f1_4_down = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.f1_4_conv = ConvBNAct(64, cat_channel, 3)

        self.o4 = ConvBNAct(5*cat_channel, 5*cat_channel, 3)

        # 3
        # self.o6_3_up = nn.ConvTranspose2d(6*cat_channel, 6*cat_channel, kernel_size=8, stride=4, padding=2)
        # self.o6_3_up = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        # self.o6_3_conv = ConvBNAct(6*cat_channel, cat_channel, 3)

        # self.o5_3_up = nn.ConvTranspose2d(6*cat_channel, 6*cat_channel, kernel_size=4, stride=2, padding=1)
        self.o5_3_up = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.o5_3_conv = ConvBNAct(5*cat_channel, cat_channel, 3)

        # self.o4_3_up = nn.ConvTranspose2d(6*cat_channel, 6*cat_channel, kernel_size=4, stride=2, padding=1)
        self.o4_3_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.o4_3_conv = ConvBNAct(5*cat_channel, cat_channel, 3)

        self.f3_3_conv = ConvBNAct(256, cat_channel, 3)

        self.f2_3_down = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.f2_3_conv = ConvBNAct(128, cat_channel, 3)

        self.f1_3_down = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.f1_3_conv = ConvBNAct(64, cat_channel, 3)

        self.o3 = ConvBNAct(5*cat_channel, 5*cat_channel, 3)

        # 2
        # self.o6_2_up = nn.ConvTranspose2d(6 * cat_channel, 6 * cat_channel, kernel_size=16, stride=8, padding=4)
        # self.o6_2_up = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False)
        # self.o6_2_conv = ConvBNAct(6 * cat_channel, cat_channel, 3)

        # self.o5_2_up = nn.ConvTranspose2d(6 * cat_channel, 6 * cat_channel, kernel_size=8, stride=4, padding=2)
        self.o5_2_up = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False)
        self.o5_2_conv = ConvBNAct(5 * cat_channel, cat_channel, 3)

        # self.o4_2_up = nn.ConvTranspose2d(6 * cat_channel, 6 * cat_channel, kernel_size=8, stride=4, padding=2)
        self.o4_2_up = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.o4_2_conv = ConvBNAct(5 * cat_channel, cat_channel, 3)

        # self.o3_2_up = nn.ConvTranspose2d(6*cat_channel, 6*cat_channel, kernel_size=4, stride=2, padding=1)
        self.o3_2_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.o3_2_conv = ConvBNAct(5*cat_channel, cat_channel, 3)

        self.f2_2_conv = ConvBNAct(128, cat_channel, 3)

        self.f1_2_down = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.f1_2_conv = ConvBNAct(64, cat_channel, 3)

        self.o2 = ConvBNAct(5 * cat_channel, 5 * cat_channel, 3)

        # 1
        # self.o6_1_up = nn.ConvTranspose2d(6 * cat_channel, 6 * cat_channel, kernel_size=16, stride=16)
        # self.o6_1_up = nn.Upsample(scale_factor=16, mode="bilinear", align_corners=False)
        # self.o6_1_conv = ConvBNAct(6 * cat_channel, cat_channel, 3)

        # self.o5_1_up = nn.ConvTranspose2d(6 * cat_channel, 6 * cat_channel, kernel_size=16, stride=8, padding=4)
        self.o5_1_up = nn.Upsample(scale_factor=16, mode="bilinear", align_corners=False)
        self.o5_1_conv = ConvBNAct(5 * cat_channel, cat_channel, 3)

        # self.o4_1_up = nn.ConvTranspose2d(6 * cat_channel, 6 * cat_channel, kernel_size=16, stride=8, padding=4)
        self.o4_1_up = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False)
        self.o4_1_conv = ConvBNAct(5 * cat_channel, cat_channel, 3)

        # self.o3_1_up = nn.ConvTranspose2d(6 * cat_channel, 6 * cat_channel, kernel_size=8, stride=4, padding=2)
        self.o3_1_up = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.o3_1_conv = ConvBNAct(5 * cat_channel, cat_channel, 3)

        # self.o2_1_up = nn.ConvTranspose2d(6*cat_channel, 6*cat_channel, kernel_size=4, stride=2, padding=1)
        self.o2_1_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.o2_1_conv = ConvBNAct(5*cat_channel, cat_channel, 3)

        self.f1_1_conv = ConvBNAct(64, cat_channel, 3)

        self.o1 = ConvBNAct(5 * cat_channel, 5 * cat_channel, 3)

    def forward(self, features_map):
        p1, p2, p3, p4, p5 = features_map
        # 6
        # f6_6 = self.f6_6(p6)
        # f5_6 = self.f5_6_conv(self.f5_6_down(p5))
        # f4_6 = self.f4_6_conv(self.f4_6_down(p4))
        # f3_6 = self.f3_6_conv(self.f3_6_down(p3))
        # f2_6 = self.f2_6_conv(self.f2_6_down(p2))
        # f1_6 = self.f1_6_conv(self.f1_6_down(p1))
        # o6 = self.o6(torch.cat([f6_6, f5_6, f4_6, f3_6, f2_6, f1_6], dim=1))

        # 5
        # f6_5 = self.o6_5_conv(self.o6_5_up(o6))
        f5_5 = self.f5_5_conv(p5)
        f4_5 = self.f4_5_conv(self.f4_5_down(p4))
        f3_5 = self.f3_5_conv(self.f3_5_down(p3))
        f2_5 = self.f2_5_conv(self.f2_5_down(p2))
        f1_5 = self.f1_5_conv(self.f1_5_down(p1))
        o5 = self.o5(torch.cat([f5_5, f4_5, f3_5, f2_5, f1_5], dim=1))

        # 4

        # f6_4 = self.o6_4_conv(self.o6_4_up(o6))
        f5_4 = self.o5_4_conv(self.o5_4_up(o5))
        f4_4 = self.f4_4_conv(p4)
        f3_4 = self.f3_4_conv(self.f3_4_down(p3))
        f2_4 = self.f2_4_conv(self.f2_4_down(p2))
        f1_4 = self.f1_4_conv(self.f1_4_down(p1))
        o4 = self.o4(torch.cat([f5_4, f4_4, f3_4, f2_4, f1_4], dim=1))

        # 3
        f5_3 = self.o5_3_conv(self.o5_3_up(o5))
        f4_3 = self.o4_3_conv(self.o4_3_up(o4))
        f3_3 = self.f3_3_conv(p3)
        f2_3 = self.f2_3_conv(self.f2_3_down(p2))
        f1_3 = self.f1_3_conv(self.f1_3_down(p1))
        o3 = self.o3(torch.cat([f5_3, f4_3, f3_3, f2_3, f1_3], dim=1))

        # 2
        f5_2 = self.o5_2_conv(self.o5_2_up(o5))
        f4_2 = self.o4_2_conv(self.o4_2_up(o4))
        f3_2 = self.o3_2_conv(self.o3_2_up(o3))
        f2_2 = self.f2_2_conv(p2)
        f1_2 = self.f1_2_conv(self.f1_2_down(p1))
        o2 = self.o2(torch.cat([f5_2, f4_2, f3_2, f2_2, f1_2], dim=1))

        # 1
        f5_1 = self.o5_1_conv(self.o5_1_up(o5))
        f4_1 = self.o4_1_conv(self.o4_1_up(o4))
        f3_1 = self.o3_1_conv(self.o3_1_up(o3))
        f2_1 = self.o2_1_conv(self.o2_1_up(o2))
        f1_1 = self.f1_1_conv(p1)
        o1 = self.o1(torch.cat([f5_1, f4_1, f3_1, f2_1, f1_1], dim=1))

        return o1


if __name__ == '__main__':
    batch_size = 1
    img_height = 320
    img_width = 480

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input = torch.rand(batch_size, 3, img_height, img_width).to(device)
    # target = torch.rand(batch_size, 1, img_height, img_width).to(device)
    print(f"input shape: {input.shape}")
    model = myModel().to(device)

    from torchstat import stat
    stat(model, (3, 320, 480))
