import logging
import torch.nn as nn
import torch
from ..common.initialization import initialize_weights
from ..encoders.wider_resnet import WiderResNetA2


def Norm2d(in_channels):
    return nn.BatchNorm2d(in_channels)


def Upsample(x, size):
    return nn.functional.interpolate(x, size=size,
                                     mode='bilinear', align_corners=True)


class _AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18)):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          Norm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                Norm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            Norm2d(reduction_dim), nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


class DeepWV3Plus(nn.Module):
    """
    WideResNet38 version of DeepLabV3
    mod1
    pool2
    mod2 bot_fine
    pool3
    mod3-7
    bot_aspp

    structure: [3, 3, 6, 3, 1, 1]
    channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048),
              (1024, 2048, 4096)]
    """

    def __init__(self, num_classes, trunk='WideResnet38', criterion=None):

        super(DeepWV3Plus, self).__init__()
        self.criterion = criterion
        logging.info("Trunk: %s", trunk)

        wide_resnet = WiderResNetA2(structure=[3, 3, 6, 3, 1, 1], classes=1000, dilation=True)
        wide_resnet = nn.DataParallel(wide_resnet)
        if criterion is not None:
            try:
                checkpoint = torch.load('./pretrained_models/wider_resnet38.pth.tar', map_location='cpu')
                wide_resnet.load_state_dict(checkpoint['state_dict'])
                del checkpoint
            except:
                print(
                    "Please download the ImageNet weights of WideResNet38 in our repo to ./pretrained_models/wider_resnet38.pth.tar.")
                raise RuntimeError(
                    "=====================Could not load ImageNet weights of WideResNet38 network.=======================")
        wide_resnet = wide_resnet.module

        self.mod1 = wide_resnet.mod1
        self.mod2 = wide_resnet.mod2
        self.mod3 = wide_resnet.mod3
        self.mod4 = wide_resnet.mod4
        self.mod5 = wide_resnet.mod5
        self.mod6 = wide_resnet.mod6
        self.mod7 = wide_resnet.mod7
        self.pool2 = wide_resnet.pool2
        self.pool3 = wide_resnet.pool3
        del wide_resnet

        self.aspp = _AtrousSpatialPyramidPoolingModule(4096, 256,
                                                       output_stride=8)

        self.bot_fine = nn.Conv2d(128, 48, kernel_size=1, bias=False)
        self.bot_aspp = nn.Conv2d(1280, 256, kernel_size=1, bias=False)

        self.final = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))

        initialize_weights(self.final)

    def forward(self, inp, gts=None):

        x_size = inp.size()
        x = self.mod1(inp)
        m2 = self.mod2(self.pool2(x))
        x = self.mod3(self.pool3(m2))
        x = self.mod4(x)
        x = self.mod5(x)
        x = self.mod6(x)
        x = self.mod7(x)
        x = self.aspp(x)
        dec0_up = self.bot_aspp(x)

        dec0_fine = self.bot_fine(m2)
        dec0_up = Upsample(dec0_up, m2.size()[2:])
        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)

        dec1 = self.final(dec0)
        out = Upsample(dec1, x_size[2:])

        if self.training:
            return self.criterion(out, gts)

        return out
