import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data.prior_box import vggstride16_config, PriorBox
from layers.functions import Detect


class VggStride16(nn.Module):
    def __init__(self, args):
        super(VggStride16, self).__init__()
        self.phase       = args.phase
        self.num_classes = args.num_classes
        self.priors      = Variable(PriorBox(vggstride16_config).forward(), volatile=True)
        self.crop_size   = args.crop_size
        self.vgg         = nn.ModuleList(vgg(base[self.crop_size], 3,))
        self.n_anchor    = len(vggstride16_config['scales']) * (len(vggstride16_config['aspect_ratios'][0] * 2) + 1)

        self.loc_layers  = nn.Conv2d(self.vgg[-2].out_channels,
                                     self.n_anchor * 4, kernel_size=3, padding=1)
        self.cls_layers  = nn.Conv2d(self.vgg[-2].out_channels,
                                     self.n_anchor * self.num_classes, kernel_size=3, padding=1)
        self.softmax     = nn.Softmax()

        if self.phase == 'test':
            self.detect      = Detect(self.num_classes, 0, 200, 0.01, 0.45)    # conf 0.01

    def forward(self, x):
        for one_layer in self.vgg:
            x = one_layer(x)

        # loc: bs x 19 x 19 x (6x4)
        # cls: bs x 19 x 19 x (6xn_class)
        loc = self.loc_layers(x).permute(0, 2, 3, 1).contiguous()
        cls = self.cls_layers(x).permute(0, 2, 3, 1).contiguous()

        output = (
            loc.view(loc.size(0), -1, 4),
            cls.view(cls.size(0), -1, self.num_classes),
            self.priors
        )

        if self.phase == 'test':
            detections = self.detect(
                loc.view(loc.size(0), -1, 4),
                self.softmax(cls.view(-1, self.num_classes)),
                self.priors.type(type(x.data))
            )
            return output, detections

        return output

class VggStride16_centerloss(VggStride16):
    def __init__(self, args):
        super(VggStride16_centerloss, self).__init__(args)
        self.center_dim = args.center_dim
        self.conv1 = nn.Conv2d(self.vgg[-2].out_channels,
                               self.center_dim, kernel_size=1, padding=0)
        self.register_buffer('centers', torch.zeros(self.num_classes, self.center_dim))

    def foward(self, x):
        for one_layer in self.vgg:
            x = one_layer(x)

        self.center_feature = x
        # loc: bs x 19 x 19 x (6x4)
        # cls: bs x 19 x 19 x (6xn_class)
        loc = self.loc_layers(x).permute(0, 2, 3, 1).contiguous()
        cls = self.cls_layers(x).permute(0, 2, 3, 1).contiguous()

        output = (
            loc.view(loc.size(0), -1, 4),
            cls.view(cls.size(0), -1, self.num_classes),
            self.priors
        )

        if self.phase == 'test':
            detections = self.detect(
                loc.view(loc.size(0), -1, 4),
                self.softmax(cls.view(-1, self.num_classes)),
                self.priors.type(type(x.data))
            )
            return output, detections

        return output

def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

base = {
300: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
        512, 512, 512],
512: [],
}