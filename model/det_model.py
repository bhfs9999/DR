import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data.prior_box import vgg_stride16, PriorBox
import os

class VggStride16(nn.Module):
    def __init__(self, args):
        super(VggStride16, self).__init__()
        self.phase       = args.phase
        self.num_classes = args.num_classes
        self.priors      = Variable(PriorBox(vgg_stride16).forward(), volatile=True)
        self.crop_size   = args.crop_size

        self.vgg = nn.ModuleList(vgg(base[self.crop_size], 3,))
        # TODO: need more general
        self.loc_layers = nn.Conv2d(self.vgg[-2].out_channels,
                                    6 * 4, kernel_size=3, padding=1)
        self.cls_layers = nn.Conv2d(self.vgg[-2].out_channels,
                                    6 * self.num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        for one_layer in self.vgg:
            x = one_layer(x)

        # loc: bs x 19 x 19 x (6x4)
        # cls: bs x 19 x 19 x (6xn_class)
        loc = self.loc_layers(x).permute(0, 2, 3, 1).contiguous()
        cls = self.cls_layers(x).permute(0, 2, 3, 1).contiguous()

        if self.phase == 'test':
            pass
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                cls.view(cls.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

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