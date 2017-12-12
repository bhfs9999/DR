import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data.prior_box import vgg_stride16, PriorBox
import os
import torch.optim as optim
from .base_model import BaseModel
from layers.modules import MultiBoxLoss
import torch.backends.cudnn as cudnn
import torch.nn.init as init
from layers.functions import Detect

class DetModel(BaseModel):
    def __init__(self, args,):
        super(DetModel, self).__init__()
        self.args       = args
        self.phase      = args.phase
        self.model_name = args.model_name
        self.cuda       = args.cuda
        self.net        = eval(self.model_name+'(args)')
        self.lr_current = self.args.lr
        self.optimizer  = optim.SGD(self.net.parameters(), lr=args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
        self.criterion  = MultiBoxLoss(args.num_classes, 0.5, True, 0, True, 3, 0.5, False, self.cuda)

    def init_model(self):
        def xavier(param):
            init.xavier_uniform(param)

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                xavier(m.weight.data)
                m.bias.data.zero_()
        self.net.apply(weights_init)

        if self.args.resume:
            print('Resuming training, loading {}...'.format(self.args.resume))
            self.load_weights(self.net, self.args.resume)
        else:
            vgg_weights = torch.load(self.args.save_folder + self.args.basenet)
            print('Loading base network...')
            self.net.vgg.load_state_dict(vgg_weights)

        if self.args.cuda:
            self.net        = self.net.cuda()
            cudnn.benchmark = True

    def test(self, x):
        assert self.phase == 'test', "Command arg phase should be 'test'. "
        self.net.val()
        out = self.net(x, self.args)
        return out

    def train(self, dataloader, epoch):
        self.net.train()
        self._adjust_learning_rate(epoch)
        for images, targets in dataloader:
            if self.cuda:
                images  = Variable(images.cuda())
                targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
            else:
                images  = Variable(images)
                targets = [Variable(anno, volatile=True) for anno in targets]

            # forward
            out = self.net(images)

            #backward
            self.optimizer.zero_grad()
            loss_l, loss_c = self.criterion(out, targets)
            loss = loss_c + loss_l
            loss.backward()
            self.optimizer.step()
            print('===loss:{}==='.format(loss.data[0]))

    def val(self, dataloader, epoch):
        self.net.val()
        for images, targets in dataloader:
            if self.cuda:
                images = Variable(images.cuda())
                targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
            else:
                images = Variable(images)
                targets = [Variable(anno, volatile=True) for anno in targets]

            # forward
            out = self.net(images)
            loss_l, loss_c = self.criterion(out, targets)
            loss = loss_c + loss_l



class VggStride16(nn.Module):
    def __init__(self, args):
        super(VggStride16, self).__init__()
        self.phase       = args.phase
        self.num_classes = args.num_classes
        self.priors      = Variable(PriorBox(vgg_stride16).forward(), volatile=True)
        self.crop_size   = args.crop_size
        self.vgg         = nn.ModuleList(vgg(base[self.crop_size], 3,))
        # TODO: need more general
        self.loc_layers  = nn.Conv2d(self.vgg[-2].out_channels,
                                    6 * 4, kernel_size=3, padding=1)
        self.cls_layers  = nn.Conv2d(self.vgg[-2].out_channels,
                                    6 * self.num_classes, kernel_size=3, padding=1)
        if self.phase == 'test':
            self.softmax = nn.Softmax()
            self.detect  = Detect(self.num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        for one_layer in self.vgg:
            x = one_layer(x)

        # loc: bs x 19 x 19 x (6x4)
        # cls: bs x 19 x 19 x (6xn_class)
        loc = self.loc_layers(x).permute(0, 2, 3, 1).contiguous()
        cls = self.cls_layers(x).permute(0, 2, 3, 1).contiguous()

        if self.phase == 'test':
            output = self.detect(
                loc.view(loc.size(0), -1, 4),
                self.softmax(cls.view(-1, self.num_classes)),
                self.priors.type(type(x.data))
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                cls.view(cls.size(0), -1, self.num_classes),
                self.priors
            )
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