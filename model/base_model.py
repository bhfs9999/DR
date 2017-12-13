import torch
import os
import torch.optim as optim
from layers.modules import MultiBoxLoss
import torch.backends.cudnn as cudnn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn as nn
from .det_model import VggStride16
from utils.summary import *
import time

def xavier(param):
    init.xavier_uniform(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

class BaseModel(object):
    def __init__(self,):
        pass

    def init_model(self):
        pass

    def train(self, dataloader, epoch):
        pass

    def val(self, dataloader, epoch):
        pass

    def load_weights(self, net, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            net.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

    def save_network(self, net, net_name, epoch, label=''):
        save_fname  = '%s_%s_%s.pth' % (epoch, net_name, label)
        save_path   = os.path.join(self.args.save_folder, self.args.exp_name, save_fname)
        torch.save(net.state_dict(), save_path)

    def _adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 at every specified step
        # Adapted from PyTorch Imagenet example:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        if epoch in self.args.stepvalues:
            self.lr_current = self.lr_current * self.args.gamma
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr_current

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
        self.iter       = 0

    def init_model(self):
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

    def train(self, dataloader, epoch, writer):
        self.net.train()
        self._adjust_learning_rate(epoch)
        t1 = time.time()
        for images, targets in dataloader:
            t2 = time.time()
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

            # log
            if self.iter % 10 == 0:
                print('train loss: {:.4f} | n_iter: {} | time: {:.2f}'.format(loss.data[0], self.iter, time.time() - t2))
                t3 = time.time()
                scalars = [loss.data[0], loss_c.data[0], loss_l.data[0]]
                names   = ['loss', 'loss_c', 'loss_l']
                write_scalars(writer, scalars, names, self.iter, tag='train_loss')

            self.iter += 1

        if self.args.log_params:
            write_hist_parameters(writer, self.net, self.iter)

        print('epoch{} train finish, cost time {:.2f}'.format(epoch, time.time()-t1))

    def val(self, dataloader, epoch, writer):
        self.net.eval()
        # self.net.phase = 'test'
        losses   = []
        losses_c = []
        losses_l = []
        t1       = time.time()
        for images, targets in dataloader:
            if self.cuda:
                images = Variable(images.cuda())
                targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
            else:
                images = Variable(images)
                targets = [Variable(anno, volatile=True) for anno in targets]

            # forward
            # out, detections = self.net(images)
            out = self.net(images)
            loss_l, loss_c = self.criterion(out, targets)
            loss = loss_c + loss_l

            losses.append(loss.data[0])
            losses_c.append(loss_c.data[0])
            losses_l.append(loss_l.data[0])

        # log
        n = len(losses)
        loss, loss_l, loss_c = sum(losses) / n, sum(losses_l) / n, sum(losses_c) / n
        scalars = [loss, loss_c, loss_l]
        names   = ['loss', 'loss_c', 'loss_l']
        write_scalars(writer, scalars, names, epoch, tag='val_loss')
        print('epoch{} val finish, cost time {:.2f}, loss: {:.4f}, loss_l: {:.4f}, loss_c: {:.4f}'.format(epoch,
                                                                            time.time()-t1, loss, loss_l, loss_c))

        # TODO: add metrics

        # self.net.phase = 'train'


