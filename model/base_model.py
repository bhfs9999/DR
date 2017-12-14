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
import numpy as np
from utils.plot import draw_bboxes_pre_label
import torchvision.utils as vutils
from torchvision.transforms import ToTensor
from eval_DR import *

def xavier(param):
    init.xavier_uniform(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

class BaseModel(object):
    def __init__(self,):
        raise NotImplementedError

    def init_model(self):
        raise NotImplementedError

    def train(self, dataloader, epoch):
        raise NotImplementedError

    def val(self, dataloader, epoch):
        raise NotImplementedError

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

        if self.args.resume and self.phase == 'train':
            print('Resuming training, loading {}...'.format(self.args.resume))
            self.load_weights(self.net, self.args.resume)
        elif self.phase == 'train':
            vgg_weights = torch.load(self.args.save_folder + self.args.basenet)
            print('Loading base network...')
            self.net.vgg.load_state_dict(vgg_weights)

        if self.phase == 'test':
            model_path = os.path.join(self.args.save_folder, self.args.exp_name, self.args.trained_model)
            print('load trained model from {}'.format(model_path))
            self.load_weights(self.net, model_path)

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
            write_hist_parameters(writer, self.net, epoch)

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

    def eval(self, dataset, writer, is_plot=True, is_save=False):
        # TODO: add metrics
        assert self.net.phase == "test" and self.phase == "test", "phase should be test during eval"
        self.net.eval()
        num_images = len(dataset)
        # num_images = 10
        # all detections are collected into:
        #    all_boxes[cls][image] = N x 5 array of detections in
        #    (x1, y1, x2, y2, score)
        all_boxes = [[[] for _ in range(num_images)]
                     for _ in range(self.args.num_classes)]

        gts = []
        annots = {}
        idx2name = {0:'MA', 1:'BS'}
        ids = []
        for i in range(num_images):
            # gt (n_obj, 5) of this im
            im, gt, id = dataset[i]
            ids.append(id)
            objects = []
            for object in gt:
                objects.append({'bbox':object[:4], 'name':idx2name[object[4]], 'difficult':0})
            annots[id] = objects
            gts.append(gt)
            x = Variable(im.unsqueeze(0))
            if self.cuda:
                x = x.cuda()

            # detections include 1 x n_classes x top_k x predict(conf, bbox of decode)
            _, detections = self.net(x)
            detections = detections.data
            # print(detections)  # 1 x 21 x 200 x 5, 1 x n_classes x top_k x predict
            scores_pred = []
            bboxes_pred = []
            classes_pred = []

            # skip j = 0, because it's the background class
            for j in range(1, detections.size(1)):
                dets = detections[0, j, :]  # dets 200 x 5
                mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                dets = torch.masked_select(dets, mask).view(-1, 5)
                if dets.dim() == 0:
                    continue
                boxes = dets[:, 1:]
                scores = dets[:, 0].cpu().numpy()
                cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])) \
                    .astype(np.float32, copy=False)
                all_boxes[j][i] = cls_dets     # cls_dets (n_obj, 5)

                scores_pred  = cls_dets[:, 4]
                bboxes_pred  = cls_dets[:, :4]
                classes_pred = [j] * len(cls_dets)

            if is_plot:
                image   = x.cpu().data.numpy()[0]
                gt_bbox = gt[:, :4]
                gt_cls  = gt[:, 4]
                image   = draw_bboxes_pre_label(image, bboxes_pred, gt_bbox, self.args.means, scores_pred, classes_pred,
                                                gt_cls)
                image   = ToTensor()(image)
                image   = vutils.make_grid([image])
                writer.add_image('Image', image, i)


            print('im_detect: {:d}/{:d}'.format(i + 1, num_images,))

        pickle.dump(annots, open('annotations_cache/annots.pkl', 'wb'))
        pickle.dump(ids, open('annotations_cache/ids.pkl', 'wb'))
        output_dir = get_output_dir('pr_result', self.args.exp_name)
        evaluate_detections(all_boxes, output_dir, dataset)

            # self.net.phase = 'train'


