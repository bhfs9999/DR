import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from .det_model import VggStride16, VggStride16_centerloss
# from layers.modules import MultiBoxLoss, CenterLoss
from utils.eval_DR import *
from utils.plot import draw_bboxes_pre_label
from utils.summary import *
import layers.modules as loss
# import model.det_model as basenet

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
            # new version
            load_dict = torch.load(base_file, map_location=lambda storage, loc: storage)
            net_state_dict = load_dict['net_state_dict']
            # model_dict = net.state_dict()
            # net_state_dict = {k: v for k, v in net_state_dict.items() if k in model_dict}
            # model_dict.update(net_state_dict)
            net.load_state_dict(net_state_dict)
            epoch = load_dict['epoch']
            self.iter = load_dict['iter']
            # ori version
            # epoch = 0
            # net.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
            return epoch
        else:
            print('Sorry only .pth and .pkl files supported.')

    def save_network(self, net, net_name, epoch, label=''):
        save_fname  = '%s_%s_%s.pth' % (epoch, net_name, label)
        save_path   = os.path.join(self.args.save_folder, self.args.exp_name, save_fname)
        save_dict = {'net_state_dict': net.state_dict(), 'exp_name': self.args.exp_name, 'epoch': epoch, 'iter': self.iter}
        torch.save(save_dict, save_path)

    def _adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 at every specified step
        # Adapted from PyTorch Imagenet example:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        if epoch in self.args.stepvalues:
            self.lr_current = self.lr_current * self.args.gamma
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr_current

    def _adjust_learning_rate_iter(self, iter):
        """Sets the learning rate to the initial LR decayed by 10 at every specified step
        # Adapted from PyTorch Imagenet example:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        if iter in self.args.stepvalues_iter:
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
        self.batch_size = args.batch_size
        self.optimizer  = optim.SGD(self.net.parameters(), lr=args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
        # TODO: more general
        self.criterion  = getattr(loss, args.loss.lower())(args)# CenterLoss(args.num_classes, 0.5, True, 0, True, 3, 0.5, False, self.cuda) # MultiBoxLoss(args.num_classes, 0.5, True, 0, True, 3, 0.5, False, self.cuda)
        self.iter       = 0

    def init_model(self):
        self.net.apply(weights_init)
        epoch = 0
        if self.args.resume and self.phase == 'train':
            resume_path = os.path.join(self.args.save_folder, self.args.exp_name, self.args.resume)
            print('Resuming training, loading {}...'.format(resume_path))
            epoch = self.load_weights(self.net, resume_path)
        elif self.phase == 'train':
            vgg_weights = torch.load(self.args.save_folder + self.args.basenet)
            print('Loading base network...')
            self.net.vgg.load_state_dict(vgg_weights)

        if self.phase == 'test':
            model_path = os.path.join(self.args.save_folder, self.args.exp_name, self.args.trained_model)
            print('load trained model from {}'.format(model_path))
            epoch = self.load_weights(self.net, model_path)

        if self.args.cuda:
            self.net        = self.net.cuda()
            cudnn.benchmark = True
        return epoch

    def train_val_byIter(self, train_dataset, train_dloader, val_dataset, val_dloader, writer):
        train_epoch_size = len(train_dataset) // self.batch_size
        val_epoch_size = len(val_dataset) // self.batch_size
        train_batch_iterator = None
        val_batch_iterator = None
        val_iter = 0
        for iteration in range(self.args.start_iter, self.args.max_iter):
            t2 = time.time()
            if (not train_batch_iterator) or (iteration % train_epoch_size == 0):
                train_batch_iterator = iter(train_dloader)
            self._adjust_learning_rate_iter(iteration)

            images, targets = next(train_batch_iterator)

            if self.cuda:
                images  = Variable(images.cuda())
                targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
            else:
                images  = Variable(images)
                targets = [Variable(anno, volatile=True) for anno in targets]

            out = self.net(images)

            # backprop
            self.optimizer.zero_grad()
            if self.args.loss == 'CenterLoss':
                loss_l, loss_c, target_fmap, have_centerloss = self.criterion(out, targets)
                center_loss, self.net._buffers['centers'] = self.criterion.get_center_loss(self.net._buffers['centers'],
                                                                                           self.net.center_feature,
                                                                                           target_fmap,
                                                                                           self.args.alpha,
                                                                                           self.args.num_classes,
                                                                                           have_centerloss,
                                                                                           )

                loss = loss_c + loss_l + self.args.centerloss_weight * center_loss
                loss.backward()
                self.optimizer.step()
                if self.iter % 10 == 0:
                    print('train loss: {:.4f} | n_iter: {} | time: {:.2f}'.format(loss.data[0], self.iter, time.time() - t2))
                    scalars = [loss.data[0], loss_c.data[0], loss_l.data[0], center_loss.data[0]]
                    names   = ['loss', 'loss_c', 'loss_l', 'center_loss']
                    write_scalars(writer, scalars, names, self.iter, tag='train_loss')
                if self.iter % self.args.val_interval == 0:
                    val_batch_iterator, val_iter = self.val_iter(val_batch_iterator, val_iter, val_dloader, val_epoch_size, writer, iteration)


            else:
                loss_l, loss_c, = self.criterion(out, targets)
                loss = loss_c + loss_l

                loss.backward()
                self.optimizer.step()

                # log
                if self.iter % 10 == 0:
                    print('train loss: {:.4f} | n_iter: {} | time: {:.2f}'.format(loss.data[0], self.iter, time.time() - t2))
                    scalars = [loss.data[0], loss_c.data[0], loss_l.data[0], ]
                    names   = ['loss', 'loss_c', 'loss_l', ]
                    write_scalars(writer, scalars, names, self.iter, tag='train_loss')
                if self.iter % self.args.val_interval == 0:
                    val_batch_iterator, val_iter = self.val_iter(val_batch_iterator, val_iter, val_dloader, val_epoch_size, writer, iteration)

            self.iter = iteration

            if (self.iter+1) % self.args.save_freq_iter == 0:
                self.save_network(self.net, 'single_feature', epoch=self.iter+1, )
                print('saving in iter {}'.format(self.iter))

    def val_iter(self, val_batch_iterator, val_iter, val_dloader, val_epoch_size, writer, train_iter):
        self.net.eval()
        losses   = []
        losses_c = []
        losses_l = []
        losses_center = []
        t1 = time.time()
        for iteration in range(val_iter, val_iter+self.args.val_iterlen):

            if (not val_batch_iterator) or (iteration % val_epoch_size == 0):
                val_batch_iterator = iter(val_dloader)

            images, targets = next(val_batch_iterator)

            if self.cuda:
                images  = Variable(images.cuda())
                targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
            else:
                images  = Variable(images)
                targets = [Variable(anno, volatile=True) for anno in targets]

            out = self.net(images)
            if self.args.loss == 'CenterLoss':
                loss_l, loss_c, target_fmap, have_centerloss = self.criterion(out, targets)
                center_loss, self.net._buffers['centers'] = self.criterion.get_center_loss(self.net._buffers['centers'],
                                                                                           self.net.center_feature,
                                                                                           target_fmap,
                                                                                           self.args.alpha,
                                                                                           self.args.num_classes,
                                                                                           have_centerloss
                                                                                           )
                loss = loss_c + loss_l + self.args.centerloss_weight * center_loss
                center_loss_s = center_loss.data[0]
            else:
                loss_l, loss_c, = self.criterion(out, targets)
                loss = loss_c + loss_l
                center_loss_s = 0

            losses.append(loss.data[0])
            losses_c.append(loss_c.data[0])
            losses_l.append(loss_l.data[0])
            losses_center.append(center_loss_s)

        # log
        n = len(losses)
        loss, loss_l, loss_c, loss_center = sum(losses) / n, sum(losses_l) / n, \
                                            sum(losses_c) / n, sum(losses_center) / n
        scalars = [loss, loss_c, loss_l, loss_center]
        names   = ['loss', 'loss_c', 'loss_l', 'loss_center']
        write_scalars(writer, scalars, names, train_iter, tag='val_loss')
        print('iter{} val finish, cost time {:.2f}, loss: {:.4f}, loss_l: {:.4f}, loss_c: {:.4f}, loss_center: {:.4f}'.format(train_iter,
                                                                                                                               time.time()-t1, loss, loss_l, loss_c, loss_center))
        self.net.train()
        return val_batch_iterator, val_iter+self.args.val_iterlen


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

            if self.args.debug:
                # plot
                print('targets', targets[0].data[0])
                if self.iter == 20:
                    return
                print('\n=============iter: ', self.iter)
                # print(images.size(), images.squeeze(0).cpu().data.numpy().shape)
                print(targets[0].size())
                gt = targets[0].cpu().data.numpy()
                gt_bbox = gt[:, :4]
                cls = gt[:, 4] + 1
                image = images.squeeze(0).cpu().data.numpy()
                image = draw_bboxes_pre_label(image, None, gt_bbox, labels=cls, means=self.args.means )

                image = ToTensor()(image)
                image = vutils.make_grid([image])
                writer.add_image('Image_{}'.format(self.iter), image)
                # write_hist_parameters(writer, self.net.conv1, self.iter)
            # print('conv1 weight', self.net.conv1.weight)

            # forward
            out = self.net(images)

            #backward
            self.optimizer.zero_grad()

            if self.args.loss == 'CenterLoss':
                loss_l, loss_c, target_fmap, have_centerloss = self.criterion(out, targets)
                center_loss, self.net._buffers['centers'] = self.criterion.get_center_loss(self.net._buffers['centers'],
                                                                            self.net.center_feature,
                                                                            target_fmap,
                                                                            self.args.alpha,
                                                                            self.args.num_classes,
                                                                            have_centerloss,
                                                                            )

                loss = loss_c + loss_l + self.args.centerloss_weight * center_loss
                loss.backward()
                self.optimizer.step()
                if self.iter % 10 == 0:
                    print('train loss: {:.4f} | n_iter: {} | time: {:.2f}'.format(loss.data[0], self.iter, time.time() - t2))
                    scalars = [loss.data[0], loss_c.data[0], loss_l.data[0], center_loss.data[0]]
                    names   = ['loss', 'loss_c', 'loss_l', 'center_loss']
                    write_scalars(writer, scalars, names, self.iter, tag='train_loss')

            else:
                loss_l, loss_c, = self.criterion(out, targets)
                loss = loss_c + loss_l

                loss.backward()
                self.optimizer.step()

                # log
                if self.iter % 10 == 0:
                    print('train loss: {:.4f} | n_iter: {} | time: {:.2f}'.format(loss.data[0], self.iter, time.time() - t2))
                    scalars = [loss.data[0], loss_c.data[0], loss_l.data[0], ]
                    names   = ['loss', 'loss_c', 'loss_l', ]
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
        losses_center = []
        t1       = time.time()
        for images, targets in dataloader:
            if self.cuda:
                images = Variable(images.cuda())
                targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
            else:
                images = Variable(images)
                targets = [Variable(anno, volatile=True) for anno in targets]

            # forward

            out = self.net(images)

            if self.args.loss == 'CenterLoss':
                loss_l, loss_c, target_fmap, have_centerloss = self.criterion(out, targets)
                center_loss, self.net._buffers['centers'] = self.criterion.get_center_loss(self.net._buffers['centers'],
                                                                                           self.net.center_feature,
                                                                                           target_fmap,
                                                                                           self.args.alpha,
                                                                                           self.args.num_classes,
                                                                                           have_centerloss
                                                                                           )


                loss = loss_c + loss_l + self.args.centerloss_weight * center_loss
            else:
                loss_l, loss_c, = self.criterion(out, targets)
                loss = loss_c + loss_l

            losses.append(loss.data[0])
            losses_c.append(loss_c.data[0])
            losses_l.append(loss_l.data[0])
            losses_center.append(center_loss.data[0])

        # log
        n = len(losses)
        loss, loss_l, loss_c, loss_center = sum(losses) / n, sum(losses_l) / n, \
                                            sum(losses_c) / n, sum(losses_center) / n
        scalars = [loss, loss_c, loss_l, loss_center]
        names   = ['loss', 'loss_c', 'loss_l', 'loss_center']
        write_scalars(writer, scalars, names, epoch, tag='val_loss')
        print('epoch{} val finish, cost time {:.2f}, loss: {:.4f}, loss_l: {:.4f}, loss_c: {:.4f}, loss_center: {:.4f}'.format(epoch,
                                                                            time.time()-t1, loss, loss_l, loss_c, loss_center))

    def eval(self, dataset, writer, is_plot=True, is_save=False, get_mAP=True, plot_which='all'):
        assert self.net.phase == "test" and self.phase == "test", "phase should be test during eval"
        self.net.eval()
        num_images = len(dataset)
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
            x = Variable(im.unsqueeze(0), volatile=True)
            if self.cuda:
                x = x.cuda()

            # detections include 1 x n_classes x top_k x predict(conf, bbox of decode)
            _, detections = self.net(x)
            detections = detections.data
            # print(detections)  # 1 x 21 x 200 x 5, 1 x n_classes x top_k x predict

            if plot_which == 'all':
                b = 1
                e = detections.size(1)
            elif plot_which == 'MA':
                b = 1
                e = 2
            elif plot_which == 'BP':
                b = 2
                e = 3

            # skip j = 0, because it's the background class
            for j in range(b, e):
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
                scores_g_idx = scores_pred > self.args.conf_th
                scores_pred  = scores_pred[scores_g_idx]
                bboxes_pred  = cls_dets[:, :4]
                bboxes_pred  = bboxes_pred[scores_g_idx]
                classes_pred = [j] * len(scores_pred)

            if is_plot:
                image   = x.cpu().data.numpy()[0]
                gt_bbox = gt[:, :4]
                gt_cls  = gt[:, 4] + 1 # 0 is bg
                image   = draw_bboxes_pre_label(image, bboxes_pred, gt_bbox, self.args.means, scores_pred, classes_pred,
                                                gt_cls)
                image   = ToTensor()(image)
                image   = vutils.make_grid([image])
                writer.add_image('Image_{}'.format(i), image, 0)

            print('im_detect: {:d}/{:d}'.format(i + 1, num_images,))

        if get_mAP:
            pickle.dump(annots, open('./annotations_cache/annots.pkl', 'wb'))
            pickle.dump(ids, open('./annotations_cache/ids.pkl', 'wb'))
            output_dir = get_output_dir('pr_result', self.args.exp_name)
            evaluate_detections(all_boxes, output_dir, dataset)

            # self.net.phase = 'train'


