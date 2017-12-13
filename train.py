import torch
from torch.autograd import Variable
import torch.utils.data as data
from data import v2, v1, AnnotationTransform, VOCDetection, detection_collate, VOCroot, VOC_CLASSES
from utils.augmentations import SSDAugmentation, BaseTransform
from model.base_model import DetModel
from options.base_options import BaseOptions
from tensorboardX import SummaryWriter
import os
from data.retinal_data import DetectionDataset

# options = BaseOptions()
# args = options.parse()
# options.setup_option()
#
# vgg16_s16_net = VggStride16(args)
# net = vgg16_s16_net
# train_sets = [('2007', 'trainval')]
#
# if args.resume:
#     print('Resuming training, loading {}...'.format(args.resume))
#     vgg16_s16_net.load_weights(args.resume)
# else:
#     vgg_weights = torch.load(args.save_folder + args.basenet)
#     print('Loading base network...')
#     vgg16_s16_net.vgg.load_state_dict(vgg_weights)
#
# if args.cuda:
#     net = net.cuda()
#
# if args.visdom:
#     import visdom
#     viz = visdom.Visdom()
#
# def xavier(param):
#     init.xavier_uniform(param)
#
# def weights_init(m):
#     if isinstance(m, nn.Conv2d):
#         xavier(m.weight.data)
#         m.bias.data.zero_()
#
# if not args.resume:
#     print('Initializing weights...')
#     # initialize newly added layers' weights with xavier method
#     vgg16_s16_net.loc_layers.apply(weights_init)
#     vgg16_s16_net.cls_layers.apply(weights_init)
#
# optimizer = optim.SGD(net.parameters(), lr=args.lr,
#                       momentum=args.momentum, weight_decay=args.weight_decay)
# criterion = MultiBoxLoss(args.num_classes, 0.5, True, 0, True, 3, 0.5, False, args.cuda)

# def train():
#     net.train()
#     # loss counters
#     loc_loss = 0  # epoch
#     conf_loss = 0
#     epoch = 0
#     print('Loading Dataset...')
#
#     dataset = VOCDetection(args.voc_root, train_sets, SSDAugmentation(
#         args.input_size, args.means), AnnotationTransform())
#
#     epoch_size = len(dataset) // args.batch_size
#     print('Training SSD on', dataset.name)
#     step_index = 0
#     if args.visdom:
#         # initialize visdom loss plot
#         lot = viz.line(
#             X=torch.zeros((1,)).cpu(),
#             Y=torch.zeros((1, 3)).cpu(),
#             opts=dict(
#                 xlabel='Iteration',
#                 ylabel='Loss',
#                 title='Current Single_feature Training Loss',
#                 legend=['Loc Loss', 'Conf Loss', 'Loss']
#             )
#         )
#         epoch_lot = viz.line(
#             X=torch.zeros((1,)).cpu(),
#             Y=torch.zeros((1, 3)).cpu(),
#             opts=dict(
#                 xlabel='Epoch',
#                 ylabel='Loss',
#                 title='Epoch Single_feature Training Loss',
#                 legend=['Loc Loss', 'Conf Loss', 'Loss']
#             )
#         )
#     batch_iterator = None
#     data_loader = data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers,
#                                   shuffle=True, collate_fn=detection_collate, pin_memory=True)
#     for iteration in range(args.start_iter, args.iterations):
#         if (not batch_iterator) or (iteration % epoch_size == 0):
#             # create batch iterator
#             batch_iterator = iter(data_loader)
#         if iteration in args.stepvalues:
#             step_index += 1
#             adjust_learning_rate(optimizer, args.gamma, step_index)
#             if args.visdom:
#                 viz.line(
#                     X=torch.ones((1, 3)).cpu() * epoch,
#                     Y=torch.Tensor([loc_loss, conf_loss,
#                         loc_loss + conf_loss]).unsqueeze(0).cpu() / epoch_size,
#                     win=epoch_lot,
#                     update='append'
#                 )
#             # reset epoch loss counters
#             loc_loss = 0
#             conf_loss = 0
#             epoch += 1
#
#         # load train data
#         images, targets = next(batch_iterator)
#         if args.cuda:
#             images = Variable(images.cuda())
#             targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
#         else:
#             images = Variable(images)
#             targets = [Variable(anno, volatile=True) for anno in targets]
#         # forward
#         t0 = time.time()
#         out = net(images)
#         # backprop
#         optimizer.zero_grad()
#         loss_l, loss_c = criterion(out, targets)
#         loss = loss_l + loss_c
#         loss.backward()
#         optimizer.step()
#         t1 = time.time()
#         loc_loss += loss_l.data[0]
#         conf_loss += loss_c.data[0]
#         if iteration % 10 == 0:
#             print('Timer: %.4f sec.' % (t1 - t0))
#             print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data[0]), end=' ')
#             if args.visdom and args.send_images_to_visdom:
#                 random_batch_index = np.random.randint(images.size(0))
#                 viz.image(images.data[random_batch_index].cpu().numpy())
#         if args.visdom:
#             viz.line(
#                 X=torch.ones((1, 3)).cpu() * iteration,
#                 Y=torch.Tensor([loss_l.data[0], loss_c.data[0],
#                     loss_l.data[0] + loss_c.data[0]]).unsqueeze(0).cpu(),
#                 win=lot,
#                 update='append'
#             )
#             # hacky fencepost solution for 0th epoch plot
#             if iteration == 0:
#                 viz.line(
#                     X=torch.zeros((1, 3)).cpu(),
#                     Y=torch.Tensor([loc_loss, conf_loss,
#                         loc_loss + conf_loss]).unsqueeze(0).cpu(),
#                     win=epoch_lot,
#                     update=True
#                 )
#         if iteration % 5000 == 0:
#             print('Saving state, iter:', iteration)
#             torch.save(vgg16_s16_net.state_dict(), 'weights/ssd300_0712_' +
#                        repr(iteration) + '.pth')
#     torch.save(vgg16_s16_net.state_dict(), args.save_folder + '' + args.version + '.pth')
#
#
# def adjust_learning_rate(optimizer, gamma, step):
#     """Sets the learning rate to the initial LR decayed by 10 at every specified step
#     # Adapted from PyTorch Imagenet example:
#     # https://github.com/pytorch/examples/blob/master/imagenet/main.py
#     """
#     lr = args.lr * (gamma ** (step))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


if __name__ == '__main__':
    # train_sets = [('2007', 'trainval')]
    options = BaseOptions()
    args = options.parse()
    options.setup_option()

    writer = SummaryWriter()

    # dataset
    existed_imgs = [fname.split('.')[0] for fname in os.listdir(args.img_root)]
    existed_xmls = [fname.split('_')[0] for fname in os.listdir(args.xml_root)]

    all_idxes  = set(existed_imgs).intersection(set(existed_xmls))
    all_xmlfnames = [idx+'_lable.xml' for idx in all_idxes]
    # print(valid_idxes, len(valid_idxes))
    n_data       = len(all_idxes)
    train_fnames = all_xmlfnames[:int(n_data*0.8)]
    val_fnames   = all_xmlfnames[int(n_data*0.8):]
    # dataset = VOCDetection(args.voc_root, train_sets, SSDAugmentation(
    #     args.input_size, args.means), AnnotationTransform())
    dataset_train = DetectionDataset(args.img_root, args.xml_root, train_fnames, args, SSDAugmentation(
        args.input_size, args.means))
    dataloader_train = data.DataLoader(dataset_train, args.batch_size, num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate, pin_memory=True)
    dataset_val = DetectionDataset(args.img_root, args.xml_root, val_fnames, args, BaseTransform(
        args.input_size, args.means))
    dataloader_val = data.DataLoader(dataset_train, args.batch_size, num_workers=args.num_workers,
                                       shuffle=False, collate_fn=detection_collate, pin_memory=True)


    model = DetModel(args)
    model.init_model()
    for i in range(args.iterations):
        model.train(dataloader_train, i, writer)

        if i % 5 == 0:
            model.save_network(model.net, 'single_feature', epoch=i, )
            print('saving in epoch {}'.format(i))

    writer.eport_scalars_to_json("./all_scalars.json")
    writer.close()