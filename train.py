import torch
from torch.autograd import Variable
import torch.utils.data as data
from data import v2, v1, AnnotationTransform, VOCDetection, detection_collate, VOCroot, VOC_CLASSES
from utils.augmentations import SSDAugmentation, BaseTransform
from model.base_model import DetModel
from options.base_options import DetOptions
from tensorboardX import SummaryWriter
import os
from data.retinal_data import DetectionDataset


if __name__ == '__main__':
    # train_sets = [('2007', 'trainval')]
    options = DetOptions()
    args = options.parse()
    options.setup_option()
    writer = SummaryWriter(os.path.join('runs', args.exp_name+'_train'))

    # dataset
    if args.voc:
        args.num_classes = 21
        train_sets = [('2007', 'trainval')]
        dataset = VOCDetection(args.voc_root, train_sets, SSDAugmentation(
            args.input_size, args.means), AnnotationTransform())
        data_loader = data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers,
                                      shuffle=True, collate_fn=detection_collate, pin_memory=True)
    else:
        existed_imgs = [fname.split('.')[0] for fname in os.listdir(args.img_root)]
        existed_xmls = [fname.split('_')[0] for fname in os.listdir(args.xml_root)]

        all_idxes  = set(existed_imgs).intersection(set(existed_xmls))
        all_xmlfnames = [idx+'_lable.xml' for idx in all_idxes]
        all_xmlfnames = sorted(all_xmlfnames)
        n_data       = len(all_xmlfnames)
        train_fnames = all_xmlfnames[:int(n_data*0.8)]
        val_fnames   = all_xmlfnames[int(n_data*0.8):]
        dataset_train = DetectionDataset(args.img_root, args.xml_root, train_fnames, args.crop_size, args.shift_rate,
                                         args.pad_value, BaseTransform(args.input_size, args.means), )#SSDAugmentation(args.input_size, args.means))
        dataloader_train = data.DataLoader(dataset_train, args.batch_size, num_workers=args.num_workers,
                                      shuffle=True, collate_fn=detection_collate, pin_memory=True)
        dataset_val = DetectionDataset(args.img_root, args.xml_root, val_fnames, args.crop_size, args.shift_rate,
                                       args.pad_value, BaseTransform(args.input_size, args.means))
        dataloader_val = data.DataLoader(dataset_val, args.batch_size, num_workers=args.num_workers,
                                           shuffle=False, collate_fn=detection_collate, pin_memory=True)
        print('train dataset len: {}'.format(len(dataset_train)))
        print('val dataset len: {}'.format(len(dataset_val)))

    model = DetModel(args)
    begin_epoch = model.init_model()
    for i in range(begin_epoch, begin_epoch+args.max_epochs):
        print('\nepoch: {}'.format(i))
        model.train(dataloader_train, i, writer)

        if not args.voc:
            model.val(dataloader_val, i, writer)
        #
        if (i+1) % args.save_freq == 0:
            model.save_network(model.net, 'single_feature', epoch=i+1, )
            print('saving in epoch {}'.format(i))

    # writer.eport_scalars_to_json("./all_scalars.json")
    writer.close()