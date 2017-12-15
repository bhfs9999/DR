import torch.utils.data as data
from data import v2, v1, AnnotationTransform, VOCDetection, detection_collate
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

    writer = SummaryWriter(comment=args.exp_name+'_eval')

    # dataset
    existed_imgs = [fname.split('.')[0] for fname in os.listdir(args.img_root)]
    existed_xmls = [fname.split('_')[0] for fname in os.listdir(args.xml_root)]

    all_idxes  = set(existed_imgs).intersection(set(existed_xmls))
    all_xmlfnames = [idx+'_lable.xml' for idx in all_idxes]
    all_xmlfnames = sorted(all_xmlfnames)
    n_data       = len(all_xmlfnames)
    eval_fnames   = all_xmlfnames[int(n_data*0.8):][:3]
    print('eval_fnames len: {}'.format(len(eval_fnames)))

    # args.means = [0, 0, 0]
    # use val dataset to eval
    dataset_eval = DetectionDataset(args.img_root, args.xml_root, eval_fnames, args.crop_size, args.shift_rate,
                                   args.pad_value, BaseTransform(args.input_size, args.means), True)
    dataloader_eval = data.DataLoader(dataset_eval, args.batch_size, num_workers=args.num_workers,
                                       shuffle=False, collate_fn=detection_collate, pin_memory=True)
    print('eval dataset len: {}'.format(len(dataset_eval)))

    model = DetModel(args)
    model.init_model()
    # model.load_weights()

    model.eval(dataset_eval, writer)

    # writer.eport_scalars_to_json("./all_scalars.json")
    writer.close()