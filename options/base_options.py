import argparse
import torch
import os
from data import VOCroot
import torch.backends.cudnn as cudnn

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

class BaseOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # basic opt
        self.parser.add_argument('phase', choices=['train', 'test'], help='choice train or test')
        self.parser.add_argument('exp_name', type=str, help='experiment name')
        self.parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
        self.parser.add_argument('--num_workers', default=2, type=int, help='Number of workers used in dataloading')
        self.parser.add_argument('--start_iter', default=0, type=int, help='Begin counting iterations starting from this value (should be used with resume)')
        self.parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
        self.parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')
        self.parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
        self.parser.add_argument('--voc_root', default=VOCroot, help='Location of VOC root directory')
        self.parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
        self.parser.add_argument('--send_images_to_visdom', type=str2bool, default=False,
                            help='Sample a random image from each 10th batch, send it to visdom after augmentations step')

        # model opt
        self.parser.add_argument('--iterations', default=120000, type=int, help='Number of training iterations')
        self.parser.add_argument('--version', default='single_feature', help='conv11_2(v2) or pool6(v1) as last layer')
        self.parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
        self.parser.add_argument('--stepvalues', default=[80000, 100000, 120000], type=list, help='# of iter to change lr')
        self.parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
        self.parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD lr')
        self.parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
        self.parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
        self.parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
        self.parser.add_argument('--model_name', default='VggStride16', type=str, choices=['VggStride16'], help='model')
        self.parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')

        # fixed args
        self.parser.add_argument('--means', type=list, default=(104, 117, 123), help='mean of now')
        self.parser.add_argument('--num_classes', default=21, type=int, help='# lesion + bg' )
        self.parser.add_argument('--crop_size', default=300, type=int, help='size of cropped image')
        self.parser.add_argument('--input_size', default=300, type=int, help='model input size')

        # eval
        self.parser.add_argument('--trained_model', type=str, help='the path of trained model')
        self.parser.add_argument('--eval_save_folder', default='eval/', type=str,
                                 help='File path to save results')
        self.parser.add_argument('--confidence_threshold', default=0.01, type=float,
                                 help='Detection confidence threshold')
        self.parser.add_argument('--top_k', default=5, type=int,
                                 help='Further restrict the number of predictions to parse')

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        args = vars(self.opt)
        print('--------------Options-------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('----------------End---------------')
        return self.opt

    def setup_option(self):
        if self.opt.cuda and torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            cudnn = True
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
        # cfg = (v1, v2)[args.version == 'v2']

        if not os.path.exists(self.opt.save_folder):
            os.mkdir(self.opt.save_folder)

        model_save_path = os.path.join(self.opt.save_folder, self.opt.exp_name)
        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)
