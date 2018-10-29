import argparse
import torch
import os
from data import VOCroot
import torch.backends.cudnn as cudnn
from data.retinal_data import dname2label
from data.prior_box import *

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

class DetOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # basic opt
        self.parser.add_argument('phase', choices=['train', 'test'], help='choice train or test')
        self.parser.add_argument('exp_name', type=str, help='experiment name')
        self.parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
        self.parser.add_argument('--num_workers', default=5, type=int, help='Number of workers used in dataloading')
        self.parser.add_argument('--start_iter', default=0, type=int, help='Begin counting iterations starting from this value (should be used with resume)')
        self.parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
        self.parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
        self.parser.add_argument('--log_params', default=False, type=str2bool, help='Whether to log params')
        self.parser.add_argument('--img_root', default='/data/grq/DR/dectection/3/images', type=str, help='img root')
        self.parser.add_argument('--xml_root', default='/data/grq/DR/dectection/newxml', type=str, help='xml_root')
        self.parser.add_argument('--debug', default=False, type=str2bool, help='Whether to debug')
        self.parser.add_argument('--train_path', default='/data/grq/DR/dectection/3/train.csv', type=str, help='train fname')
        self.parser.add_argument('--val_path', default='/data/grq/DR/dectection/3/val.csv', type=str, help='val fname')
        self.parser.add_argument('--eval_path', default='/data/grq/DR/dectection/3/test.csv', type=str, help='test fname')
        self.parser.add_argument('--loss', default='MultiBoxLoss', choices=['MultiBoxLoss', 'CenterLoss'], type=str, help='choose loss')

        # train by iter
        self.parser.add_argument('--train_by_iter', default=False, type=str2bool, help='Whether to train by iter')
        self.parser.add_argument('--max_iter', default=100000, type=int, help='Number of training iterations')
        self.parser.add_argument('--stepvalues_iter', default=[25000, 50000, 75000], type=list, help='# of iter to change lr')
        self.parser.add_argument('--val_interval', default=100, type=int, help='every # train iter to val')
        self.parser.add_argument('--val_iterlen', default=20, type=int, help='how many iter to val')
        self.parser.add_argument('--save_freq_iter', default=5000 , type=int, help='how many iter to val')


        # train opt
        self.parser.add_argument('--max_epochs', default=40, type=int, help='Number of training iterations')
        self.parser.add_argument('--version', default='single_feature', help='conv11_2(v2) or pool6(v1) as last layer')
        self.parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
        self.parser.add_argument('--stepvalues', default=[10, 20, 30], type=list, help='# of iter to change lr')
        self.parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float, help='Weight decay for SGD')
        self.parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD lr')
        self.parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
        self.parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
        self.parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
        self.parser.add_argument('--model_name', default='VggStride16', type=str, choices=['VggStride16', 'VggStride16_centerloss'], help='model')
        self.parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
        self.parser.add_argument('--save_freq', default=5, type=int, help='save weights every # epochs')

        # crop args
        self.parser.add_argument('--pad_value', default=0, type=int, help='pad_value of crop')
        self.parser.add_argument('--shift_rate', default=0.8, type=float, help='shift rate of crop')
        self.parser.add_argument('--crop_size', default=300, type=int, help='size of cropped image')

        # fixed args
        self.parser.add_argument('--means', type=list, default=(104, 117, 123), help='mean of now')
        self.parser.add_argument('--num_classes', default=len(dname2label.keys())+1, type=int, help='# lesion + bg' )
        self.parser.add_argument('--input_size', default=300, type=int, help='model input size')

        # eval
        self.parser.add_argument('--trained_model', type=str, help='the path of trained model')
        self.parser.add_argument('--comment', type=str, default='', help='plot comment')
        self.parser.add_argument('--conf_th', default=0.01, type=float, help='the path of trained model')
        self.parser.add_argument('--is_plot', default=False, type=str2bool, help='Whether use center loss')
        self.parser.add_argument('--top_k', default=5, type=int,
                                 help='Further restrict the number of predictions to parse')
        self.parser.add_argument('--plot_which', type=str, default='all', choices=['all', 'MA', 'BP'],
                                 help='choose which lesion to visualize')

        # voc
        self.parser.add_argument('--voc_root', default='/home/xav/data/VOCdevkit', help='Location of VOC root directory')
        self.parser.add_argument('--voc', default=False, type=str2bool, help='whether to use voc')
        self.parser.add_argument('--voc_eval_save_folder', default='eval/', type=str,
                                 help='File path to save results')

        # center loss
        self.parser.add_argument('--alpha', default=0.5, type=float, help='center loss hp alpha')
        self.parser.add_argument('--centerloss_weight', default=0.1, type=float, help='center loss hp weight')
        self.parser.add_argument('--center_dim', default=128, type=int, help='center loss hp center dim')
        self.parser.add_argument('--only_pos_centerloss', default=False, type=str2bool, help='Whether only use pos to center loss')
        self.parser.add_argument('--only_centerloss', default=False, type=str2bool, help='Whether only use pos to center loss')

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
            cudnn.benchmark = True
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
        # cfg = (v1, v2)[args.version == 'v2']

        if self.opt.phase == 'train':
            if not os.path.exists(self.opt.save_folder):
                os.mkdir(self.opt.save_folder)

            # mk log save path
            model_save_path = os.path.join(self.opt.save_folder, self.opt.exp_name)
            print('model save path: ', model_save_path)
            if not os.path.exists(model_save_path):
                os.mkdir(model_save_path)

            # save option log at model save path
            option_fpath = os.path.join(model_save_path, 'options.log')
            with open(option_fpath, 'w') as f:

                model_name = self.opt.model_name.lower()
                hyper_param = eval(model_name+'_config')
                for k, v in sorted(hyper_param.items()):
                    f.write("%s: %s\n" % (str(k), str(v)))

                args = vars(self.opt)
                for k, v in sorted(args.items()):
                    f.write("%s: %s\n" % (str(k), str(v)))

            # mkdir log_dir
            if not os.path.exists('runs'):
                os.mkdir('runs')

            log_path = os.path.join('runs', self.opt.exp_name)
            print('log path: ',log_path)
            # if not os.path.exists(log_path):
            #     os.mkdir(log_path)
