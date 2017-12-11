import argparse

class BaseOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--version', default='single_feature', help='conv11_2(v2) or pool6(v1) as last layer')
        self.parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
        self.parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
        self.parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
        self.parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
        self.parser.add_argument('--num_workers', default=2, type=int, help='Number of workers used in dataloading')
        self.parser.add_argument('--iterations', default=120000, type=int, help='Number of training iterations')
        self.parser.add_argument('--start_iter', default=0, type=int, help='Begin counting iterations starting from this value (should be used with resume)')
        self.parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
        self.parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
        self.parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
        self.parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
        self.parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
        self.parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')
        self.parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
        self.parser.add_argument('--send_images_to_visdom', type=str2bool, default=False, help='Sample a random image from each 10th batch, send it to visdom after augmentations step')
        self.parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
        self.parser.add_argument('--voc_root', default=VOCroot, help='Location of VOC root directory')
        self.parser.add_argument('--crop_size', default=300, help='size of cropped image')

