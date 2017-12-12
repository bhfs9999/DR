import torch
import os

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
