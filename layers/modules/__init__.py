# from .l2norm import L2Norm
from .multibox_loss import MultiBoxLoss
from .center_loss import CenterLoss
# __all__ = ['L2Norm', 'MultiBoxLoss']

multiboxloss = lambda args: MultiBoxLoss(args.num_classes, 0.5, True, 0, True, 3, 0.5, False, args.cuda)
centerloss = lambda args: CenterLoss(args.num_classes, 0.5, True, 0, True, 3, 0.5, False, args.cuda, args.only_pos_centerloss)