import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data.prior_box import vggstride16_config
from ..box_utils import match, log_sum_exp
import numpy as np
# from train import args

class CenterLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True, only_pos_centerloss=False):
        super(CenterLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = vggstride16_config['variance']
        self.only_pos_centerloss = only_pos_centerloss

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)      # batch size
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        # print('priors size', priors.size())       # 4332 4
        # print('loc_data size', loc_data.size())   # 16 4332 4
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data  # gt loc
            labels = targets[idx][:, -1].data   # gt label
            defaults = priors.data
            # get loc_t: the offset to be learnt
            #    conf_t: the label to be learnt
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        # print('conf_t', conf_t.size())     # 16 4332
        # pos mean object
        pos = conf_t > 0
        # num_pos = pos.sum(keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)

        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        # print(loss_c)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)

        # print(pos_idx[0])       # 4332 3
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        # print('pos neg gt0 size', (pos+neg).gt(0).size(), (pos+neg).gt(0))   # bs x (19 x 19 x 9)
        n_anchos = len(vggstride16_config['scales']) * (len(vggstride16_config['aspect_ratios'][0] * 2) + 1)
        # determine whether only pos sample have center loss or both pos and neg
        if self.only_pos_centerloss:
            have_centerloss = torch.max((pos).gt(0).view(conf_t.size(0), 19, 19, n_anchos), dim=3)[0].view(-1)
        else:
            have_centerloss = torch.max((pos+neg).gt(0).view(conf_t.size(0), 19, 19, n_anchos), dim=3)[0].view(-1)
        # print('have_centerloss size', have_centerloss.size())
        # TODO: so many bounder is 1, big bug, maybe neg? why most neg is bounder
        # print('have_centerloss',torch.max((pos+neg).gt(0).view(conf_t.size(0), 19, 19, n_anchos), dim=3)[0][0])
        # print('conf_p size', conf_p.size(), 'targets_weighted size', targets_weighted.size())   # pos and neg (after hem)
        # print('conf_t size', conf_t.size(), torch.max(conf_t[0]))     # 16, 4332
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # TODO: is it good to use max? some maybe 1 both 2
        # print('conf_t size', conf_t.size())
        # TODO: the order of conf_t_feature map is right
        conf_t_featuremap = torch.max(conf_t.view(conf_t.size(0), 19, 19, n_anchos), dim=3)[0].view(-1)
        # print('conf_t_featuremap', torch.max(conf_t.view(conf_t.size(0), 19, 19, n_anchos), dim=3)[0])
        # print('conf_t_featuremap size', conf_t_featuremap.size())
        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c, conf_t_featuremap, have_centerloss

    def get_center_loss(self, centers, features, target, alpha, num_classes, have_centerloss):
        '''
        :param centers: num_classes x dim_center
        :param features: (bsx19x19) x dim_feature
        :param target: (bsx19x19)
        :param alpha:
        :param num_classes:
        :return:
        '''
        # have_centerloss bs'
        # print('have_centerloss', have_centerloss)

        # the order of target and have_centerloss is correct
        # extract hc idx is correct
        target = target[have_centerloss]
        # print('target: ', target)
        # print('target size, have_centerloss size', target.size(), have_centerloss.size())  # target len(hc)  ;  hc 19 x 19
        hc_expand = have_centerloss.view(have_centerloss.size(0), 1).expand(have_centerloss.size(0), features.size(1))
        # print('hc_expand size', hc_expand.size())    # (19x19) x dim_feature
        features = features[hc_expand].view(target.size(0), -1)
        # print('target size, feature size', target.size(), features.size())    # len(hc) x dim_feature

        batch_size = target.size(0)
        features_dim = features.size(1)

        target_expand = target.view(batch_size, 1).expand(batch_size, features_dim)
        centers_var = Variable(centers)
        centers_batch = centers_var.gather(0, target_expand)
        criterion = nn.MSELoss()
        center_loss = criterion(features, centers_batch)

        diff = centers_batch - features
        unique_label, unique_reverse, unique_count = np.unique(target.cpu().data.numpy(), return_inverse=True, return_counts=True)
        appear_times = torch.from_numpy(unique_count).gather(0,torch.from_numpy(unique_reverse))
        appear_times_expand = appear_times.view(-1,1).expand(batch_size,features_dim).type(torch.FloatTensor)
        diff_cpu = diff.cpu().data / appear_times_expand.add(1e-6)
        diff_cpu = alpha * diff_cpu
        for i in range(batch_size):
            centers[target.data[i]] -= diff_cpu[i].type(centers.type())

        return center_loss, centers