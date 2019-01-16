'''
@Author: guoweiyu
@Date: 2019-01-15 11:38:52
@LastEditors: Jilong Wang
@Email: jilong.wang@watrix.ai
@LastEditTime: 2019-01-15 12:04:47
@Description: file content
'''
# coding: utf-8
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from util.utils import get_scheduler, init_weights, print_network

def create_models(opt):
    if opt.model_name == 'seg':
        from networks import SegNet
        model = SegNet(opt.input_chs, opt.filter_num)
    elif opt.model_name == 'seg_half':
        from networks import SegNetHalf
        model = SegNetHalf(opt.input_chs, opt.filter_num)
    elif opt.model_name == 'seg_mob':
        from networks import SegMobileNet
        model = SegMobileNet(opt.input_chs, opt.filter_num)
    elif opt.model_name == 'seg_mob_half':
        from networks import SegMobileNetHalf
        model = SegMobileNetHalf(opt.input_chs, opt.filter_num)
    elif opt.model_name == 'seg_upsample':
        from networks import SegUpsample
        model = SegUpsample(opt.input_chs, opt.filter_num)
    elif opt.model_name == 'seg_shuffle':
        from networks import SegShuffle
        model = SegShuffle(opt.input_chs, opt.filter_num, opt.typ)
    elif opt.model_name == 'seg_shuffle_add':
        from networks import SegShuffleAdd
        model = SegShuffleAdd(opt.input_chs, opt.filter_num, opt.typ)
    elif opt.model_name == 'seg_shuffle_half':
        from networks import SegShuffleHalf
        model = SegShuffleHalf(opt.input_chs, opt.filter_num, opt.typ)
    elif opt.model_name == 'seg_shuffle_mob':
        from networks import SegShuffleMob
        model = SegShuffleMob(opt.input_chs, opt.filter_num)
    else:
        raise ValueError('Model [%s] not recognized.' % opt.model_name)
    return model


def define_model(opt):
    net = create_models(opt)
    model = _segModel(net)
    use_gpu = len(opt.gpus) > 0
    if use_gpu:
        assert (torch.cuda.is_available())

    if len(opt.gpus) > 0:
        model.cuda(opt.gpus[0])

    init_weights(model, init_type=opt.init_type)
    return model


class _segModel(nn.Module):
    def __init__(self, model, gpus=[]):
        super(_segModel, self).__init__()
        self.gpu_ids = gpus
        self.model = model

    def forward(self, x):
        if len(self.gpu_ids) > 1 and isinstance(x.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, x, self.gpu_ids)
        else:
            return self.model(x)


class SegmentModel:
    imgs = None
    gts = None
    loss = float("inf")

    def __init__(self, opt, is_train=True):
        self.model = define_model(opt)
        self.save_dir = os.path.join(opt.train_root, opt.model_name)

        self.gpus = opt.gpus
        self.use_gpu = (len(opt.gpus) > 0 and torch.cuda.is_available())

        which_model = opt.which_model
        self.load_network(opt.model_name, which_model)

        self.class_weight = Variable(torch.FloatTensor([1, 10]))
        if self.use_gpu:
            self.class_weight = self.class_weight.cuda()

    def set_test_input(self, imgs):
        self.model.eval()
        if self.use_gpu:
            imgs = imgs.cuda(self.gpus[0], async=True)
        self.imgs = Variable(imgs, requires_grad=False)

    def forward(self):
        return self.model(self.imgs)

    # helper loading function that can be used by subclasses
    def load_network(self, network_label, record_label):
        save_filename = '%s_%s.pth' % (network_label, record_label)
        save_path = os.path.join(self.save_dir, save_filename)
        if os.path.exists(save_path):
            print('loading pretrained model...')
            self.model.load_state_dict(torch.load(save_path))