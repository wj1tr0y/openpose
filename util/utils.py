# -*- coding=utf-8 -*-
"""
Created on Dec 15, 2017

@author: guoweiyu
"""

import cv2
import os
import cPickle
import math
import json
from numpy.random.mtrand import shuffle
from torch.nn import init
from torch.optim import lr_scheduler
from random import shuffle


def batch_serialize(data_root, data_tab, compressParams=['.jpg', 80], batch_size=1024, neg_ratio=0.1, save_path='.'):
    data_set = []
    data_list = [line.strip().split(' ') for line in open(os.path.join(data_root, data_tab))]
    shuffle(data_list)
    neg_samples = []
    pos_samples = []
    for d in data_list:
        if d[1] == 'null':
            neg_samples.append(d)
        else:
            pos_samples.append(d)

    i = 0
    cur = 0
    num_neg = 0
    if neg_ratio > 0 and len(neg_samples) > 0:
        num_neg = int(neg_ratio * batch_size) + 1
    num_pos = batch_size - num_neg

    for d in pos_samples:
        sample = {'_id': d[0]}
        bbox = [int(float(d[3])), int(float(d[4])), int(float(d[5])), int(float(d[6]))]
        sample['bbox'] = bbox
        print(d[1])
        gt = cv2.imread(os.path.join(data_root, d[1]), cv2.IMREAD_GRAYSCALE)
        ret, buf = cv2.imencode(compressParams[0], gt, [cv2.IMWRITE_JPEG_QUALITY, compressParams[1]])
        if not ret:
            print 'Image: %s compression error!!' % (os.path.join(data_root, d[1]))
            continue
        sample['gt'] = gt  # buf

        img = cv2.imread(os.path.join(data_root, d[0]))
        ret, buf = cv2.imencode(compressParams[0], img, [cv2.IMWRITE_JPEG_QUALITY, compressParams[1]])
        if not ret:
            print 'Image: %s compression error!!' % (os.path.join(data_root, d[1]))
            continue
        sample['img'] = img  # buf

        data_set.append(sample)
        i += 1
        cur += 1
        if i % num_pos == 0 and len(neg_samples) > 0:
            shuffle(neg_samples)
            j = num_neg
            for nd in neg_samples:
                n_sample = {'_id': nd[0], 'gt': None}
                img = cv2.imread(os.path.join(data_root, nd[0]))
                ret, buf = cv2.imencode(compressParams[0], img, [cv2.IMWRITE_JPEG_QUALITY, compressParams[1]])
                if not ret:
                    print 'Image: %s compression error!!' % (os.path.join(data_root, nd[0]))
                    continue
                n_sample['img'] = img  # buf

                data_set.append(n_sample)
                j -= 1
                if j <= 0:
                    break
                cur += 1

        if cur % batch_size == 0:
            write_file = open(os.path.join(save_path, 'dataset_' + str(cur / batch_size) + '.pkl'), 'wb')
            cPickle.dump(data_set, write_file, -1)
            write_file.close()
            data_set = []
            print 'batch:', cur / batch_size

    write_file = open(os.path.join(save_path, 'dataset_' + str((i / batch_size) + 1) + '.pkl'), 'wb')
    cPickle.dump(data_set, write_file, -1)
    write_file.close()
    print 'finish'


def de_batch_serialize(batch_data):
    f = open(batch_data, 'rb')
    data = cPickle.load(f)
    f.close()
    return data


def serialize_train_data():
    data_root = '/public_datasets/people_sementation_data/data_new'
    data_tab_train = 'train_multi_add_dark_list_2018.10.09.txt'  # 'train_multi_list_2017.2.8.txt'
    save_path_train = '/home/guoweiyu/people_segmentation/train_set_dark'
    batch_serialize(data_root, data_tab_train, ['.jpg', 80], 2048, 0.1, save_path_train)


def serialize_test_data():
    data_root = '/public_datasets/people_sementation_data/data_new'
    data_tab_test = 'benchmark_multi_list_2017.1.13.txt'
    save_path_test = '/home/guoweiyu/people_segmentation/test_set'
    batch_serialize(data_root, data_tab_test, ['.jpg', 80], 2048, 0.1, save_path_test)


def get_list_from_json():
    # prefix = '02'
    # data_root = os.path.join('/home/guoweiyu/people_segmentation/seg_dark_night', prefix)
    data_root = '/home/guoweiyu/people_segmentation/seg_dark_night2'
    seg_root = os.path.join(data_root, 'seg2')
    data_list = os.path.join(data_root, 'img/multi_person.json')
    output = 'seg_dark_night2.txt'
    with open(data_list, 'r') as f:
        data = json.loads(f.read())
    print(data[0]['filename'])
    print(data[0]['annotations'][0]['x'], data[0]['annotations'][0]['y'], data[0]['annotations'][0]['width'],
          data[0]['annotations'][0]['height'])
    print(len(data))
    dat = []
    for d in data:
        filename = d['filename'][:-4]
        for idx, bbox in enumerate(d['annotations']):
            seg_file = filename + '_' + str(idx) + '.jpg'
            if os.path.exists(os.path.join(seg_root, seg_file)) is not True:
                continue

            x = str(bbox['x'])
            y = str(bbox['y'])
            w = str(bbox['width'])
            h = str(bbox['height'])
            # im_pth = os.path.join('seg_dark_night', prefix, 'img', d['filename'])
            # seg_pth = os.path.join('seg_dark_night', prefix, 'seg', seg_file)
            im_pth = os.path.join('seg_dark_night2', 'img', d['filename'])
            seg_pth = os.path.join('seg_dark_night2', 'seg2', seg_file)
            dat.append([im_pth, seg_pth, '1', x, y, w, h])
    print(len(dat))
    with open(output, 'a') as f:
        for d in dat:
            txt = ''
            for t in d:
                txt += t + ' '
            f.write(txt[:-1] + '\n')
    print('finish making list.')


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'poly':
        def lambda_rule(cur_iter):
            lr_l = math.pow(1.0 + opt.gamma * cur_iter / opt.max_iter, -1.0 * opt.power)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def print_network(net, opt):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

    expr_dir = os.path.join(opt.train_root, opt.model_name)
    if not os.path.isdir(expr_dir):
        os.mkdir(expr_dir)

    file_name = os.path.join(expr_dir, 'net.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Network -------------\n')
        opt_file.write('%s\n' % net)
        opt_file.write('Total number of parameters: %d \n' % num_params)
        opt_file.write('-------------- End ----------------\n')


def change_file_name():
    data_root = '/home/guoweiyu/people_segmentation/seg_dark_night2/seg'
    for _, _, file_names in os.walk(data_root):
        fnames = file_names
        break
    out_dir = '/home/guoweiyu/people_segmentation/seg_dark_night2/seg2'
    data_list = []
    for idx, f in enumerate(fnames):
        fn = f[:-4]
        pth = os.path.join(data_root, f)
        img = cv2.imread(pth)
        # out_pth = os.path.join(out_dir, str(idx) + '.jpg')
        out_pth = os.path.join(out_dir, fn + '.jpg')
        cv2.imwrite(out_pth, img)
        data_list.append(os.path.join('seg_dark_night2/seg2', str(idx) + '.jpg'))

    with open('neg_data_list.txt', 'w') as f:
        for d in data_list:
            f.write(d + ' ' + 'null' + ' ' + '0 0 0 0 0\n')


'''
class adjustLearningRate:
    def __init__(self, optimizer, params={}, lr_policy='fixed'):
        self.optimizer = optimizer
        self.gamma = 1
        self.power = 1
        self.step_size = 1
        self.base_lr = 0.1
        self.stepvalue = []
        self.max_iter = 1
        self.lowest_lr = 0.000001
        self.cur_iter = 0
        self.lr_policy = lr_policy

        if params.has_key('gamma'):
            self.gamma = params['gamma']
        if params.has_key('step_size'):
            self.step_size = params['step_size']
        if params.has_key('base_lr'):
            self.base_lr = params['base_lr']
        if params.has_key('power'):
            self.power = params['power']
        if params.has_key('stepvalue'):
            self.stepvalue = params['stepvalue']
        if params.has_key('max_iter'):
            self.max_iter = params['max_iter']
        if params.has_key('lowest_lr'):
            self.lowest_lr = params['lowest_lr']

        self.rate = self.base_lr

        self.policy = None
        if lr_policy == 'fixed':
            self.policy = self.fixed_policy
        elif lr_policy == 'step':
            self.policy = self.step_policy
        elif lr_policy == 'exp':
            self.policy = self.exp_policy
        elif lr_policy == 'inv':
            self.policy = self.inv_policy
        elif lr_policy == 'multistep':
            self.policy = self.multistep_policy
        elif lr_policy == 'poly':
            self.policy = self.poly_policy
        elif lr_policy == 'sigmoid':
            self.policy = self.sigmoid_policy
        else:
            raise ValueError('Unknown learning rate policy: %s' % (lr_policy))

    def setLearningRate(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.rate

    def fixed_policy(self):
        return False

    def step_policy(self):
        cur_step = self.cur_iter // self.step_size
        self.rate = self.base_lr * math.pow(self.gamma, cur_step)
        return True

    def exp_policy(self):
        self.rate = self.base_lr * math.pow(self.gamma, self.cur_iter)
        return True

    def inv_policy(self):
        self.rate = self.base_lr * math.pow(1.0 + self.gamma * self.cur_iter, -1.0 * self.power)
        return True

    def multistep_policy(self):
        for i in range(len(self.stepvalue)):
            if self.cur_iter > self.stepvalue[i]:
                break
        rate = self.base_lr * math.pow(self.gamma, i + 1)
        if rate < self.rate:
            self.rate = rate
            return True
        else:
            return False

    def poly_policy(self):
        self.rate = self.base_lr * math.pow((1.0 - 1.0 * self.cur_iter / self.max_iter), self.power)
        return True

    def sigmoid_policy(self):
        self.rate = self.base_lr * (1.0 / (1.0 + math.exp(-1 * self.gamma * (self.cur_iter - self.step_size))))
        return True

'''

'''
    def __call__(self) :
        self.policy()
        self.setLearningRate()
        self.cur_iter += 1 
        print self.rate,self.lr_policy
'''
'''

    def update_lr(self):
        if self.cur_iter >= self.max_iter or self.rate <= self.lowest_lr:
            return False
        if self.policy():
            self.setLearningRate()
        self.cur_iter += 1
        return True
'''

if __name__ == '__main__':
    '''
    change_file_name()
    exit(88)
    
    
    get_list_from_json()
    exit(99)
    '''

    '''
    serialize_train_data()
    print 'training set finished'
    exit(66)
    '''

    '''
    serialize_test_data()
    print 'testing set finished'
    exit(0)
    '''

    # for debug
    data_root = '/public_datasets/people_sementation_data/data_new'
    data_tab_debug = 'seg_dark_night2.txt'
    save_path_debug = '/home/guoweiyu/people_segmentation/debug'
    batch_serialize(data_root, data_tab_debug, ['.jpg', 80], 20, 0.1, save_path_debug)
    data = de_batch_serialize(os.path.join(save_path_debug, 'dataset_1.pkl'))
    for c, d in enumerate(data):
        print d['_id']
        img = d['img']
        # img = cv2.imdecode(img, cv2.CV_LOAD_IMAGE_COLOR)
        if d['gt'] is not None:
            print d['bbox'], img.shape

            cv2.imshow('image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            x = d['bbox'][0]
            y = d['bbox'][1]
            w = d['bbox'][2]
            h = d['bbox'][3]
            cv2.imshow('image', img[y:y + h, x:x + w, :])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            gt = d['gt']
            # gt = cv2.imdecode(gt, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            cv2.imshow('gt', gt)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # break
