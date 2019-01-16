'''
@Author: Jilong Wang
@Date: 2019-01-15 11:51:01
@LastEditors: Jilong Wang
@Email: jilong.wang@watrix.ai
@LastEditTime: 2019-01-15 12:38:53
@Description: file content
'''
# coding: utf-8

class BaseOptions:
    def __init__(self, gpuid=[0], modelfile='models/checkpoint/'):
        self.gpus = gpuid
        self.train_root = modelfile
        self.resume = True
        self.model_name = 'seg_shuffle'
        self.which_model = 'best'
        self.typ = 'conv'
        self.input_chs = 3
        self.filter_num = 64
        self.init_type = 'xavier'