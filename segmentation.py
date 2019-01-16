'''
@Author: guoweiyu
@Date: 2019-01-15 11:06:20
@LastEditors: Jilong Wang
@Email: jilong.wang@watrix.ai
@LastEditTime: 2019-01-16 14:27:44
@Description: A Segmentation class which contains get_area method and forward method
'''
# coding: utf-8

import os
import cv2
import numpy
import torch
import sys
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from util.models import SegmentModel
from util.options import BaseOptions

class Segmentation:
    def __init__(self, gpuid=[0]):
        if type(gpuid) != list:
            gpuid = [gpuid]
        torch.cuda.set_device(gpuid[0])
        opt = BaseOptions()
        self.model = SegmentModel(opt, False)
        self.transform = transforms.Normalize(mean=(104.008, 116.669, 122.675), std=(1, 1, 1))
        self.input_height = 150
        self.input_width = 100
        
    def get_area(self, im, bboxes):
        '''
        @description: According bboxes, use seg_net to get person segment image then use the number of white pixels as bbox area.
        @param {image, bboxes} 
        @return: area list, max_area_person's seg image
        '''
        area = []
        segs = []
        for coord in bboxes:
            im_crop = im[coord[2]:coord[3], coord[0]:coord[1], :].copy()
            segResults = self.forward(im_crop)
            segs.append(segResults)
            area.append(numpy.sum(segResults/255))
        loc = numpy.argmax(area)
        result = segs[loc]
        result = self.resize_and_pad(segs[loc])
        return area, result

    def forward(self, im_crop):
        '''
        @description: use seg_net and get im_crop's person segmentation images
        @param {image array} 
        @return: seg_image array
        '''

        # image process
        im_width, im_height = im_crop.shape[1], im_crop.shape[0]
        im_crop = cv2.resize(im_crop, (self.input_width, self.input_height))
        im_crop = torch.from_numpy(im_crop.transpose((2, 0, 1)))
        im_crop = self.transform(im_crop.float())
        im_input = torch.zeros(1, 3, self.input_height, self.input_width)
        im_input[0] = im_crop

        self.model.set_test_input(im_input)
        pred = self.model.forward()

        # get output
        binary_seg_pred = pred['binary_seg_pred'].cpu()
        binary_seg_pred.view(binary_seg_pred.shape[0],binary_seg_pred.shape[2],binary_seg_pred.shape[3])
        binary_seg_pred = binary_seg_pred.data.numpy()
        pred_img = binary_seg_pred[0].reshape((self.input_height, self.input_width))
        pred_img = cv2.resize(pred_img, (im_width, im_height))

        thresholds = 0.9
        pred_img[pred_img > thresholds] = 255
        pred_img[pred_img < thresholds] = 0
        pred_img = cv2.erode(pred_img, kernel=(3,3), iterations=5)
        pred_img = cv2.morphologyEx(pred_img, cv2.MORPH_OPEN, kernel=(5,5))
        pred_img = self.get_largest_connected_region(pred_img)
        pred_img = cv2.dilate(pred_img, kernel=(5,5))

        return pred_img

    def get_largest_connected_region(self,inputImage):
        '''
        @description: when there are more than one person in an image, only maintain the largest one as the result
        @param {seg_image array} 
        @return: image array
        '''
        inputImage = inputImage.astype('uint8')
        img = inputImage.copy()
        _, contours, _ = cv2.findContours(inputImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return img
        areas = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            areas.append(area)

        del contours[numpy.argmax(areas)]
        temp = numpy.zeros((img.shape[0], img.shape[1]), dtype=numpy.uint8)
        cv2.fillPoly(temp, contours, color=(255), lineType=4)
        img[temp == 255] = 0

        return img
        
    def resize_and_pad(self, image):
        '''
        @description: resize and pad an image into 256x256
        @param {image array} 
        @return: resize_and_pad_image array
        '''
        w, h = image.shape[1], image.shape[0]
        m = max(w, h)
        ratio = 256.0 / m
        new_w, new_h = int(ratio * w), int(ratio *h)
        assert new_w > 0 and new_h > 0
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        W, H = 256, 256
        top = (H - new_h) // 2
        bottom = (H - new_h) // 2
        if top + bottom + h < H:
            bottom += 1

        left = (W - new_w) // 2
        right = (W - new_w) // 2
        if left + right + w < W:
            right += 1

        pad_image = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value = (0))
        return pad_image

if __name__ == '__main__':
    seg = Segmentation()
    img = cv2.imread('videoframes/videoframe-001_scene3_nm_L_090_1_s/frame40.jpg')
    print(seg.get_area(img, [[973, 1167, 259, 906],[1797, 1912, 449, 665]]))