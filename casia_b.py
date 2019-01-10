'''
@Author: Jilong Wang
@Date: 2019-01-10 13:21:08
@LastEditors: Jilong Wang
@Email: jilong.wang@watrix.ai
@LastEditTime: 2019-01-10 17:28:06
@Description: file content
'''
from __future__ import print_function
import argparse
import os
import sys
import numpy as np
import skimage.io as io
import cv2
import time
import shutil
import matplotlib.pyplot as plt
# Remember to add your installation path here
sys.path.append('./build/python')
# Parameters for OpenPose. Take a look at C++ OpenPose example for meaning of components. Ensure all below are filled
try:
    from openpose import *
except:
    raise Exception('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')


class GaitExtractor:
    def __init__(self, gpuid, det_batch_size=20):
        self.op_net = self.net_init(gpu_id=gpuid)
        self.threshold = 0.10
    def net_init(self, gpu_id=0):
        '''
        @description: load detection & openpose & segementation models
        @param {None} 
        @return: instance of op_net
        '''
        # load openpose model
        params = dict()
        params["logging_level"] = 3
        params["output_resolution"] = "-1x-1"
        params["net_resolution"] = "-1x240"
        params["model_pose"] = "BODY_25"
        params["alpha_pose"] = 0.6
        params["scale_gap"] = 0.25
        params["scale_number"] = 1
        params["render_threshold"] = 0.10
        # If GPU version is built, and multiple GPUs are available, set the ID here
        params["num_gpu_start"] = int(gpu_id)
        params["disable_blending"] = False
        # Ensure you point to the correct path where models are located
        params["default_model_folder"] = "/home/wjltroy/codeground/openpose/models/"
        # Construct OpenPose object allocates GPU memory
        op_net = OpenPose(params)

        return op_net
    
    def find_first_role(self, frame_result):
        for i, pack in enumerate(frame_result):
            _, keypoints = pack
            if keypoints == 7:
                return i + 5
        return -1
        
    def check_integrity(self, points):
        '''
        @description: get an image then return how many keypoints were detected
        @param {nparray image} 
        @return: number of keypoints
        '''
        count = 0
        if sum(points[1]) > 0 and points[1][2] > self.threshold: # Neck
            count += 1
        if sum(points[10]) >0 and sum(points[13]) >0 and points[10][2] > self.threshold and points[13][2] > self.threshold: # Knee
            count += 1
        if sum(points[11]) > 0 and sum(points[14]) >0 and points[11][2] > self.threshold and points[14][2] > self.threshold: # Ankle
            count += 1
        if sum(points[19]) >0 and sum(points[22]) >0 and points[19][2] > self.threshold and points[22][2] > self.threshold: # BigToe
            count += 1
        else:
            count -= 1
        if sum(points[21]) >0 and sum(points[24]) >0 and points[21][2] > self.threshold and points[24][2] > self.threshold: # Heel
            count += 1
        else:
            count -= 1
        if sum(points[9]) >0 and sum(points[8]) >0 and sum(points[12]) >0 and points[9][2] > self.threshold and points[8][2] > self.threshold and points[12][2] > self.threshold: # Hip
            count += 1
        if sum(points[2]) >0 and sum(points[5]) >0 and points[2][2] > self.threshold and points[5][2] > self.threshold: # double Shoulder
            count += 1
        else:
            count -= 1
        return count

    def find_max(self, keypoints, shape):
        role = 0
        max_area = 0
        Xmax, Xmin, Ymax, Ymin = 0, 0, 0, 0
        for i in range(keypoints.shape[0]):
            coords = keypoints[i, ...]
            if self.check_integrity(coords) < 5:
                continue
            xmin, xmax, ymin, ymax = 10000, 0, 10000, 0
            
            # find rectangle
            for coord in coords:
                x, y, conf = coord
                if conf > self.threshold:
                    if x < xmin:
                        xmin = x
                    if x > xmax:
                        xmax = x
                    if y < ymin:
                        ymin = y
                    if y > ymax:
                        ymax = y
            area = (ymax - ymin) * (xmax - xmin)
            
            if area > max_area:
                max_area = area
                role = i
                Xmin = xmin
                Xmax = xmax
                Ymin = ymin
                Ymax = ymax
                
        height = Ymax - Ymin
        width = Xmax - Xmin
        h = int(round(height * 0.1))
        w = int(round(width * 0.1))
        Xmin -= 2 * w
        Ymin -= 2 * h
        Xmax += 2 * w
        Ymax += h
        # check border
        Xmin = 0 if Xmin < 0 else Xmin
        Xmax = 0 if Xmax < 0 else Xmax
        Xmin = shape[1] if Xmin > shape[1] else Xmin
        Xmax = shape[1] if Xmax > shape[1] else Xmax
        Ymin = 0 if Ymin < 0 else Ymin
        Ymax = 0 if Ymax < 0 else Ymax
        Ymin = shape[0] if Ymin > shape[0] else Ymin
        Ymax = shape[0] if Ymax > shape[0] else Ymax
        return role, [Xmin, Xmax, Ymin, Ymax]

    def save_results(self, frame_result, first_frame, last_frame, img_dir, save_dir):
        # if having enough frames, then abort the first 5 frame and last 25 frames in order to have intact person
        if last_frame - first_frame < 15:
            return
        last_frame -= 3
        print("the frist frame is {}, the last frame is {}".format(frame_result[first_frame][0][5:-4], frame_result[last_frame-1][0][5:-4]))
        for im_name, _ in frame_result[first_frame: last_frame]:
            img = cv2.imread(os.path.join(img_dir, im_name), cv2.IMREAD_COLOR)
            cv2.imwrite(os.path.join(save_dir, im_name[:-4] + '_dets.jpg'), img)
            print('Saved: ' + os.path.join(save_dir, im_name[:-4] + '_dets.jpg'))
            
    def extract(self, img_dir, save_dir):
        frame_result = []
        print('Processing {}:'.format(img_dir))

        # get all image names and sorted by name
        im_names = os.listdir(img_dir)
        im_names = [x for x in im_names if 'jpg' in x]
        im_names.sort(key=lambda x: int(x[5:-4]))

        frame_result = []
        for im_name in im_names:
            img = cv2.imread(os.path.join(img_dir, im_name), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            keypoints = self.op_net.forward(img, False)
            if keypoints.shape[0] > 0:
                if keypoints.shape[0] > 1:
                    role, coord = self.find_max(keypoints, img.shape)
                else:
                    role = 0
                count = self.check_integrity(keypoints[role, ...])
                frame_result.append((im_name, count))


        first_frame = self.find_first_role(frame_result)
        last_frame = len(frame_result) - self.find_first_role(reversed(frame_result))
        self.save_results(frame_result, first_frame, last_frame, img_dir, save_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "run test and get all result")
    parser.add_argument("--gpuid",
        help = "The gpu chosen to run the model.", required=True)
    parser.add_argument("--out-dir",
        help = "The output directory where we store the result.", required=True)
    parser.add_argument("--test-set", 
        help = "which sets your wanna run test.", required=True)

    args = parser.parse_args()
    # gpu preparation
    assert len(args.gpuid) == 1, "You only need to choose one gpu. But {} gpus are chosen.".format(args.gpuid)
    
    save_dir = args.out_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    img_dir = args.test_set
    if not os.path.exists(img_dir):
        print("{} doesn't exists".format(img_dir))
        sys.exit(0)

    gait = GaitExtractor(args.gpuid)
    gait.extract(img_dir, save_dir)
