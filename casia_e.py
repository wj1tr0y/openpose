'''
@Author: Jilong Wang
@Date: 2019-01-10 13:21:08
@LastEditors: Jilong Wang
@Email: jilong.wang@watrix.ai
@LastEditTime: 2019-01-10 17:36:34
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
        self.threshold = 0.20
        self.op_net = self.net_init(gpu_id=gpuid)

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
        params["net_resolution"] = "-1x368"
        params["model_pose"] = "BODY_25"
        params["alpha_pose"] = 0.6
        params["scale_gap"] = 0.25
        params["scale_number"] = 1
        params["render_threshold"] = self.threshold
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
            if keypoints.shape[0] == 0:
                continue

            for j in range(keypoints.shape[0]):
                if self.check_integrity(keypoints[j,...]) == 7:
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

    def is_main_role(self, coord1, coord2):
        '''
        @description: using Euclidean Distance to jugde wether the role is main role or not  
        @param {coord1, coord2} 
        @return: True or False
        '''
        x1_center = (coord1[0] + coord1[1])
        y1_center = (coord1[2] + coord1[3])

        x2_center = (coord2[0] + coord2[1])
        y2_center = (coord2[2] + coord2[3])
        distance = (x1_center - x2_center) ** 2 + (y1_center - y2_center) ** 2
        if distance < (1080*0.1)**2 and x1_center != 0 and y1_center != 0:
            return True
        else:
            return False
            
    def find_first_main_role(self, frame_result):
        '''
        @description: find the first frame where the main role appear
        @param {frame_result:(im_name, result, shape)} 
        @return: the first frame where the main role appear, coord of the main role in this frame
        '''
        print('finding the first frame of the main role.')
        roles = []
        # find main role in each frame
        for i in range(0, len(frame_result), 5):
            im_name, count, coord = frame_result[i]
            xmin, xmax, ymin, ymax = coord
            if xmin + xmax + ymin + ymax != 0 and count == 7:
                roles.append((i, [xmin, xmax, ymin, ymax]))

        # find the largest role which definitely is the main role
        max_area = 0.0
        max_index = 0
        main_role_coord = []
        for i, coord in roles:
            xmin, xmax, ymin, ymax = coord
            area = (xmax - xmin) * (ymax - ymin)
            if max_area < area:
                max_area = area
                max_index = i
                main_role_coord = [xmin, xmax, ymin, ymax]

        first_index = roles[0][0]

        # find the first frame of the largest role
        if first_index == max_index:
            return max_index, main_role_coord
            
        search_frame = frame_result[first_index: max_index]
        search_frame.reverse()
        for i in range(len(search_frame)):
            im_name, count, coord = search_frame[i]
            xmin, xmax, ymin, ymax = coord
            if self.is_main_role([xmin, xmax, ymin, ymax], main_role_coord):
                first_frame = max_index - i
                main_role_coord = [xmin, xmax, ymin, ymax]

        return first_frame, [xmin, xmax, ymin, ymax]

    def find_main_role_in_each_frame(self, frame_result):
        '''
        @description: when more than one person exist in one frame, find out which person is our main role. And track it from the first frame it appears to the frame it disappears.if the frame doesn't contain the main role, abort it
        @param {detection result(im_name, result, shape)} 
        @return main role's coord in each frame(im_name, coord), 
        '''
        frame_main_role = []
        first_frame, main_role_coord = self.find_first_main_role(frame_result)

        for im_name, count, coord in frame_result[first_frame:]:
            xmin, xmax, ymin, ymax = coord

            if self.is_main_role([xmin, xmax, ymin, ymax], main_role_coord):
                main_role_coord = [xmin, xmax, ymin, ymax]
                frame_main_role.append((im_name, count, [int(xmin), int(xmax), int(ymin), int(ymax)]))
            else:
                print(im_name + ' is not main role')
                break

        return frame_main_role

    def is_moving(self, coord, still_coord):
        x1_center = (coord[0] + coord[1])
        y1_center = (coord[2] + coord[3])

        x2_center = (still_coord[0] + still_coord[1])
        y2_center = (still_coord[2] + still_coord[3])
        distance = (x1_center - x2_center) ** 2 + (y1_center - y2_center) ** 2
        if distance > (1080*0.008)**2:
            return True
        else:
            return False
        
    def delete_still_frame(self, frame_main_role):
        first_frame = 0
        last_frame = len(frame_main_role)
        main_role_coord = frame_main_role[0][2]
        for i, pack in enumerate(frame_main_role):
            _, _, coord = pack
            if self.is_moving(coord, main_role_coord):
                first_frame = i
                break

        reversed_frame_main_role = frame_main_role[:]
        reversed_frame_main_role.reverse()
        main_role_coord = reversed_frame_main_role[0][2]
        
        for i, pack in enumerate(reversed_frame_main_role):
            _, _, coord = pack
            if self.is_moving(coord, main_role_coord):
                last_frame = len(reversed_frame_main_role) - i
                break
        return first_frame, last_frame
        
    def save_results(self, frame_main_role, first_frame, last_frame, img_dir, save_dir):
        # if having enough frames, then abort the first 5 frame and last 25 frames in order to have intact person
        if last_frame - first_frame < 15:
            return
        last_frame -= 3
        print("the frist frame is {}, the last frame is {}".format(frame_main_role[first_frame][0][5:-4], frame_main_role[last_frame-1][0][5:-4]))
        for im_name, _, coord in frame_main_role[first_frame: last_frame]:
            img = cv2.imread(os.path.join(img_dir, im_name), cv2.IMREAD_COLOR)
            img = img[coord[2]:coord[3], coord[0]:coord[1]]
            # plt.imshow(img)
            # plt.show()
            cv2.imwrite(os.path.join(save_dir, im_name[:-4] + '_dets.jpg'), img)
            print('Saved: ' + os.path.join(save_dir, im_name[:-4] + '_dets.jpg'))
    
    def extract(self, img_dir, save_dir):
        frame_result = []
        print('Processing {}:'.format(img_dir))

        # get all image names and sorted by name
        im_names = os.listdir(img_dir)
        im_names = [x for x in im_names if 'jpg' in x]
        im_names.sort(key=lambda x: int(x[5:-4]))

        # do openpose
        frame_result = []
        for im_name in im_names:
            img = cv2.imread(os.path.join(img_dir, im_name), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.shape = img.shape
            keypoints = self.op_net.forward(img, False)
            frame_result.append((im_name, keypoints))
        
        # delete background frame
        first_frame = self.find_first_role(frame_result)
        last_frame = len(frame_result) - self.find_first_role(reversed(frame_result))
        frame_result = frame_result[first_frame:last_frame]

        frame_role = []
        # find largest role in each frame
        for im_name, keypoints in frame_result:
            if keypoints.shape[0] > 0:
                role, coord = self.find_max(keypoints, self.shape)
                count = self.check_integrity(keypoints[role, ...])
                frame_role.append((im_name, count, coord))

        # find main role
        frame_main_role = self.find_main_role_in_each_frame(frame_role)

        first_frame, last_frame = self.delete_still_frame(frame_main_role)
        self.save_results(frame_main_role, first_frame, last_frame, img_dir, save_dir)

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
