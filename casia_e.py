'''
@Author: Jilong Wang
@Date: 2019-01-10 13:21:08
@LastEditors: Jilong Wang
@Email: jilong.wang@watrix.ai
@LastEditTime: 2019-01-16 14:43:57
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
import subprocess

# Remember to add your installation path here
sys.path.append('./build/python')
# Parameters for OpenPose. Take a look at C++ OpenPose example for meaning of components. Ensure all below are filled
try:
    from openpose import *
except:
    raise Exception('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')

import segmentation

class GaitExtractor:
    def __init__(self, gpuid, det_batch_size=20):
        # openpose threshold
        self.threshold = 0.15
        self.op_net, self.seg_net = self.net_init(gpu_id=gpuid)
        # save images and segmentation in order to reduce imread time cost
        self.img = {}
        self.seg = {}

    def net_init(self, gpu_id=0):
        '''
        @description: load openpose & segementation models
        @param {gpu_id} 
        @return: instance of op_net, seg_net
        '''
        # load openpose model
        params = dict()
        params["logging_level"] = 3
        params["output_resolution"] = "-1x-1"
        params["net_resolution"] = "-1x256"
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

        seg_net = segmentation.Segmentation(gpuid=int(gpu_id))
        return op_net, seg_net
    
    def find_first_role(self, frame_result):
        '''
        @description: Strip blank frames and frames of incomplete bodys from both ends.
        @param {openpose result([frames x num_of_people x 25 x 3])} 
        @return: first frame(int), last frame(int)
        '''
        first_frame, last_frame = -1, -1
        for i, pack in enumerate(frame_result):
            _, keypoints = pack
            if keypoints.shape[0] == 0:
                continue
            for j in range(keypoints.shape[0]):
                # If there is a role's body is complete regardless of whether it is main role or not, stop looping
                if self.check_integrity(keypoints[j,...]) == 7:
                    first_frame = i
                    break
            if first_frame > -1:
                break
    
        for i, pack in enumerate(reversed(frame_result)):
            _, keypoints = pack
            if keypoints.shape[0] == 0:
                continue
            for j in range(keypoints.shape[0]):
                # If there is a role's body is complete regardless of whether it is main role or not, stop looping
                if self.check_integrity(keypoints[j,...]) == 7:
                    last_frame = i
                    break
            if last_frame > -1:
                break
        assert first_frame != -1 or last_frame != -1, "Didn't find any frame that contains a complete body"

        return first_frame, last_frame
        
    def check_integrity(self, points):
        '''
        @description: get each person's keypoints array then check 7 main parts of human body--Neck, Knees, Ankles, BigToes, Heels, Hips and Shoulders.
        @param {openpose format keypoints array} 
        @return: number of keypoints(max is 7)
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

        if len([x for x in points if (x[0]!=0 or x[1]!=0) and x[2] > self.threshold]) > 22:
            count = 7
        return count

    def find_max(self, im_name, keypoints, shape):
        '''
        @description: Find the largest person in a frame.
        @param {im_name, openpose_keypoints, img_shape} 
        @return: largest person's keypoints count(max is 7), bbox coord
        '''
        Xmax, Xmin, Ymax, Ymin = 0, 0, 0, 0
        count = 0
        people_rect = []
        people_count = []

        # get all people's bbox
        for i in range(keypoints.shape[0]):
            coords = keypoints[i, ...]
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

            # expand bbox border                        
            # height = ymax - ymin
            # width = xmax - xmin
            # h = int(round(height * 0.1))
            # w = int(round(width * 0.1))
            xmin -= 70
            ymin -= 100
            xmax += 70
            ymax += 100

            # check border
            xmin = 0 if xmin < 0 else xmin
            xmax = 0 if xmax < 0 else xmax
            xmin = shape[1] if xmin > shape[1] else xmin
            xmax = shape[1] if xmax > shape[1] else xmax
            ymin = 0 if ymin < 0 else ymin
            ymax = 0 if ymax < 0 else ymax
            ymin = shape[0] if ymin > shape[0] else ymin
            ymax = shape[0] if ymax > shape[0] else ymax

            # check if the bbox is valid
            if int(xmin) >= int(xmax) or int(ymin) >= int(ymax):
                continue
            people_rect.append([int(xmin), int(xmax), int(ymin), int(ymax)])
            people_count.append(self.check_integrity(coords))
        
        # find the largest person
        if len(people_rect) > 0:
            area, self.seg[im_name] = self.seg_net.get_area(self.img[im_name], people_rect)
            if sum(area) > 0:
                loc = np.argmax(area)
                Xmin = people_rect[loc][0]
                Xmax = people_rect[loc][1]
                Ymin = people_rect[loc][2]
                Ymax = people_rect[loc][3]
                count = people_count[loc]

        return count, [Xmin, Xmax, Ymin, Ymax]

    def is_main_role(self, coord1, coord2):
        '''
        @description: using Euclidean Distance to check wether the role is main role or not  
        @param {coord1, coord2} 
        @return: True or False
        '''
        x1_center = (coord1[0] + coord1[1])
        y1_center = (coord1[2] + coord1[3])

        x2_center = (coord2[0] + coord2[1])
        y2_center = (coord2[2] + coord2[3])
        distance = (x1_center - x2_center) ** 2 + (y1_center - y2_center) ** 2
        if distance < (self.shape[0]*0.12)**2 and x1_center != 0 and y1_center != 0:
            return True
        else:
            return False
            
    def find_first_main_role(self, frame_role):
        '''
        @description: Find the index of first frame where the MAIN role appears
        @param {frame_role:(im_name, result, shape)} 
        @return: index of the first frame where the MAIN role appear, bbox's coords of the MAIN role in this frame
        '''
        print('Finding the first frame of the main role.')
        roles = []
        # find main role in each frame
        for i in range(0, len(frame_role), 5):
            _, count, coord = frame_role[i]
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
        
        search_frame = frame_role[first_index: max_index]
        search_frame.reverse()
        for i in range(len(search_frame)):
            _, count, coord = search_frame[i]
            xmin, xmax, ymin, ymax = coord
            if self.is_main_role([xmin, xmax, ymin, ymax], main_role_coord):
                first_frame = max_index - i
                main_role_coord = [xmin, xmax, ymin, ymax]

        return first_frame, main_role_coord

    def find_main_role_in_each_frame(self, frame_role):
        '''
        @description: when more than one person exist in one frame, find out which person is our main role. 
        And track it from the first frame it appears to the last frame it disappears. If the frame doesn't contain the main role, stop and return.
        @param {find_max result(im_name, count, bbox's coords)} 
        @return main role's coord in each frame(im_name, count, bbox's coords), 
        '''
        frame_main_role = []
        first_frame, main_role_coord = self.find_first_main_role(frame_role)
        for im_name, count, coord in frame_role[first_frame:]:
            xmin, xmax, ymin, ymax = coord
            if self.is_main_role([xmin, xmax, ymin, ymax], main_role_coord):
                main_role_coord = [xmin, xmax, ymin, ymax]
                frame_main_role.append((im_name, count, [int(xmin), int(xmax), int(ymin), int(ymax)]))
            else:
                print([xmin, xmax, ymin, ymax], main_role_coord)
                print(im_name + ' is not main role')
                break

        return frame_main_role

    def is_moving(self, coord, still_coord):
        '''
        @description: Using Euclidean distance to check wether the role is moving.  
        @param {coord1, coord2} 
        @return: True or False
        '''
        x1_center = (coord[0] + coord[1])
        y1_center = (coord[2] + coord[3])

        x2_center = (still_coord[0] + still_coord[1])
        y2_center = (still_coord[2] + still_coord[3])
        distance = (x1_center - x2_center) ** 2 + (y1_center - y2_center) ** 2
        if distance > (self.shape[0]*0.018)**2:
            return True
        else:
            return False
        
    def delete_still_frame(self, frame_main_role):
        '''
        @description: Strip frames where MAIN role is still from both ends.
        @param {frame_main_role([im_name, count, bbox's coords])} 
        @return: index of first frame, index of last frame
        '''
        first_frame = 0
        last_frame = len(frame_main_role)
        main_role_coord = frame_main_role[0][2]
        for i, pack in enumerate(frame_main_role):
            _, count, coord = pack
            if self.is_moving(coord, main_role_coord) and count == 7:
                first_frame = i
                break

        reversed_frame_main_role = frame_main_role[:]
        reversed_frame_main_role.reverse()
        main_role_coord = reversed_frame_main_role[0][2]
        
        for i, pack in enumerate(reversed_frame_main_role):
            _, count, coord = pack
            if self.is_moving(coord, main_role_coord) and count == 7:
                last_frame = len(reversed_frame_main_role) - i
                break
        return first_frame, last_frame
        
    def save_results(self, frame_main_role, first_frame, last_frame, save_dir):
        '''
        @description: save MAIN role's continuous segmentation image
        @param {frame_main_role, first_frame, last_frame, save_dir} 
        @return: None
        '''
        # if having less than 10 frames, discard this video.
        if last_frame - first_frame < 10:
            return
        # if having enough frames, then abort the first 3 frame and last 3 frames in order to have intact person.
        first_frame += 3
        last_frame -= 3
        print("the frist frame is {}, the last frame is {}".format(frame_main_role[first_frame][0][5:-4], frame_main_role[last_frame-1][0][5:-4]))
        for im_name, _, _ in frame_main_role[first_frame: last_frame]:
            cv2.imwrite(os.path.join(save_dir, im_name[:-4] + '_dets.jpg'), self.seg[im_name])
            print('Saved: ' + os.path.join(save_dir, im_name[:-4] + '_dets.jpg'))
    
    def extract(self, img_dir, save_dir):
        '''
        @description: Extract a continuous gait from frames image and save MAIN role segmentation image
        @param {img_dir(where frames are stored), save_dir} 
        @return: None
        '''
        frame_result = []
        print('Processing {}:'.format(img_dir))

        # get all image names and sorted by name
        im_names = os.listdir(img_dir)
        im_names = [x for x in im_names if 'jpg' in x]
        im_names.sort(key=lambda x: int(x[5:-4]))
         
        # do openpose
        frame_result = []
        for im_name in im_names:
            img = cv2.imread(os.path.join(img_dir, im_name))
            # save images for later processing in order to reduce time cost
            self.img[im_name] = img
            self.shape = img.shape
            keypoints = self.op_net.forward(img, False)
            frame_result.append((im_name, keypoints))
        
        # delete background frame
        first_frame, last_frame = self.find_first_role(frame_result)

        frame_role = []
        # find largest role in each frame
        for im_name, keypoints in frame_result:
            if keypoints.shape[0] > 0:
                count, coord = self.find_max(im_name, keypoints, self.shape)         
                frame_role.append((im_name, count, coord))

        # find main role
        frame_main_role = self.find_main_role_in_each_frame(frame_role)
        
        first_frame, last_frame = self.delete_still_frame(frame_main_role)
        self.save_results(frame_main_role, first_frame, last_frame, save_dir)
        self.img = {}

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
