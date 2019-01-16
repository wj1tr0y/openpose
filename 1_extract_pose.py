'''
@Author: Jilong Wang
@Date: 2019-01-10 14:12:01
@LastEditors: Jilong Wang
@Email: jilong.wang@watrix.ai
@LastEditTime: 2019-01-15 11:16:42
@Description: file content
'''
# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import numpy as np
# Remember to add your installation path here
# Option a
dir_path = os.path.dirname(os.path.realpath(__file__))
if platform == "win32": sys.path.append(dir_path + '/../../python/openpose/');
else: sys.path.append('./build/python');
# Option b
# If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
# sys.path.append('/usr/local/python')

# Parameters for OpenPose. Take a look at C++ OpenPose example for meaning of components. Ensure all below are filled
try:
    from openpose import *
except:
    raise Exception('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    
def init():
    params = dict()
    params["logging_level"] = 3
    params["output_resolution"] = "-1x-1"
    params["net_resolution"] = "-1x256"
    params["model_pose"] = "BODY_25"
    params["alpha_pose"] = 0.6
    params["scale_gap"] = 0.25
    params["scale_number"] = 1
    params["render_threshold"] = 0.15
    # If GPU version is built, and multiple GPUs are available, set the ID here
    params["num_gpu_start"] = 0
    params["disable_blending"] = False
    # Ensure you point to the correct path where models are located
    params["default_model_folder"] = "./models/"
    # Construct OpenPose object allocates GPU memory
    openpose = OpenPose(params)
    return openpose

def check_integrity(points):
        '''
        @description: get an image then return how many keypoints were detected
        @param {nparray image} 
        @return: number of keypoints
        '''
        threshold = 0.15
        count = 0
        if sum(points[1]) > 0 and points[1][2] > threshold: # Neck
            count += 1
        if sum(points[10]) >0 and sum(points[13]) >0 and points[10][2] > threshold and points[13][2] > threshold: # Knee
            count += 1
        if sum(points[11]) > 0 and sum(points[14]) >0 and points[11][2] > threshold and points[14][2] > threshold: # Ankle
            count += 1
        if sum(points[19]) >0 and sum(points[22]) >0 and points[19][2] > threshold and points[22][2] > threshold: # BigToe
            count += 1
        else:
            count -= 1
        if sum(points[21]) >0 and sum(points[24]) >0 and points[21][2] > threshold and points[24][2] > threshold: # Heel
            count += 1
        else:
            count -= 1
        if sum(points[9]) >0 and sum(points[8]) >0 and sum(points[12]) >0 and points[9][2] > threshold and points[8][2] > threshold and points[12][2] > threshold: # Hip
            count += 1
        if sum(points[2]) >0 and sum(points[5]) >0 and points[2][2] > threshold and points[5][2] > threshold: # double Shoulder
            count += 1
        else:
            count -= 1

        if len([x for x in points if (x[0]!=0 or x[1]!=0) and x[2] > threshold]) > 22:
            print(len([x for x in points if (x[0]!=0 or x[1]!=0) and x[2] > threshold]))
            count = 7
        return count
if __name__ == '__main__':

    openpose = init()
    # Read new image
    # img = cv2.imread("../../../examples/media/COCO_val2014_000000000192.jpg")
    img = cv2.imread("videoframes/videoframe-001_scene3_nm_L_090_1_s/frame108.jpg")
    # Output keypoints and the image with the human skeleton blended on it
    keypoints, output_image = openpose.forward(img, True)
    # Print the human pose keypoints, i.e., a [#people x #keypoints x 3]-dimensional numpy object with the keypoints of all the people on that image
    for j in range(keypoints.shape[0]):
        print(check_integrity(keypoints[j, ...]), np.sum(keypoints[j,:,2]))
    print(keypoints)
    # Display the image
    cv2.imshow("output", output_image)
    cv2.imwrite('38.jpg', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
