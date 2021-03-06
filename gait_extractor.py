'''
@Author: Jilong Wang
@Date: 2019-01-05 17:47:31
@LastEditors: Jilong Wang
@Email: jilong.wang@watrix.ai
@LastEditTime: 2019-01-16 17:56:48
@Description: Gait extractor. Supporting single video file extraction{pass the video file path} and mutli-videos extraction{pass the video folder path}
'''
import cv2
import argparse
import os
import shutil
import subprocess
import zipfile
import time
import sys

def split_video(video_name, frame_save_dir):
    '''
    @description: split video into frames
    @param {video path, frame save dir} 
    @return: None
    '''
    if not os.path.exists(frame_save_dir):
        os.makedirs(frame_save_dir)
        cap = cv2.VideoCapture(video_name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        frame_count = 1
        success = True
        while(success):
            success, frame = cap.read()
            if success:
                print 'Reading frames: {}\r'.format(frame_count),
                cv2.imwrite(os.path.join(frame_save_dir, 'frame{}.jpg'.format(frame_count)), frame)
                frame_count += 1
            else:
                print ''
        cap.release()
    else:
        print('Video had already split in frames stored in {}'.format(frame_save_dir))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "run test and get all result")
    parser.add_argument("video", 
        help = "path of test video")
    parser.add_argument("dataset",
        help = "use which dataset.")
    parser.add_argument("--gpuid",
        help = "The gpu chosen to run the model.", required=True)

    args = parser.parse_args()
    assert len(args.gpuid) == 1, "You only need to choose one gpu. But {} gpus are chosen.".format(args.gpuid)

    # single video or videos folder
    SET_TEST = False
    if os.path.isdir(args.video):
        SET_TEST = True
    elif not os.path.exists(args.video):
        print "{} doesn't exist.".format(args.video)

    video_names = [args.video]
    if SET_TEST:
        video_names = os.listdir(args.video)
        video_names = [os.path.join(args.video, x) for x in video_names]
    
    # choose which dataset to use. Different dataset will use different algorithm
    if args.dataset == 'casia_b':
        import util.casia_b as casia
        video_names = [x for x in video_names if 'bkgrd' not in x and 'avi' in x]
    elif args.dataset == 'casia_e':
        import util.casia_e as casia
        video_names = [x for x in video_names if 'mp4' in x]

    # initialize openpose and detect net 
    gait_extractor = casia.GaitExtractor(args.gpuid)
    
    time_cost = 0
    for video_name in video_names:
        frame_save_dir = './videoframes/videoframe-'+ os.path.basename(video_name)[:-4]
        # split video into frame pictures
        split_video(video_name, frame_save_dir)

        img_dir = frame_save_dir
        if not os.path.exists(img_dir):
            print("{} doesn't exists".format(img_dir))
            sys.exit(0)

        # gait save path
        if args.dataset == 'casia_b':
            basename = os.path.basename(video_name)[:-4].split('-')
            save_dir = os.path.join('./results', basename[0], basename[1]+'_'+basename[2], basename[3])
        elif args.dataset == 'casia_e':
            basename = os.path.basename(video_name)[:-4].split('_')
            if len(basename) == 7:
                save_dir = os.path.join('./results', basename[0], basename[1], basename[2]+'_'+basename[3], basename[4], basename[5] + '_' + basename[6])
            else:
                save_dir = os.path.join('./results', basename[0], basename[1], basename[2]+'_'+basename[3], basename[4], basename[5])
        else:
            save_dir = os.path.join('./results', os.path.basename(video_name)[:-4])
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)

        start_time = time.time()

        print 'Extracting gait.....'
        # do extraction
        gait_extractor.extract(img_dir, save_dir)
        
        time_cost += time.time() - start_time
    print(time_cost)