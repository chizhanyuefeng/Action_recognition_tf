import os
import cv2

DATASET_PATH = '/home/tony/motion_research/FutureCamp_ActionRecognitionData_TrainVal'
SAVE_PATH = '../dataset/'
FRAMES_SPACE = 3

def save_frames_from_video(video_path, save_path, frames_space):
    capture = cv2.VideoCapture(video_path)
    n = 0
    i = 0
    while (True):
        n=n+1
        ret, frame = capture.read()
        img = frame
        if frame is None:
            break
        if n%frames_space == 0:
            i = i + 1
            cv2.imwrite(save_path + str(i)+'.jpg', img)
    capture.release()

def extract_all_videos(path):
    if not os.path.exists(path):
        print('path is not exist：', path)
        return None

    for i in os.listdir(path):
        cur_path = path + "/" + i
        if os.path.isfile(cur_path) and ('avi' in i):
            video_name = cur_path.replace(DATASET_PATH, '').replace('.avi', '')
            save_path = SAVE_PATH + video_name + '/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_frames_from_video(cur_path, save_path, FRAMES_SPACE)
        else:
            extract_all_videos(cur_path)

extract_all_videos(DATASET_PATH)
