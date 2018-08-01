import numpy as np
import cv2

VIDEO_PATH = '/home/tony/motion_research/FutureCamp_ActionRecognitionData_TrainVal'

def save_frames_from_video(video_path,frames_space):

    capture = cv2.VideoCapture(video_path)
    n = 0
    while (True):
        n=n+1
        ret, frame = capture.read()
        img = frame
        if frame is None:
            break
        if n%frames_space == 0:
            cv2.imwrite(str(n)+'.jpg', img)
    capture.release()

save_frames_from_video(VIDEO_PATH+'/GolfSwing/v_GolfSwing_g01_c01.avi', 3)