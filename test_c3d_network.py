import os
import time
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from c3d_network import C3D_Network,class_label
from utils.clip_videos import *
from collections import Counter


class Run_C3D_model(object):
    net_predict = None
    _sess = None

    def __init__(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 16, 112, 112, 3])
        self.dropout_pro = tf.placeholder(tf.float32)

        network = C3D_Network(self.x, 1, trainable=False)
        net_out = network.contruct_graph()
        self.net_predict = tf.argmax(net_out, 1)

        saver = tf.train.Saver()
        #saver.restore(self._get_session(), "./models/test_model/model.ckpt")
        saver.restore(self._get_session(), "/home/tony/motion_research/models/90/model.ckpt")

    def _get_session(self):
        if self._sess is None:
            self._sess = tf.Session()
        return self._sess

    def run_video_from_frames(self, video_dir):
        """
        测试视频
        :param video_dir: 视频地址
        :param v_class:
        :return:
        """
        frames_np = self._get_video_frames(video_dir)

        frame_num = np.shape(frames_np)[0]

        if frame_num<16:
            return -1

        batch_num = int(frame_num/16)
        test_list = [15 for _ in range(int(batch_num/15))]

        if batch_num%15 != 0:
            test_list.append(batch_num%15)
        predict_res = []
        for i in range(len(test_list)):
            x = np.zeros([test_list[i], 16, 112, 112, 3])
            for b in range(test_list[i]):
                if i !=0:
                    start = i*16*15+16*b
                else:
                    start = i*16*0+16*b
                x[b, :, :, :, :] = frames_np[start:start + 16, :, :, :]

            feed_dict = {self.x: x, self.dropout_pro: 1}

            net_predict = self._get_session().run(self.net_predict, feed_dict=feed_dict)
            predict_res.extend(net_predict)

        a = Counter(predict_res)

        try:
            max(a.values())
        except ValueError:
            print(video_dir)
        else:
            max_num = max(a.values())

        for i in a.keys():
            if a[i] == max_num:
                return i



    def _get_video_frames(self, video_dir):
        """
        读取一个视频，并将视频帧数shape转为 [frames_num , w, w, 3]
        :param video_dir:
        :return:
        """
        capture = cv2.VideoCapture(video_dir)
        frame_num = 0
        frames = []
        while (True):
            ret, frame = capture.read()
            if frame is None:
                break
            frame_num = frame_num + 1
            if frame_num%2 == 0:
                img = cv2.resize(frame, (112, 112))
                img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames.append(img_RGB)

        capture.release()

        video_frames = np.array(frames)

        return video_frames


if __name__ == "__main__":
    a = Run_C3D_model()
    start = time.time()
    test_path = '/home/tony/motion_research/FutureCamp_ActionRecognitionData_Test/'

    # test_vide = '/home/tony/motion_research/FutureCamp_ActionRecognitionData_Test/LongJump/lIF5SyZn-ZQ_000003_000013.mp4'
    # a.run_video_from_frames(test_vide)
    video_num = 0
    p = 0

    for i in os.listdir(test_path):
        video_class = class_label[i]

        for j in os.listdir(test_path+i):
            video_num += 1
            p_c = a.run_video_from_frames(test_path+i+'/'+j)
            if p_c == -1:
                print('视频太短:',j)
                continue
            if video_class == p_c:
                p +=1

        print("当前%s准确度为%6f"%(i,p/video_num))
    print("最终准确度为%6f" % (p / video_num))

    during = time.time() - start
    print(during)
