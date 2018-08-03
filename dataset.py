import os
import time
import cv2
import pandas as pd
import numpy as np


train_data_list_csv = './list/train_data.csv'
validation_list_csv = './list/validation_data.csv'

class Dataset(object):
    """
    从划分好的数据集中提取所需数据。
    获取训练数据，使用get_next_batch方法来得到
    获取验证集，使用 get_valiation_data方法
    """

    def __init__(self, depth, img_size):
        self.depth = depth
        self.img_size = img_size
        self.epoch = 0
        self.validation_epoch = 0


        train_data_paths = pd.read_csv(train_data_list_csv, index_col=False)
        self.train_data_num = train_data_paths.path.size
        validation_data_paths = pd.read_csv(validation_list_csv, index_col=False)
        self.validation_data_num = validation_data_paths.path.size
        # 洗牌
        self.train_data_paths = train_data_paths.sample(frac=1).values
        self.validation_data_paths = validation_data_paths.sample(frac=1).values

        # 记录下次获取batch的位置
        self.train_next_pos = 0
        self.validation_next_pos = 0

    def get_next_batch(self, batch_size):
        img_batch = np.zeros(shape=[batch_size, self.depth,
                                    self.img_size, self.img_size,
                                    3])
        img_label = np.zeros(shape=[batch_size, 45])

        for i in range(batch_size):
            label = int(self.train_data_paths[self.train_next_pos][0])
            label_list = [0 for _ in range(45)]
            label_list[label] = 1
            img_label[i] = np.array(label_list)

            img_path_list = self._get_img_path_list(self.train_data_paths[self.train_next_pos][1])
            frames_num = len(img_path_list)
            if self.train_next_pos == self.train_data_num-1:
                self.train_next_pos = 0
                self.epoch += 1
            else:
                self.train_next_pos += 1

            #print(img_path_list)
            assert frames_num-self.depth >= 0, print(frames_num, img_path_list)
            if frames_num-self.depth == 0:
                start = 0
            else:
                start = np.random.randint(0, frames_num-self.depth+1)
            for j in range(self.depth):
                img = self._read_img(img_path_list[start+j])
                img_batch[i, j, :, :, :] = img

        return img_batch, img_label

    def get_valiation_data(self, batch_size):
        img_batch = np.zeros(shape=[batch_size, self.depth,
                                    self.img_size, self.img_size,
                                    3])
        img_label = np.zeros(shape=[batch_size, 45])

        for i in range(batch_size):
            label = int(self.validation_data_paths[self.validation_next_pos][0])
            label_list = [0 for _ in range(45)]
            label_list[label] = 1
            img_label[i] = np.array(label_list)

            img_path_list = self._get_img_path_list(self.validation_data_paths[self.validation_next_pos][1])
            frames_num = len(img_path_list)
            if self.validation_next_pos == self.validation_data_num-1:
                self.validation_next_pos = 0
                self.validation_epoch = 1
            else:
                self.validation_next_pos += 1

            assert frames_num-self.depth >= 0, print(frames_num, img_path_list)
            if frames_num-self.depth == 0:
                start = 0
            else:
                start = np.random.randint(0, frames_num-self.depth+1)
            for j in range(self.depth):
                img = self._read_img(img_path_list[start+j])
                img_batch[i, j, :, :, :] = img

        return img_batch, img_label

    def _get_img_path_list(self, video_path):
        """
        获取当前从这个视频中提取的帧图像的路径
        :param video_path:
        :return:
        """
        img_path = os.listdir(video_path)
        img_path.sort()
        img_path = [os.path.join(video_path, _) for _ in img_path]

        return img_path

    def _read_img(self, img_path):
        """
        获取一张图片，并将数据进行resize
        :param img_path:
        :return: [112,112,3]
        """
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_np = np.asarray(img_RGB)

        return img_np

if __name__=="__main__":

    a = Dataset(16, 112)
    start = time.time()
    b, c = a.get_next_batch(20)
    print(c)
    during = time.time() - start
    print(during)