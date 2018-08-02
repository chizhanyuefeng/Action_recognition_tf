import os
import cv2
import pandas as pd
import numpy as np


train_data_list_csv = './list/train_data.csv'
validation_list_csv = './list/validation_data.csv'

class Dataset(object):

    def __init__(self, batch_size, depth, img_size):
        self.batch_size = batch_size
        self.depth = depth
        self.img_size = img_size

        train_data_paths = pd.read_csv(train_data_list_csv, index_col=False)
        self.train_data_num = train_data_paths.path.size
        validation_data_paths = pd.read_csv(validation_list_csv, index_col=False)
        self.validation_data_num = validation_data_paths.path.size
        # 洗牌
        self.train_data_paths = train_data_paths.sample(frac=1).values
        self.validation_data_paths = validation_data_paths.sample(frac=1)

        # 记录下次获取batch的位置
        self.train_next_pos = 0

        img_path_list = self._get_img_path_list(self.train_data_paths[0][1])


    def _get_img_path_list(self, video_path):

        img_path = os.listdir(video_path)
        img_path = sorted(img_path)
        img_path = [os.path.join(video_path, _) for _ in img_path]

        return img_path

    def _read_img(self, img_path):
        """
        获取一张图片
        :param img_path:
        :return: [112,112,3]
        """
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_np = np.asarray(img_RGB)

        return img_np


    def get_next_batch(self):
        img_batch = np.zeros(shape=[self.batch_size, self.depth,
                                    self.img_size, self.img_size,
                                    3])
        img_label = np.zeros(shape=[self.batch_size])

        for i in range(self.batch_size):
            img_label[i] = self.train_data_paths[self.train_next_pos][0]
            img_path_list = self._get_img_path_list(self.train_data_paths[self.train_next_pos][1])
            self.train_next_pos += 1
            frames_num = len(img_path_list)
            start = np.random.randint(0, frames_num-self.depth)
            for j in range(self.depth):
                img = self._read_img(img_path_list[start+j])
                img_batch[i, j, :, :, :] = img

        return img_batch, img_label

    def get_valiation_data(self):
        pass

if __name__=="__main__":
    a = Dataset(4,1,112)

    b, c= a.get_next_batch()
    print(c)