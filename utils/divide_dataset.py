import sys
sys.path.append("../")
import os
import pandas as pd
from c3d_network import class_label

extracted_frames_path = '/home/tony/motion_research/dataset/'
train_file = '../list/train_data.csv'
validation_file = '../list/validation_data.csv'

def divide_dataset(data_path, train_data_proportion=0.9):
    """
    数据集划分，根据训练集和测试集的划分比例进行划分，
    将划分好的训练集数据绝对路径存放至train_file中，
    验证集数据的绝对路径存放至validation_file中
    :param data_path: 需要划分的数据集路径
    :return:
    """

    train_data_path = []
    train_data_class = []

    validation_data_path = []
    validation_data_class = []

    # 获取所有类文件夹的路径
    class_files_list = os.listdir(data_path)
    for i in class_files_list:
        # 根据每个类文件夹的路径查找文件
        if os.path.isdir(os.path.join(data_path, i)):
            # 划分训练和验证集的个数
            videos_num = len(os.listdir(os.path.join(data_path, i)))
            train_data_num = int(videos_num * train_data_proportion)

            videos_list = os.listdir(os.path.join(data_path, i))

            for p in range(train_data_num):
                train_data_path.append(os.path.join(extracted_frames_path,i,videos_list[p]))
                train_data_class.append(class_label[i])

            for p in range(train_data_num, videos_num):
                validation_data_path.append(os.path.join(extracted_frames_path,i,videos_list[p]))
                validation_data_class.append(class_label[i])

    train_data = pd.DataFrame({"path": train_data_path,
                               "label": train_data_class})
    train_data.to_csv(train_file, index=False)

    validation_data = pd.DataFrame({"path": validation_data_path,
                                    "label": validation_data_class})
    validation_data.to_csv(validation_file, index=False)

if __name__ == '__main__':
    divide_dataset(extracted_frames_path)
