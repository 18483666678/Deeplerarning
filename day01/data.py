import torch,os
from torch.utils.data import Dataset
import cv2
import numpy as np

class MNISTDataset(Dataset):
    def __init__(self,root,is_train=True):
        self.dataset = [] # 记录所有数据
        sub_dir = "TRAIN" if is_train else "TEST"
        for tag in os.listdir(f"{root}/{sub_dir}"):
            img_dir = f"{root}/{sub_dir}/{tag}"
            for img_filename in os.listdir(img_dir):
                img_pat = f"{img_dir}/{img_filename}"
                self.dataset.append((img_pat,tag))
                print(self.dataset)
            print(tag)


    # 数据集有多少数据
    def __len__(self):
       return len(self.dataset)

    # 每条数据的处理方式
    def __getitem__(self, index):
        data = self.dataset[index]
        img_data = cv2.imread(data[0],cv2.IMREAD_GRAYSCALE)
        img_data = img_data.resape(-1)
        img_data = img_data / 255

        tag_one_ot = np.zeros(10)
        print(tag_one_ot)
        tag_one_ot[int(data[1])] = 1

        return np.float32(img_data),np.float(tag_one_ot)


if __name__ == '__main__':
    root = r"G:\github_pycharm\deeplearning\MNIST_IMG"
    dataset = MNISTDataset(root)
    # len(dataset)
    # print(len(dataset))
    print(dataset[30000])