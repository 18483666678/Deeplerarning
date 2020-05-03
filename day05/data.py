from torch.utils.data import Dataset
from torch import nn
import cv2, os
import torch
import numpy as np


class FaceMyData(Dataset):
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.dataset = os.listdir(root)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pic_name = self.dataset[index]
        img_data = cv2.imread(f"{self.root}/{pic_name}")
        print(img_data.shape)

        # 颜色空间转换
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
        # img_data = img_data[...,::,-1]

        img_data = img_data.transpose([2, 0, 1])
        print(img_data.shape)
        # img_data = (img_data / 255.).astype(np.float32)  # (0,1)
        img_data = ((img_data / 255. - 0.5) * 2).astype(np.float32)  # (-1,1)

        print(img_data.dtype)

        return img_data


if __name__ == '__main__':
    faceMyData = FaceMyData("./faces")
    print(faceMyData[0])
    print(faceMyData[0].max()) # 1
    print(faceMyData[0].min()) # -1
