from torch.utils.data import Dataset
import torch
from torch import nn
import os
from PIL import Image
from torchvision import transforms
import numpy as np

tf = transforms.ToTensor()


class MyDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.dataset = os.listdir(root)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        fn = self.dataset[index]
        strs = fn.split('.')[0]
        labels = np.array([int(x) for x in strs])
        img_data = tf(Image.open(f"{self.root}/{fn}"))
        return img_data, labels


if __name__ == '__main__':
    myDataset = MyDataset("./code")
    print(myDataset[0])
    print(myDataset[0][0].shape) # [3,60,240]
