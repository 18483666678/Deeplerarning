from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms

tf = transforms.Compose([
    transforms.ToTensor()]
)
# tf = transforms.ToTensor()

class MyDataset(Dataset):
    def __init__(self,root,img_size):
        self.root_dir = root
        self.img_size = img_size
        self.dataset = []

        # positive_file = open(f"{root}/{img_size}/positive.txt","r")
        # negative_file = open(f"{root}/{img_size}/negative.txt","r")
        # part_file = open(f"{root}/{img_size}/part.txt","r")
        #
        # self.dataset.extend(positive_file)
        # self.dataset.extend(negative_file)
        # self.dataset.extend(part_file)
        #
        # positive_file.close()
        # negative_file.close()
        # part_file.close()

        with open(f"{root}/{img_size}/positive.txt") as f:
            self.dataset.extend(f.readlines())

        with open(f"{root}/{img_size}/negative.txt") as f:
            self.dataset.extend(f.readlines())

        with open(f"{root}/{img_size}/part.txt") as f:
            self.dataset.extend(f.readlines())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        # print(data)
        strs = data.split()

        img_path = None

        if strs[1] == "1":
            img_path = f"{self.root_dir}/{self.img_size}/positive/{strs[0]}"
        elif strs[1] == "2":
            img_path = f"{self.root_dir}/{self.img_size}/negative/{strs[0]}"
        else:
            img_path = f"{self.root_dir}/{self.img_size}/part/{strs[0]}"

        img_data = tf(Image.open(img_path))
        print(img_data.shape)
        c,x1,y1,x2,y2 = float(strs[1]),float(strs[2]),float(strs[3]),float(strs[4]),float(strs[5])

        return img_data,np.array([c,x1,y1,x2,y2],dtype=np.float32())

if __name__ == '__main__':
    dataset = MyDataset(r"",12)
    print(dataset[0])
