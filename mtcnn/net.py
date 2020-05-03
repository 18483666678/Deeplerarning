import torch
import torch.nn as nn


class PNet(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Conv2d(3, 10, 3),
                         nn.ReLU(),
                         nn.MaxPool2d(kernel_size=2),
                         nn.Conv2d(10, 16, 3),
                         nn.ReLU(),
                         nn.Conv2d(16, 32, 3),
                         nn.ReLU(),
                         nn.Conv2d(32, 5, 1))

class RNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3,28,3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(28,48,3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48,64,2)
        )
        self.output_layer = nn.Sequential(
            nn.Linear(3*3*64,128),
            nn.ReLU(),
            nn.Linear(128,5)
        )
    def forward(self,x):
        h = self.backbone(x)
        h = h.reshape(-1,3*3*64)
        return self.output_layer(h)

class ONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,64,3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,2),
            nn.ReLU()
        )
        self.output_layer = nn.Sequential(
            nn.Linear(3*3*128,256),
            nn.ReLU(),
            nn.Linear(256,5)
        )
    def forward(self,x):
        h = self.backbone(x)
        h = h.reshape(-1,3*3*128)
        return self.output_layer(h)

if __name__ == '__main__':
    # pnet = PNet()
    # y = pnet(torch.randn(1,3,12,12))
    # print(y.shape) #[1,5,1,1]

    # rnet = RNet()
    # y = rnet(torch.randn(1, 3, 24, 24))
    # print(y.shape) # [1,5]

    onet = ONet()
    y = onet(torch.randn(1, 3, 48, 48))
    print(y.shape)  # [1,5]