import torch
from torch import nn


class DNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(3, 64, 5, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, padding=0, bias=False),
            nn.Sigmoid() # (0,1)
        )

    def forward(self, img):
        h = self.sequential(img)
        print(h.shape)
        return h.reshape(-1)


class GNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.ConvTranspose2d(128,512,4,1,padding=0,bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(64,3,5,3,padding=1,bias=False),
            nn.Tanh() # (-1,1)
        )

    def forward(self,nosie):
        return self.sequential(nosie)


class DCGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gnet = GNet()
        self.dnet = DNet()

        self.loss_fn = nn.BCEWithLogitsLoss() # 二值交叉熵

    # def forward(self, noise_d, noise_g, real_img):
    #     real_y = self.dnet(real_img)
    #     g_img = self.gnet(noise_d)
    #     fake_y = self.dnet(g_img)
    #
    #     real_tag = torch.ones(real_img.size(0)) # .cuda()
    #     fake_tag = torch.zeros(noise_d.size(0)) # .cuda()
    #
    #     loss_real = self.loss_fn(real_y, real_tag)
    #     loss_fake = self.loss_fn(fake_y, fake_tag)
    #
    #     loss_d = loss_fake + loss_real
    #
    #     _g_img = self.gnet(noise_g)
    #     _real_y = self.dnet(g_img)
    #     _real_tag = torch.ones(real_img.size(0)) # .cuda()
    #
    #     # 训练生成器
    #     loss_g = self.loss_fn(_real_y, _real_tag)
    #
    #     return loss_d, loss_g

    def forward(self,noise):
        return self.gnet(noise)

    def get_D_loss(self,noise_d,real_img):
        real_y = self.dnet(real_img)
        g_img = self.gnet(noise_d)
        fake_y = self.dnet(g_img)

        real_tag = torch.ones(real_img.size(0))  # .cuda()
        fake_tag = torch.zeros(noise_d.size(0))  # .cuda()

        loss_real = self.loss_fn(real_y, real_tag)
        loss_fake = self.loss_fn(fake_y, fake_tag)

        loss_d = loss_fake + loss_real

        return loss_d

    def get_G_loss(self,noise_g):
        _g_img = self.gnet(noise_g)
        _real_y = self.dnet(_g_img)
        _real_tag = torch.ones(noise_g.size(0)) # .cuda()

        loss_g = self.loss_fn(_real_y,_real_tag)

        return loss_g

if __name__ == '__main__':
    # dnet = DNet()
    # x = torch.randn(1,3,96,96)
    # y = dnet(x)

    # gnet = GNet()
    # x = torch.randn(2,128,1,1)
    # y = gnet(x) # [2,3,96,96]

    dcgan = DCGAN()
    x = torch.randn(2,128,1,1)
    r = torch.randn(4,3,96,96)
    loss1,loss2 = dcgan(x,x,r)