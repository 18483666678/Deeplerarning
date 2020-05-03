from day05.data import *
from day05.net import *
from torch.utils.data import DataLoader
from torch import optim
from torchvision import utils
torch.autograd.set_detect_anomaly(True)

class Trainer:
    def __init__(self, root):
        faceMyData = FaceMyData(root)
        self.train_dataloader = DataLoader(faceMyData, 100, shuffle=True)

        self.net = DCGAN()
        # self.net.cuda()

        self.d_opt = optim.Adam(self.net.dnet.parameters(), 0.0002, betas=(0.5, 0.9))
        self.g_opt = optim.Adam(self.net.gnet.parameters(), 0.0002, betas=(0.5, 0.9))

    def __call__(self):
        for epoch in range(10000):
            for i, real_img in enumerate(self.train_dataloader):
                # img = img.cuda()
                # 1.15版本不适用
                # noise_d = torch.normal(0, 0.1, (100, 128, 1, 1)) # .cuda()
                # noise_g = torch.normal(0, 0.1, (100, 128, 1, 1)) # .cuda()
                # loss_d,loss_g = self.net(noise_d,noise_g,real_img)

                noise_d = torch.normal(0, 0.1, (100, 128, 1, 1)) # .cuda()
                loss_d = self.net.get_D_loss(noise_d,real_img)

                self.d_opt.zero_grad()
                loss_d.backward()
                self.d_opt.step()

                noise_g = torch.normal(0, 0.1, (100, 128, 1, 1)) # .cuda()
                loss_g = self.net.get_G_loss(noise_g)

                self.g_opt.zero_grad()
                loss_g.backward()
                self.g_opt.step()

                print(loss_d.cpu().detach().item(),loss_g.cpu().detach().item,)

            # 查看生成图片
            print(".............")
            noise = torch.normal(0, 0.1, (8, 128, 1, 1))  # .cuda()
            y = self.net(noise)
            utils.save_image(y,"1.jpg",range=(-1,1),normalize=True)




if __name__ == '__main__':
    train = Trainer("./faces")
    train()