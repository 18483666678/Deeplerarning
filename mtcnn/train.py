from mtcnn.data import *
from mtcnn.net import *
from torch.utils.data import DataLoader
from torch import optim


class Train:
    def __init__(self, root, img_size):

        self.img_size = img_size

        self.mydataset = MyDataset(root, img_size)
        self.dataloader = DataLoader(self.mydataset, batch_size=1000, shuffle=True)

        if img_size == 12:
            self.net = PNet()
        elif img_size == 24:
            self.net = RNet()
        elif img_size == 48:
            self.net = ONet()

        self.opt = optim.Adam(self.net.parameters())

    def __call__(self, epochs):

        for epoch in range(epochs):
            for i, (img, tag) in enumerate(self.dataloader):
                predict = self.net(img)

                if self.img_size == 12:
                    predict = predict.reshape(-1,5)

                torch.sigmoid_(predict[:,0])

                c_mask = tag[:, 0] < 2
                c_predict = predict[c_mask]
                c_tag = tag[c_mask]
                loss_c = torch.mean((c_predict[:, 0] - c_tag[:, 0]) ** 2)

                off_mask = tag[:, 0] > 0
                off_predict = predict[off_mask]
                off_tag = tag[off_mask]
                loss_off = torch.mean((off_predict[:, 1:] - off_tag[:, 1:]) ** 2)

                loss = loss_c + loss_off

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                print(loss)
            if self.img_size == 12:
                torch.save(self.net.state_dict(),"pnet.pt")
            elif self.img_size == 24:
                torch.save(self.net.state_dict(), "rnet.pt")
            elif self.img_size == 48:
                torch.save(self.net.state_dict(), "onet.pt")


if __name__ == '__main__':
    train = Train(r"",12)
    train(1)
