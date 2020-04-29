from torchvision import datasets,transforms
import torch
from day01.data import *
from day02.net import *
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter

# DEVICE = torch.device("cuda:0")
class Train:

    def __init__(self, root):

        self.summaryWriter = SummaryWriter("./logs")

        # 加载训练数据
        self.train_dataset = datasets.CIFAR10(root, True,transform=transforms.ToTensor(),download=True)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=100, shuffle=True, )

        # 加载测试数据
        self.test_dataset = datasets.CIFAR10(root, False,transform=transforms.ToTensor(),download=True)
        self.test_dataloder = DataLoader(self.test_dataset, 100, )

        # 创建模型
        self.net = NetV2()
        # self.net.load_state_dict(torch.load("./checkpoint/2.t"))
        # self.net.to(DEVICE)

        # 创建优化器
        self.opt = optim.Adam(self.net.parameters())

        self.loos_fn = nn.CrossEntropyLoss()

    # 训练代码
    def __call__(self):
        for epoch in range(100000):
            self.net.train()
            sum_loss = 0.
            for i, (imgs, tags) in enumerate(self.train_dataloader):
                y = self.net(imgs)

                #正则化
                # L2 = []
                # for param in self.net.parameters():
                #     L2 += torch.sum(param ** 2)

                # loss = torch.mean((tags - y) ** 2)
                loss = self.loos_fn(y,tags)
                # loss = self.loos_fn(y,tags) + 0.01*L2

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                sum_loss += loss.cpu().detach().item()

            avg_loss = sum_loss / len(self.train_dataloader)

            self.net.eval()
            sum_score = 0.
            test_sum_loss = 0.
            for i, (imgs, tags) in enumerate(self.test_dataloder):
                # imgs,tags = imgs.to(DRVICE),tags.to(DEVICE)
                test_y = self.net(imgs)
                # test_loss = torch.mean((tags - test_y) ** 2)
                test_loss = self.loos_fn(test_y,tags)
                test_sum_loss += test_loss.cpu().detach().item()

                pred_tags = torch.argmax(test_y, dim=1)
                # label_tags = torch.argmax(tags, dim=1)
                sum_score += torch.sum(torch.eq(pred_tags, tags).float()).cpu().detach().item()

            # 加载测试图片
            self.summaryWriter.add_images("imgs",imgs[:10],epoch)

            test_avg_loss = test_sum_loss / len(self.test_dataloder)
            score = sum_score / len(self.test_dataset)

            self.summaryWriter.add_scalars("loss",{"train_loss":avg_loss,"test_loss":test_avg_loss},epoch)
            self.summaryWriter.add_scalar("score",score,epoch)

            layer1_weight = self.net.seq[1].weight
            layer2_weight = self.net.seq[5].weight
            layer3_weight = self.net.seq[9].weight

            self.summaryWriter.add_histogram("later1", layer1_weight,epoch)
            self.summaryWriter.add_histogram("later2", layer2_weight,epoch)
            self.summaryWriter.add_histogram("later3", layer3_weight,epoch)

            print(epoch, avg_loss, test_avg_loss, score)

            torch.save(self.net.state_dict(),f"./checkpoint/{epoch}.t")


if __name__ == '__main__':
    train = Train("G:\github_pycharm\deeplearning\data\CIFAR10")
    train()