import torch
from net import *
from torch.utils.data import DataLoader
from data import *
from torch import optim


class Trainer:
    def __init__(self):
        train_dataset = MyDataset("./code")
        self.train_dataloader = DataLoader(train_dataset, 100, True)

        self.net = Cnn2SEQ()

        self.opt = optim.Adam(self.net.parameters())

        self.loss_fn = nn.CrossEntropyLoss()

    def __call__(self,EPOCHS):
        for epoch in range(EPOCHS):
            for i, (img, tag) in enumerate(self.train_dataloader):
                output = self.net(img)  # [100,4,10]
                output = output.reshape(-1, 10)  # [400,10]

                tag = tag.reshape(-1)  # [100,4]-->[400]

                loss = self.loss_fn(output, tag.long())

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                output = torch.argmax(output, dim=1)
                bool = tag.eq(output.data).cpu().sum()
                accuracy = int(bool) / tag.size(0)

                print(f"epoch:{epoch}/{EPOCHS} | loss:{loss} | accuracy:{accuracy}")
                print('output:', output[:4])
                print("label:", tag[:4])



if __name__ == '__main__':
    train = Trainer()
    train(10000)
