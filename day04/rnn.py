import torch.nn as nn
import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import functional as F

class MyRnn(nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn = nn.LSTM(28*1,128,2,batch_first=True,bidirectional=False)
        self.output_layer = nn.Linear(128,10)

    def forward(self,x):
        # nchw-->nsv
        n,c,h,w = x.shape
        x = x.permute(0,2,3,1)
        x = x.reshape(n,h,w*c)
        h0 = torch.zeros(2*1,n,128)
        c0 = torch.zeros(2*1,n,128)
        hsn,(hn,cn) = self.rnn(x,(h0,c0))
        out = self.output_layer(hsn[:,-1,:])
        return out

class MyRnn2(nn.Module):
    def __init__(self):
        super().__init__()

        self.rnnCell_1 = nn.GRUCell(28*1,128)
        # self.relu_1 = nn.ReLU()
        self.rnnCell_2 = nn.GRUCell(128,128)

        self.output_layer = nn.Linear(128,10)

    def forward(self,x):
        n, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(n, h, w * c)

        hx_1 = torch.zeros(n,128)
        hx_2 = torch.zeros(n,128)
        for i in range(h):
            # hx_1 = self.rnnCell_1(x[:,i,:],hx_1)
            hx_1 = F.relu(self.rnnCell_1(x[:,i,:],hx_1))
            hx_2 = F.relu(self.rnnCell_2(hx_1,hx_2))
        out = self.output_layer(hx_2)
        return out



# if __name__ == '__main__':
#     x = torch.randn(2,3,28,28)
#     myRnn = MyRnn()
#     y = myRnn(x)
#     print(y.shape)

if __name__ == '__main__':
    train_dataset = datasets.MNIST(root='../data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='../data', train=False, transform=transforms.ToTensor(), download=True)

    train_dataloader = DataLoader(train_dataset,100,shuffle=True)
    test_dataloader = DataLoader(train_dataset,100,shuffle=True)

    # myRnn = MyRnn()
    myRnn = MyRnn2()

    opt = optim.Adam(myRnn.parameters())

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10000):
        for i,(img,tag) in enumerate(train_dataloader):
            output = myRnn(img)
            loss = loss_fn(output,tag)

            opt.zero_grad()
            loss.backward()
            opt.step()

            print(loss)
