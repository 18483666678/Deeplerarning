import torch
from torch import nn

class Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Conv2d(3,16,7,2,3),
                         nn.ReLU(),
                         nn.Conv2d(16, 32, 3, 1, 1),
                         nn.ReLU(),
                         nn.MaxPool2d(2),
                         nn.Conv2d(32, 64, 3, 1, 1),
                         nn.ReLU(),
                         nn.MaxPool2d(2),
                         nn.Conv2d(64,128,3,1,1),
                         nn.ReLU(),
                         nn.Conv2d(128,128,1,1,0)
                         )

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(32*7*30,10,2,batch_first=True)

    def forward(self,x):
        x = x.reshape(-1,32*7*30)
        x = x[:,None,:].repeat(1,4,1)
        h0 = torch.randn(2,x.size(0),10)
        output,h0 = self.rnn(x,h0)
        return output

class Cnn2SEQ(nn.Module):
    def __init__(self):
        super().__init__()

        self.encode = Encoder()
        self.decode = Decoder()

    def forward(self,x):
        f = self.encode(x)
        f = self.decode(x)
        return y