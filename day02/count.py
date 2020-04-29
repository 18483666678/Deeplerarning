import torch,thop
from torch import nn

if __name__ == '__main__':
    conv = nn.Conv2d(3,60,10,2,padding=0)
    x = torch.randn(1,3,16,16)

    # 参数量
    flops,params = thop.profile(conv,(x,))
    flops,params = thop.clever_format((flops,params),"%.3f")
    print(flops,params)