import torchvision
import torch
from torch import nn

# print(torchvision.models.densenet121())
# print(torchvision.models.inception_v3())
# print(torchvision.models.squeezenet1_0())
print(torchvision.models.mobilenet_v2())
# import thop
# conv_1 = nn.Conv2d(4,20,3,1)
# conv_2 = nn.Conv2d(4,20,3,1,groups=2)
# conv_3 = nn.Conv2d(4,20,3,1,groups=4)
#
# x = torch.randn(1, 4, 112, 112)
#
# # 参数量
# flops,params = thop.profile(conv_1,(x,))
# flops,params = thop.clever_format((flops,params),"%.3f")
# print(flops,params)
#
# print(thop.clever_format(thop.profile(conv_2,(x,)),"%.3f"))
# print(thop.clever_format(thop.profile(conv_3,(x,)),"%.3f"))