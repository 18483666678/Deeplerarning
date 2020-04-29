from day02.net import *
from torch.utils.tensorboard import SummaryWriter
import cv2

if __name__ == '__main__':
    net = NetV2()
    net.load_state_dict(torch.load("./checkpoint/9.t"))

    summaryWriter = SummaryWriter("./logs")

    layer1_weight = net.seq[1].weight
    layer2_weight = net.seq[5].weight
    layer3_weight = net.seq[9].weight

    summaryWriter.add_histogram("later1",layer1_weight)
    summaryWriter.add_histogram("later2",layer2_weight)
    summaryWriter.add_histogram("later3",layer3_weight)

    cv2.waitKey(0)