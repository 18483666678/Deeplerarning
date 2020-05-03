from mtcnn.net import *
from mtcnn.data import tf
from PIL import Image, ImageDraw
from mtcnn import utils
import numpy as np


class Detector:
    def __init__(self):
        self.pnet = PNet()
        self.pnet.load_state_dict(torch.load("pnet.pt"))

        self.rnet = RNet()
        self.pnet.load_state_dict(torch.load("rnet.pt"))

        self.onet = ONet()
        self.pnet.load_state_dict(torch.load("onet.pt"))

    def __call__(self, img):
        boxes = self.detPnet(img)
        if boxes is None: return []

        boxes = self.detRnet(img, boxes)
        if boxes is None: return []

        boxes = self.detOnet(img, boxes)
        if boxes is None: return []
        return boxes

    def detPnet(self, img):
        w, h = img.size
        scale = 1
        img_scale = img

        min_side = min(w, h)

        _boxes = []
        while min_side > 12:
            _img_scale = tf(img_scale)
            y = self.pnet(_img_scale[None, ...])
            torch.sigmoid_(y[:,0,...])
            c = y[0, 0]
            # print(c.shape)
            c_mask = c > 0.4  # ０.４～0.65
            # print(c_mask)
            idxs = c_mask.nonzero()  # 筛选索引
            _x1, _y1 = idxs[:, 1] * 2, idxs[:, 0] * 2  # 2是P网络步长
            _x2, _y2 = _x1 + 12, _y1 + 12

            p = y[0, 1:, c_mask]  # 筛选值
            # print(p.shape)

            # 跟gendata.py生成数据有关
            # x1 = (p[0, :] * 12 + _x1) / scale
            # y1 = (p[1, :] * 12 + _y1) / scale
            # x2 = (p[2, :] * 12 + _x2) / scale
            # y2 = (p[3, :] * 12 + _y2) / scale

            x1 = (_x1 - (p[0, :] * 12)) / scale
            y1 = (_y1 - (p[1, :] * 12)) / scale
            x2 = (_x2 - p[2, :] * 12) / scale
            y2 = (_y2 - p[3, :] * 12) / scale
            # print(x1.shape)

            cc = y[0, 0, c_mask]

            # boxes = torch.stack([x1,y1,x2,y2,cc],dim=1)
            # print(boxes)
            _boxes.append(torch.stack([x1, y1, x2, y2, cc], dim=1))

            # 图像金字塔
            scale *= 0.702
            w, h = int(w * scale), int(h * scale)
            img_scale = img_scale.resize((w, h))
            min_side = min(w, h)

        boxes = torch.cat(_boxes, dim=0)
        return utils.nms(boxes.cpu().detach().numpy(), 0.7)

    def detRnet(self, img, boxes):
        _boxes = self._rnet_onet(img, boxes, 24)
        return utils.nms(_boxes, 0.7)

    def detOnet(self, img, boxes):
        _boxes = self._rnet_onet(img, boxes, 48)
        _boxes = utils.nms(_boxes, 0.7)
        _boxes = utils.nms(_boxes, 0.7, is_min=True)
        return _boxes

    def _rnet_onet(self, img, boxes, s):

        imgs = []
        for box in boxes:
            img = img.crop(box[0:4])
            img = img.resize((s, s))
            imgs.append(tf(img))
        _imgs = torch.stack(imgs, dim=0)
        if s == 24:
            y = self.rnet(_imgs)
        else:
            y = self.onet(_imgs)

        #　训练加了sigmoid　侦测也要加
        torch.sigmoid_(y[:,0])

        y = y.cpu().detach().numpy()

        # c_mask = y[:,4] > 0.01
        c_mask = y[:, 0] > -1000
        _boxes = boxes[c_mask]
        _y = y[c_mask]

        _w, _h = _boxes[:, 2] - _boxes[:, 0], _boxes[:, 3] - _boxes[:, 1]
        x1 = _boxes[:, 0] - _y[:, 1] * _w
        y1 = _boxes[:, 1] - _y[:, 2] * _h
        x2 = _boxes[:, 2] - _y[:, 3] * _w
        y2 = _boxes[:, 3] - _y[:, 4] * _h
        cc = _y[:, 0]

        _boxes = np.stack([x1, y1, x2, y2, cc], axis=1)
        # print(_boxes.shape)
        return _boxes


if __name__ == '__main__':
    img_path = r""
    img = Image.open(img_path)
    detector = Detector()
    detector(img)
