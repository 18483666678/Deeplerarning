import random

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter
from PIL import ImageFont
import os


def randomChar():
    '''
    随机生成chr
    :return:返回一个随机生成的chr
    '''
    return chr(random.randint(48, 57))


def randomNum():
    """
    随机数字
    :return:
    """
    a = str(random.randint(0, 9))
    a = chr(random.randint(48, 57))  # ASCII编码表
    b = chr(random.randint(65, 90))  # 大写字母
    c = chr(random.randint(97, 122))  # 小写字母
    d = ord(a)
    return a


def randomBgColor():
    '''
    Color1
    随机生成验证码的背景色
    :return:
    '''
    return (random.randint(65, 255), random.randint(65, 255), random.randint(65, 255))


def randomTextColor():
    '''
    Color2
    随机生成验证码的文字颜色
    :return:
    '''
    return (random.randint(32, 127), random.randint(32, 127), random.randint(32, 127))


w = 60 * 4
h = 60

# 设置字体类型及大小
font = ImageFont.truetype(font='C:\\Windows\\Fonts\\arial.ttf', size=40)

for _ in range(500):
    # 创建一张图片，指定图片mode，长宽
    image = Image.new('RGB', (w, h), (255, 255, 255))

    # 创建Draw对象
    draw = ImageDraw.Draw(image)
    # 遍历给图片的每个像素点着色
    for x in range(w):
        for y in range(h):
            draw.point((x, y), fill=randomBgColor())

    # 将随机生成的chr，draw如image
    filename = ""
    for j in range(4):
        ch = randomNum()
        filename += ch
        draw.text((60 * j + 10, 10), ch, font=font, fill=randomTextColor())

    # 设置图片模糊
    image = image.filter(ImageFilter.BLUR)
    # 保存图片
    image_path = "./code"
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    # print("{0}/{1}.jpg".format(image_path, filename))
    image.save("{0}/{1}.jpg".format(image_path, filename))
