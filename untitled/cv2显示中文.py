import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

img = cv2.imread("img/1.jpg")

def draw_box_string(img, x, y, string):
    """
    img: imread读取的图片;
    x,y:字符起始绘制的位置;
    string: 显示的文字;
    return: img
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    # simhei.ttf 是字体，你如果没有字体，需要下载
    font = ImageFont.truetype("simhei.ttf", 50, encoding="utf-8")
    draw.text((x, y - 50), string, (255, 255, 255), font=font)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img


# 三基色=白色
img = draw_box_string(img, 300, 200, "abc中文显示china")
# 显示图片
cv2.imshow("image", img)
cv2.waitKey(0)