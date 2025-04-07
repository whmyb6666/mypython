import cv2
import tkinter as tk
from tkinter import filedialog


def img_test():
    # 获取选择文件路径
    # 实例化
    root = tk.Tk()
    root.withdraw()
    # 获取文件的绝对路径路径
    return filedialog.askopenfilename()


def is_inside(o, i):
    # 判断矩形o是否在矩形i中
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i
    return ox > ix and oy > iy and ox + ow < ix + iw and oy + oh < iy + ih


def detect_test(img):
    #img = cv2.imread(img_test())
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    # 使用默认的HOG特征描述符
    hog = cv2.HOGDescriptor()
    # cv2.HOGDescriptor_getDefaultPeopleDetector函数返回为行人检测训练的分类器的系数
    detector = cv2.HOGDescriptor_getDefaultPeopleDetector()
    # 使用默认的行人分类器（检测窗口64x128）
    hog.setSVMDetector(detector)

    # 使用detecMultiScale函数检测图像中的行人，返回值为行人对应的矩形框和权重值
    found, weight = hog.detectMultiScale(img_gray, scale=1.02)
    found_filtered = []
    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            # r在q内？
            if ri != qi and is_inside(r, q):
                break
        else:
            found_filtered.append(r)

    for person in found_filtered:
        x, y, w, h = person
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
    return img


if __name__ == '__main__':
    image = cv2.imread('img/xr.jpeg')
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', image)
    while 1:
        k = cv2.waitKey()
        if k == ord('q'):
            break
        elif k == ord('n'):
            image = detect_test(image)
            cv2.imshow('image', image)
    cv2.destroyAllWindows()