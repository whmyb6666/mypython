# -*- coding: utf-8 -*-

import cv2

import matplotlib.pyplot as plt

def zh_ch(string):
    return string.encode("gbk").decode(errors="ignore")

img =cv2.imread("mh.JPG")
img3 = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

b,g,r = cv2.split(img)
h,s,v = cv2.split(img3)
print("h色调（H）",h)
print("s饱和度（S）",s)
print("v亮度（V）",v)

img2 = cv2.merge((r,g,b))

'''
使用切片来进行倒序排列

my_list = [1, 2, 3, 4, 5]
my_list = my_list[::-1]
print(my_list)  # 输出 [5, 4, 3, 2, 1]
'''

cv2.imshow("img2",img2[:,:,::-1])

cv2.imshow(zh_ch("H色调"),h)
cv2.imshow(zh_ch("S饱和度"),s)
cv2.imshow(zh_ch("V亮度"),v)

cv2.imshow("img3",img3)


#

img4 = cv2.pyrUp(img)
cv2.imshow('img4',img4)
#plt.imshow(img2)

cv2.waitKey(0)

cv2.xfeatures2d.SIFT_create()