import cv2
import numpy as np
from matplotlib import pyplot as plt

# 加载图像
image = cv2.imread('mh.jpg')

# 创建一个与原始图像相同大小的掩模
mask = np.zeros(image.shape[:2], dtype=np.uint8)

# 使用高斯模糊去除噪声
blurred = cv2.GaussianBlur(image, (3, 3), 0)

# 将模糊图像转换为灰度图像
gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

# 通过阈值分割创建二值图像
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 执行形态学操作来消除噪声
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

# 数组的逻辑运算来与原始图像创建掩模
mask[cleaned == 255] = 255

# 使用掩模来恢复图像的清晰度
restored = cv2.inpaint(image, mask, 5, cv2.INPAINT_NS)

# 将模糊图像转换为灰度图像
gray1= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# 直方图均衡化,使用OpenCV中的直方图均衡化或对比度增强函数增强图像
equalized = cv2.equalizeHist(gray1)
# 对比度增强
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
contrast = clahe.apply(gray1)





# 显示结果
plt.subplot(2,2,1)
plt.imshow( image)
plt.title("1")

plt.subplot(2,2,2)
plt.imshow( restored)
plt.title("2")

plt.subplot(2,2,3)
plt.imshow( gray1)
plt.title("3")

plt.subplot(2,2,4)
plt.imshow( contrast)
plt.title("4")

plt.tight_layout()
plt.show()

cv2.imshow("1",image)
cv2.imshow("2",restored)
cv2.imshow("3",gray1)
cv2.imshow("4",contrast)

cv2.waitKey(0)
cv2.destroyAllWindows()