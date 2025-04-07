# 导入所需库
import cv2

# 读取输入图像
img = cv2.imread('e1b2942258f5a6e9b4df31231a5d145.jpg')

# 将图像转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用默认值初始化SIFT对象
sift = cv2.SIFT_create()

# 在图像（灰度）中查找关键点
kp,d = sift.detectAndCompute(gray, None)

# 在图像中绘制关键点
img2 = cv2.drawKeypoints(gray, kp, None, flags=0)
print(d.shape,d[40],d[41])
# 显示绘制了关键点的图像
cv2.imshow("Keypoints", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()