import cv2
import numpy as np
import face_recognition

# filename = './img/多人.png'
# img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
# cv2.imshow('中文', img)
# cv2.waitKey()

#curImg = cv2.imread('ImagesBasic/zzy.jpeg')
#cv2.imshow('c', curImg)
#cv2.waitKey()
img1 = face_recognition.load_image_file('img/1.jpg')
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img2 = face_recognition.load_image_file('img/z1.jpeg')
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

facelocs =face_recognition.face_locations(img1) # 定位人脸位置,检测模型，默认是hog机器学习模型
#基于卷积神经网络实现人脸检测
#facelocs =face_recognition.face_locations(img1,number_of_times_to_upsample=0, model="cnn") # 定位人脸位置

print(len(facelocs))
encodeImg1 = face_recognition.face_encodings(img1)[0]# 提取人脸的面部特征
for faceloc in facelocs: # 框出人脸
    cv2.rectangle(img1,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)
faceloc = facelocs[0]

facelocs2 = face_recognition.face_locations(img2)
print(len(facelocs2))
encodeImg2 = face_recognition.face_encodings(img2)[0]
for faceloc2 in facelocs2:
    cv2.rectangle(img2,(faceloc2[3],faceloc2[0]),(faceloc2[1],faceloc2[2]),(255,0,255),2)
faceloc2=facelocs2[0]

result = face_recognition.compare_faces([encodeImg1],encodeImg2) # 比较人脸编码的相似度
faceDis = face_recognition.face_distance([encodeImg1],encodeImg2) # 计算两个人脸的欧氏距离（欧氏距离用于计算样本之间的相似度或距离）

print(result,faceDis)

cv2.putText(img1,f'{result}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)# 显示比对结果

cv2.imshow('img1',img1)
cv2.imshow('img2',img2)
key = cv2.waitKey(0)


print("马赛克")
frame = cv2.imread("img/1.jpg")
small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
# 找到人脸
face_locations = face_recognition.face_locations(small_frame, model="cnn")
print(face_locations)
for top, right, bottom, left in face_locations:
    # 提取边界框在原图比例的边界框
    top *= 4
    right *= 4
    bottom *= 4
    left *= 4
# 提取人脸
face_image = frame[top:bottom, left:right]
# 高斯模糊人脸
face_image = cv2.GaussianBlur(face_image, (99, 99), 30)
# 原图人脸替换
frame[top:bottom, left:right] = face_image
# 展示图像
#img = img1[:,:,::-1]
#plt.axis('off')
#plt.imshow(img)
cv2.imshow("img",frame)
key = cv2.waitKey(0)
if key == 27: # 按ESC键退出
    cv2.destroyAllWindows()

print("提取图像中的人脸关键点")
from PIL import Image, ImageDraw
# 通过PIL加载图片
image = face_recognition.load_image_file("img/1.jpg")
# 找到图像中所有人脸的所有面部特征，返回字典
face_landmarks_list = face_recognition.face_landmarks(image)
# 发现人脸数print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))
# 创建展示结果的图像
pil_image = Image.fromarray(image)
d = ImageDraw.Draw(pil_image)
## 找到图像中所有人脸的所有面部特征，返回字典
face_landmarks_list = face_recognition.face_landmarks(image)
# 发现人脸数
print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))
# 创建展示结果的图像
pil_image = Image.fromarray(image)
d = ImageDraw.Draw(pil_image)
# 绘制关键点
for face_landmarks in face_landmarks_list:

    # 打印此图像中每个面部特征的位置
    # for facial_feature in face_landmarks.keys():
       # print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))

    # 用一条线勾勒出图像中的每个面部特征
    for facial_feature in face_landmarks.keys():
        d.line(face_landmarks[facial_feature], width=5)

# jupyter 绘图# pil_image.show()
img = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)  # PIL图片格式转换为CV2图片格式代码
cv2.imshow("img",img)
key = cv2.waitKey(0)
if key == 27: # 按ESC键退出
    cv2.destroyAllWindows()

print("人脸涂色")
# 通过PIL加载图片
image = face_recognition.load_image_file("img/1.jpg")

# 找到图像中所有人脸的所有面部特征，返回字典
face_landmarks_list = face_recognition.face_landmarks(image)

pil_image = Image.fromarray(image)

# 绘图
for face_landmarks in face_landmarks_list:
    d = ImageDraw.Draw(pil_image, 'RGBA')

    # 眉毛涂色
    d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
    d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
    d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
    d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)

    # 嘴唇涂色
    d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
    d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
    d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
    d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)

    # 眼睛涂色
    d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
    d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

    # 眼线涂色
    d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
    d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)

img = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)  # PIL图片格式转换为CV2图片格式代码
cv2.imshow("img",img)
key = cv2.waitKey(0)
if key == 27: # 按ESC键退出
    cv2.destroyAllWindows()
