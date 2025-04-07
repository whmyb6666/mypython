import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import distance as dist
import ctypes
import math
from imutils.video import FPS
import threading
from copy import deepcopy

user32 =ctypes.WinDLL("user32",use_last_error=True)

path = 'ImagesBasic'  # 人像存储位置
images = []           # 标准人像的缓冲区
className = []        # 标准人像的名称
cn_name_dicts= {}     # 标准人像的名称 英文名与中文名的对应关系字典
cn_name_postion_dicts= {}     # 中文名的与检测的坐标对应关系字典
scale = 2  #缩放系数  2:表示缩小2倍，提示处理效率，降低采样准确度
myList = os.listdir(path)  # 返回指定文件目录下的列表，这里返回的是人像图片
# 人眼闭上次数超过设定阈值EYES_CLOSED_SECONDS，判定人眼处于闭眼状态
EYES_CLOSED_SECONDS = 2
closed_count = 0

thread_exit = False

print(myList)

# 把配置文件英文名与中文名对应关系形成字典
def get_cn_name_dicts():
    with open(f'name.txt',"r",encoding="utf-8") as f:
        cn_names = f.readlines()
        for cn_name in cn_names:
            cn_name = cn_name.split(":")
            cn_name_dicts[cn_name[0]] =  cn_name[1].replace("\n","")
        f.close()
    print(cn_name_dicts)

# 定义神经网络的多线程类
class myNetInference(threading.Thread):
    def __init__(self):
        super(myNetInference, self).__init__()
        self.image =None
        self.face_landmarks_list = []
        self.faceCurFrame =[]
        self.encodesCurFrame=[]

    #返回推理结果
    def get_faceCurFrame(self):
        return self.faceCurFrame

    # 返回推理结果
    def get_encodesCurFrame(self):
        return self.encodesCurFrame

    def get_face_landmarks_list(self):
        return self.face_landmarks_list

    # 返回推理运算的次数===
    def get_count(self):
        return self.count

    # 设置新图片，并打开进行推理的开关
    def set_image(self,image):
        self.image = image
        self.startCalu = True

    # 线程运行过程，长期运行一直到程序终止
    def run(self):
        # 设置全局变量，用于程序退出
        global thread_exit
        #循环判断是否存在新图片，如果存在开始推理
        self.count =0
        while not thread_exit:
            if self.startCalu:
                self.startCalu = False
                print('myNetInference is run......')
                if scale > 1:
                    imgs = cv2.resize(self.image, (0, 0), None, 1 / scale, 1 / scale)  # 0.25, 0.25)  # 调整图片大小，缩小4倍
                else:
                    imgs = self.image

                imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

                faceCurFrame = deepcopy(face_recognition.face_locations(imgs))  # 获取人脸位置信息
                encodesCurFrame = deepcopy(face_recognition.face_encodings(imgs, self.faceCurFrame))  # 获取人脸编码
                face_landmarks_list = deepcopy(face_recognition.face_landmarks(imgs))  # 人脸关键点检测

                self.faceCurFrame = faceCurFrame
                self.encodesCurFrame = encodesCurFrame
                self.face_landmarks_list = face_landmarks_list
                self.count+=1
                if self.count > 100:
                    self.count=0
        #time.sleep(1)

def getListName():
    for cl in myList:  # 获取每张人像的名称
        #print("cl=",cl)
        curImg = cv2.imread(f'{path}/{cl}')
        #cv2.imshow('c',curImg)
        #cv2.waitKey()
        images.append(curImg)
        name =os.path.splitext(cl)[0]
        cname = cn_name_dicts.get(name,name)
        className.append(cname)
    print(className)

def write_chinese_string(img,string,point, fontScale=1, color=(255, 255, 255)):
    """
    img: imread读取的图片;
    point(x,y):字符起始绘制的位置;
    string: 显示的文字;
    return: img
    """
    x,y = point
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    # simhei.ttf 是字体，你如果没有字体，需要下载
    fontSize = 10
    fontSize = fontSize * fontScale
    font = ImageFont.truetype("simhei.ttf", fontSize, encoding="utf-8")
    draw.text((x, y-fontSize), string, color, font=font)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img

def findEncodings(images):  # 获取所有存储的人像编码
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name,isFocursWrite = False ,event =None):  # 打卡，生成记录
    with open('Attendance.csv', 'r+',encoding="utf-8") as f:
        myDatalist = f.readlines()  # 读取文件中所有的行
        nameList = []
        if not isFocursWrite: # 对用户名判断是否存存在
            for line in myDatalist:
                entry = line.split(',')
                nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%m/%d/%Y %H:%M:%S')  # 将日期时间格式化成字符串
                f.writelines(f'\n{name},{dtString}')  # 将包含多个字符串的可迭代对象写入文件中，这里是记录人名
        else: #直接写入不判断
            now = datetime.now()
            dtString = now.strftime('%m/%d/%Y %H:%M:%S')  # 将日期时间格式化成字符串
            f.writelines(f'\n{name},{dtString},{event}')  # 将包含多个字符串的可迭代对象写入文件中，这里是记录人名

# 计算人眼纵横比
def get_ear(eye):
    # 计算眼睛轮廓垂直方向上下关键点的距离
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # 计算水平方向上的关键点的距离
    C = dist.euclidean(eye[0], eye[3])

    # 计算眼睛的纵横比
    ear = (A + B) / (2.0 * C)

    # 返回眼睛的纵横比
    return ear
# 计算左右眼打开夹角
def get_eye_angle(left_eye,right_eye):
    #计算左眼 内夹角
    l_p_a = left_eye[2]
    l_p_b = left_eye[3]
    l_p_c = left_eye[0]
    l_angle = cal_angle(l_p_a,l_p_b,l_p_c)

    #计算右眼 内夹角
    r_p_a = right_eye[1]
    r_p_b = right_eye[0]
    r_p_c = right_eye[3]
    r_angle = cal_angle(r_p_a,r_p_b,r_p_c)

    return l_angle,r_angle

# 计算根据三点坐标计算夹角
def cal_angle(point_a, point_b, point_c):
    """
    根据三点坐标计算夹角
                  点a
           点b ∠
                   点c
    :param point_a、point_b、point_c: 数据类型为list,二维坐标形式[x、y]或三维坐标形式[x、y、z]
    :return: 返回角点b的夹角值
    数学原理：
    设m,n是两个不为0的向量，它们的夹角为<m,n> (或用α ,β, θ ,..,字母表示)
    1、由向量公式：cos<m,n>=m.n/|m||n|
    2、若向量用坐标表示，m=(x1,y1,z1), n=(x2,y2,z2),
    则,m.n=(x1x2+y1y2+z1z2).
    |m|=√(x1^2+y1^2+z1^2), |n|=√(x2^2+y2^2+z2^2).
    将这些代入②得到：
    cos<m,n>=(x1x2+y1y2+z1z2)/[√(x1^2+y1^2+z1^2)*√(x2^2+y2^2+z2^2)]
    上述公式是以空间三维坐标给出的，令坐标中的z=0,则得平面向量的计算公式。
    两个向量夹角的取值范围是：[0,π].
    夹角为锐角时，cosθ>0；夹角为钝角时,cosθ<0.
    """
    a_x, b_x, c_x = point_a[0], point_b[0], point_c[0]  # 点a、b、c的x坐标
    a_y, b_y, c_y = point_a[1], point_b[1], point_c[1]  # 点a、b、c的y坐标

    if len(point_a) == len(point_b) == len(point_c) == 3:
        # print("坐标点为3维坐标形式")
        a_z, b_z, c_z = point_a[2], point_b[2], point_c[2]  # 点a、b、c的z坐标
    else:
        a_z, b_z, c_z = 0, 0, 0  # 坐标点为2维坐标形式，z 坐标默认值设为0
        # print("坐标点为2维坐标形式，z 坐标默认值设为0")

    # 向量 m=(x1,y1,z1), n=(x2,y2,z2)
    x1, y1, z1 = (a_x - b_x), (a_y - b_y), (a_z - b_z)
    x2, y2, z2 = (c_x - b_x), (c_y - b_y), (c_z - b_z)

    # 两个向量的夹角，即角点b的夹角余弦值
    cos_b = (x1 * x2 + y1 * y2 + z1 * z2) / (
        math.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2) * (math.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2)))  # 角点b的夹角余弦值
    B = math.degrees(math.acos(cos_b))  # 角点b的夹角值
    return B

# 通过器官坐标判断人名
def find_face_landmarks_man_name(face_landmark):
    find_face_name = '无名'
    left_eye = face_landmark['left_eye'][4]
    right_eye = face_landmark['right_eye'][0]
    top_lip = face_landmark['top_lip'][6]
    bottom_lip = face_landmark['bottom_lip'][6]

    for key, face_loc in cn_name_postion_dicts.items():
        y1, x2, y2, x1 = face_loc
        find = False
        # 判断左眼角是否在人脸框内
        if left_eye[0] >x1 and left_eye[0] < x2 and left_eye[1] >y1 and left_eye[1]< y2:
            find =True
        else:
            continue
        # 判断右眼角是否在人脸框内
        if right_eye[0] >x1 and right_eye[0] < x2 and right_eye[1] >y1 and right_eye[1]< y2:
            find =True
        else:
            continue
        # 判断上嘴唇是否在人脸框内
        if top_lip[0] >x1 and top_lip[0] < x2 and top_lip[1] >y1 and top_lip[1]< y2:
            find =True
        else:
            continue
        # 判断下嘴唇是否在人脸框内
        if bottom_lip[0] >x1 and bottom_lip[0] < x2 and bottom_lip[1] >y1 and bottom_lip[1]< y2:
            find =True
        else:
            continue
        if find:
            find_face_name = key
            print(f"find_face_name = {find_face_name}")
            break
    return find_face_name

#使用cv2获取屏幕大小的整个过程
def get_screen_size():
    width = user32.GetSystemMetrics(0)
    height = user32.GetSystemMetrics(1)
    # 因为usb 摄像头只能产生4:3 的图片，所以根据屏幕大小，自动选择最合适的尺寸
    t_width = int(height * 4 / 3)
    t_hight = int(width * 3 / 4)

    if t_width > width :
        height = t_hight
    if t_hight > height:
        width = t_width
    return width, height

# 设置usb摄像头，fullScreen 确定是否打开全屏
def open_usb_Camera(fullScreen = False):
    cap = cv2.VideoCapture(0) #开启摄像头
    if fullScreen:
        screen_width, screen_height = get_screen_size()
        # 设置想要的分辨率宽度和高度
        #desired_width = 1024
        #desired_height = 768

        #desired_width = 1280
        #desired_height = 960

        desired_width = screen_width
        desired_height = screen_height

        # 设置摄像头分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

        # 检查分辨率是否被摄像头支持
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Desired Resolution: {desired_width}x{desired_height}")
        print(f"Actual Resolution: {actual_width}x{actual_height}")
    return cap

def main():
    global thread_exit
    # 读取中文与英文名称的对应字典
    get_cn_name_dicts()
    # 把对应的英文图片文件与中文名关联起来
    getListName()
    # 获取所有存储的人像编码
    encodeListKnown = findEncodings(images)
    print('encoding complete')
    name = ''

    # 打开摄像头
    cap = open_usb_Camera(True )
    # 设置窗口的尺寸
    window_name = 'Image Display'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    # 计算总FPS
    fps = FPS().start()

    # 启动检测线程
    success, img = cap.read()  # 获取摄像头
    myInference = myNetInference()
    myInference.set_image(img)
    myInference.start()
    my_count =-1
    while True:
        success, img = cap.read() #获取摄像头
        # if scale > 1:
        #     imgs = cv2.resize(img, (0, 0), None,1/scale,1/scale   )#  0.25, 0.25)  # 调整图片大小，缩小4倍
        # else:
        #     imgs = img
        #
        # imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
        #
        net_count = myInference.get_count()
        myInference.set_image(img)
        if net_count != my_count:
            my_count = net_count
            faceCurFrame = myInference.get_faceCurFrame()  # 获取人脸位置信息
            encodesCurFrame = myInference.get_encodesCurFrame() #face_recognition.face_encodings(imgs, faceCurFrame)  # 获取人脸编码
            #
            face_landmarks_list =  myInference.get_face_landmarks_list() #face_recognition.face_landmarks(imgs)  # 人脸关键点检测
        # 没有检测到关键点
        if len(face_landmarks_list) > 0:
            # 获得人眼特征点位置
            for face_landmark in face_landmarks_list:
                findname = find_face_landmarks_man_name(face_landmark)
                # 每只眼睛有六个关键点，以眼睛最左边顺时针顺序排列
                left_eye = face_landmark['left_eye']
                right_eye = face_landmark['right_eye']

                # # 计算眼睛的纵横比ear，ear这里不是耳朵的意思
                # ear_left = get_ear(left_eye)
                # ear_right = get_ear(right_eye)
                # print(f"ear_left:{ear_left}   eye_right:{ear_right}")
                # # 判断眼睛是否闭上
                # # 如果两只眼睛纵横比小于0.2，视为眼睛闭上
                # closed = ear_left < 0.2 and ear_right < 0.2

                # 判断眼睛是否闭上
                # 如果两只眼睛角度小于20，视为眼睛闭上
                ear_left, ear_right = get_eye_angle(left_eye, right_eye)
                print(f"ear_left_角度:{ear_left}   eye_right角度:{ear_right}")
                closed = ear_left < 20.0 and ear_right < 20.0

                # 设置眼睛检测闭上次数
                if closed:
                    closed_count += 1
                else:
                    closed_count = 0
                # 如果眼睛检测闭上次数大于EYES_CLOSED_SECONDS，输出眼睛闭上
                if closed_count > EYES_CLOSED_SECONDS:
                    eye_status = "闭眼"
                    eye_color = (255,0,0)
                    if closed_count == EYES_CLOSED_SECONDS +1:
                        markAttendance(name,True,eye_status)
                elif closed_count > 0:
                    eye_status = "可能闭眼"
                    if closed_count == 1:
                        markAttendance(findname,True,eye_status)
                    eye_color = (255, 250, 0)
                else:
                    eye_status = "睁眼"
                    eye_color = (155, 250, 0)
                print(eye_status)

                # 左右眼轮廓第一个关键点颜色为red，最后一个关键点颜色为blue，其他关键点为yellow
                color = [(0, 0, 255)] + [(0,255,255)] * 2 + [(255,0,0)] + [(0,255,255)] * 2
                # 按照顺序依次绘制眼睛关键点
                for index in range(len(left_eye)):
                    leye = [num *scale for num in left_eye[index]]
                    reye = [num *scale for num in right_eye[index]]
                    # 常用的marker："o"实心点，"."点，","极小像素点，"^"上三角，">"右三角，"+"十字……等

                    #plt.plot(leye[0], leye[1], '.', color=color[index])
                    #plt.plot(reye[0], reye[1], '.', color=color[index])
                    #plt.title(eye_status)
                    # 画一个实心圆点
                    cv2.circle(img, (leye[0], leye[1]), 2, color[index], -1)
                    cv2.circle(img, (reye[0], reye[1]), 2, color[index], -1)

                    #用左右眉毛的中点的坐标显示眼睛开关状态
                    ex =(face_landmark['left_eyebrow'][4][0] + face_landmark['right_eyebrow'][0][0]) /2
                    ey =(face_landmark['left_eyebrow'][4][1] + face_landmark['right_eyebrow'][0][1]) /2
                    img = write_chinese_string(img, f"{findname} {eye_status} 左:{ear_left:.2f} 右:{ear_right:0.2f}", ((ex-20)*scale, (ey-30)*scale), 2, eye_color)
                #plt.show()
        #print(faceCurFrame)
        for faceCur in faceCurFrame:
            cv2.rectangle(img, (faceCur[3]*scale, faceCur[0]*scale), (faceCur[1]*scale, faceCur[2]*scale), (255, 0, 255), 2) # 放大4倍

        for encodeFace, faceLoc in zip(encodesCurFrame, faceCurFrame):  # zip函数，连接成字典
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)  # 人脸匹配度
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)  # 欧式距离

            # print(faceDis)
            matchIndex = np.argmin(faceDis)  # 返回数组中小元素的索引
            if matches[matchIndex]:
                name = className[matchIndex].upper()

                print(f"{name}  差别距离：{faceDis[matchIndex]:.3f}")
                y1, x2, y2, x1 = faceLoc  # 人脸位置
                y1, x2, y2, x1 = y1 * scale, x2 * scale, y2 * scale, x1 * scale
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                #cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                #cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                cv2.rectangle(img, (x1, y2 ), (x2, y2+35), (10, 155, 10), cv2.FILLED)
                #cv2.putText(img, name, (x1 + 6, y2 + 28), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                img = write_chinese_string(img, f"{name} 差别距离:{faceDis[matchIndex]:.3f}",(x1 + 6, y2 + 28),1.5)

                #保存人脸框坐标信息
                cn_name_postion_dicts[name]= faceLoc
                markAttendance(name)  # 记录人名
                print(cn_name_postion_dicts)

        cv2.imshow(window_name, img)
        window_info = cv2.getWindowImageRect( window_name) #  str('Face_Detector'))
        print("window_info:",window_info)
        #按ESC退出
        key = cv2.waitKey(1)
        key = key & 0xff
        if key == 27:
            thread_exit = True
            break
        #print(key & 0xff)
        #cv2.waitKey(0)
        if key ==ord('s') or key ==ord('S') or key == ord(' '): # 按下's键'  or 按下 space键 保存当前屏幕:
            now = datetime.now()
            formatted_date_time = now.strftime("%Y-%m-%d %H-%M-%S")
            #print(formatted_date_time)
            cv2.imwrite(f"img/{formatted_date_time}.jpg",img)
            print(f"保存当前屏幕到./img/{formatted_date_time}.jpg，按任意键继续...")
            cv2.waitKey(0)
        fps.update()
    cap.release()
    cv2.destroyAllWindows()
    fps.stop()
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

if __name__=='__main__':
    main()