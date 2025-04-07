import matplotlib.pylab as plt
import face_recognition
import cv2
import math
from scipy.spatial import distance as dist
# 这是一个检测眼睛状态的演示
# 人眼闭上次数超过设定阈值EYES_CLOSED_SECONDS，判定人眼处于闭眼状态
EYES_CLOSED_SECONDS = 2

def main():
    # 闭眼次数
    closed_count = 0
    # 读取两张图像模仿人睁闭眼
    img_eye_opened = cv2.imread('img/mi.jpeg')
    img_eye_closed = cv2.imread('img/cccc.jpg')
    # cv2.imshow("open",img_eye_opened)
    # cv2.imshow("close",img_eye_closed)
    # cv2.waitKey(0)
    # 设置图像输入序列，前1张睁眼，中间3张闭眼，最后1张睁眼
    frame_inputs = [img_eye_opened] + [img_eye_closed] * 3 + [img_eye_opened] * 1
    print(len(frame_inputs))

    for frame_num, frame in enumerate(frame_inputs):
        # 缩小图片
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        #small_frame =frame
        # bgr通道变为rgb通道
        rgb_small_frame = small_frame[:, :, ::-1]
        # 人脸关键点检测
        face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)
        # 没有检测到关键点
        if len(face_landmarks_list) < 1:
            continue

        # 获得人眼特征点位置
        for face_landmark in face_landmarks_list:
            # 每只眼睛有六个关键点，以眼睛最左边顺时针顺序排列
            print(face_landmark)
            left_eye = face_landmark['left_eye']
            right_eye = face_landmark['right_eye']

            # 计算眼睛的纵横比ear，ear这里不是耳朵的意思
            ear_left = get_ear(left_eye)
            ear_right = get_ear(right_eye)
            print(f"ear_left:{ear_left}   eye_right:{ear_right}")
            # 判断眼睛是否闭上
            # 如果两只眼睛纵横比小于0.2，视为眼睛闭上
            closed = ear_left < 0.2 and ear_right < 0.2

            left_rangle,right_rangle = get_eye_angle(left_eye,right_eye)
            print(f"ear_left_角度:{left_rangle}   eye_right角度:{right_rangle}")
            # 判断眼睛是否闭上
            # 如果两只眼睛角度小于20，视为眼睛闭上
            closed = left_rangle < 20.0 and right_rangle < 20.0

            # 设置眼睛检测闭上次数
            if closed:
                closed_count += 1
            else:
                closed_count = 0
            # 如果眼睛检测闭上次数大于EYES_CLOSED_SECONDS，输出眼睛闭上
            if closed_count > EYES_CLOSED_SECONDS:
                eye_status = "frame {} | EYES CLOSED".format(frame_num)
            elif closed_count > 0:
                eye_status = "frame {} | MAYBE EYES CLOSED ".format(frame_num)
            else:
                eye_status = "frame {} | EYES OPENED ".format(frame_num)
            print(eye_status)

            plt.imshow(rgb_small_frame)
            # 左右眼轮廓第一个关键点颜色为red，最后一个关键点颜色为blue，其他关键点为yellow
            color = ['red'] + ['yellow'] *2  + ['blue'] + ['yellow'] *2
            # 按照顺序依次绘制眼睛关键点
            for index in range(len(left_eye)):
                #leye = left_eye[index]
                reye = right_eye[index]
                # 常用的marker："o"实心点，"."点，","极小像素点，"^"上三角，">"右三角，"+"十字……等
                #plt.plot(leye[0], leye[1], '.', color=color[index])
                plt.plot(reye[0], reye[1], '.', color=color[index])
                plt.title(eye_status)
            plt.show()
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

if __name__ == "__main__":
    main()
    #print(cal_angle((3 ** 0.5, 1), (0, 0), (3 ** 0.5, 0)) ) # 结果为 30°
