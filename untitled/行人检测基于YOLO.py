import cv2
import sys
import time

# 初始化OpenCV的视频捕捉设备
cap = cv2.VideoCapture(0)

# 检查是否成功打开摄像头
if not cap.isOpened():
    print("无法打开摄像头")
    sys.exit()

# 加载YOLOv5模型
net = cv2.dnn.readNet('yolov5s.weights', 'yolov5s.cfg')
#net = cv2.dnn.readNet('models/yolov5s.pt')
# 获取命令行参数（可选的）
confThreshold = 0.5  # 置信度阈值
nmsThreshold = 0.4  # NMS阈值

while True:
    # 读取一帧图像
    ret, frame = cap.read()
    if not ret:
        print("无法读取帧")
        break

    # 获取图像的宽度和高度
    (h, w) = frame.shape[:2]

    # 构造blob从帧读取
    blob = cv2.dnn.blobFromImage(frame, 0.003922, (w, h), (0, 0, 0), True, crop=False)

    # 设置网络输入
    net.setInput(blob)

    # 运行前向传递
    start = time.time()
    layersNames = net.getLayerNames()
    outputNames = [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    end = time.time()

    # 解析YOLO输出
    classIds = []
    confidences = []
    boxes = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                width = int(detection[2] * w)
                height = int(detection[3] * h)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # 非极大值抑制
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    # 在框外画出行人
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        cv2.rectangle(frame, (left, top), (left + width, top + height), (255, 0, 0), 2)

    # 显示结果
    cv2.imshow("Frame", frame)
    cv2.imshow("Detected People", frame)

    # 按'q'退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放捕捉设备
cap.release()
# 关闭所有OpenCV窗口
cv2.destroyAllWindows()