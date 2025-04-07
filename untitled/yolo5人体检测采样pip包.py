#pip install -U torch torchvision
#pip install -U yolo5

import torch
from PIL import Image
from yolov5 import YOLOv5
import cv2
im = 'img/xr.jpeg'  # file, Path, PIL.Image, OpenCV, nparray, list
img = cv2.imread(im)
# 初始化YOLOv5模型，这里以YOLOv5s为例，你可以根据需要选择其他规模的模型（如YOLOv5m, YOLOv5l）
#model = yolo5('yolov5s')# 载入模型
model = YOLOv5('yolov5s.pt' )

results = model(img, augment=True)

# 检测结果数据解析，所属类别、置信度、目标位置信息
predictions = results.pred[0]
boxes = predictions[:, :4]  # x1, x2, y1, y2
scores = predictions[:, 4]
categories = predictions[:, 5]

# 显示检测结果
results.show()
