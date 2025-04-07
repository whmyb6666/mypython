import torch
import cv2

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6 or custom
im = 'img/xr.jpeg'  # file, Path, PIL.Image, OpenCV, nparray, list
img = cv2.imread(im)

results = model(img) # inference
print(results.pandas().xyxy[0])
#results.show()
cv2.imshow('img',img)
cv2.waitKey(0)
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
