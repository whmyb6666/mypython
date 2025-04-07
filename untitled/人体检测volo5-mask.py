import torch
import cv2
import numpy as np
import datetime
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from utils.segment.general import masks2segments, process_mask, process_mask_native
from utils.general import (
    non_max_suppression,
    scale_boxes,
)
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """Resizes and pads image to new_shape with stride-multiple constraints, returns resized image, ratio, padding."""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def warmup( model,augment,visualize,device,imgsz=(1, 3, 640, 640)):
    """Performs a single inference warmup to initialize model weights, accepting an `imgsz` tuple for image size."""
    im = torch.empty(*imgsz, dtype= torch.float, device=device)  # input
    #forward(im)  # warmup
    y = model(im, augment=augment, visualize=visualize)

if __name__=="__main__":
    #load model
    ckpt = torch.load('models/yolov5s-seg.pt')
    device = torch.device("cuda")

    ckpt = ckpt["model"].to(device).float()  # FP32 model
    model =ckpt.eval()

    print(model.names)

    # 参数设置
    imgsz =(640,640)
    stride = 32
    pt= True
    vid_stride =1
    augment =False
    visualize =False
    ## NMS
    conf_thres = 0.25  # confidence threshold
    iou_thres = 0.45  # NMS IOU threshold
    max_det = 1000  # maximum detections per image
    classes = None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms = False  # class-agnostic NMS

    # 预热模型，便于加速
    warmup(model,augment,visualize,device=device)

    path = 'data/images/006.png'

    # 设置窗口的尺寸
    window_name = 'Image Display'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    #导入图片
    #img = cv2.imread(path)  # BGR
    # 定义视频编码器并创建VideoWriter对象
    fource = cv2.VideoWriter_fourcc(*'mp4v')
    nowtime = str(datetime.datetime.now())[:19].replace(':', "_")
    filename = f'runs/{nowtime}.mp4'
    out = cv2.VideoWriter(filename, fource, 20.0, (640, 480))

    videoCapture = cv2.VideoCapture(0)
    # model = torch.hub.load("ultralytics/yolov5","yolov5l",pretrained = True)
    if not videoCapture.isOpened():
        print("无法打开摄像头")
        exit()
    success, img = videoCapture.read()
    cv2.imshow(window_name, img)
    cv2.waitKey(1)
    while True:
        success,img = videoCapture.read()
        #调整图片便于yolo5 推理
        im = letterbox(img, imgsz, stride, auto=pt)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        im = torch.from_numpy(im).to(device)
        #print(im)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        #print(im.shape)
        if len(im.shape) == 3:  #扩展维度[3, 480, 640] -> [1, 3, 480, 640]
            im = im[None]  # expand for batch dim
        #print(im.shape)
        #print(im)

        # Inference 推理 ============
        pred, proto = model(im,augment=augment, visualize=visualize)[:2]
        #print(pred.size())

        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
        #print(f'perd = {len(pred)}')

        # 画框
        # Process predictions
        for i, det in enumerate(pred):  # per image
            #print(f'i={i}')
            im0 = img
            names =model.names
            line_thickness = 2
            annotator = Annotator(im0, line_thickness, example=str(names))
            if len(det):
                masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                # Mask plotting 画MASK
                annotator.masks(
                    masks,
                    colors=[colors(x, True) for x in det[:, 5]],
                    im_gpu= im[i] ) #torch.as_tensor(im[i], dtype=torch.float16).to(device).permute(2, 0, 1).flip(0).contiguous()/255)

                # 画框及标识
                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    # Add bbox to image
                    c = int(cls)  # integer class
                    label = f"{names[c]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(c, True))

        # Stream results Return annotated image as array
        im0 = annotator.result()

        # 保存到视频文件
        out.write(im0)

        # 显示视频帧
        cv2.imshow(window_name, im0)
        #save_path='runs/1.jpg'
        #cv2.imwrite(save_path, im0)
        if cv2.waitKey(1) == ord("q"):  # 1 millisecond 按q键退出
            out.release()
            break
    cv2.destroyAllWindows()
