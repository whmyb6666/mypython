import numpy as np
import cv2 as cv
import time
import datetime

from yolox import YoloX

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]

classes = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
           'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

def letterbox(srcimg, target_size=(640, 640)):
    padded_img = np.ones((target_size[0], target_size[1], 3)).astype(np.float32) * 114.0
    ratio = min(target_size[0] / srcimg.shape[0], target_size[1] / srcimg.shape[1])
    resized_img = cv.resize(
        srcimg, (int(srcimg.shape[1] * ratio), int(srcimg.shape[0] * ratio)), interpolation=cv.INTER_LINEAR
    ).astype(np.float32)
    padded_img[: int(srcimg.shape[0] * ratio), : int(srcimg.shape[1] * ratio)] = resized_img

    return padded_img, ratio

def unletterbox(bbox, letterbox_scale):
    return bbox / letterbox_scale

def vis(dets, srcimg, letterbox_scale, fps=None):
    res_img = srcimg.copy()

    if fps is not None:
        fps_label = "FPS: %.2f" % fps
        cv.putText(res_img, fps_label, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    timestr = time.strftime("%Y-%m-%d %X", time.localtime())
    cv.putText(res_img, timestr, (400, 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    havePerson= False
    #print(dets)
    for det in dets:
        box = unletterbox(det[:4], letterbox_scale).astype(np.int32)
        score = det[-2]
        cls_id = int(det[-1])
        if cls_id ==0 : havePerson =True
        x0, y0, x1, y1 = box

        text = '{}:{:.1f}%'.format(classes[cls_id], score * 100)
        font = cv.FONT_HERSHEY_SIMPLEX
        txt_size = cv.getTextSize(text, font, 0.4, 1)[0]
        cv.rectangle(res_img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv.rectangle(res_img, (x0, y0 + 1), (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])), (255, 255, 255), -1)
        cv.putText(res_img, text, (x0, y0 + txt_size[1]), font, 0.4, (0, 0, 0), thickness=1)

    return res_img,havePerson

if __name__=='__main__':
    model_net = YoloX(modelPath= 'object_detection_yolox_2022nov.onnx',
                      confThreshold=0.65,
                      nmsThreshold=0.5,
                      objThreshold=0.5,
                      backendId=cv.dnn.DNN_BACKEND_CUDA,  # 开启GPU
                      targetId=cv.dnn.DNN_TARGET_CUDA)    # 开启GPU

    # 设置窗口的尺寸
    window_name = 'Image Display'
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, 800, 600)

    deviceId = 0
    cap = cv.VideoCapture(deviceId)

    #cap.set(cv.CAP_PROP_FPS, 120) #这个有时候生效，有时候不生效不知道是什么原因
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    #cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    tm = cv.TickMeter()
    tm.reset()

    #mp4v– >.mp4
    #MJPG– >.avi

    fource = cv.VideoWriter_fourcc(*'mp4v')
    outmp4 = None

    allow_writeMP4 = False  #是否保存视频的标记
    TOTOL_DELAY = 60.0      # 设置检测到人体保存视频后，延时60s，没有检测到人体，关闭视频保存
    curr_timelength = 0.0   # 记录当前的时长
    framesBuffers =[]       # 视频帧的缓冲区
    havePersonsFlagsBuffers =[] # 记录存在人的缓冲区
    BufferLength = 10           # 缓冲区大小
    #ji_count = 4
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        input_blob = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        input_blob, letterbox_scale = letterbox(input_blob)

        # Inference
        if allow_writeMP4 : tm.start()

        #if ji_count > 0:
        preds = model_net.infer(input_blob)
        #    ji_count -= 1
        #print('preds',type(preds),preds.shape)
        #preds=[]
        if allow_writeMP4 : tm.stop()
        if allow_writeMP4 : curr_timelength += tm.getTimeSec()
        if allow_writeMP4 : print(f'curr_timelength={curr_timelength}  curr_FPS = {tm.getFPS()}')
        frame,havePersonFlag = vis(preds, frame, letterbox_scale)#, fps=tm.getFPS())
        #havePersonFlag =True
        # 保存检测到人的标记到缓冲区 =================
        havePersonsFlagsBuffers.append(havePersonFlag)
        # 保存视频帧到缓冲区 =================
        framesBuffers.append(frame)

        # 保证缓冲区的大小
        if len(havePersonsFlagsBuffers) > BufferLength:
            print(f'删除缓冲区第一条数据，buffer length={len(havePersonsFlagsBuffers)} havePerson ={havePerson}')
            del havePersonsFlagsBuffers[0]
            del framesBuffers[0]

        cv.imshow(window_name, frame)

        # 连续帧检测到人的次数大于3，说明存在有效的人，防止误检（说明可能偶尔存在非人检测成人）
        havePerson =True if sum(havePersonsFlagsBuffers) >= 3 else False

        # 检测到有人，当前计时器清零，运行保存视频标记设置为Ture
        if havePerson:
            allow_writeMP4 = True
            curr_timelength = 0.0

        # 开始新的视频文件句柄
        if outmp4 == None and allow_writeMP4:
            nowtime = str(datetime.datetime.now())[:19].replace(':', "_")
            filename = f'runs/detect/{nowtime}.mp4'
            outmp4 = cv.VideoWriter(filename, fource, 5.0, (640, 480))
            # 把缓冲区的视频保存到视频文件
            for frame in framesBuffers:
                outmp4.write(frame)
        # 保存视频
        if outmp4 != None :
            if curr_timelength < TOTOL_DELAY :
                outmp4.write(frame)
            else: # 计时器超过时延设置，释放资源，关闭视频保存
                outmp4.release()
                outmp4=None
                allow_writeMP4 =False

        tm.reset()

    outmp4.release()
    cap.release()
