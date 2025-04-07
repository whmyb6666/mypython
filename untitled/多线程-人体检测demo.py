import numpy as np
import threading
from copy import deepcopy
from imutils.video import FPS
from moviepy.editor import *

import gc

import cv2 as cv
import time
import datetime

import pyaudio
import wave

from yolox import YoloX
#from concurrent.futures import ThreadPoolExecutor

#thread_lock = threading.Lock()
thread_exit = False  # 多线程退出标记
system_exit = False  # 系统因资源受限，强制退出

import psutil
import os
import wmi

logfilename =f"runs/detect/person_{str(datetime.datetime.now())[:19].replace(':', '_')}.log"

# # 获取另一个Python进程的进程对象
# process = psutil.Process(pid)
#
# # 读取进程内存信息
# memory_info = process.memory_info()
#
# # 输出内存信息
# print(f"RSS: {memory_info.rss}")  # 常驻内存集大小
# print(f"VMS: {memory_info.vms}")  # 虚拟内存集大小



# 定义合并视频与音频的多线程类
class myThreadSyetemInfo(threading.Thread):
    def __init__(self,pid):
        super(myThreadSyetemInfo, self).__init__()
        self.one_info = None
        self.pid = pid
        self.count =0
        self.jl_exit = 0

    def get_cpu_temperature(self):
        w = wmi.WMI(namespace="root\OpenHardwareMonitor")
        temperature_infos = w.Sensor()
        resultstr = ''
        for sensor in temperature_infos:
            if sensor.SensorType == 'Temperature' and 'Core' in sensor.Name:
                #print('{} : {}°C'.format(sensor.Name, sensor.Value))
                resultstr += f'{sensor.Name} : {sensor.Value}°C '
                #ret.update({sensor.Name: sensor.Value})
        return resultstr
    # # 获取 CPU 温度
    # def get_cpu_temperature(self):
    #     try:
    #         temperatures = psutil.sensors_temperatures()
    #         if 'coretemp' in temperatures:
    #             for entry in temperatures['coretemp']:
    #                 if entry.label == 'Package id 0':
    #                     return entry.current
    #     except Exception as e:
    #         print(f"Error getting CPU temperature: {e}")
    #     return None

    # 获取当前进程的内存占用
    def get_currprocess_use(self):
        # 获取当前进程
        current_process = psutil.Process(self.pid)
        # 获取当前进程的内存使用信息
        memory_info = current_process.memory_info()
        # 返回内存使用信息
        return memory_info.rss , memory_info.vms

    def get_cpu_usage(self):
        cpu_usage = psutil.cpu_percent(interval=1)
        return cpu_usage

    def get_memory_usage(self):
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        return memory_usage

    def showSystemInfo(self):
        global system_exit
        rss,vms = self.get_currprocess_use()
        cpu_usage = self.get_cpu_usage()
        memory_usage = self.get_memory_usage()
        get_cpu_temperature =self.get_cpu_temperature()

        if self.count ==10: # 延时到第10秒给参数赋值
            self.one_info = rss,vms,cpu_usage,memory_usage

        if self.one_info ==None:
            currInfo= f"当前程序 常驻内存: size={rss / 1024 ** 2:.2f} MB, 虚拟内存={vms / 1024 ** 2:.2f} MB"
            print(f"{currInfo}\n系统CPU占用率：{cpu_usage}% \n系统内存占用率：{memory_usage}%")
            print( get_cpu_temperature)
        else:
            rss1, vms1,cpu_usage1, memory_usage1 = self.one_info
            currInfo= f"当前程序 常驻内存: size={rss / 1024 ** 2:.2f} MB 增长{ (rss-rss1)/ 1024 ** 2:.2f} MB , 虚拟内存={vms / 1024 ** 2:.2f} MB 增长{ (vms-vms1)/ 1024 ** 2:.2f} MB 系统CPU占用率：{cpu_usage}% { get_cpu_temperature}"
            print(f"{currInfo}\n系统CPU占用率：{cpu_usage}%  增长 {cpu_usage-cpu_usage1:.2f}% \n系统内存占用率：{memory_usage}% 增长 {memory_usage-memory_usage1:.2f}%")
            # 当前程序的虚拟内存占用量的超过量大于程序首次运行的内存占用量退出
            if vms-vms1 > vms1: #  vms1:
                self.jl_exit +=1
            else:
                self.jl_exit = 0

            if self.jl_exit > 10:
                system_exit =True
            else:
                system_exit =False

            saveLogs(currInfo)
        #print("-" * 30)

    def run(self) -> None:
        while not thread_exit:
            self.showSystemInfo()
            if self.count < 30:
                self.count += 1
            #print(f'系统信息{gc.get_count()}:垃圾回收：{gc.collect()}')
            time.sleep(5)
# audio_record_flag =True
#
# # 停止记录声音
# def stop_wave(stream,p,wf):
#     audio_record_flag = False
#     #while stream.is_active():
#     #    time.sleep(1)
#
#     stream.stop_stream()
#     stream.close()
#     wf.close()
#     p.terminate()
#     print("audio recording done!!!!!")
#
# # #开启音频流
# # def start_wave_stream():
# #     FORMAT = pyaudio.paInt16
# #     CHANNELS = 2
# #     RATE = 44100
# #     p = pyaudio.PyAudio()
# #     audio_record_flag = True # 控制音乐是否录音的标记
# #     def callback(in_data, frame_count, time_info, status):
# #         #wf.writeframes(in_data)
# #         if audio_record_flag:
# #             return (in_data, pyaudio.paContinue)
# #         else:
# #             return (in_data, pyaudio.paComplete)
# #     # 音频保存的回调函数
# #     stream = p.open(format=p.get_sample_size(FORMAT),
# #         channels=CHANNELS,
# #         rate=RATE,
# #         input=True,
# #         stream_callback=callback)
# #
# #     stream.start_stream()
#
#
# # 开始记录声音到文件
# def start_wave(filename):
#     CHUNK = 1024
#     FORMAT = pyaudio.paInt16
#     CHANNELS = 2
#     RATE = 44100
#
#     p = pyaudio.PyAudio()
#     #nowtime = str(datetime.datetime.now())[:19].replace(':', "_")
#     # f'runs/detect/audio/{nowtime}.wav'
#     wf = wave.open(filename, 'wb')
#     wf.setnchannels(CHANNELS)
#     wf.setsampwidth(p.get_sample_size(FORMAT))
#     wf.setframerate(RATE)
#
#     audio_record_flag = True # 控制音乐是否录音的标记
#
#     def callback(in_data, frame_count, time_info, status):
#         wf.writeframes(in_data)
#         if audio_record_flag:
#             return (in_data, pyaudio.paContinue)
#         else:
#             return (in_data, pyaudio.paComplete)
#     # 音频保存的回调函数
#     print(f"音频文件开始录音:{filename}")
#
#     stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
#         channels=wf.getnchannels(),
#         rate=wf.getframerate(),
#         input=True,
#         stream_callback=callback)
#
#     stream.start_stream()
#     return stream,p,wf

# def run_detect_img(image):
#     #print(image)
#     print('run_detect_img  futures  is run......')
#     input_blob = cv.cvtColor(image, cv.COLOR_BGR2RGB)
#     input_blob, letterbox_scale = letterbox(input_blob)
#     result = model_net.infer(input_blob)
#     #time.sleep(1)
#     print('run_detect_img result',result)
#     letterbox_scale = letterbox_scale
#     return [result,letterbox_scale]

# 音频参数
wavframes = []  # 音频缓冲区
wf = None  # 音频文件句柄
audio_record_flag = True


# 开启音频流监测  ====
def startWareStream():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100

    # 设置音频缓冲记录的时长5秒
    timeDealy = 5

    totalbuffer = int(RATE / CHUNK * timeDealy)
    print(f'totalbuffer = {totalbuffer}')

    p = pyaudio.PyAudio()

    # 定义回调函数，用于音频记录的保存
    def callback(in_data, frame_count, time_info, status):
        global wf
        # 把音频记录保存到音频缓冲区
        if wf == None:
            # print(len(in_data))
            wavframes.append(in_data)
            # 如果超过时长，删除第一条记录，确保缓冲区的音频时长为设置的时长
            if len(wavframes) > totalbuffer:
                del wavframes[0]
                #gc.collect()
                # print(f"删除第一个wavframes，wavframes length={len(wavframes)}")
        else:  # 把音频记录保存到文件===
            try:
                wf.writeframes(in_data)
            except Exception as e:
                print("  音频记录回调函数出现异常:",e)
                saveLogs( f"音频记录回调函数出现异常:{str(e)}")
        if audio_record_flag:
            return (in_data, pyaudio.paContinue)
        else:
            return (in_data, pyaudio.paComplete)

    # 开启音频麦克风=====
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    stream_callback=callback)
    stream.start_stream()
    return stream, p

# 音频流写入文件
def streamToFile(filename, p):
    CHANNELS = 2
    FORMAT = pyaudio.paInt16
    RATE = 44100
    wfile = wave.open(filename, 'wb')
    wfile.setnchannels(CHANNELS)
    wfile.setsampwidth(p.get_sample_size(FORMAT))
    wfile.setframerate(RATE)
    print(f'frames length = {len(wavframes)}')
    # 保存缓冲区的音频记录====
    wfile.writeframes(b''.join(wavframes))
    # 清空缓冲区
    wavframes.clear()
    return wfile


# 关闭音频文件
def stopWriteFile(wfile):
    if wfile !=None:
        wfile.close()
    return None

# 关闭音频流
def stopWareStream(stream, p):
    print("* recording")
    # while stream.is_active():
    #  time.sleep(1)
    #  time_count += 1

    stream.stop_stream()
    stream.close()
    p.terminate()
    print("* recording done!")

# 把视频与音频文件进行合并的功能
def mergeMP4andAudio(mp4_filename,wav_filename,mp4_audio_filename):
    audioclip = AudioFileClip(wav_filename)
    videoclip = VideoFileClip(mp4_filename)
    print(videoclip)
    # 设置音频和视频的时间偏移，确保音频在视频的第3秒开始播放
    #audio_delay = 5
    # 应用时间偏移到音频
    #offset_audio_clip = audioclip.set_start(audio_delay)
    #final_clip = videoclip.set_audio(offset_audio_clip)
    #final_clip = videoclip.set_audio(audioclip)
    #video = CompositeVideoClip([final_clip])
    # 将音频的时长调整为与视频一致
    chang_rate = audioclip.duration / videoclip.duration
    new_audio = audioclip.fl_time(lambda t: chang_rate * t, apply_to=['mask', 'audio'])
    new_audio = new_audio.set_duration(videoclip.duration)
    print(videoclip.duration, audioclip.duration, new_audio.duration)
    # 将音频和视频进行合并
    final_clip = videoclip.set_audio(new_audio)
    # 将音频和视频进行合并
    video = CompositeVideoClip([final_clip])
    video.write_videofile(mp4_audio_filename, codec='mpeg4',bitrate='5000k')
    final_clip.close()
    audioclip.close()
    videoclip.close()
    new_audio.close()
    video.close()
    del final_clip
    del audioclip
    del videoclip
    del new_audio
    del video

# 定义合并视频与音频的多线程类
class myThreadMP4andAudio(threading.Thread):
    def __init__(self):
        super(myThreadMP4andAudio,self).__init__()
        self.workstatus = True   # 线程工作运作标志
        self.filenameList=[]     # 未处理的音视频文件列表

    # 把需要处理的音视频文件放入缓冲区，等待线程逐一处理
    def setFilename(self,mp4_filename,wav_filename,mp4_audio_filename):
        #thread_lock.acquire()
        self.filenameList.append([mp4_filename,wav_filename,mp4_audio_filename])
        # self.mp4_filename  = mp4_filename
        # self.wav_filename = wav_filename
        # self.mp4_audio_filename =mp4_audio_filename
        #thread_lock.release()

    # 设置线程工作状态标记为假，退出线程
    def exit(self):
        #thread_lock.acquire()
        self.workstatus = False
        #thread_lock.release()

    def run(self):
        # 如果音视频文件列表为空且工作状态为false，退出线程...
        global system_exit
        self.status = "RINNING"
        try:
            while self.workstatus or len(self.filenameList) > 0:
                if len(self.filenameList) > 0:
                    mp4_filename, wav_filename, mp4_audio_filename = self.filenameList[0]
                    mergeMP4andAudio(mp4_filename,wav_filename ,mp4_audio_filename)
                    # 处理完成，删除缓冲区
                    #thread_lock.acquire()
                    del self.filenameList[0]
                    #thread_lock.release()
                    print(f'合并文件完成 {mp4_audio_filename}!')
                    print(f'合并音视频文件 {gc.get_count()} 垃圾回收：{gc.collect()} {gc.get_count()}')
                    saveLogs(f'合并文件完成 {mp4_audio_filename}! 合并音视频文件 {gc.get_count()} 垃圾回收：{gc.collect()} {gc.get_count()} ')
                else:
                    print('~~~~~~~~~~~~~~~<<<<<<<<<<<<<<< 等待处理音频与视频文件合并 >>>>>>>>>>>>>>>~~~~~~~~~~~~~ ')
                time.sleep(1)
        except Exception as e:
            self.status= 'EXCEPTION'
            print(f'myThreadMP4andAudio 进程出错：{e}')
            saveLogs(f'合并音频与视频 ：myThreadMP4andAudio 进程出错：{e}')
            system_exit = True
        finally:
            self.status = 'TERMINATED'

    def restart(self):
        if self.status =='TERMINATED':
            self.start()

# 定义神经网络的多线程类
class myNetInference(threading.Thread):
    def __init__(self,model):
        super(myNetInference, self).__init__()
        self.model =model
        self.image =None
        self.result = []
        self.letterbox_scale =[]

    #返回推理结果
    def get_result(self):
        return self.result

    # 返回推理结果
    def get_letterbox_scale(self):
        return self.letterbox_scale

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
                input_blob = cv.cvtColor(self.image, cv.COLOR_BGR2RGB)
                input_blob, letterbox_scale = letterbox(input_blob)
                self.result = deepcopy(self.model.infer(input_blob))
                self.letterbox_scale = deepcopy(letterbox_scale)
                del input_blob
                del letterbox_scale
                self.count+=1
                if self.count > 100:
                    self.count=0
                #print(f'神经网络推理 {gc.get_count()} 垃圾回收：{gc.collect()}')
        #time.sleep(1)

# 摄像头的多线程
class myThreadCamera(threading.Thread):
    def __init__(self, camera_id, img_height, img_width):
        super(myThreadCamera, self).__init__()
        self.camera_id = camera_id
        self.img_height = img_height
        self.img_width = img_width
        self.getframe = True
        self.frame = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    def get_frame(self):
        #self.getframe = True
        return self.frame

    def run(self):
        global thread_exit
        cap = cv.VideoCapture(self.camera_id)
        # 设置摄像头分辨率
        cap.set(cv.CAP_PROP_FRAME_WIDTH, self.img_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.img_height)

        #cap.set(cv.CAP_PROP_FPS, 30)
        while not thread_exit:
            #if self.getframe:
                #self.getframe =False
                ret, frame = cap.read()
                if ret:
                    #frame = cv.resize(frame, (self.img_width, self.img_height))
                    #thread_lock.acquire()
                    self.frame = deepcopy(frame)
                    #thread_lock.release()
                    del frame
                    time.sleep(0.01)
                    #print(f'摄像头的多线程 {gc.get_count()} 垃圾回收：{gc.collect()}')

                else:
                    # 如果读取摄像头失败，反复关闭摄像头，重新打开
                    timeout =1
                    while True:
                        print(f"{str(datetime.datetime.now())[:19]} 读取摄像头失败。。。,重新打开摄像头")
                        cap.release()
                        time.sleep(timeout)
                        cap = cv.VideoCapture(self.camera_id)
                        if cap.isOpened():
                            break
                        else:
                            if timeout < 5:
                                timeout+=1
                    #thread_exit = True
        cap.release()

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


#画选定框 ====
def vis(dets, srcimg, letterbox_scale,drawCircle, fps=None,haveError = False):
    #res_img = srcimg.copy()
    res_img = srcimg
    if fps is not None:
        fps_label = "FPS: %.2f" % fps
        cv.putText(res_img, fps_label, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    timestr = time.strftime("%Y-%m-%d %X", time.localtime())
    # 圆的参数
    center_coordinates = (390, 20)  # 圆心坐标
    radius = 5  # 半径
    color = (0, 255, 0)  # 绿色
    colortext = (0, 255, 255) # 黄色
    if haveError:
        color = (0, 0, 255)  # 绿色
        colortext = (0, 0, 255)  # 绿色

    thickness = -1  # 线条宽度

    cv.putText(res_img, timestr, (400, 25), cv.FONT_HERSHEY_SIMPLEX, 0.6,colortext, 1)
    if drawCircle:
        # 画圆
        cv.circle(res_img, center_coordinates, radius, color, thickness)

    #cv.putText(res_img, "时间："+timestr, (400, 55), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    havePerson= False
    #print(dets)
    personBoxs =[]

    for det in dets:

        box = unletterbox(det[:4], letterbox_scale).astype(np.int32)
        #print('box',box,type(box))
        score = det[-2]
        cls_id = int(det[-1])
        if cls_id ==0 :
            havePerson =True
            personBoxs.append(box.tolist())
        x0, y0, x1, y1 = box

        text = '{}:{:.1f}%'.format(classes[cls_id], score * 100)
        font = cv.FONT_HERSHEY_SIMPLEX
        txt_size = cv.getTextSize(text, font, 0.4, 1)[0]
        cv.rectangle(res_img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv.rectangle(res_img, (x0, y0 + 1), (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])), (255, 255, 255), -1)
        cv.putText(res_img, text, (x0, y0 + txt_size[1]), font, 0.4, (0, 0, 0), thickness=1)

    return res_img,havePerson,personBoxs

# # 获取神经网络推理的最后结果
# def getNetDetect(executor,futures,frame):
#     print(frame)
#     f_dones = []
#     result=[[],[]]
#     print(f'futures length = {len(futures)}')
#     for i in range(len(futures)):
#         f = futures[i]
#         if f.done():
#             result = f.result()
#             print(f'future{i} result = {f.result()}')
#             f_dones.append(i)
#         else:
#             print(f'future{i} run status = {f.done()}')
#     f_dones_length = len(f_dones)
#     result if f_dones_length == 0 else futures[f_dones_length-1].result()
#     if f_dones_length >= 1:
#         print(f_dones)
#         for i in range(f_dones_length - 1):
#             fi = f_dones[i]
#             print(f"delete future{fi}")
#             futures[fi] = None
#             del futures[fi]
#             future = executor.submit(run_detect_img, frame)
#             futures.append(future)
#     return result

# 保存日志文件
def saveLogs(logs):
    global logfilename
    nowtime = str(datetime.datetime.now())[:19]
    with open(logfilename,'a') as f:
        f.write(nowtime + " " + logs +"\r")

# 判断线程是否正常
def detectThreadIsActive(threads):
    print("----------------------------------------------------------------------")
    haveERROR = False
    errorDict=[]
    global system_exit
    for key in threads:
        #print(f"键: {key}, 值: {threads[key]}")
        if threads[key].is_alive():
            print(f"OK,线程{key},运行正常！！！")
        else:
            print(f"ERROR:线程{key}，运行异常####")
            saveLogs(f"ERROR:线程{key}，运行异常####")

            threads[key].restart()
            saveLogs(f"重启:线程{key}，**************")

            haveERROR = True
            system_exit = True
            errorDict.append(key)
    print("")
    return haveERROR,errorDict

# 主程序函数
def main():
    global thread_exit,wf,system_exit
    threadsDict ={}
    returncode = 0

    camera_id = 0
    img_height = 480
    img_width = 640

    # 获取当前进程的PID
    pid = os.getpid()
    # 启动系统内存检测线程
    threadSystemInfo = myThreadSyetemInfo(pid)
    threadSystemInfo.start()
    threadsDict["threadSystemInfo系统资源"]= threadSystemInfo
    # 启动音视频处理多线程
    threadMP4_Audio =myThreadMP4andAudio()
    threadMP4_Audio.start()
    threadsDict["threadMP4_Audio合并音视频"] =threadMP4_Audio

    #启动摄像头多线程
    thread = myThreadCamera(camera_id, img_height, img_width)
    thread.start()
    threadsDict["thread摄像头"] = thread

    model_net = YoloX(modelPath='object_detection_yolox_2022nov.onnx',
                      confThreshold=0.75,
                      nmsThreshold=0.6,
                      objThreshold=0.6,
                      backendId=cv.dnn.DNN_BACKEND_CUDA,  # 开启GPU
                      targetId=cv.dnn.DNN_TARGET_CUDA)    # 开启GPU

    #启动神经网络推理的多线程
    threadNet = myNetInference(model_net)
    frame = thread.get_frame()
    threadNet.set_image(frame)
    #threadNet.daemon = True
    #threadNet.run()
    threadNet.start()

    threadsDict["threadNet神级网络"] = threadNet

    # #1.创建线程池
    # executor = ThreadPoolExecutor(max_workers=3)
    # futures =[]
    # future = executor.submit(run_detect_img, frame)
    # futures.append(future)

    #开启计数器
    tm = cv.TickMeter()
    tm.reset()

    # 设置视频文件格式
    fource = cv.VideoWriter_fourcc(*'mp4v')
    outmp4 = None

    allow_writeMP4 = False        # 是否保存视频的标记
    TOTOL_DELAY = 120.0           # 设置检测到人体保存视频后，延时60s，没有检测到人体，关闭视频保存
    curr_timelength = 0.0         # 记录当前的时长
    framesBuffers =[]             # 视频帧的缓冲区
    havePersonsFlagsBuffers =[]   # 记录存在人的缓冲区
    personBoxsBuffers = []        # 记录存在人的坐标位置缓冲区
    BufferLength = 300            #  视频帧缓冲区大小
    havePersonsBufferLength =10   #  检测到人的缓冲区大小

    # 设置窗口的尺寸 =======
    window_name = 'Video Display ' +   str(datetime.datetime.now())[:19]
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, 800, 600)

    # 音视频文件名===
    mp_filename =''
    mp4_audio_filename =''
    wave_filename = ''

    # 开启音频流及麦克风 ==============
    stream, paudio = startWareStream()

    # 计算总FPS
    fps = FPS().start()

    # 初始化变量
    preds =[]
    letterbox_scale=[]
    havePersonsCount = -1
    havePersonFlag = False
    personBoxs =[]
    detectThreadCount = 0
    haveThreadError =False
    haveDrawCircle = True
    # 系统循环过程-----------------
    while not thread_exit:
        #thread_lock.acquire()
        frame = thread.get_frame()
        #thread_lock.release()
        if frame.any() == 0:
            continue
        else:
            # Inference

            #检测线程是否正常运行
            detectThreadCount +=1
            if detectThreadCount > 1000:
                detectThreadCount = 0
                haveThreadError, errorThread = detectThreadIsActive(threadsDict)
                # if haveError:
                #     detectThreadCount = 1000
                    #saveLogs()

            #获取神经网络多线程推理的结果
            curr_havePersonsCount = threadNet.get_count()

            # 如果神经网络的多进程的计算进行了更新，获取新的计算结果，缓冲检测人的标记 =================
            if havePersonsCount != curr_havePersonsCount:
                del preds
                del letterbox_scale
                preds = threadNet.get_result()
                letterbox_scale = threadNet.get_letterbox_scale()

                havePersonsCount = curr_havePersonsCount
                # 保存检测到人的标记到缓冲区 =================
                havePersonsFlagsBuffers.append(havePersonFlag)
                # 保存人的坐标信息保存到缓冲区
                personBoxsBuffers.append(personBoxs)
                print('保存发现人脸的标志信息。。。')

            # 设置新的视频帧，用于神经网络多线程开启新的推理
            threadNet.set_image(frame)
            # 根据推理的结果，在视频帧上画框级打点=====
            frame, havePersonFlag,personBoxs = vis(preds, frame, letterbox_scale, haveDrawCircle,haveError=haveThreadError)  # , fps=tm.getFPS())
            if detectThreadCount % 20 ==0:
                haveDrawCircle = not haveDrawCircle
            # 保存视频帧到缓冲区 =================
            framesBuffers.append(frame)
            # 保证缓冲区的大小
            if len(havePersonsFlagsBuffers) > havePersonsBufferLength:
                print(f'删除缓冲区第一条数据，buffer length={len(havePersonsFlagsBuffers)} havePerson ={havePerson}')
                del havePersonsFlagsBuffers[0]
                del personBoxsBuffers[0]
                #gc.collect()

            if len(framesBuffers) >BufferLength:
                del framesBuffers[0]
                #gc.collect()

            cv.imshow(window_name , frame)

            # 连续帧检测到人的视频帧次数大于3，说明存在有效的人，防止误检（说明可能偶尔存在非人检测成人）
            # 并且判断人的坐标变化大于3 ，说明人在运动，通过除去重复的坐标
            lst = personBoxsBuffers
            cleaned_list = list(filter(lambda x: lst.count(x) == 1, lst))
            #print(f'lst========={lst}  cleaned_list = {cleaned_list} ')
            # if len(cleaned_list) >3:
            #     pa = cleaned_list[0]
            #     pb = cleaned_list[-1]
            #     for i in range(len(pa)):
            #         for j in range(len(pb)):
            #             distance = np.linalg.norm(np.array (pa[i]) - np.array(pb[j]))
            #             print(f'distance[{i}{j}]={distance}')
            havePerson = True if sum(havePersonsFlagsBuffers) > 3 and len(cleaned_list) >3  else False

            # 检测到有人，当前计时器清零，运行保存视频标记设置为Ture
            if havePerson:
                allow_writeMP4 = True
                tm.reset()    # 计时器清零
                tm.start()    # 开始计时

            # 开始新的视频文件句柄
            if outmp4 == None and allow_writeMP4:
                nowtime = str(datetime.datetime.now())[:19].replace(':', "_")
                mp_filename   =      f'runs/detect/mp4/{nowtime}.mp4'
                wave_filename =      f'runs/detect/audio/{nowtime}.wav'
                mp4_audio_filename = f'runs/detect/{nowtime}_audio.mp4'

                outmp4 = cv.VideoWriter(mp_filename, fource, 60.0, (640, 480)) # 开始录制视频文件
                #stream,paudio,wf = start_wave(wave_filename) #开始音频录音
                # 开始保存音频文件
                wf = streamToFile(wave_filename, paudio)
                # 把缓冲区的视频保存到视频文件
                for frame in framesBuffers:
                    outmp4.write(frame)
            # 保存视频
            if outmp4 != None:
                if curr_timelength < TOTOL_DELAY:
                    outmp4.write(frame)
                else:  # 计时器超过时延设置，释放资源，关闭视频保存，关闭音频，并把视频与音频文件合并
                    #计时器变量清零
                    curr_timelength =0.0
                    # 释放视频与音频资源
                    outmp4.release()
                    outmp4 = None
                    #del outmp4
                    allow_writeMP4 = False

                    #stop_wave(stream,paudio,wf)
                    # 停止音频文本保存
                    wf = stopWriteFile(wf)
                    #del wf
                    #megreMP4andAudio(mp_filename, mp4_audio_filename)
                    # 合并音视频文件
                    if threadMP4_Audio.is_alive():
                        threadMP4_Audio.setFilename(mp_filename,wave_filename,mp4_audio_filename)
                    else:
                        threadMP4_Audio.restart()
                        #threadMP4_Audio.start()
                        threadMP4_Audio.setFilename(mp_filename,wave_filename,mp4_audio_filename)


            if allow_writeMP4:
                tm.stop()    # 暂停计时
                curr_timelength = tm.getTimeSec()  #获取时长
                print(f'curr_timelength={curr_timelength}  curr_FPS = {tm.getFPS()}')
                tm.start()
        frame =None
        del frame

        # mp_filename = 'runs/detect/2024-07-05 18_21_32.mp4'
        # mp4_audio_filename = 'runs/detect/2024-07-05 18_21_32_audio_new_test`.mp4'
        # threadMP4_Audio.setFilename(mp_filename, WAVE_OUTPUT_FILENAME, mp4_audio_filename)

        # 如果当前程序内存资源，超过设置门槛，且当前没有检测到人，强制退出程序
        system_exit_flag = False
        #print(f'system_exit={system_exit}')
        if system_exit and not havePerson:
            system_exit_flag = True
            saveLogs("当前程序内存资源，超过设置门槛，且当前没有检测到人，强制退出程序!")


        # 通过waitkey调整帧率，按q键或者ESC键退出
        keyboard = cv.waitKey(5) & 0xFF
        if keyboard == ord('q')  or keyboard == 27 or system_exit_flag:
            if keyboard == ord('q')  or keyboard == 27:
                returncode = 100
            # 停止音频文本保存
            if wf !=None:
                wf = stopWriteFile(wf)
            # 停止音频流检测
            stopWareStream(stream, paudio)
            # 多线程退出
            thread_exit = True
        fps.update()

    fps.stop()
    tm.stop()
    # 如何当前存在没有处理的音视频资源，处理完毕后退出
    if outmp4 != None:
        outmp4.release()
        #megreMP4andAudio(mp_filename,mp4_audio_filename)
        threadMP4_Audio.setFilename(mp_filename, wave_filename, mp4_audio_filename)
        # 设置延时防止线程没有正常运行 ，快速退出
        time.sleep(2)
        #threadMP4_Audio.run()
        threadMP4_Audio.exit()
    else:
        threadMP4_Audio.exit()
    # 等着线程终止
    threadSystemInfo.join()
    thread.join()
    threadNet.join()
    threadMP4_Audio.join()
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    return returncode

# 主线程=====
if __name__ == "__main__":
    r= main()
    exit(r)