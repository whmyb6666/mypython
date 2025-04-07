import pyaudio
import wave
import datetime
import time
import random

# 音频参数
wavframes =[]  # 音频缓冲区
wf =None       # 音频文件句柄
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

        # 把音频记录保存到音频缓冲区
        if wf == None :
            #print(len(in_data))
            wavframes.append(in_data)
            # 如果超过时长，删除第一条记录，确保缓冲区的音频时长为设置的时长
            if len(wavframes) >= totalbuffer:
                del wavframes[0]
                #print(f"删除第一个wavframes，wavframes length={len(wavframes)}")
        else: # 把音频记录保存到文件===
            wf.writeframes(in_data)
        if audio_record_flag :
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
    return stream,p

# 音频流写入文件
def streamToFile(filename,p):
    CHANNELS = 2
    FORMAT = pyaudio.paInt16
    RATE = 44100
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    print(f'frames length = {len(wavframes)}')
    #保存缓冲区的音频记录====
    wf.writeframes(b''.join(wavframes))
    # 清空缓冲区
    wavframes.clear()
    return wf

# 关闭音频文件
def stopWriteFile():
    wf.close()
    return None

# 关闭音频流
def stopWareStream(stream,p):
    print("* recording")
    # while stream.is_active():
    #  time.sleep(1)
    #  time_count += 1

    stream.stop_stream()
    stream.close()
    p.terminate()
    print("* recording done!")

if __name__=='__main__':
    stream,paudio = startWareStream()
    random.seed()
    for i in range(3):
        nowtime = str(datetime.datetime.now())[:19].replace(':', "_")
        wave_filename = f'runs/{nowtime}.wav'

        sleeptime = random.randint(1,10)
        print(f"start recode to buffer {sleeptime} {wave_filename}-------")
        time.sleep(sleeptime)
        print("start recode 5 to file ..")

        # 开始保存音频文件
        wf = streamToFile(wave_filename,paudio)
        #wavframes.clear()
        time.sleep(5)
        #停止音频文本保存
        wf = stopWriteFile()
        print(f"end recode file {wave_filename}")

    #audio_record_flag = False
    # 停止音频流检测
    stopWareStream(stream,paudio)