import cv2 as cv
import time

tm=cv.TickMeter()
tm.reset()   # 清零
tm.start()   #开始计时
time.sleep(1)
tm.stop()    #暂停及时
print(tm.getTimeSec())

tm.reset()   #清零
tm.start()   #开始计时
time.sleep(1.5)
tm.stop()    #暂停计时
print(tm.getTimeSec())
