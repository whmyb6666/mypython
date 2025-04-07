import threading
import time

def bigclau():
    i =0
    while True:
        i=i+1
        if i >100000:
            break


class myThread(threading.Thread):
    def __init__(self, a1, a2):
        threading.Thread.__init__(self)
        self.a1 = a1
        self.a2=  a2

    def geta1(self):
        return self.a1

    def geta2(self):
        return self.a2

    def set(self,a1,a2):
        self.a1 = a1
        self.a2=  a2

    def run(self):
        print(f"{self.a1}  {self.a2} thread started...")
        bigclau()
        time.sleep(1)
        print(f"{self.a1}  {self.a2} thread stoped!!!")
def thread_func(a1,a2):
    print(f"{a1}  {a2} thread started...")
    time.sleep(10)
    print(f"{a1}  {a2} thread stoped!!!")

thread = threading.Thread(target=thread_func,args=(1,3))
thread.daemon =True
thread .start()

thread1 = myThread(100,300)
thread1.run()
thread1.start()

while thread.is_alive():
    if thread.is_alive():
        print("thread is still running ")
        time.sleep(0.1)

    if not thread1.is_alive():
        a1 = thread1.geta1()+1
        a2 = thread1.geta2()+1
        thread1.set( a1,a2)
        print('重新启动thread1，，，')
        thread1.run()

while thread1.is_alive():
    pass
print("thread is stop run ")