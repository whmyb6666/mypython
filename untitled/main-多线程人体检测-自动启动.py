# import os
# import sys
# import subprocess
#
# def restart():
#     print('重启程序……')
#     print(sys.argv)
#     try:
#         #os.system(sys.executable , sys.argv[0])
#         python = sys.executable
#         os.execl(python,"%s",* sys.argv)%(python)
#         #result = subprocess.run([sys.executable , sys.argv[0]] ,capture_output=True, text=True)
#         #print(result.stdout)
#     except Exception as e :
#         print(e)
#     #subprocess.Popen([sys.executable, "m", "main"])
#
# if __name__=="__main__":
#     print("start....",sys.executable)
#     restart()
from subprocess import Popen
import sys
filename = "多线程-人体检测demo.py"
#filename = "test1.py"
while True:
    print("\n启动 " + filename)
    # 自动运行程序
    p = Popen("python " + filename, shell=True)
    #等待程序运行结束
    p.wait()
    print(f"returncode : {p.returncode}")
    if p.returncode ==100:
        print("\n 系统全局退出！" )
        break
    else:
        print("\n停止 " + filename)