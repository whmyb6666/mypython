import psutil
import time
import os

# 获取当前进程的PID
pid = os.getpid()

# # 获取另一个Python进程的进程对象
# process = psutil.Process(pid)
#
# # 读取进程内存信息
# memory_info = process.memory_info()
#
# # 输出内存信息
# print(f"RSS: {memory_info.rss}")  # 常驻内存集大小
# print(f"VMS: {memory_info.vms}")  # 虚拟内存集大小


# 获取当前进程的内存占用
def get_currprocess_use():
    # 获取当前进程
    current_process = psutil.Process(pid)
    # 获取当前进程的内存使用信息
    memory_info = current_process.memory_info()

    # 打印内存使用信息
    print(f"当前程序 常驻内存: size={memory_info.rss / 1024 ** 2:.2f} MB, 虚拟内存={memory_info.vms / 1024 ** 2:.2f} MB")


def get_cpu_usage():
    cpu_usage = psutil.cpu_percent(interval=1)
    return cpu_usage

def get_memory_usage():
    memory = psutil.virtual_memory()
    memory_usage = memory.percent
    return memory_usage

def showSystemInfo():
    get_currprocess_use()
    cpu_usage = get_cpu_usage()
    memory_usage = get_memory_usage()
    print(f"系统CPU占用率：{cpu_usage}%")
    print(f"系统内存占用率：{memory_usage}%")
    print("-" * 30)


if __name__ == "__main__":
    while True:
        showSystemInfo()
        time.sleep(1)


# import os
# import psutil
# import time
# import multiprocessing
#
#
# def get_pid_memory(pid):
#     """
#     根据进程号来获取进程的内存大小
#     :param pid: 进程id
#     :return: pid内存大小/MB
#     """
#     process = psutil.Process(pid)
#     mem_info = process.memory_info()
#     return mem_info.rss / 1024 / 1024
#
#
# def get_process_memory(process_name):
#     """
#     获取同一个进程名所占的所有内存
#     :param process_name:进程名字
#     :return:同一个进程名所占的所有内存/MB
#     """
#     total_mem = 0
#     for i in psutil.process_iter():
#         if i.name() == process_name:
#             total_mem += get_pid_memory(i.pid)
#     print('{:.2f} MB'.format(total_mem))
#     return total_mem
#
#
# def test(n):
#     pid = os.getpid()
#     print("Test PID:{}".format(pid))
#
#     for i in range(n):
#         print("Test i{}...".format(i))
#     return pid
#
#
# if __name__ == '__main__':
#     get_process_memory('python')
#
#     test_p = multiprocessing.Process(target=test, args=(10,))
#
#     test_p.start()
#     print("test_p PID:{}".format(test_p.pid))
#     print("Main PID:{}".format(os.getpid()))
#
#     test_m = multiprocessing.Process(target=get_pid_memory, args=(test_p.pid,))
#     test_m.start()