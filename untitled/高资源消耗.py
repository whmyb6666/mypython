'''
在Python中实现异步运行可以使用concurrent.futures模块中的ThreadPoolExecutor或ProcessPoolExecutor。
如果计算任务可以被分解到多个独立的线程或进程中，可以使用这些执行器来异步执行计算密集型任务。

以下是使用ThreadPoolExecutor异步执行高CPU消耗的计算任务的示例代码：

'''

import concurrent.futures
import random

# 高CPU消耗的计算任务函数
def cpu_intensive_task(number ) :
    random.seed()
    x = random.randint(1,10)
    #x =10
    print(f"Task {number} starting random={x}")
    # 这里执行一些高CPU消耗的计算
    result = sum([i ** x for i in range(10000000)])
    print(f"Task {number} finished with result {result}")
    return result


# 使用ThreadPoolExecutor异步执行任务
def run_tasks_asynchronously(tasks):
    #with concurrent.futures.ThreadPoolExecutor() as executor:
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        #ProcessPoolExecutor
        #future_to_task = {executor.submit(cpu_intensive_task, task): task for task in tasks }
        future_to_task = [executor.submit(cpu_intensive_task, task) for task in tasks ]
        print(future_to_task)
        #exit()
        while True:
            fflag = True
            for f in future_to_task:     # 非阻塞状态
                cfflag = f.done()
                print(f'当前任务：{f} 运行状态： {cfflag}')
                if cfflag: print(f'当前任务：{f} 计算完成结果为 {f.result()} ')
                fflag &= cfflag

            if fflag:break;
        #print(concurrent.futures.Future.done())
        for future in concurrent.futures.as_completed(future_to_task): # 阻塞状态
            #print(future.done())
            #task = future_to_task[future]
            #print(task)
            try:
                data = future.result()
            except Exception as e:
                print(f"Task {future} failed: {e}")
            else:
                print(f"Task {future} got result: {data}")


# 要异步执行的任务列表
tasks = [1, 2, 3]

if __name__ == '__main__':
    # 运行异步任务
    run_tasks_asynchronously(tasks)


'''
在这个例子中，cpu_intensive_task函数是一个高CPU消耗的计算任务。run_tasks_asynchronously函数接收一个任务列表，
并使用ThreadPoolExecutor来异步执行这些任务。当一个任务完成时，它会通过concurrent.futures.as_completed生成器来通知。

这个例子展示了如何简单地将多线程应用于高CPU消耗的任务，以实现异步运行。如果你的任务需要更多的资源，可以考虑使用ProcessPoolExecutor，
它将任务分配给独立的进程来执行。

******************************************************************************************************
在使用多进程时，必须把 多进程代码 写在 if __name__ == '__main__' 下面，否则异常，甚至报错

concurrent.futures.process.BrokenProcessPool: A process in the process pool was terminated abruptly while the future was running or pending.

'''