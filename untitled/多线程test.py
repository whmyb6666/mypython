import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import random
import time
def sleep_time(sleep_seconds) :
    #time.sleep(sleep_seconds)
    #print(f'I have sleep {sleep_seconds} seconds')
    random.seed()
    x = random.randint(1, 10)
    # x =10
    print(f"Task {sleep_seconds} starting random={x}")
    # 这里执行一些高CPU消耗的计算
    result = sum([i ** x for i in range(10000000)])
    print(f"Task {sleep_seconds} finished with result {result}")
    return [result,sleep_seconds]

    #return sleep_seconds

def run():
    # with ProcessPoolExecutor(max_workers=3) as executor:
    #     futures = [executor.submit(sleep_time, time_seconds) for time_seconds in sleep_list]
    #     while True:
    #         fflag = True
    #         for f in futures:  # 非阻塞状态
    #             cfflag = f.done()
    #             print(f'当前任务：{f} 运行状态： {cfflag}')
    #             if cfflag: print(f'当前任务：{f} 计算完成结果为 {f.result()} ')
    #             fflag &= cfflag
    #
    #         if fflag: break;
    #
    #
    #     print('ProcessPoolExecutor', [future.result() for future in as_completed(futures)])

    #1.创建线程池
    from concurrent.futures import ProcessPoolExecutor

    executor = ProcessPoolExecutor(max_workers=3)
    futures =[]

    future = executor.submit(sleep_time, 1)
    futures.append(future)
    future = executor.submit(sleep_time, 2)
    futures.append(future)
    future = executor.submit(sleep_time, 3)
    futures.append(future)
    ti = 3

    while True:
        f_dones=[]
        for i in range(len(futures)):
            f = futures[i]
            if f.done():
                #result = f.result()
                print(f'future{i} result = {f.result()}')
                f_dones.append(i)
            else:
                print(f'future{i} run startuse = {f.done()}')
        f_dones_length= len(f_dones)
        if f_dones_length >=2:
            print(f_dones)

            for i in range(f_dones_length-1):
                fi = f_dones[i]
                print(f"delete future{fi}")
                futures[fi] =None
                del futures[fi]
                ti+=1
                future = executor.submit(sleep_time, ti)
                futures.append(future)
            #time.sleep(2)
        #if len(futures)==1:time.sleep(5)



sleep_list = [6, 4, 3, 2]


if __name__ == '__main__':
    # with ThreadPoolExecutor(max_workers=3) as executor:
    #     result = executor.map(sleep_time, sleep_list)
    #     print(result)
    #     print('ThreadPoolExecutor',list(result))

    #future_list = []
    run()