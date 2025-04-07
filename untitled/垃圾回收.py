'''
说明：
垃圾回收后的对象会放在gc.garbage列表里面
gc.collect()会返回不可达的对象数目，4等于两个对象以及它们对应的dict。
有三种情况会触发垃圾回收：
调用gc.collect(),
当gc模块的计数器达到阀值的时候。
程序退出的时候

四.gc模块常用功能解析
gc模块提供一个接口给开发者设置垃圾回收的选项。上面说到，采用引用计数的方法管理内存的一个缺陷是循环引用，
而gc模块的一个主要功能就是解决循环引用的问题。

常用函数：
1、gc.set_debug(flags) 设置gc的debug日志，一般设置为gc.DEBUG_LEAK。
2、gc.collect([generation]) 显式进行垃圾回收，可以输入参数，0代表只检查第一代的对象，1代表检查一，二代的对象，2代表检查一，二，三代的对象，如果不传参数，执行一个full collection，也就是等于传2。返回不可达（unreachable objects）对象的数目。
3、gc.get_threshold() 获取的gc模块中自动执行垃圾回收的频率。
4、gc.set_threshold(threshold0[, threshold1[, threshold2]) 设置自动执行垃圾回收的频率。
5、gc.get_count() 获取当前自动执行垃圾回收的计数器，返回一个长度为3的列表。

gc模块的自动垃圾回收机制：
必须要import gc模块，并且is_enable()=True才会启动自动垃圾回收。
这个机制的主要作用就是发现并处理不可达的垃圾对象。
垃圾回收=垃圾检查+垃圾回收
在Python中，采用分代收集的方法。把对象分为三代，一开始，对象在创建的时候，放在一代中，
如果在一次一代的垃圾检查中，改对象存活下来，就会被放到二代中，
同理在一次二代的垃圾检查中，该对象存活下来，就会被放到三代中。

gc模块里面会有一个长度为3的列表的计数器，可以通过gc.get_count()获取。
例如(488,3,0)，其中488是指距离上一次一代垃圾检查，
Python分配内存的数目减去释放内存的数目，注意是内存分配，而不是引用计数的增加。例如：
3是指距离上一次二代垃圾检查，一代垃圾检查的次数，同理，0是指距离上一次三代垃圾检查，二代垃圾检查的次数。
gc模快有一个自动垃圾回收的阀值，即通过gc.get_threshold函数获取到的长度为3的元组，
例如(700,10,10) 每一次计数器的增加，gc模块就会检查增加后的计数是否达到阀值的数目，
如果是，就会执行对应的代数的垃圾检查，然后重置计数器

例如，假设阀值是(700,10,10)：

注意点：
gc模块唯一处理不了的是循环引用的类都有__del__方法，所以项目中要避免定义__del__方法。
'''
import gc

class ClassA():
    def __init__(self):
        print('对象产生，id:%s' % str(hex(id(self))))


def f1():
    print("------0------")
    #print(gc.get_count())
    #print(gc.collect())
    print(gc.get_count())
    c1=ClassA()
    print(gc.get_count())
    c2=ClassA()
    c1.t = c2
    c2.t = c1
    print(gc.get_count())
    print("------1------")
    #print(gc.collect())
    del c1
    del c2
    print(gc.get_count())

    print("------2------")
    print(gc.garbage)
    print(gc.get_count())
    print("------3------")
    print(gc.collect())
    print("------4------")
    print(gc.garbage)
    print(gc.get_count())
    print("------5------")

if __name__=='__main__':
    #gc.set_debug(gc.DEBUG_LEAK)
    f1()



