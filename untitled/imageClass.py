import os
import cv2
import numpy as np
from sklearn.cluster import KMeans

import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input


def aHash(img):
    # 均值哈希算法
    # 缩放为8*8
    img = cv2.resize(img, (8, 8))
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # s为像素和初值为0，hash_str为hash值初值为''
    s = 0
    hash_str = ''
    # 遍历累加求像素和
    for i in range(8):
        for j in range(8):
            s = s + gray[i, j]
    # 求平均灰度
    avg = s / 64
    # 灰度大于平均值为1相反为0生成图片的hash值
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


def dHash(img):
    # 差值哈希算法
    # 缩放8*8
    img = cv2.resize(img, (9, 8))
    # 转换灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


def pHash(img):
    # 感知哈希算法
    # 缩放32*32
    img = cv2.resize(img, (32, 32))  # , interpolation=cv2.INTER_CUBIC

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 将灰度图转为浮点型，再进行dct变换
    dct = cv2.dct(np.float32(gray))
    # opencv实现的掩码操作
    dct_roi = dct[0:8, 0:8]

    hash = []
    avreage = np.mean(dct_roi)
    for i in range(dct_roi.shape[0]):
        for j in range(dct_roi.shape[1]):
            if dct_roi[i, j] > avreage:
                hash.append(1)
            else:
                hash.append(0)
    return hash

def cmpHash(hash1, hash2):
    # Hash值对比
    # 算法中1和0顺序组合起来的即是图片的指纹hash。顺序不固定，但是比较的时候必须是相同的顺序。
    # 对比两幅图的指纹，计算汉明距离，即两个64位的hash值有多少是不一样的，不同的位数越小，图片越相似
    # 汉明距离：一组二进制数据变成另一组数据所需要的步骤，可以衡量两图的差异，汉明距离越小，则相似度越高。汉明距离为0，即两张图片完全一样
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        if hash1[i] != hash2[i]:
            n = n + 1
    return n

# 欧氏距离：可以使用scipy.spatial.distance.euclidean函数来计算两个向量之间的欧氏距离。示例代码如下：
from scipy.spatial.distance import euclidean

# 加载预训练的VGG16模型：
model = VGG16(weights='imagenet', include_top=False)

''''''
# 定义一个函数来提取图像特征：
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x)
    features = np.reshape(features, (features.shape[0], -1))
    print(features)
    return features
''''''

def compute_image_hashs(image):
    ahash=aHash(image)
    dhash=dHash(image)
    phash=pHash(image)
    len(ahash)
    a0 = cmpHash(ahash,"0"*len(ahash))
    d0 = cmpHash(dhash,"0"*len(ahash))
    p0 = cmpHash(phash,[0]*len(phash))

    a1 = cmpHash(ahash,"1"*len(ahash))
    d1 = cmpHash(dhash,"1"*len(ahash))
    p1 = cmpHash(phash,[1]*len(phash))
    return [a0,d0,p0,a1,d1,p1]

#定义计算图片特征值的函数
def compute_image_feature(image):
    #image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(400)
    key_points, descriptors = sift.detectAndCompute(gray_image, None)
    feature = np.reshape(descriptors, -1).tolist()
    print(len(feature))
    return  feature[:51200]

# 定义一个函数来提取图像特征：
def extract_feature(img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    feature = model.predict(x)
    feature = np.reshape(feature, -1)

    print(feature.shape)
    print(feature[0:100])
    return feature.tolist()



# 加载图像数据,返回图像特征
def load_images(image_dir):
    image_files = os.listdir(image_dir)
    features = []
    for image_file in image_files:
        print(image_file )
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        #feature = extract_feature(image)
        #feature = compute_image_feature(image)
        feature= compute_image_hashs(image)
        print(feature)
        features.append(feature)
    return features,image_files


# 提取图像特征
def extract_features(images):
    # 使用预训练的VGG16模型提取图像特征
    # TODO: 调用相应的深度学习库加载预训练模型，并提取特征
    features = []
    for image in images:
        feature = extract_feature(image) # 提取图像特征的代码
        features.append(feature)
    return features


# 计算相似度度量
def compute_similarity(feature1, feature2):
    # TODO: 根据特征向量计算相似度度量的代码
    similarity = euclidean(feature1, feature2)
    # print(distance) # 相似度度量的结果
    return similarity


# 图像聚类
def image_clustering(features, num_clusters):
    # 使用K-Means聚类算法将图像分为不同的类别
    print("kmeans",np.array(features).shape)
    kmeans = KMeans(n_clusters=num_clusters)
    print("kmeans.fit...")
    labels = kmeans.fit_predict(features)
    print(labels)
    return labels


# 可视化结果
def visualize_results(images, labels):
    image_dir = '201201'
    # 根据聚类结果将相似的图像显示在一起
    clusters = [[] for _ in range(max(labels) + 1)]
    print(clusters)
    for i, label in enumerate(labels):
        clusters[label].append(images[i])
    print(clusters)

    for cluster in clusters:
        for image in cluster:
            # 使用图像处理库显示图像
            #cv2.imshow('Similar Images', image)
            #cv2.waitKey(0)
            imagef = os.path.join(image_dir, image)
            print(imagef)
            #img = cv2.imread(imagef)
            #cv2.imshow('Similar Images', img)
            #cv2.waitKey(0)
        print("  ===  ")


# 主函数
if __name__ == '__main__':
    image_dir = '201201'
    features,imagefiles = load_images(image_dir)
    #features = extract_features(images)
    print("使用K-Means聚类算法将图像分为不同的类别")
    labels = image_clustering(features, num_clusters=7)
    visualize_results(imagefiles, labels)