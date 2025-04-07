from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import cv2


#欧氏距离：可以使用scipy.spatial.distance.euclidean函数来计算两个向量之间的欧氏距离。示例代码如下：
from scipy.spatial.distance import euclidean

import numpy as np
#加载预训练的VGG16模型：
model = VGG16(weights='imagenet', include_top=False)
#定义一个函数来提取图像特征：
def extract_features(img_path):
    #img = image.load_img(img_path)
    img = cv2.imread(img_path)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x)
    features = np.reshape(features, (features.shape[0], -1))

    return features
#调用这个函数来提取图像的特征：
img_path = 'e1b2942258f5a6e9b4df31231a5d145.jpg'
features = extract_features(img_path)
print(features)