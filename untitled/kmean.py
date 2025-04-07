import numpy as np
from sklearn.cluster import KMeans
Data =[[1,4,5,1],[1,5,4,3],[15,23,18,4],[15,48,5,3],[100,5,48,1],[2,2,3,4],[122,122,3,1],[2,33,122,12333]]
Km = KMeans(n_clusters=3) #设置分类成几个簇
Lables = Km.fit_predict(Data)
print(Lables)