from plyfile import PlyData
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import json
from PIL import Image
from sklearn.cluster import KMeans
from ground import Surface,Surface2,Surface3
import time



def cluster_filter(p):
    cluster = KMeans(n_clusters=2)
    s = cluster.fit(p)
    label = cluster.predict(p)
    new_p = []
    if np.sum(label) > len(label)-np.sum(label):
        for j in range(len(p)):
            if label[j] == 1:
                new_p.append(p[j])
    else:
        for j in range(len(p)):
            if label[j] == 0:
                new_p.append(p[j])
    new_p = np.array(new_p)
    return new_p


def distance(filename):

  cloud_np= o3d.io.read_point_cloud(filename+'.pcd')
  image= plt.imread(filename+".jpg")

  u=image.shape[0]
  v=image.shape[1]
  pc=np.array(cloud_np.points)
  pc=np.reshape(pc,[u,v,3])
  pc=Image.fromarray(np.uint8(pc))

  with open(filename+'.json','r',encoding='utf8')as fp:
      label_2d = json.load(fp)
  num_box=len(label_2d['result'])
  nut=[]
  for i in range(num_box):
    if label_2d['result'][i]['tagtype']=='nut':
        croped_data=[]
        cur_data=label_2d['result'][i]['data']
        x1=cur_data[0]
        y1=cur_data[1]
        x2=cur_data[2]+x1
        y2=cur_data[3]+y1

        p = pc.crop((x1-20,y1-20,x2+20,y2+20))

        p= np.asarray(p)
        p=np.reshape(p,[-1,3])
        p_new=[]
        for j in range(len(p)):
          if p[j,-1]!=255:
             p_new.append(p[j])

        p=cluster_filter(p)

        knn = NearestNeighbors(n_neighbors=200, algorithm='kd_tree').fit(p)
        distance, index = knn.kneighbors(p)
        temp = p[index]
        nut_temp = np.mean(temp, axis=1)
        p = np.concatenate((p[:, 0:2], np.expand_dims(nut_temp[:, -1], -1)), axis=1)
        nut.append(p)



  result=[]

  for i in range(len(nut)):
    m4 = nut[i]
    abc2,d2,label2=Surface2(m4,10,threshold=0.2)
    abc1, d1, label1 = Surface3(m4, 10, threshold=1,abc=abc2)
    result.append(abs(d1-d2)/np.sqrt(np.sum(np.square(abc1-np.array([0,0,0])))))
    print("done 1")
  return result
start =time.time()
filename1='D:/科研项目/格灵深瞳/螺丝松动/16_rapid/114/106'
filename2='D:/科研项目/格灵深瞳/螺丝松动/16_rapid/114/106'

result1=distance(filename1)
result2=distance(filename2)
print(np.array(result1)-np.array(result2))
end =time.time()
print('Running time: {} Seconds'.format(end-start))









