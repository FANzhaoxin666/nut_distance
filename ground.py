import struct
import numpy as np

def knn(train, test, k):
    d = train - test
    dicedence = np.sum(d ** 2, axis=1) ** 0.5
    Dis = dicedence.argsort()
    result1 = []
    for i in range(k):
        result1.append(train[Dis[i]])
    result1 = np.array(result1)

    result2 = Dis[0:k]

    return result1, result2


def knn_one(train, test):
    d = train - test
    dicedence = np.sum(d ** 2, axis=1) ** 0.5
    Dis = dicedence.argsort()
    train[Dis[0]]
    return train[Dis[0]], Dis[0]


def zeroMean(dataMat):
    meanVal = np.mean(dataMat, axis=0)  # 按列求均值，即求各个特征的均值
    newData = dataMat - meanVal
    return newData, meanVal


def pca(dataMat):
    newData, meanVal = zeroMean(dataMat)
    covMat = np.cov(newData, rowvar=0)  # 求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本

    eigVals, eigVects = np.linalg.eig(np.mat(covMat))  # 求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
    eigValIndice = np.argsort(eigVals)  # 对特征值从小到大排序
    n_eigVect = eigVects[:, eigValIndice[0]]  # 最小的n个特征值对应的特征向量
    return n_eigVect

def Surface(pcdata,epoches,threshold):
    '''
    :param data:x.y.z
    :param label:ground is 1 and others is 0
    :param epoches:
    :return:a,b,c,d
    '''

    label = []
    for i in range(len(pcdata)):
        label.append('blue')
    index = np.argsort(pcdata[:, -1])
    for i in reversed(index[-3000:-1]):
        label[i] = 'orange'




    result=[]
    for ite in range(epoches):
        orange_label = []
        blue_label = []
        for i in range(len(label)):
            if (label[i] == 'orange'):
                orange_label.append(pcdata[i])
        C_map = np.zeros([3, 3])
        # print("C_map",C_map)
        # print(np.array(orange_label).shape)
        orange_label = np.array(orange_label)
        x = np.mean(orange_label[:, 0])
        y = np.mean(orange_label[:, 1])
        z = np.mean(orange_label[:, 2])
        S = np.array([x, y, z])
        for each in orange_label:
            tmp_con = np.array([(np.array(each) - S)])
            C_map = C_map + np.dot(tmp_con.T, tmp_con)
        con_C = pca(C_map)
        con_C = np.array(con_C)
        d=-np.dot(con_C.T, S)

        for each in range(pcdata.shape[0]):
            node_array = np.array(pcdata[each, 0:3])
            # print(node_array)
            # print(con_C)
           # print(np.dot(con_C.T, node_array)+d)
            dis=abs(np.dot(con_C.T, node_array)+d)/np.sqrt(np.sum(np.square(con_C-np.array([0,0,0]))))
            if (dis<threshold):
                label[each] = 'orange'

            else:
                label[each] = 'blue'
                blue_label.append(pcdata[each])
        result=con_C

    return result,d,label

def Surface2(pcdata,epoches,threshold):
    '''
    :param data:x.y.z
    :param label:ground is 1 and others is 0
    :param epoches:
    :return:a,b,c,d
    '''

    label = []
    for i in range(len(pcdata)):
        label.append('blue')
    index = np.argsort(pcdata[:, -1])
    for i in index[0:1000]:
        label[i] = 'orange'



    result=[]
    for ite in range(epoches):
        orange_label = []
        blue_label = []
        for i in range(len(label)):
            if (label[i] == 'orange'):
                orange_label.append(pcdata[i])
        C_map = np.zeros([3, 3])
        # print("C_map",C_map)
        # print(np.array(orange_label).shape)
        orange_label = np.array(orange_label)
        x = np.mean(orange_label[:, 0])
        y = np.mean(orange_label[:, 1])
        z = np.mean(orange_label[:, 2])
        S = np.array([x, y, z])
        for each in orange_label:
            tmp_con = np.array([(np.array(each) - S)])
            C_map = C_map + np.dot(tmp_con.T, tmp_con)
        con_C = pca(C_map)
        con_C = np.array(con_C)
        d=-np.dot(con_C.T, S)

        for each in range(pcdata.shape[0]):
            node_array = np.array(pcdata[each, 0:3])
            # print(node_array)
            # print(con_C)
           # print(np.dot(con_C.T, node_array)+d)
            dis=abs(np.dot(con_C.T, node_array)+d)/np.sqrt(np.sum(np.square(con_C-np.array([0,0,0]))))
            if (dis<threshold ):
                label[each] = 'orange'

            else:
                label[each] = 'blue'
                blue_label.append(pcdata[each])
        result=con_C

    return result,d,label

def Surface3(pcdata,epoches,threshold,abc):
    '''
    :param data:x.y.z
    :param label:ground is 1 and others is 0
    :param epoches:
    :return:a,b,c,d
    '''

    label = []
    for i in range(len(pcdata)):
        label.append('blue')
    index = np.argsort(pcdata[:, -1])
    for i in reversed(index[-3000:-1]):
        label[i] = 'orange'




    result=[]
    for ite in range(epoches):
        orange_label = []
        blue_label = []
        for i in range(len(label)):
            if (label[i] == 'orange'):
                orange_label.append(pcdata[i])
        # print("C_map",C_map)
        # print(np.array(orange_label).shape)
        orange_label = np.array(orange_label)
        x = np.mean(orange_label[:, 0])
        y = np.mean(orange_label[:, 1])
        z = np.mean(orange_label[:, 2])
        S = np.array([x, y, z])
        con_C = abc
        d=-np.dot(con_C.T, S)

        for each in range(pcdata.shape[0]):
            node_array = np.array(pcdata[each, 0:3])
            # print(node_array)
            # print(con_C)
           # print(np.dot(con_C.T, node_array)+d)
            dis=abs(np.dot(con_C.T, node_array)+d)/np.sqrt(np.sum(np.square(con_C-np.array([0,0,0]))))
            if (dis<threshold ):
                label[each] = 'orange'

            else:
                label[each] = 'blue'
                blue_label.append(pcdata[each])
        result=con_C
        #threshold=threshold*0.8

    return result,d,label