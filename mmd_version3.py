# Compute MMD (maximum mean discrepancy) using numpy and scikit-learn.
"""
这是一个封装好的版本，调用的时候直接传入俩个待比较的分布的样本集即可，具体方式见最后的例子以及"mmb_rbf"

"""
import numpy as np
from sklearn import metrics
import torch
import time

def gamma_estimation(source,target):
    k = int(min(source.shape[0],target.shape[0])/3)
    x1 = 0
    x2 = 0
    x3 = 0
    for i in range(3):
        x1 += np.linalg.norm(source[k * i] - target[k * i])**2* (source.shape[0] * target.shape[0] * 2)
        x2 += np.linalg.norm(source[k * i] - source[-k * (i + 1)])**2 * (source.shape[0] ** 2)
        x3 += np.linalg.norm(target[k * i] - target[-k * (i + 1)])**2 * (target.shape[0] ** 2)
    x = (x1 + x2 + x3) / 3
    x=x.item()
    print(x)
    return ((source.shape[0]+target.shape[0])**2)/x
def mmd_rbf(X, Y,gamma):




    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()
def mkmmd_rbf(X,Y):
    #gamma=gamma_estimation(X,Y)
    gamma=1/10000#(如何要指定gamma就用这行)


    gamma_list=[gamma*(2**(i-2)) for i in range(5)]
    sum=0
    print(gamma_list)
    for each in gamma_list:
        print("each",each)
        sum+=mmd_rbf(X,Y,each)
    return sum





if __name__ == '__main__':
    time0 = time.time()
    data_1 = np.random.normal(loc=0, scale=10, size=(300, 500))
    data_2 =np.random.normal(loc=10, scale=10, size=(300, 500))
    time0 = time.time()
    gamma=gamma_estimation(data_1,data_2)
    print("MMD Loss:", mkmmd_rbf(data_1, data_2))
    time1 = time.time()
    print("time", time1 - time0)
    data_1 = np.random.normal(loc=0, scale=10, size=(100, 500))
    data_2 = np.random.normal(loc=0, scale=9, size=(80, 500))
    print("MMD Loss:", mkmmd_rbf(data_1, data_2))