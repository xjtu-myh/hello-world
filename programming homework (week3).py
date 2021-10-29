import numpy as np
from activation_function import *
import matplotlib.pyplot as plt

import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

#%matplotlib inline #如果你使用用的是Jupyter Notebook的话请取消注释。
L=3
m=400
learning_rate=0.05
iteration=1000
z=[0,0,0]

A=[0,0,0]

X, Y = load_planar_dataset()
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral) #绘制散点图
def Relu(Z):
    return np.maximum(0,Z)
def Relu_1(Z):
    return np.maximum()
def tanh(Z):
    result=(np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z))
    return result
def tanh_1(Z):
    s=4/((np.exp(Z)+np.exp(-Z))*(np.exp(Z)+np.exp(-Z)))
    return s
def initialize(n_1):
    np.random.seed(2)
    
    n=[2,n_1+1,1]
    shape_w=[0]
    shape_b=[0]
    w=[0]
    b=[0]
    for i in range(1,3):
        tempw=(n[i],n[i-1])
        tempb=(n[i],1)
        shape_w.append(tempw)
        shape_b.append(tempb)
    for i in range(1,3):
        t=np.random.randn(shape_w[i][0],shape_w[i][1])

        w.append(t)

        b.append(np.zeros((shape_b[i][0],shape_b[i][1])))
    return w,b
def forward_propagate(w,b,x,z,A):
    A[0]=x

    z[1]=np.dot(w[1],A[0])+b[1]
    A[1]=relu(z[1])
    z[2]=np.dot(w[2],A[1])+b[2]
    A[2]=sigmoid(z[2])
    return z,A
def backforward_propagate(A,Y,w,z,b,X):
    dZ=[0,0,0]
    dW=[0,0,0]
    db=[0,0,0]
    dZ[2]=A[2]-Y

    dW[2]=(1/m)*(np.dot(dZ[2],A[1].T))
    
    db[2]=(1/m)*np.sum(dZ[2],axis=1,keepdims=True)
    dZ[1]=np.dot(w[2].T,dZ[2])*relu_1(z[1])
    dW[1]=(1/m)*np.dot(dZ[2],X.T)
    db[1]=(1/m)*np.sum(dZ[1],axis=1,keepdims=True)
    for i in range(1,3):

        
        w[i] = w[i] -0.05 * dW[i]
        b[i] = b[i] -0.05 * db[i]
    return w,b
def predict(w,b,x,z,A):

    w,A=forward_propagate(w,b,x,z,A)


    pre=np.round(A[2])
    return pre



w,b=initialize(5)
for i in range(1000):
    z,A=forward_propagate(w, b, X,z,A)
    w,b=backforward_propagate(A,Y,w,z,b,X)
plot_decision_boundary(lambda x: predict(w,b,x.T,z,A), X, np.squeeze(Y))
plt.title("Decision Boundary for hidden layer size " + str(4))

predictions = predict(w,b,X,z,A)
print ('准确率: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')





plt.waitforbuttonpress()

