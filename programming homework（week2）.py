#identify cats
import numpy as np
from lr_utils import load_dataset
from PIL import Image
import h5py
import matplotlib.pyplot as plt
train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes = load_dataset()
learning_rate=0.005
#打印出当前的训练标签值
#使用np.squeeze的目的是压缩维度，【未压缩】train_set_y[:,index]的值为[1] , 【压缩后】np.squeeze(train_set_y[:,index])的值为1
#print("【使用np.squeeze：" + str(np.squeeze(train_set_y[:,index])) + "，不使用np.squeeze： " + str(train_set_y[:,index]) + "】")
#只有压缩后的值才能进行解码操作
train_set_x_orig_flatten=train_set_x_orig.reshape(209,12288).T
test_set_x_orig_flatten=test_set_x_orig.reshape(50,12288).T
train_set_x=train_set_x_orig_flatten/255
test_set_x=test_set_x_orig_flatten/255
#def sigmoid 函数
def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s
#初始化参数
def initialize_parameter():
    w=np.zeros((12288,1))
    b=0
    return w,b
def forward(w,b,X):
    Z=np.dot(w.T,X)+b
    A=sigmoid(Z)
    cost=-(1/50)*(np.dot(train_set_y,np.log(A).T)+np.dot(1-train_set_y,np.log(1-A).T))

    return A,cost[0][0]

def backward(w,b,X,Y,A,learning_rate):
    dz = Y - A

    dw=(1/209)*np.dot(X,dz.T)
    w+=dw*learning_rate
    db=(1/209)*np.sum(dz)
    b+=db*learning_rate
    return w,b
def propogate(w,b,X,Y,learning_rate):
    A=forward(w,b,X)[0]
    return backward(w,b,X,Y,A,learning_rate)


def test(w, b, X):
    z = np.dot(w.T, X) + b


    if sigmoid(z)>0.5:
        return 1
    else :return 0
w,b=initialize_parameter()
cost=[]
for i in range(2000):
    w,b=propogate(w,b,train_set_x,train_set_y,learning_rate)
    cost.append(forward(w,b,train_set_x)[1])
count=0


for index in range(50):
    if test_set_y[0][index]==test(w,b,test_set_x[:,index]):
        count+=1



plt.waitforbuttonpress()