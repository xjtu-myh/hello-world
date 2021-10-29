import numpy as np
from lr_utils import load_dataset
from PIL import Image
import h5py
import matplotlib.pyplot as plt
from testCases import *
from activation_function import *
train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes = load_dataset()
train_set_x_orig_flatten=train_set_x_orig.reshape(209,12288).T
test_set_x_orig_flatten=test_set_x_orig.reshape(50,12288).T
train_set_x=train_set_x_orig_flatten/255
#(12288,209)
test_set_x=test_set_x_orig_flatten/255
L=4
learning_rate=0.05
iteration=5000
n1=5
n2=4
n3=1
x_dim=train_set_x.shape[0]
n=[x_dim,n1,n2,n3]
m=train_set_x.shape[1]
w_shape=[x_dim]
b_shape=[0]
for i in range(1,L):
    w_shape.append((n[i],n[i-1]))
    b_shape.append((n[i],1))
parameter={"n":n,"w_shape":w_shape,"b_shape":b_shape}
def initialize(parameter):



    n=parameter["n"]
    w_shape=parameter["w_shape"]
    b_shape=parameter["b_shape"]
    w=[0]
    b=[0]
    for i in range(1,L):
        w.append(np.random.randn(w_shape[i][0],w_shape[i][1]))
        b.append(np.zeros(b_shape[i]))
    parameter["w"]=w
    parameter["b"]=b
    print(w)
    return parameter
def forward_propagate(X,Y,parameter):
    w=parameter["w"]
    b=parameter["b"]
    Z=[0]*L
    A=[X]+[0]*(L-1)

    for i in range(1,L-1):
        Z[i]=np.dot(w[i],A[i-1])+b[i]
        A[i]=relu(Z[i])
    Z[L-1]=np.dot(w[L-1],A[L-2])+b[L-1]
    A[L-1]=sigmoid(Z[L-1])
    parameter["A"]=A
    parameter["Z"]=Z
    return parameter
def backward_propagate(X,Y,parameter):
    dZ=[0]*L
    dA=[0]*L
    dw=[0]*L
    db=[0]*L
    A=parameter["A"]
    w=parameter["w"]
    Z=parameter["Z"]
    b=parameter["b"]
    dZ[L-1]=A[L-1]-Y
    dw[L-1]=1/m*np.dot(dZ[L-1],A[L-2].T)
    db[L-1]=(1/m)*np.sum(dZ[L-1],axis=1,keepdims=True)
    for i in range(L-2,0,-1):
        dA[i]=np.dot(w[i+1].T,dZ[i+1])
        dZ[i]=dA[i]*relu_1(Z[i])
        dw[i]=1/m*np.dot(dZ[i],A[i-1].T)
        db[i]=1/m*np.sum(dZ[i],axis=1,keepdims=True)
    for i in range(1,L):
        w[i]=w[i]-learning_rate*dw[i]
        b[i]=b[i]-learning_rate*db[i]
    return parameter
parameter=initialize(parameter)
for i in range(iteration):
    parameter=forward_propagate(train_set_x,train_set_y,parameter)
    parameter=backward_propagate(train_set_x,train_set_y,parameter)
A=parameter["A"]
w=parameter["w"]
b=parameter["b"]
count=0
print(w)
def test(X,Y,parameter):
    count=0
    parameter=forward_propagate(X,Y,parameter)
    A=parameter["A"]
    pre=np.round(A[L-1])
    for i in range(50):
        if(pre[0][i]==Y[0][i]):
            count+=1
    count*=2
    print(count)
test(train_set_x,train_set_y,parameter)
plt.waitforbuttonpress()








