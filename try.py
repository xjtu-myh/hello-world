import numpy as np
from time import time
b=[i for i in range(10000)]
a=np.array([i for i in range(10000)])
x=0
time0=time()
a+=1
time1=time()
print("numpy",time1-time0)
for each in b:
    each+=1
time2=time()
print("list",time2-time1)