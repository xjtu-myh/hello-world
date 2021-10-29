import matplotlib.pyplot as plt
import numpy as np
x=np.array([365,405,436,546,577])
v=3/x*10**17
y=np.array([1.812,1.505,1.272,0.703,0.525])
plt.scatter(v,y)

xx=np.polyfit(v,y,1)
print(xx)
p=np.poly1d(xx)
plt.plot(v,p(v))
plt.xlim(xmin=0)


plt.show()