from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
import math

# Define the model inputs
problem = {
    'num_vars': 2,
    'names': ['x1', 'x2'],
    'bounds': [[0, 10],
               [-1, 1]]
}
def evaluate(X):  # 这里是我们要进行灵敏度分析的模型,接受一个数组,每个数组元素作为模型的一个输入,模型的输出是一个float,干函数返回的时候再讲所有输出并起来
    return np.array([0.118*x[0]+0.668*math.exp(x[1])+7.99 for x in X])


# Generate samples
param_values = saltelli.sample(problem, 1000)

# Run model (example)
Y = evaluate(param_values)
print(param_values.shape, Y.shape)
# Perform analysis (这里运行完成后会自动对结果进行展示)
Si = sobol.analyze(problem, Y, print_to_console=True)
print()

# Print the first-order sensitivity indices  一阶灵敏度
print('S1:', Si['S1'])

# Print the second-order sensitivity indices   二阶灵敏度
print("x1-x2:", Si['S2'][0, 1])

from SALib.plotting.bar import plot as barplot
import matplotlib.pyplot as plot

Si_df = Si.to_df()
barplot(Si_df[0])
plot.show()