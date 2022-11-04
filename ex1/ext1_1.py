import matplotlib.pyplot as plt  #绘图模块
from scipy import interpolate  #插值模块
import numpy as np  #数值计算模块
from numpy import *
from pylab import *

file = 'E:\Desktop\MachineLearning\ex1/ex1data1.txt' #以高岭石波谱曲线为例
a = np.loadtxt(file,delimiter=',')
x = a[:,0]  #读取第一列所有数据
y = a[:,1]  #读取第二列所有数据


w = sum(y*(x-sum(x)/len(x)))/(sum(x**2)-(sum(x))**2/len(x))
b = sum(y-w*x)/len(x)
print(w,b)

f = w*x + b
plot(x,f,'r-')
plot(x, y,'gx', 10)

xlabel("Population of City in 10,000s")
ylabel("Profit in $10,000s")
title('MarkerSize')

show()
