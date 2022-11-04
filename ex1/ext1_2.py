from pylab import *
import numpy as np
import normalEqn

file = 'E:\Desktop\MachineLearning\ex1/ex1data2.txt' #以高岭石波谱曲线为例
a = np.loadtxt(file,delimiter=',',dtype=int)
x = a[:,0:2]  #读取第一、二列所有数据
y = a[:,2].reshape((-1, 1))  #读取第三列所有数据
m = len(y)

#Print out some data points
print('First 10 examples from the dataset: ')
for i in range(0,10):
    print(' x = {}, y = {} '.format(x[i,:],y[i,:]))


#Add intercept term to X
X = np.insert(x,2,1,axis = 1)

#Calculate the parameters from the normal equation
#theta = normalEqn.normalEqn(X, y)
w_ = normalEqn.normalEqn(X,y)
print(w_)

b = w_[-1]
w = w_[:-1]

x_test = np.array([1650, 3])

y_pred = np.dot(x_test, w) + b
#Display normal equation's result



print(' y_pred = {}'.format(y_pred))


