from pylab import *
import numpy as np
#import normalEqn

file = 'E:\Desktop\MachineLearning\ex1/ex1data2.txt' #以高岭石波谱曲线为例
a = np.loadtxt(file,delimiter=',',dtype=int)
x_data = a[:,0:2]  #读取第一、二列所有数据
y_data = a[:,2].reshape((-1, 1))  #读取第三列所有数据
m = len(y_data)

#Print out some data points
print('First 10 examples from the dataset: ')
for i in range(0,10):
    print(' x = {}, y = {} '.format(x_data[i,:],y_data[i,:]))

'''#取前四十组进行训练
x = x_data[0:40,:]
y = y_data[0:40,:]
'''
#取所有数据进行训练
x = x_data[0:47,:]
y = y_data[0:47,:]

#取后七组做预测值对比
x_test = x_data[40:47,:]
y_test = y_data[40:47,:]

#Add intercept term to X
X = np.insert(x,2,1,axis = 1)

#Calculate the parameters from the normal equation
#theta = normalEqn.normalEqn(X, y)

#Display normal equation's result
#print("Theta computed from the normal equations: %f"%theta)
w_ = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
#print(w_)

b = w_[-1]
w = w_[:-1]

y_pred = np.dot(x_test, w) + b


print('7组预测值与测试值：')
for i in range(0,len(y_test)):
    print(' y_pred = {}, y_test = {} '.format(y_pred[i,:],y_test[i,:]))

#print(X,y,m)
#print(type(x))
#print(len(x),len(y),len(X))
