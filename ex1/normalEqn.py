import numpy as np
def normalEqn(X,y):
    #theta = zeros(size(X, 2), 1);
    w_ = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)


    return w_