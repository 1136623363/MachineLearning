import scipy.io as scio         #用于读取.mat格式的文件
import copy
from copy import deepcopy
import operator
import math
import TreePlotter


def loadData(mat_select): #输入参数为watermelon或breastcancer # 加载mat格式数据，并转换成列表格式
    data_mat = f'./{mat_select}.mat'    #.mat格式文件路径
    a = scio.loadmat(data_mat)
    data_np = a[list(a.keys())[-1]]     #读取到的数据为numpy.ndarray格式

    #将数组转化成列表格式方便后续操作
    data_list = []
    temp = []
    for i in data_np:
        for j in i:
            temp.append(j[0])
        data_list.append(copy.deepcopy(temp))
        temp.clear()

    #数据集
    dataset = data_list[1:]
    # 特征值列表
    labels = data_list[0]
    return dataset, labels

def calShannonEnt(dataset):  #计算信息熵
    numEntries=len(dataset)  #数据集行数
    labelCounts={}           #用于保存每个标签出现次数

    # 对每组特征向量进行统计
    for featVec in dataset:
        currentLable=featVec[-1]  #提取标签的信息

        if currentLable not in labelCounts.keys():
            labelCounts[currentLable]=0     #如果标签不在统计里面
        labelCounts[currentLable]+=1        #如果在，计数
    #print(labelCounts)

    shannonEnt=0  #信息熵
    # 计算信息熵
    for key in labelCounts:
        # 选择该标签(label)的概率
        prob=float(labelCounts[key])/numEntries
        shannonEnt-=prob*math.log(prob,2)

    return shannonEnt

def splitDataset(dataset,axis,value):
    # 按照给定特征划分 数据集
    #  dataSet - 待划分的数据集
    #  axis - 划分数据集的特征
    #  value - 需要返回的特征的值
    retDataset=[]   #创建用于返回的数据集列表

    #遍历数据集
    for featVec in dataset:

        if featVec[axis]==value:   #如果划分的等于那个特征
            reducedFeatVec=featVec[:axis]    #去掉axis特征
            reducedFeatVec.extend(featVec[axis+1:])    #将符合条件的添加到返回的数据集
            retDataset.append(reducedFeatVec)
    return retDataset     #返回划分后的数据集

def chooseBestFeatureToSplit(dataset):  #选择信息增益最大的属性特征
    numFeatures=len(dataset[0])-1  #得到特征个数
    baseEntropy=calShannonEnt(dataset)  #计算当前节点的信息熵
    bestInfoGain=0.0  #信息增益
    bestFeature=-1    #最优特征索引值

    # 遍历所有特征
    for i in range(numFeatures):
        featList=[example[i] for example in dataset]
        uniqueVals=set(featList)     #创建SET集合，元素不可重复
        newEntropy=0.0

        # 计算信息增益
        for value in uniqueVals:
            subDataSet=splitDataset(dataset,i,value)    #划分子集
            prob=len(subDataSet)/float(len(dataset))    #计算子集的概率
            newEntropy+=prob*calShannonEnt(subDataSet)  #根据公式计算经验条件熵

        infoGain=baseEntropy-newEntropy #信息增益

        # 计算最优特征的信息增益
        if infoGain>bestInfoGain:
            bestInfoGain=infoGain   #更新信息增益，找到最大的信息增益
            bestFeature=i   #记录信息增益最大的特征的索引值
    return  bestFeature     #返回信息增益最大的特征的索引值

def majorityCnt(classList):    #统计classList中出现此处最多的元素(类标签)
    classCount={}

    # 统计classList中每个元素出现的次数
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1

    # 根据字典的值降序排序
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]   #返回classList中出现次数最多的元素

def createTree(dataset, labels):  #生成树
    classList=[example[-1] for example in dataset]  #取分类标签(是否好瓜)

    if classList.count(classList[0])==len(dataset): #如果类别完全相同则停止继续划分
        return classList[0]

    if len(dataset[0])==1:      #遍历完所有特征时返回出现次数最多的类标签
        return majorityCnt(classList)

    bestFeat=chooseBestFeatureToSplit(dataset) #找到信息增益最大的特征
    bestFeatLabel=labels[bestFeat] #得到信息增益最大的特征的名字，即为接下来要删除的特征
    myTree={bestFeatLabel:{}}
    del(labels[bestFeat])  #删除该特征

    featValues=[example[bestFeat] for example in dataset]
    uniqueVals=set(featValues)

    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataset(dataset,bestFeat,value),subLabels)
    return myTree

def classify(decisionTree, test, labels):
    classLabel = 0

    labels = list(labels)
    # 获取特征
    feature = list(decisionTree.keys())[0]
    # 决策树对于该特征的值的判断字段
    featDict = decisionTree[feature]
    # 获取特征的列
    feat = labels.index(feature)
    # 获取数据该特征的值
    featVal = test[feat]
    # 根据特征值查找结果，如果结果是字典说明是子树，调用本函数递归
    if featVal in featDict.keys():
        if type(featDict[featVal]).__name__ == 'dict': #判断输入值是否为“dict”
            classLabel = classify(featDict[featVal], test, labels)
        else:
            classLabel = featDict[featVal]
    return classLabel

if __name__ == '__main__':
    data, labels  = loadData('watermelon')
    #data, labels  = loadData('breastcancer')
    dataset = deepcopy(data[0:-2])
    x_test =deepcopy(data[-2:])
    labels_full = deepcopy(labels)
    myTree = createTree(dataset,copy.deepcopy(labels)) #copy.deepcopy()是深度拷贝，可防止creatTree函数中删除lables中元素
    TreePlotter.createPlot(myTree)
    print(myTree)

    #按照生成的决策树分类西瓜测试集
    for l in x_test:
        #print(l)
        print(classify(myTree,l,labels_full))