from copy import deepcopy
from numpy import *
from main import loadData,createTree,classify
import TreePlotter

#dataset,labels = loadData('watermelon')    #读取西瓜数据集和特征列表
dataset,labels = loadData('breastcancer')    #读取乳腺癌数据集和特征列表

dataset_full = deepcopy(dataset) #拷贝一份数据集
labels_full = deepcopy(labels)  #拷贝一份特征列表

#交叉验证
k_times = 10      #次
k_fold = 10       #折
accuracy_list_all=[] #用于存储所有次十折精度

print('10次10折交叉验证的精度结果为:')
for h in range(k_times):

    dataset=deepcopy(dataset_full)      #重置数据集
    length = int(len(dataset)/k_fold)   #取整之后的每折数据长度

    #对数据集进行随机十折拆分
    per_set = []    #用于存储每折数据集
    ten_set = []    #用于存储十折的数据集
    for i in range(k_fold):
        #把每一折数据逐一添加到per_set
        for j in range(length):
            random_num = random.randint(0, len(dataset)-1)      #生成一个(0,数据集长度-1)的随机整数
            per_set.append(dataset[random_num])
            dataset.pop(random_num)  #已添加的数据出栈

        ten_set.append(deepcopy(per_set)) #把每折添加到ten_set
        per_set.clear()     #清空
    #得到的ten_set的len为10

    accuracy_list = []  #用于存储每次十折精度
    #每折数据轮流当作测试集，其余九折当作训练集
    print(f'第{h+1}次十折精度：')
    for i in range(k_fold):
        #把随机十折拆分的数据集分成训练集和测试集
        temp = deepcopy(ten_set)
        x_test = deepcopy(temp[i]) #轮流当作测试集
        del temp[i]                 #测试集出栈
        x_train = deepcopy(temp)    #剩下的作为训练集，此时剩下的9折数据为9个分开列表形式

        x_train_full = []
        for j in x_train:           #将9个列表合并成一个
            x_train_full+=j

        labels = deepcopy(labels_full)
        #生成决策树
        myTree = createTree(x_train_full, labels)
        #TreePlotter.createPlot(myTree)
        #print(myTree)

        #计算精度
        count = 0 #计数分类正确的个数
        for l in x_test:
            testClass = classify(myTree,l,labels_full)
            if str(testClass) == str(l[-1]):
                count += 1

        accuracy_list.append(float(count / length))

    print(accuracy_list)
    accuracy_list_all.append(sum(accuracy_list) / k_fold)
    print(f'平均精度{sum(accuracy_list) / k_fold}')

print(f'{k_times}次十折总平均精度为:{sum(accuracy_list_all) / k_times}')
