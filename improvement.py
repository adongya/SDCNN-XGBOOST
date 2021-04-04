import numpy as np
import pandas as pd
from pandas import read_csv
global num
from tensorflow.keras.models import Model
import xgboost as xgb
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2
# from keras.models import *
from sklearn.metrics import accuracy_score,f1_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from random import sample
# from keras.models import *
from tensorflow.keras.layers import Permute,Lambda,RepeatVector,Multiply
from keras.layers.core import K


min_max_scaler = preprocessing.MinMaxScaler()
minmaxscaler = MinMaxScaler()
num =4
global num1
num1 =4
feature = 8
sum1=65
name = 'Level1'
def show():
    lev3_lab0 = pd.read_csv(r'data_set_1000train\\' + name + '_test.csv').values
    # data = lev3_lab0[:, [6, 4, 48, 8, 18, 2, 28, 46]]
    data = lev3_lab0[:, 1:]
    tsne = TSNE(n_components=3, init='pca', random_state=1)
    result = tsne.fit_transform(data)
    x_min, x_max = np.min(result), np.max(result)
    result = (result - x_min) / (x_max - x_min)# 这一步似乎让结果都变为0-1的数字
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(result[:100, 0], result[:100, 1], result[:100, 2], c='#00CED1', s=8, marker='o', label="normal")
    ax.scatter(result[100:200, 0], result[100:200, 1], result[100:200, 2], c='Chocolate', s=8, marker='o', label="CF")
    ax.scatter(result[200:300, 0], result[200:300, 1], result[200:300, 2], c='#DC143C', s=8, marker='o', label="EO")
    ax.scatter(result[300:400, 0], result[300:400, 1], result[300:400, 2], c='#A9A9A9', s=8, marker='o', label="NCR")
    ax.scatter(result[400:500, 0], result[400:500, 1], result[400:500, 2], c='#556B2F', s=8, marker='o', label="RCW")
    ax.scatter(result[500:600, 0], result[500:600, 1], result[500:600, 2], c='#9932CC', s=8, marker='o', label="REW")
    ax.scatter(result[600:700, 0], result[600:700, 1], result[600:700, 2], c='Gold', s=8, marker='o', label="RL")
    ax.scatter(result[700:800, 0], result[700:800, 1], result[700:800, 2], c='Indigo', s=8, marker='o', label="RO")
    ax.legend(loc=2)
    plt.title('Classification data visualization of ' + name)
    # plt.savefig("C:\\Users\\dong\\Desktop\\SD-XGboost\\"+str(name)+"show.png")
    plt.show(fig)

def show1(data):
    lev3_lab0 = pd.read_csv(r'data_set_1000train\\' + name + '_test.csv').values
    # data = lev3_lab0[:, [6, 4, 48, 8, 18, 2, 28, 46]]
    data = lev3_lab0[:, 1:]
    tsne = TSNE(n_components=3, init='pca', random_state=1)
    result = tsne.fit_transform(data)
    x_min, x_max = np.min(result), np.max(result)
    result = (result - x_min) / (x_max - x_min)# 这一步似乎让结果都变为0-1的数字
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(result[:100, 0], result[:100, 1], result[:100, 2], c='#00CED1', s=8, marker='o', label="normal")
    ax.scatter(result[100:200, 0], result[100:200, 1], result[100:200, 2], c='Chocolate', s=8, marker='o', label="CF")
    ax.scatter(result[200:300, 0], result[200:300, 1], result[200:300, 2], c='#DC143C', s=8, marker='o', label="EO")
    ax.scatter(result[300:400, 0], result[300:400, 1], result[300:400, 2], c='#A9A9A9', s=8, marker='o', label="NCR")
    ax.scatter(result[400:500, 0], result[400:500, 1], result[400:500, 2], c='#556B2F', s=8, marker='o', label="RCW")
    ax.scatter(result[500:600, 0], result[500:600, 1], result[500:600, 2], c='#9932CC', s=8, marker='o', label="REW")
    ax.scatter(result[600:700, 0], result[600:700, 1], result[600:700, 2], c='Gold', s=8, marker='o', label="RL")
    ax.scatter(result[700:800, 0], result[700:800, 1], result[700:800, 2], c='Indigo', s=8, marker='o', label="RO")
    ax.legend(loc=2)
    plt.title('Classification data visualization of ' + name)
    # plt.savefig("C:\\Users\\dong\\Desktop\\SD-XGboost\\"+str(name)+"show.png")
    plt.show(fig)

from matplotlib.font_manager import FontProperties
font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=15)
def plot_confusion_matrix(cm, labels_name, title):
    # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm,interpolation='nearest',)    # 在特定的窗口上显示图像
    plt.title(title,fontproperties=font_set)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('真实标签',fontproperties=font_set)
    plt.xlabel('预测标签',fontproperties=font_set)

def accuracy1(a,b):
    L1 = len(a)
    count =0
    # print(L1)
    for j in range(len(b)):
        if b[j]==a[j][0]:
            count=count+1
    return (count/L1)*100

def projection(real_data, predict_data,name1,a,b):
    L1 = len(real_data)
    L2 = len(predict_data)
    real_list = []
    real_list.append(real_data[:1000])
    real_list.append(real_data[1000:2000])
    real_list.append(real_data[2000:3000])
    real_list.append(real_data[3000:4000])
    real_list.append(real_data[4000:5000])
    real_list.append(real_data[5000:6000])
    real_list.append(real_data[6000:7000])
    real_list.append(real_data[7000:])
    # print(len(real_list))#8
    # print(real_list[6][40][0])#500
    # print(len(real_list[7]))#496
    predict_list = []
    predict_list.append(predict_data[:1000])
    predict_list.append(predict_data[1000:2000])
    predict_list.append(predict_data[2000:3000])
    predict_list.append(predict_data[3000:4000])
    predict_list.append(predict_data[4000:5000])
    predict_list.append(predict_data[5000:6000])
    predict_list.append(predict_data[6000:7000])
    predict_list.append(predict_data[7000:])
    # print(len(predict_list))#8
    # print(predict_list[6][40])#500
    # print(predict_list[7])#496

    error_list = []
    for i in range(len(predict_list)):##24
        accuracy2 = accuracy1(real_list[i], predict_list[i])
        error_list.append(accuracy2)
    error_list.append(error_list[0])
    print(len(error_list))##len(error_list)=9
    print(error_list)



    titles = np.arange(0, 8, 1).tolist()
    theta = np.arange(0, 2 * np.pi, (2/8) * np.pi)
    theta = theta.tolist()
    theta.append(0)
    #
    plt.figure(1, figsize=(5, 4))     # wide

    plt.rc('font', family='Times New Roman')
    ax1 = plt.subplot(projection='polar')#极坐标图
    ax1.set_thetagrids(np.arange(0.0, 360.0, 45.0), labels=titles, weight="bold", color="black", fontsize=16)
    if a==6:
        ax1.set_rticks(np.arange(75, 100, 5))  # (0, 100, 25)
        ax1.set_rlabel_position(0)  # 从哪一个开始标
        ax1.set_rlim(75, 100)  # 从圆心开始到最外圈的范围
    elif a==8:
        ax1.set_rticks(np.arange(90, 100, 2))#(0, 100, 25)
        ax1.set_rlabel_position(0)#从哪一个开始标
        ax1.set_rlim(90, 100)#从圆心开始到最外圈的范围
    elif a==10:
        ax1.set_rticks(np.arange(95, 100, 1))#(0, 100, 25)
        ax1.set_rlabel_position(0)#从哪一个开始标
        ax1.set_rlim(95, 100)#从圆心开始到最外圈的范围

    ax1.set_theta_direction(-1)
    ax1.set_theta_zero_location('N')
    ax1.plot(theta, error_list, '--', linewidth=2.5, marker='o',color="black")
    plt.title(name1, fontsize=24, y=1.1)
    #
    plt.subplots_adjust(left=0.0, bottom=0.1, right=1.0, top=0.84)     # wide
    # plt.subplots_adjust(left=0.0, bottom=0.08, right=1.0, top=0.85)     # full
    #
    # if a==6:
    #     plt.savefig('D:\\python\\kong\\FDD\\improvement\\6show\\ '+str(name1)+'_'+str(b)+'.png')
    if a==8:
        plt.savefig('D:\\python\\kong\\FDD\\improvement\\8show\\ '+str(name1)+'_'+str(b)+'.png')
    # if a==10:
    #     plt.savefig('D:\\python\\kong\\FDD\\improvement\\10show\\ '+str(name1)+'_'+str(b)+'.png')

    # plt.savefig('result\\pics\\projection_' + str(name) + '_wide.png')
    # plt.savefig('C:\\Users\\dong\\Desktop\\improvement\\Polar\\fig_'+str(name1)+'_'+str(name)+'_'+str(feature)+'.png')
    #
    # plt.show()
    plt.close(1)

def feature_select():
    filename = r'data_set_5000train\\' + name + '_train.csv'
    # filename = "data_set_5000train\\Level1_train.csv"
    dataFame = read_csv(filename)
    column_headers = list(dataFame.columns.values)
    select = dataFame.values
    train1 = select[0:4000, :]
    test1 = select[4000:5000, :]
    train2 = select[5000:9000, :]
    test2 = select[9000:10000, :]
    train3 = select[10000:14000, :]
    test3 = select[14000:15000, :]
    train4 = select[15000:19000, :]
    test4 = select[19000:20000, :]
    train5 = select[20000:24000, :]
    test5 = select[24000:25000, :]
    train6 = select[25000:29000, :]
    test6 = select[29000:30000, :]
    train7 = select[30000:34000, :]
    test7 = select[34000:35000, :]
    train8 = select[35000:39000, :]
    test8 = select[39000:40000, :]

    train = np.concatenate((train1, train2, train3, train4, train5, train6, train7, train8,), axis=0)
    test = np.concatenate((test1, test2, test3, test4, test5, test6, test7, test8,), axis=0)
    trainx = train[:,1:]
    testx = test[:,1:]
    trainx = minmaxscaler.fit_transform(trainx)
    testx = minmaxscaler.fit_transform(testx)
    trainy = train[:,0]
    testy = test[:,0]
    trainy = trainy.reshape(trainy.shape[0],1)
    aa = SelectKBest(chi2, k=8)
    X_new =aa.fit_transform(trainx, trainy)
    # X_new =SelectKBest(lambda X, Y: array(map(lambda x: pearsonr(x, Y), X.T)).T, k=2).fit_transform(trainx, trainy)
    # print(trainx.shape)
    # print(X_new.shape)
    # print("aa.scores_:",aa.scores_)
    aa.scores_[53] = 0
    # print("11111111111",len(aa.scores_))#65
    # print("11111111111", type(aa.scores_))
    # print("aa.scores_:", aa.scores_)
    # print("aa.scores_:", sum(aa.scores_))
    for i in range(len(aa.scores_)):
        aa.scores_[i] = aa.scores_[i]/sum(aa.scores_);
    aa.scores_ = aa.scores_.tolist()
    print(aa.scores_)
    a = sorted(aa.scores_,reverse=True)
    print("分数从大到小排列为：",a)
    order=[]
    for i in range(len(aa.scores_)):
        order.append(aa.scores_.index(a[i])-1)
    print(order)
    features = []
    for i in range(len(order)):
        features.append(column_headers[order[i]+1])
    print("选择的特征是:",features)

    plt.bar(range(len(aa.scores_)), aa.scores_)
    plt.show()

    # trainx = train[:, [56, 62, 55, 57, 27, 58, 45, 52, 32, 60]]
    trainx = train[:, [6, 4, 48, 8, 18, 2, 28, 46, 30, 61, 5, 25, 32, 26, 9, 33]]
    #[6, 4, 48, 8, 18, 2, 28, 46, 30, 61, 5, 25, 32, 26, 9, 33]
    pcc = np.corrcoef(trainx.T) * 0.5 + 0.5
    print(pcc)
    labels_name = ['1', '2', '3', '4', '5', '6', '7', '8','9', '10', '11', '12', '13', '14', '15', '16']
    plot_confusion_matrix(pcc, labels_name, "混淆矩阵")
    # plt.savefig('/HAR_cm.png', format='png')
    plt.show()

    # indices = np.argsort(aa.scores_)[::-1]
    # k_best_features = list(trainx.tolist.columns.values[indices[0:8]])
    # print('k best features are: ', k_best_features)

    return None

def attention_3d_block2(inputs, single_attention_vector=True):
    # 如果上一层是LSTM，需要return_sequences=True
    # inputs.shape = (batch_size, time_steps, input_dim)
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    a = Permute((2, 1))(inputs)
    a = tf.keras.layers.Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)#RepeatVector不改变我们的步长，改变我们的每一步的维数（即：属性长度）
    a_probs = Permute((2, 1))(a)
    # 乘上了attention权重，但是并没有求和，好像影响不大
    # 如果分类任务，进行Flatten展开就可以了
    # element-wise
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

def load_ML1():
    np.random.seed(2)
    lev1_lab0 = pd.read_csv(r'data_set0\\level1\\lev1_lab0.csv')
    lev1_lab1 = pd.read_csv(r'data_set0\\level1\\lev1_lab1.csv')
    lev1_lab2 = pd.read_csv(r'data_set0\\level1\\lev1_lab2.csv')
    lev1_lab3 = pd.read_csv(r'data_set0\\level1\\lev1_lab3.csv')
    lev1_lab4 = pd.read_csv(r'data_set0\\level1\\lev1_lab4.csv')
    lev1_lab5 = pd.read_csv(r'data_set0\\level1\\lev1_lab5.csv')
    lev1_lab6 = pd.read_csv(r'data_set0\\level1\\lev1_lab6.csv')
    lev1_lab7 = pd.read_csv(r'data_set0\\level1\\lev1_lab7.csv')
    # print(type(lev1_lab0))#<class 'pandas.core.frame.DataFrame'>
    lev1_lab0 = lev1_lab0.sample(n=5000)
    print(type(lev1_lab0))#<class 'pandas.core.frame.DataFrame'>
    lev1_lab1 = lev1_lab1.sample(n=5000)
    lev1_lab2 = lev1_lab2.sample(n=5000)
    lev1_lab3 = lev1_lab3.sample(n=5000)
    lev1_lab4 = lev1_lab4.sample(n=5000)
    lev1_lab5 = lev1_lab5.sample(n=5000)
    lev1_lab6 = lev1_lab6.sample(n=5000)
    lev1_lab7 = lev1_lab7.sample(n=5000)

    lev1_lab0_train = lev1_lab0.iloc[0:3000, :]
    lev1_lab0_val = lev1_lab0.iloc[3000:4000, :]
    lev1_lab0_test = lev1_lab0.iloc[4000:5000, :]

    lev1_lab1_train = lev1_lab1.iloc[0:3000, :]
    lev1_lab1_val = lev1_lab1.iloc[3000:4000, :]
    lev1_lab1_test = lev1_lab1.iloc[4000:5000, :]

    lev1_lab2_train = lev1_lab2.iloc[0:3000, :]
    lev1_lab2_val = lev1_lab2.iloc[3000:4000, :]
    lev1_lab2_test = lev1_lab2.iloc[4000:5000, :]

    lev1_lab3_train = lev1_lab3.iloc[0:3000, :]
    lev1_lab3_val = lev1_lab3.iloc[3000:4000, :]
    lev1_lab3_test = lev1_lab3.iloc[4000:5000, :]

    lev1_lab4_train = lev1_lab4.iloc[0:3000, :]
    lev1_lab4_val = lev1_lab4.iloc[3000:4000, :]
    lev1_lab4_test = lev1_lab4.iloc[4000:5000, :]

    lev1_lab5_train = lev1_lab5.iloc[0:3000, :]
    lev1_lab5_val = lev1_lab5.iloc[3000:4000, :]
    lev1_lab5_test = lev1_lab5.iloc[4000:5000, :]

    lev1_lab6_train = lev1_lab6.iloc[0:3000, :]
    lev1_lab6_val = lev1_lab6.iloc[3000:4000, :]
    lev1_lab6_test = lev1_lab6.iloc[4000:5000, :]

    lev1_lab7_train = lev1_lab7.iloc[0:3000, :]
    lev1_lab7_val = lev1_lab7.iloc[3000:4000, :]
    lev1_lab7_test = lev1_lab7.iloc[4000:5000, :]

    # "#合并训练集测试集\n",
    lev1_train = np.concatenate([lev1_lab0_train, lev1_lab1_train, lev1_lab2_train, lev1_lab3_train,
                            lev1_lab4_train, lev1_lab5_train, lev1_lab6_train, lev1_lab7_train], axis=0)

    lev1_val = np.concatenate([lev1_lab0_val, lev1_lab1_val, lev1_lab2_val, lev1_lab3_val,
                            lev1_lab4_val, lev1_lab5_val, lev1_lab6_val, lev1_lab7_val], axis=0)

    lev1_test = np.concatenate([lev1_lab0_test, lev1_lab1_test, lev1_lab2_test, lev1_lab3_test,
                           lev1_lab4_test, lev1_lab5_test, lev1_lab6_test, lev1_lab7_test], axis=0)

    # 故障等级2中，读取原始数据集\n",
    lev2_lab0 = pd.read_csv(r'data_set0\\level2\\lev2_lab0.csv')
    lev2_lab1 = pd.read_csv(r'data_set0\\level2\\lev2_lab1.csv')
    lev2_lab2 = pd.read_csv(r'data_set0\\level2\\lev2_lab2.csv')
    lev2_lab3 = pd.read_csv(r'data_set0\\level2\\lev2_lab3.csv')
    lev2_lab4 = pd.read_csv(r'data_set0\\level2\\lev2_lab4.csv')
    lev2_lab5 = pd.read_csv(r'data_set0\\level2\\lev2_lab5.csv')
    lev2_lab6 = pd.read_csv(r'data_set0\\level2\\lev2_lab6.csv')
    lev2_lab7 = pd.read_csv(r'data_set0\\level2\\lev2_lab7.csv')

    lev2_lab0 = lev2_lab0.sample(n=5000)
    lev2_lab1 = lev2_lab1.sample(n=5000)
    lev2_lab2 = lev2_lab2.sample(n=5000)
    lev2_lab3 = lev2_lab3.sample(n=5000)
    lev2_lab4 = lev2_lab4.sample(n=5000)
    lev2_lab5 = lev2_lab5.sample(n=5000)
    lev2_lab6 = lev2_lab6.sample(n=5000)
    lev2_lab7 = lev2_lab7.sample(n=5000)

    # 将5000个数据划分为训练集、验证集、测试集\n",
    lev2_lab0_train = lev2_lab0.iloc[0:3000, :]
    lev2_lab0_val = lev2_lab0.iloc[3000:4000, :]
    lev2_lab0_test = lev2_lab0.iloc[4000:5000, :]

    lev2_lab1_train = lev2_lab1.iloc[0:3000, :]
    lev2_lab1_val = lev2_lab1.iloc[3000:4000, :]
    lev2_lab1_test = lev2_lab1.iloc[4000:5000, :]

    lev2_lab2_train = lev2_lab2.iloc[0:3000, :]
    lev2_lab2_val = lev2_lab2.iloc[3000:4000, :]
    lev2_lab2_test = lev2_lab2.iloc[4000:5000, :]

    lev2_lab3_train = lev2_lab3.iloc[0:3000, :]
    lev2_lab3_val = lev2_lab3.iloc[3000:4000, :]
    lev2_lab3_test = lev2_lab3.iloc[4000:5000, :]

    lev2_lab4_train = lev2_lab4.iloc[0:3000, :]
    lev2_lab4_val = lev2_lab4.iloc[3000:4000, :]
    lev2_lab4_test = lev2_lab4.iloc[4000:5000, :]

    lev2_lab5_train = lev2_lab5.iloc[0:3000, :]
    lev2_lab5_val = lev2_lab5.iloc[3000:4000, :]
    lev2_lab5_test = lev2_lab5.iloc[4000:5000, :]

    lev2_lab6_train = lev2_lab6.iloc[0:3000, :]
    lev2_lab6_val = lev2_lab6.iloc[3000:4000, :]
    lev2_lab6_test = lev2_lab6.iloc[4000:5000, :]

    lev2_lab7_train = lev2_lab7.iloc[0:3000, :]
    lev2_lab7_val = lev2_lab7.iloc[3000:4000, :]
    lev2_lab7_test = lev2_lab7.iloc[4000:5000, :]

    # 合并训练集测试集\n",
    lev2_train = np.concatenate([lev2_lab0_train, lev2_lab1_train, lev2_lab2_train, lev2_lab3_train,
                            lev2_lab4_train, lev2_lab5_train, lev2_lab6_train, lev2_lab7_train], axis=0)

    lev2_val = np.concatenate([lev2_lab0_val, lev2_lab1_val, lev2_lab2_val, lev2_lab3_val,
                            lev2_lab4_val, lev2_lab5_val, lev2_lab6_val, lev2_lab7_val], axis=0)

    lev2_test = np.concatenate([lev2_lab0_test, lev2_lab1_test, lev2_lab2_test, lev2_lab3_test,
                           lev2_lab4_test, lev2_lab5_test, lev2_lab6_test, lev2_lab7_test], axis=0)

    # 故障等级3中，读取原始数据集\n",
    lev3_lab0 = pd.read_csv(r'data_set0\\level3\\lev3_lab0.csv')
    lev3_lab1 = pd.read_csv(r'data_set0\\level3\\lev3_lab1.csv')
    lev3_lab2 = pd.read_csv(r'data_set0\\level3\\lev3_lab2.csv')
    lev3_lab3 = pd.read_csv(r'data_set0\\level3\\lev3_lab3.csv')
    lev3_lab4 = pd.read_csv(r'data_set0\\level3\\lev3_lab4.csv')
    lev3_lab5 = pd.read_csv(r'data_set0\\level3\\lev3_lab5.csv')
    lev3_lab6 = pd.read_csv(r'data_set0\\level3\\lev3_lab6.csv')
    lev3_lab7 = pd.read_csv(r'data_set0\\level3\\lev3_lab7.csv')

    lev3_lab0 = lev3_lab0.sample(n=5000)
    lev3_lab1 = lev3_lab1.sample(n=5000)
    lev3_lab2 = lev3_lab2.sample(n=5000)
    lev3_lab3 = lev3_lab3.sample(n=5000)
    lev3_lab4 = lev3_lab4.sample(n=5000)
    lev3_lab5 = lev3_lab5.sample(n=5000)
    lev3_lab6 = lev3_lab6.sample(n=5000)
    lev3_lab7 = lev3_lab7.sample(n=5000)

    lev3_lab0_train = lev3_lab0.iloc[0:3000, :]
    lev3_lab0_val = lev3_lab0.iloc[3000:4000, :]
    lev3_lab0_test = lev3_lab0.iloc[4000:5000, :]

    lev3_lab1_train = lev3_lab1.iloc[0:3000, :]
    lev3_lab1_val = lev3_lab1.iloc[3000:4000, :]
    lev3_lab1_test = lev3_lab1.iloc[4000:5000, :]

    lev3_lab2_train = lev3_lab2.iloc[0:3000, :]
    lev3_lab2_val = lev3_lab2.iloc[3000:4000, :]
    lev3_lab2_test = lev3_lab2.iloc[4000:5000, :]

    lev3_lab3_train = lev3_lab3.iloc[0:3000, :]
    lev3_lab3_val = lev3_lab3.iloc[3000:4000, :]
    lev3_lab3_test = lev3_lab3.iloc[4000:5000, :]

    lev3_lab4_train = lev3_lab4.iloc[0:3000, :]
    lev3_lab4_val = lev3_lab4.iloc[3000:4000, :]
    lev3_lab4_test = lev3_lab4.iloc[4000:5000, :]

    lev3_lab5_train = lev3_lab5.iloc[0:3000, :]
    lev3_lab5_val = lev3_lab5.iloc[3000:4000, :]
    lev3_lab5_test = lev3_lab5.iloc[4000:5000, :]

    lev3_lab6_train = lev3_lab6.iloc[0:3000, :]
    lev3_lab6_val = lev3_lab6.iloc[3000:4000, :]
    lev3_lab6_test = lev3_lab6.iloc[4000:5000, :]

    lev3_lab7_train = lev3_lab7.iloc[0:3000, :]
    lev3_lab7_val = lev3_lab7.iloc[3000:4000, :]
    lev3_lab7_test = lev3_lab7.iloc[4000:5000, :]
    # 合并训练集测试集\n",
    lev3_train = np.concatenate([lev3_lab0_train, lev3_lab1_train, lev3_lab2_train, lev3_lab3_train,
                            lev3_lab4_train, lev3_lab5_train, lev3_lab6_train, lev3_lab7_train], axis=0)

    lev3_val = np.concatenate([lev3_lab0_val, lev3_lab1_val, lev3_lab2_val, lev3_lab3_val,
                            lev3_lab4_val, lev3_lab5_val, lev3_lab6_val, lev3_lab7_val], axis=0)

    lev3_test = np.concatenate([lev3_lab0_test, lev3_lab1_test, lev3_lab2_test, lev3_lab3_test,
                           lev3_lab4_test, lev3_lab5_test, lev3_lab6_test, lev3_lab7_test], axis=0)

    # "#故障等级4中，读取原始数据集\n",
    lev4_lab0 = pd.read_csv(r'data_set0\\level4\\lev4_lab0.csv')
    lev4_lab1 = pd.read_csv(r'data_set0\\level4\\lev4_lab1.csv')
    lev4_lab2 = pd.read_csv(r'data_set0\\level4\\lev4_lab2.csv')
    lev4_lab3 = pd.read_csv(r'data_set0\\level4\\lev4_lab3.csv')
    lev4_lab4 = pd.read_csv(r'data_set0\\level4\\lev4_lab4.csv')
    lev4_lab5 = pd.read_csv(r'data_set0\\level4\\lev4_lab5.csv')
    lev4_lab6 = pd.read_csv(r'data_set0\\level4\\lev4_lab6.csv')
    lev4_lab7 = pd.read_csv(r'data_set0\\level4\\lev4_lab7.csv')

    lev4_lab0 = lev4_lab0.sample(n=5000)
    lev4_lab1 = lev4_lab1.sample(n=5000)
    lev4_lab2 = lev4_lab2.sample(n=5000)
    lev4_lab3 = lev4_lab3.sample(n=5000)
    lev4_lab4 = lev4_lab4.sample(n=5000)
    lev4_lab5 = lev4_lab5.sample(n=5000)
    lev4_lab6 = lev4_lab6.sample(n=5000)
    lev4_lab7 = lev4_lab7.sample(n=5000)

    # 讲5000个数据划分为训练集、验证集、测试集\n",
    lev4_lab0_train = lev4_lab0.iloc[0:3000, :]
    lev4_lab0_val = lev4_lab0.iloc[3000:4000, :]
    lev4_lab0_test = lev4_lab0.iloc[4000:5000, :]

    lev4_lab1_train = lev4_lab1.iloc[0:3000, :]
    lev4_lab1_val = lev4_lab1.iloc[3000:4000, :]
    lev4_lab1_test = lev4_lab1.iloc[4000:5000, :]

    lev4_lab2_train = lev4_lab2.iloc[0:3000, :]
    lev4_lab2_val = lev4_lab2.iloc[3000:4000, :]
    lev4_lab2_test = lev4_lab2.iloc[4000:5000, :]

    lev4_lab3_train = lev4_lab3.iloc[0:3000, :]
    lev4_lab3_val = lev4_lab3.iloc[3000:4000, :]
    lev4_lab3_test = lev4_lab3.iloc[4000:5000, :]

    lev4_lab4_train = lev4_lab4.iloc[0:3000, :]
    lev4_lab4_val = lev4_lab4.iloc[3000:4000, :]
    lev4_lab4_test = lev4_lab4.iloc[4000:5000, :]

    lev4_lab5_train = lev4_lab5.iloc[0:3000, :]
    lev4_lab5_val = lev4_lab5.iloc[3000:4000, :]
    lev4_lab5_test = lev4_lab5.iloc[4000:5000, :]

    lev4_lab6_train = lev4_lab6.iloc[0:3000, :]
    lev4_lab6_val = lev4_lab6.iloc[3000:4000, :]
    lev4_lab6_test = lev4_lab6.iloc[4000:5000, :]

    lev4_lab7_train = lev4_lab7.iloc[0:3000, :]
    lev4_lab7_val = lev4_lab7.iloc[3000:4000, :]
    lev4_lab7_test = lev4_lab7.iloc[4000:5000, :]

    # 合并训练集测试集\n",
    lev4_train = np.concatenate([lev4_lab0_train, lev4_lab1_train, lev4_lab2_train, lev4_lab3_train,
                            lev4_lab4_train, lev4_lab5_train, lev4_lab6_train, lev4_lab7_train], axis=0)

    lev4_val = np.concatenate([lev4_lab0_val, lev4_lab1_val, lev4_lab2_val, lev4_lab3_val,
                            lev4_lab4_val, lev4_lab5_val, lev4_lab6_val, lev4_lab7_val], axis=0)

    lev4_test = np.concatenate([lev4_lab0_test, lev4_lab1_test, lev4_lab2_test, lev4_lab3_test,
                           lev4_lab4_test, lev4_lab5_test, lev4_lab6_test, lev4_lab7_test], axis=0)

    return lev1_train,lev1_val,lev1_test,lev2_train,lev2_val,lev2_test,lev3_train,lev3_val,lev3_test,lev4_train,lev4_val,lev4_test

def load_ML():
    a = 2000
    b = 3500
    m = 5000

    np.random.seed(2)
    lev1_lab0 = pd.read_csv(r'data_set0\\level1\\lev1_lab0.csv')
    lev1_lab1 = pd.read_csv(r'data_set0\\level1\\lev1_lab1.csv')
    lev1_lab2 = pd.read_csv(r'data_set0\\level1\\lev1_lab2.csv')
    lev1_lab3 = pd.read_csv(r'data_set0\\level1\\lev1_lab3.csv')
    lev1_lab4 = pd.read_csv(r'data_set0\\level1\\lev1_lab4.csv')
    lev1_lab5 = pd.read_csv(r'data_set0\\level1\\lev1_lab5.csv')
    lev1_lab6 = pd.read_csv(r'data_set0\\level1\\lev1_lab6.csv')
    lev1_lab7 = pd.read_csv(r'data_set0\\level1\\lev1_lab7.csv')
    # print(type(lev1_lab0))#<class 'pandas.core.frame.DataFrame'>
    # lev1_lab0 = lev1_lab0.sample(n=5000)
    # print(type(lev1_lab0))#<class 'pandas.core.frame.DataFrame'>
    # lev1_lab1 = lev1_lab1.sample(n=5000)
    # lev1_lab2 = lev1_lab2.sample(n=5000)
    # lev1_lab3 = lev1_lab3.sample(n=5000)
    # lev1_lab4 = lev1_lab4.sample(n=5000)
    # lev1_lab5 = lev1_lab5.sample(n=5000)
    # lev1_lab6 = lev1_lab6.sample(n=5000)
    # lev1_lab7 = lev1_lab7.sample(n=5000)

    lev1_lab0_train = lev1_lab0.iloc[0:a, :]
    # lev1_lab0_train = lev1_lab0_train.sample(n=a)
    lev1_lab0_val = lev1_lab0.iloc[a:b, :]
    # lev1_lab0_val = lev1_lab0_val.sample(n=1000)
    lev1_lab0_test = lev1_lab0.iloc[b:m, :]

    lev1_lab1_train = lev1_lab1.iloc[0:a, :]
    # lev1_lab1_train = lev1_lab1_train.sample(n=a)
    lev1_lab1_val = lev1_lab1.iloc[a:b, :]
    # lev1_lab1_val = lev1_lab1_val.sample(n=1000)
    lev1_lab1_test = lev1_lab1.iloc[b:m, :]

    lev1_lab2_train = lev1_lab2.iloc[0:a, :]
    # lev1_lab2_train = lev1_lab2_train.sample(n=a)
    lev1_lab2_val = lev1_lab2.iloc[a:b, :]
    # lev1_lab2_val = lev1_lab2_val.sample(n=1000)
    lev1_lab2_test = lev1_lab2.iloc[b:m, :]

    lev1_lab3_train = lev1_lab3.iloc[0:a, :]
    # lev1_lab3_train = lev1_lab3_train.sample(n=a)
    lev1_lab3_val = lev1_lab3.iloc[a:b, :]
    # lev1_lab3_val = lev1_lab3_val.sample(n=1000)
    lev1_lab3_test = lev1_lab3.iloc[b:m, :]

    lev1_lab4_train = lev1_lab4.iloc[0:a, :]
    # lev1_lab4_train = lev1_lab4_train.sample(n=a)
    lev1_lab4_val = lev1_lab4.iloc[a:b, :]
    # lev1_lab4_val = lev1_lab4_val.sample(n=1000)
    lev1_lab4_test = lev1_lab4.iloc[b:m, :]

    lev1_lab5_train = lev1_lab5.iloc[0:a, :]
    # lev1_lab5_train = lev1_lab5_train.sample(n=a)
    lev1_lab5_val = lev1_lab5.iloc[a:b, :]
    # lev1_lab5_val = lev1_lab5_val.sample(n=1000)
    lev1_lab5_test = lev1_lab5.iloc[b:m, :]

    lev1_lab6_train = lev1_lab6.iloc[0:a, :]
    # lev1_lab6_train = lev1_lab6_train.sample(n=a)
    lev1_lab6_val = lev1_lab6.iloc[a:b, :]
    # lev1_lab6_val = lev1_lab6_val.sample(n=1000)
    lev1_lab6_test = lev1_lab6.iloc[b:m, :]

    lev1_lab7_train = lev1_lab7.iloc[0:a, :]
    # lev1_lab7_train = lev1_lab7_train.sample(n=a)
    lev1_lab7_val = lev1_lab7.iloc[a:b, :]
    # lev1_lab7_val = lev1_lab7_val.sample(n=1000)
    lev1_lab7_test = lev1_lab7.iloc[b:m, :]

    # "#合并训练集测试集\n",
    lev1_train = np.concatenate([lev1_lab0_train, lev1_lab1_train, lev1_lab2_train, lev1_lab3_train,
                            lev1_lab4_train, lev1_lab5_train, lev1_lab6_train, lev1_lab7_train], axis=0)

    lev1_val = np.concatenate([lev1_lab0_val, lev1_lab1_val, lev1_lab2_val, lev1_lab3_val,
                            lev1_lab4_val, lev1_lab5_val, lev1_lab6_val, lev1_lab7_val], axis=0)

    lev1_test = np.concatenate([lev1_lab0_test, lev1_lab1_test, lev1_lab2_test, lev1_lab3_test,
                           lev1_lab4_test, lev1_lab5_test, lev1_lab6_test, lev1_lab7_test], axis=0)

    # 故障等级2中，读取原始数据集\n",
    lev2_lab0 = pd.read_csv(r'data_set0\\level2\\lev2_lab0.csv')
    lev2_lab1 = pd.read_csv(r'data_set0\\level2\\lev2_lab1.csv')
    lev2_lab2 = pd.read_csv(r'data_set0\\level2\\lev2_lab2.csv')
    lev2_lab3 = pd.read_csv(r'data_set0\\level2\\lev2_lab3.csv')
    lev2_lab4 = pd.read_csv(r'data_set0\\level2\\lev2_lab4.csv')
    lev2_lab5 = pd.read_csv(r'data_set0\\level2\\lev2_lab5.csv')
    lev2_lab6 = pd.read_csv(r'data_set0\\level2\\lev2_lab6.csv')
    lev2_lab7 = pd.read_csv(r'data_set0\\level2\\lev2_lab7.csv')

    # lev2_lab0 = lev2_lab0.sample(n=5000)
    # lev2_lab1 = lev2_lab1.sample(n=5000)
    # lev2_lab2 = lev2_lab2.sample(n=5000)
    # lev2_lab3 = lev2_lab3.sample(n=5000)
    # lev2_lab4 = lev2_lab4.sample(n=5000)
    # lev2_lab5 = lev2_lab5.sample(n=5000)
    # lev2_lab6 = lev2_lab6.sample(n=5000)
    # lev2_lab7 = lev2_lab7.sample(n=5000)

    # 将5000个数据划分为训练集、验证集、测试集\n",
    lev2_lab0_train = lev2_lab0.iloc[0:a, :]
    # lev2_lab0_train = lev2_lab0_train.sample(n=a)
    lev2_lab0_val = lev2_lab0.iloc[a:b, :]
    # lev2_lab0_val = lev2_lab0_val.sample(n=1000)
    lev2_lab0_test = lev2_lab0.iloc[b:m, :]

    lev2_lab1_train = lev2_lab1.iloc[0:a, :]
    # lev2_lab1_train = lev2_lab1_train.sample(n=a)
    lev2_lab1_val = lev2_lab1.iloc[a:b, :]
    # lev2_lab1_val = lev2_lab1_val.sample(n=1000)
    lev2_lab1_test = lev2_lab1.iloc[b:m, :]

    lev2_lab2_train = lev2_lab2.iloc[0:a, :]
    # lev2_lab2_train = lev2_lab2_train.sample(n=a)
    lev2_lab2_val = lev2_lab2.iloc[a:b, :]
    # lev2_lab2_val = lev2_lab2_val.sample(n=1000)
    lev2_lab2_test = lev2_lab2.iloc[b:m, :]

    lev2_lab3_train = lev2_lab3.iloc[0:a, :]
    # lev2_lab3_train = lev2_lab3_train.sample(n=a)
    lev2_lab3_val = lev2_lab3.iloc[a:b, :]
    # lev2_lab3_val = lev2_lab3_val.sample(n=1000)
    lev2_lab3_test = lev2_lab3.iloc[b:m, :]

    lev2_lab4_train = lev2_lab4.iloc[0:a, :]
    # lev2_lab4_train = lev2_lab4_train.sample(n=a)
    lev2_lab4_val = lev2_lab4.iloc[a:b, :]
    # lev2_lab4_val = lev2_lab4_val.sample(n=1000)
    lev2_lab4_test = lev2_lab4.iloc[b:m, :]

    lev2_lab5_train = lev2_lab5.iloc[0:a, :]
    # lev2_lab5_train = lev2_lab5_train.sample(n=a)
    lev2_lab5_val = lev2_lab5.iloc[a:b, :]
    # lev2_lab5_val = lev2_lab5_val.sample(n=1000)
    lev2_lab5_test = lev2_lab5.iloc[b:m, :]

    lev2_lab6_train = lev2_lab6.iloc[0:a, :]
    # lev2_lab6_train = lev2_lab6_train.sample(n=a)
    lev2_lab6_val = lev2_lab6.iloc[a:b, :]
    # lev2_lab6_val = lev2_lab6_val.sample(n=1000)
    lev2_lab6_test = lev2_lab6.iloc[b:m, :]

    lev2_lab7_train = lev2_lab7.iloc[0:a, :]
    # lev2_lab7_train = lev2_lab7_train.sample(n=a)
    lev2_lab7_val = lev2_lab7.iloc[a:b, :]
    # lev2_lab7_val = lev2_lab7_val.sample(n=1000)
    lev2_lab7_test = lev2_lab7.iloc[b:m, :]

    # 合并训练集测试集\n",
    lev2_train = np.concatenate([lev2_lab0_train, lev2_lab1_train, lev2_lab2_train, lev2_lab3_train,
                            lev2_lab4_train, lev2_lab5_train, lev2_lab6_train, lev2_lab7_train], axis=0)

    lev2_val = np.concatenate([lev2_lab0_val, lev2_lab1_val, lev2_lab2_val, lev2_lab3_val,
                            lev2_lab4_val, lev2_lab5_val, lev2_lab6_val, lev2_lab7_val], axis=0)

    lev2_test = np.concatenate([lev2_lab0_test, lev2_lab1_test, lev2_lab2_test, lev2_lab3_test,
                           lev2_lab4_test, lev2_lab5_test, lev2_lab6_test, lev2_lab7_test], axis=0)

    # 故障等级3中，读取原始数据集\n",
    lev3_lab0 = pd.read_csv(r'data_set0\\level3\\lev3_lab0.csv')
    lev3_lab1 = pd.read_csv(r'data_set0\\level3\\lev3_lab1.csv')
    lev3_lab2 = pd.read_csv(r'data_set0\\level3\\lev3_lab2.csv')
    lev3_lab3 = pd.read_csv(r'data_set0\\level3\\lev3_lab3.csv')
    lev3_lab4 = pd.read_csv(r'data_set0\\level3\\lev3_lab4.csv')
    lev3_lab5 = pd.read_csv(r'data_set0\\level3\\lev3_lab5.csv')
    lev3_lab6 = pd.read_csv(r'data_set0\\level3\\lev3_lab6.csv')
    lev3_lab7 = pd.read_csv(r'data_set0\\level3\\lev3_lab7.csv')

    # lev3_lab0 = lev3_lab0.sample(n=5000)
    # lev3_lab1 = lev3_lab1.sample(n=5000)
    # lev3_lab2 = lev3_lab2.sample(n=5000)
    # lev3_lab3 = lev3_lab3.sample(n=5000)
    # lev3_lab4 = lev3_lab4.sample(n=5000)
    # lev3_lab5 = lev3_lab5.sample(n=5000)
    # lev3_lab6 = lev3_lab6.sample(n=5000)
    # lev3_lab7 = lev3_lab7.sample(n=5000)

    lev3_lab0_train = lev3_lab0.iloc[0:a, :]
    # lev3_lab0_train = lev3_lab0_train.sample(n=a)
    lev3_lab0_val = lev3_lab0.iloc[a:b, :]
    # lev3_lab0_val = lev3_lab0_val.sample(n=1000)
    lev3_lab0_test = lev3_lab0.iloc[b:m, :]

    lev3_lab1_train = lev3_lab1.iloc[0:a, :]
    # lev3_lab1_train = lev3_lab1_train.sample(n=a)
    lev3_lab1_val = lev3_lab1.iloc[a:b, :]
    # lev3_lab1_val = lev3_lab1_val.sample(n=1000)
    lev3_lab1_test = lev3_lab1.iloc[b:m, :]

    lev3_lab2_train = lev3_lab2.iloc[0:a, :]
    # lev3_lab2_train = lev3_lab2_train.sample(n=a)
    lev3_lab2_val = lev3_lab2.iloc[a:b, :]
    # lev3_lab2_val = lev3_lab2_val.sample(n=1000)
    lev3_lab2_test = lev3_lab2.iloc[b:m, :]

    lev3_lab3_train = lev3_lab3.iloc[0:a, :]
    # lev3_lab3_train = lev3_lab3_train.sample(n=a)
    lev3_lab3_val = lev3_lab3.iloc[a:b, :]
    # lev3_lab3_val = lev3_lab3_val.sample(n=1000)
    lev3_lab3_test = lev3_lab3.iloc[b:m, :]

    lev3_lab4_train = lev3_lab4.iloc[0:a, :]
    # lev3_lab4_train = lev3_lab4_train.sample(n=a)
    lev3_lab4_val = lev3_lab4.iloc[a:b, :]
    # lev3_lab4_val = lev3_lab4_val.sample(n=1000)
    lev3_lab4_test = lev3_lab4.iloc[b:m, :]

    lev3_lab5_train = lev3_lab5.iloc[0:a, :]
    # lev3_lab5_train = lev3_lab5_train.sample(n=a)
    lev3_lab5_val = lev3_lab5.iloc[a:b, :]
    # lev3_lab5_val = lev3_lab5_val.sample(n=1000)
    lev3_lab5_test = lev3_lab5.iloc[b:m, :]

    lev3_lab6_train = lev3_lab6.iloc[0:a, :]
    # lev3_lab6_train = lev3_lab6_train.sample(n=a)
    lev3_lab6_val = lev3_lab6.iloc[a:b, :]
    # lev3_lab6_val = lev3_lab6_val.sample(n=1000)
    lev3_lab6_test = lev3_lab6.iloc[b:m, :]

    lev3_lab7_train = lev3_lab7.iloc[0:a, :]
    # lev3_lab7_train = lev3_lab7_train.sample(n=a)
    lev3_lab7_val = lev3_lab7.iloc[a:b, :]
    # lev3_lab7_val = lev3_lab7_val.sample(n=1000)
    lev3_lab7_test = lev3_lab7.iloc[b:m, :]
    # 合并训练集测试集\n",
    lev3_train = np.concatenate([lev3_lab0_train, lev3_lab1_train, lev3_lab2_train, lev3_lab3_train,
                            lev3_lab4_train, lev3_lab5_train, lev3_lab6_train, lev3_lab7_train], axis=0)

    lev3_val = np.concatenate([lev3_lab0_val, lev3_lab1_val, lev3_lab2_val, lev3_lab3_val,
                            lev3_lab4_val, lev3_lab5_val, lev3_lab6_val, lev3_lab7_val], axis=0)

    lev3_test = np.concatenate([lev3_lab0_test, lev3_lab1_test, lev3_lab2_test, lev3_lab3_test,
                           lev3_lab4_test, lev3_lab5_test, lev3_lab6_test, lev3_lab7_test], axis=0)

    # "#故障等级4中，读取原始数据集\n",
    lev4_lab0 = pd.read_csv(r'data_set0\\level4\\lev4_lab0.csv')
    lev4_lab1 = pd.read_csv(r'data_set0\\level4\\lev4_lab1.csv')
    lev4_lab2 = pd.read_csv(r'data_set0\\level4\\lev4_lab2.csv')
    lev4_lab3 = pd.read_csv(r'data_set0\\level4\\lev4_lab3.csv')
    lev4_lab4 = pd.read_csv(r'data_set0\\level4\\lev4_lab4.csv')
    lev4_lab5 = pd.read_csv(r'data_set0\\level4\\lev4_lab5.csv')
    lev4_lab6 = pd.read_csv(r'data_set0\\level4\\lev4_lab6.csv')
    lev4_lab7 = pd.read_csv(r'data_set0\\level4\\lev4_lab7.csv')

    lev4_lab0 = lev4_lab0.sample(n=5000)
    lev4_lab1 = lev4_lab1.sample(n=5000)
    lev4_lab2 = lev4_lab2.sample(n=5000)
    lev4_lab3 = lev4_lab3.sample(n=5000)
    lev4_lab4 = lev4_lab4.sample(n=5000)
    lev4_lab5 = lev4_lab5.sample(n=5000)
    lev4_lab6 = lev4_lab6.sample(n=5000)
    lev4_lab7 = lev4_lab7.sample(n=5000)

    # 讲5000个数据划分为训练集、验证集、测试集\n",
    lev4_lab0_train = lev4_lab0.iloc[0:3000, :]
    lev4_lab0_val = lev4_lab0.iloc[3000:4000, :]
    lev4_lab0_test = lev4_lab0.iloc[4000:5000, :]

    lev4_lab1_train = lev4_lab1.iloc[0:3000, :]
    lev4_lab1_val = lev4_lab1.iloc[3000:4000, :]
    lev4_lab1_test = lev4_lab1.iloc[4000:5000, :]

    lev4_lab2_train = lev4_lab2.iloc[0:3000, :]
    lev4_lab2_val = lev4_lab2.iloc[3000:4000, :]
    lev4_lab2_test = lev4_lab2.iloc[4000:5000, :]

    lev4_lab3_train = lev4_lab3.iloc[0:3000, :]
    lev4_lab3_val = lev4_lab3.iloc[3000:4000, :]
    lev4_lab3_test = lev4_lab3.iloc[4000:5000, :]

    lev4_lab4_train = lev4_lab4.iloc[0:3000, :]
    lev4_lab4_val = lev4_lab4.iloc[3000:4000, :]
    lev4_lab4_test = lev4_lab4.iloc[4000:5000, :]

    lev4_lab5_train = lev4_lab5.iloc[0:3000, :]
    lev4_lab5_val = lev4_lab5.iloc[3000:4000, :]
    lev4_lab5_test = lev4_lab5.iloc[4000:5000, :]

    lev4_lab6_train = lev4_lab6.iloc[0:3000, :]
    lev4_lab6_val = lev4_lab6.iloc[3000:4000, :]
    lev4_lab6_test = lev4_lab6.iloc[4000:5000, :]

    lev4_lab7_train = lev4_lab7.iloc[0:3000, :]
    lev4_lab7_val = lev4_lab7.iloc[3000:4000, :]
    lev4_lab7_test = lev4_lab7.iloc[4000:5000, :]

    # 合并训练集测试集\n",
    lev4_train = np.concatenate([lev4_lab0_train, lev4_lab1_train, lev4_lab2_train, lev4_lab3_train,
                            lev4_lab4_train, lev4_lab5_train, lev4_lab6_train, lev4_lab7_train], axis=0)

    lev4_val = np.concatenate([lev4_lab0_val, lev4_lab1_val, lev4_lab2_val, lev4_lab3_val,
                            lev4_lab4_val, lev4_lab5_val, lev4_lab6_val, lev4_lab7_val], axis=0)

    lev4_test = np.concatenate([lev4_lab0_test, lev4_lab1_test, lev4_lab2_test, lev4_lab3_test,
                           lev4_lab4_test, lev4_lab5_test, lev4_lab6_test, lev4_lab7_test], axis=0)

    return lev1_train,lev1_val,lev1_test,lev2_train,lev2_val,lev2_test,lev3_train,lev3_val,lev3_test,lev4_train,lev4_val,lev4_test


def load_data_det_6(train_data,val_data,test_data):

    # train_X = train_data[['FWC', 'FWE', 'TCA', 'Tolerance%', 'PO_feed', 'PO_net', 'TWCD', 'VE']]
    # test_X = test_data[['FWC', 'FWE', 'TCA', 'Tolerance%', 'PO_feed', 'PO_net', 'TWCD', 'VE']]
    # train_X = train_data[['TCI', 'TEO', 'PO_feed', 'TCO','Evap Tons', 'TEI', 'TCA', 'TO_feed']]
    # test_X = test_data[['TCI', 'TEO', 'PO_feed', 'TCO','Evap Tons', 'TEI', 'TCA', 'TO_feed']]

    #[6, 4, 48, 8, 18, 2, 28, 46, 30, 61, 5, 25, 32, 26, 9, 33]
    train_X = train_data[:, [6, 4, 48, 8, 18, 2]]
    val_X = val_data[:, [6, 4, 48, 8, 18, 2]]
    test_X = test_data[:, [ 6, 4,48, 8, 18, 2]]
    # train_X = train_data[:, [6, 4, 48, 8, 18, 2, 28, 46]]
    # val_X = val_data[:, [6, 4, 48, 8, 18, 2, 28, 46]]
    # test_X = test_data[:, [ 6, 4,48, 8, 18, 2, 28, 46]]
    # train_X = train_data[:, [6, 4,  48, 8, 18, 2, 28, 46, 30, 61]]
    # val_X = val_data[:, [6, 4,  48, 8, 18, 2, 28, 46, 30, 61]]
    # test_X = test_data[:,[6, 4,  48, 8, 18, 2, 28, 46, 30, 61]]
    train_X = train_X

    test_X = test_X

    train_X = train_X.reshape(train_X.shape[0],train_X.shape[1],1)
    val_X = val_X.reshape(val_X.shape[0],val_X.shape[1],1)
    test_X = test_X.reshape(test_X.shape[0],test_X.shape[1],1)
    train_Y = train_data[:,0]
    val_Y = val_data[:,0]
    test_Y = test_data[:,0]

    train_Y = np.array(train_Y)
    val_Y = np.array(val_Y)
    test_Y = np.array(test_Y)
    train_Y = train_Y.reshape(-1,1)
    val_Y = val_Y.reshape(-1,1)
    test_Y = test_Y.reshape(-1,1)
    return train_X,val_X,test_X,train_Y,val_Y,test_Y

def load_data_det_8(train_data,val_data,test_data):

    # train_X = train_data[['FWC', 'FWE', 'TCA', 'Tolerance%', 'PO_feed', 'PO_net', 'TWCD', 'VE']]
    # test_X = test_data[['FWC', 'FWE', 'TCA', 'Tolerance%', 'PO_feed', 'PO_net', 'TWCD', 'VE']]
    # train_X = train_data[['TCI', 'TEO', 'PO_feed', 'TCO','Evap Tons', 'TEI', 'TCA', 'TO_feed']]
    # test_X = test_data[['TCI', 'TEO', 'PO_feed', 'TCO','Evap Tons', 'TEI', 'TCA', 'TO_feed']]

    #[6, 4, 48, 8, 18, 2, 28, 46, 30, 61, 5, 25, 32, 26, 9, 33]
    # [55, 61, 54, 56, 26, 57, 44, 51]
    # [26,44,51,54,55,56,57,61]
    # train_X = train_data[:, [6, 4, 48, 8, 18, 2, 28, 46]]
    # val_X = val_data[:, [6, 4, 48, 8, 18, 2, 28, 46]]
    # test_X = test_data[:, [ 6, 4,48, 8, 18, 2, 28, 46]]
    # train_X = train_data[:, [2, 4, 6, 8, 18, 28, 46, 48]]
    # val_X = val_data[:, [2, 4, 6, 8, 18, 28, 46, 48]]
    # test_X = test_data[:, [2, 4, 6, 8, 18, 28, 46, 48]]
    train_X = train_data[:, [25,26,28,43,48,49,50,57]]
    val_X = val_data[:, [25,26,28,43,48,49,50,57]]
    test_X = test_data[:, [25,26,28,43,48,49,50,57]]
    # train_X = train_data[:, [26,44,51,54,55,56,57,61]]
    # val_X = val_data[:, [26,44,51,54,55,56,57,61]]
    # test_X = test_data[:, [26,44,51,54,55,56,57,61]]
    # train_X = train_data[:, [3,7,24,30,35,47,56,58]]
    # val_X = val_data[:, [3,7,24,30,35,47,56,58]]
    # test_X = test_data[:, [3,7,24,30,35,47,56,58]]

    # train_X = train_data[:, [6, 4,  48, 8, 18, 2, 28, 46, 30, 61]]
    # val_X = val_data[:, [6, 4,  48, 8, 18, 2, 28, 46, 30, 61]]
    # test_X = test_data[:,[6, 4,  48, 8, 18, 2, 28, 46, 30, 61]]
    train_X = train_X

    test_X = test_X

    train_X = train_X.reshape(train_X.shape[0],train_X.shape[1],1)
    val_X = val_X.reshape(val_X.shape[0],val_X.shape[1],1)
    test_X = test_X.reshape(test_X.shape[0],test_X.shape[1],1)
    train_Y = train_data[:,0]
    val_Y = val_data[:,0]
    test_Y = test_data[:,0]

    train_Y = np.array(train_Y)
    val_Y = np.array(val_Y)
    test_Y = np.array(test_Y)
    train_Y = train_Y.reshape(-1,1)
    val_Y = val_Y.reshape(-1,1)
    test_Y = test_Y.reshape(-1,1)
    return train_X,val_X,test_X,train_Y,val_Y,test_Y

def load_data_det_10(train_data,val_data,test_data):

    # train_X = train_data[['FWC', 'FWE', 'TCA', 'Tolerance%', 'PO_feed', 'PO_net', 'TWCD', 'VE']]
    # test_X = test_data[['FWC', 'FWE', 'TCA', 'Tolerance%', 'PO_feed', 'PO_net', 'TWCD', 'VE']]
    # train_X = train_data[['TCI', 'TEO', 'PO_feed', 'TCO','Evap Tons', 'TEI', 'TCA', 'TO_feed']]
    # test_X = test_data[['TCI', 'TEO', 'PO_feed', 'TCO','Evap Tons', 'TEI', 'TCA', 'TO_feed']]

    #[6, 4, 48, 8, 18, 2, 28, 46, 30, 61, 5, 25, 32, 26, 9, 33]
    # train_X = train_data[:, [6, 4, 48, 8, 18, 2]]
    # val_X = val_data[:, [6, 4, 48, 8, 18, 2]]
    # test_X = test_data[:, [ 6, 4,48, 8, 18, 2]]
    # train_X = train_data[:, [6, 4, 48, 8, 18, 2, 28, 46]]
    # val_X = val_data[:, [6, 4, 48, 8, 18, 2, 28, 46]]
    # test_X = test_data[:, [ 6, 4,48, 8, 18, 2, 28, 46]]
    train_X = train_data[:, [6, 4,  48, 8, 18, 2, 28, 46, 30, 61]]
    val_X = val_data[:, [6, 4,  48, 8, 18, 2, 28, 46, 30, 61]]
    test_X = test_data[:,[6, 4,  48, 8, 18, 2, 28, 46, 30, 61]]
    train_X = train_X

    test_X = test_X

    train_X = train_X.reshape(train_X.shape[0],train_X.shape[1],1)
    val_X = val_X.reshape(val_X.shape[0],val_X.shape[1],1)
    test_X = test_X.reshape(test_X.shape[0],test_X.shape[1],1)
    train_Y = train_data[:,0]
    val_Y = val_data[:,0]
    test_Y = test_data[:,0]

    train_Y = np.array(train_Y)
    val_Y = np.array(val_Y)
    test_Y = np.array(test_Y)
    train_Y = train_Y.reshape(-1,1)
    val_Y = val_Y.reshape(-1,1)
    test_Y = test_Y.reshape(-1,1)
    return train_X,val_X,test_X,train_Y,val_Y,test_Y

def load_data_det_12(train_data,val_data,test_data):

    # train_X = train_data[['FWC', 'FWE', 'TCA', 'Tolerance%', 'PO_feed', 'PO_net', 'TWCD', 'VE']]
    # test_X = test_data[['FWC', 'FWE', 'TCA', 'Tolerance%', 'PO_feed', 'PO_net', 'TWCD', 'VE']]
    # train_X = train_data[['TCI', 'TEO', 'PO_feed', 'TCO','Evap Tons', 'TEI', 'TCA', 'TO_feed']]
    # test_X = test_data[['TCI', 'TEO', 'PO_feed', 'TCO','Evap Tons', 'TEI', 'TCA', 'TO_feed']]

    #[6, 4, 48, 8, 18, 2, 28, 46, 30, 61, 5, 25, 32, 26, 9, 33]
    # train_X = train_data[:, [6, 4, 48, 8, 18, 2]]
    # val_X = val_data[:, [6, 4, 48, 8, 18, 2]]
    # test_X = test_data[:, [ 6, 4,48, 8, 18, 2]]
    # train_X = train_data[:, [6, 4, 48, 8, 18, 2, 28, 46]]
    # val_X = val_data[:, [6, 4, 48, 8, 18, 2, 28, 46]]
    # test_X = test_data[:, [ 6, 4,48, 8, 18, 2, 28, 46]]
    # train_X = train_data[:, [6, 4,  48, 8, 18, 2, 28, 46, 30, 61]]
    # val_X = val_data[:, [6, 4,  48, 8, 18, 2, 28, 46, 30, 61]]
    # test_X = test_data[:,[6, 4,  48, 8, 18, 2, 28, 46, 30, 61]]
    train_X = train_data[:, [6, 4, 48, 8, 18, 2, 28, 46, 30, 61, 5, 25]]
    val_X = val_data[:, [6, 4, 48, 8, 18, 2, 28, 46, 30, 61, 5, 25]]
    test_X = test_data[:,[6, 4, 48, 8, 18, 2, 28, 46, 30, 61, 5, 25]]
    train_X = train_X

    test_X = test_X

    train_X = train_X.reshape(train_X.shape[0],train_X.shape[1],1)
    val_X = val_X.reshape(val_X.shape[0],val_X.shape[1],1)
    test_X = test_X.reshape(test_X.shape[0],test_X.shape[1],1)
    train_Y = train_data[:,0]
    val_Y = val_data[:,0]
    test_Y = test_data[:,0]

    train_Y = np.array(train_Y)
    val_Y = np.array(val_Y)
    test_Y = np.array(test_Y)
    train_Y = train_Y.reshape(-1,1)
    val_Y = val_Y.reshape(-1,1)
    test_Y = test_Y.reshape(-1,1)
    return train_X,val_X,test_X,train_Y,val_Y,test_Y

def ML_61():
    # start_time = time.time()#参数调优要花费1小时之久~~
    print("ML_61 starting")
    lev1_train,lev1_val,lev1_test,lev2_train,lev2_val,lev2_test,lev3_train,lev3_val,lev3_test,lev4_train,lev4_val,lev4_test = load_ML()
    train_X_det1, val_X_det1,test_X_det1,train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_6(lev1_train,lev1_val,lev1_test)
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    train_X_det1 = train_X_det1.reshape(train_X_det1.shape[0],train_X_det1.shape[1])#变成2维
    test_X_det1 = test_X_det1.reshape(test_X_det1.shape[0],test_X_det1.shape[1])

    classifier = LogisticRegression(random_state=5,solver='sag',C=10,)
    classifier.fit(train_X_det1, train_Y_det1.ravel())
    lg_predict = classifier.predict(test_X_det1)
    np.set_printoptions(threshold=100000000)
    # print("lg_predict:", lg_predict)
    LG_AC = accuracy_score(test_Y_det1, lg_predict)
    LG_f1 = f1_score(test_Y_det1, lg_predict, average='macro')
    print("LR_AC_61:", LG_AC)


    dtc = DecisionTreeClassifier(criterion="gini",max_features=6,min_samples_split=6)
    dtc.fit(train_X_det1, train_Y_det1.ravel())
    dt_pre = dtc.predict(test_X_det1)
    np.set_printoptions(threshold=100000000)
    # print("dt_pre:", dt_pre)
    DT_AC = accuracy_score(test_Y_det1, dt_pre)
    DT_f1 = f1_score(test_Y_det1, dt_pre, average='macro')
    print("DT_AC_61:", DT_AC)

    knn = KNeighborsClassifier(weights='distance',p=2,leaf_size=50)
    knn.fit(train_X_det1, train_Y_det1.ravel())
    knn_predict = knn.predict(test_X_det1)
    np.set_printoptions(threshold=100000000)
    # print("knn_predict:", knn_predict)
    KNN_AC = accuracy_score(test_Y_det1, knn_predict)
    KNN_f1 = f1_score(test_Y_det1, knn_predict, average='macro')
    print("KNN_AC_61:", KNN_AC)
    _ = projection(test_Y_det1, knn_predict,"KNN",6,61)
    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': 10,
        'gamma': 0.1,
        'max_depth': 10,
        'lambda': 2,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 1,
        'silent': 1,
        'eta': 0.3,
        'seed': 1000,
        'nthread': 4,
    }

    dtrain = xgb.DMatrix(train_X_det1, train_Y_det1.ravel())
    num_rounds = 20
    plst = list(params.items())
    xgb1 = xgb.train(plst, dtrain, num_rounds)
    dtest = xgb.DMatrix(test_X_det1)
    xgb1_p_temp = xgb1.predict(dtest)
    xgb1_p = xgb1_p_temp.reshape(-1, 1)
    xgb1_acc = accuracy_score(xgb1_p, test_Y_det1)
    print('xgb1_acc_61=', xgb1_acc)
    _ = projection(test_Y_det1, xgb1_p,"XGboost",6,61)
def ML_62():
    # start_time = time.time()#参数调优要花费1小时之久~~
    print("ML_62 starting")
    lev1_train,lev1_val,lev1_test,lev2_train,lev2_val,lev2_test,lev3_train,lev3_val,lev3_test,lev4_train,lev4_val,lev4_test = load_ML()
    train_X_det1, val_X_det1,test_X_det1,train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_6(lev2_train,lev2_val,lev2_test)
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    train_X_det1 = train_X_det1.reshape(train_X_det1.shape[0],train_X_det1.shape[1])#变成2维
    test_X_det1 = test_X_det1.reshape(test_X_det1.shape[0],test_X_det1.shape[1])

    classifier = LogisticRegression(random_state=5,solver='sag',C=10,)
    classifier.fit(train_X_det1, train_Y_det1.ravel())
    lg_predict = classifier.predict(test_X_det1)
    np.set_printoptions(threshold=100000000)
    # print("lg_predict:", lg_predict)
    LG_AC = accuracy_score(test_Y_det1, lg_predict)
    LG_f1 = f1_score(test_Y_det1, lg_predict, average='macro')
    print("LR_AC_62:", LG_AC)

    dtc = DecisionTreeClassifier(criterion="gini",max_features=6,min_samples_split=6)
    dtc.fit(train_X_det1, train_Y_det1.ravel())
    dt_pre = dtc.predict(test_X_det1)
    np.set_printoptions(threshold=100000000)
    # print("dt_pre:", dt_pre)
    DT_AC = accuracy_score(test_Y_det1, dt_pre)
    DT_f1 = f1_score(test_Y_det1, dt_pre, average='macro')
    print("DT_AC_62:", DT_AC)

    knn = KNeighborsClassifier(weights='distance',p=2,leaf_size=50)
    knn.fit(train_X_det1, train_Y_det1.ravel())
    knn_predict = knn.predict(test_X_det1)
    np.set_printoptions(threshold=100000000)
    # print("knn_predict:", knn_predict)
    KNN_AC = accuracy_score(test_Y_det1, knn_predict)
    KNN_f1 = f1_score(test_Y_det1, knn_predict, average='macro')
    print("KNN_AC_62:", KNN_AC)
    _ = projection(test_Y_det1, knn_predict,"KNN",6,62)
    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': 10,
        'gamma': 0.1,
        'max_depth': 10,
        'lambda': 2,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 1,
        'silent': 1,
        'eta': 0.3,
        'seed': 1000,
        'nthread': 4,
    }

    dtrain = xgb.DMatrix(train_X_det1, train_Y_det1.ravel())
    num_rounds = 20
    plst = list(params.items())
    xgb1 = xgb.train(plst, dtrain, num_rounds)
    dtest = xgb.DMatrix(test_X_det1)
    xgb1_p_temp = xgb1.predict(dtest)
    xgb1_p = xgb1_p_temp.reshape(-1, 1)
    xgb1_acc = accuracy_score(xgb1_p, test_Y_det1)
    print('xgb1_acc_62=', xgb1_acc)
    _ = projection(test_Y_det1, xgb1_p,"XGboost",6,62)

def ML_63():
    # start_time = time.time()#参数调优要花费1小时之久~~
    print("ML_63 starting")
    lev1_train,lev1_val,lev1_test,lev2_train,lev2_val,lev2_test,lev3_train,lev3_val,lev3_test,lev4_train,lev4_val,lev4_test = load_ML()
    train_X_det1, val_X_det1,test_X_det1,train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_6(lev3_train,lev3_val,lev3_test)
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    train_X_det1 = train_X_det1.reshape(train_X_det1.shape[0],train_X_det1.shape[1])#变成2维
    test_X_det1 = test_X_det1.reshape(test_X_det1.shape[0],test_X_det1.shape[1])

    classifier = LogisticRegression(random_state=5,solver='sag',C=10,)
    classifier.fit(train_X_det1, train_Y_det1.ravel())
    lg_predict = classifier.predict(test_X_det1)
    np.set_printoptions(threshold=100000000)
    # print("lg_predict:", lg_predict)
    LG_AC = accuracy_score(test_Y_det1, lg_predict)
    LG_f1 = f1_score(test_Y_det1, lg_predict, average='macro')
    print("LR_AC_63:", LG_AC)

    dtc = DecisionTreeClassifier(criterion="gini",max_features=6,min_samples_split=6)
    dtc.fit(train_X_det1, train_Y_det1.ravel())
    dt_pre = dtc.predict(test_X_det1)
    np.set_printoptions(threshold=100000000)
    # print("dt_pre:", dt_pre)
    DT_AC = accuracy_score(test_Y_det1, dt_pre)
    DT_f1 = f1_score(test_Y_det1, dt_pre, average='macro')
    print("DT_AC_63:", DT_AC)

    knn = KNeighborsClassifier(weights='distance',p=2,leaf_size=50)
    knn.fit(train_X_det1, train_Y_det1.ravel())
    knn_predict = knn.predict(test_X_det1)
    np.set_printoptions(threshold=100000000)
    # print("knn_predict:", knn_predict)
    KNN_AC = accuracy_score(test_Y_det1, knn_predict)
    KNN_f1 = f1_score(test_Y_det1, knn_predict, average='macro')
    print("KNN_AC_63:", KNN_AC)
    _ = projection(test_Y_det1, knn_predict,"KNN",6,63)
    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': 12,
        'gamma': 0.1,
        'max_depth': 12,
        'lambda': 2,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 1,
        'silent': 1,
        'eta': 0.3,
        'seed': 1000,
        'nthread': 4,
    }

    dtrain = xgb.DMatrix(train_X_det1, train_Y_det1.ravel())
    num_rounds = 20
    plst = list(params.items())
    xgb1 = xgb.train(plst, dtrain, num_rounds)
    dtest = xgb.DMatrix(test_X_det1)
    xgb1_p_temp = xgb1.predict(dtest)
    xgb1_p = xgb1_p_temp.reshape(-1, 1)
    xgb1_acc = accuracy_score(xgb1_p, test_Y_det1)
    print('xgb1_acc_63=', xgb1_acc)
    _ = projection(test_Y_det1, xgb1_p,"XGboost",6,63)
def ML_64():
    # start_time = time.time()#参数调优要花费1小时之久~~
    print("ML_64 starting")
    lev1_train,lev1_val,lev1_test,lev2_train,lev2_val,lev2_test,lev3_train,lev3_val,lev3_test,lev4_train,lev4_val,lev4_test = load_ML()
    train_X_det1, val_X_det1,test_X_det1,train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_6(lev4_train,lev4_val,lev4_test)
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    train_X_det1 = train_X_det1.reshape(train_X_det1.shape[0],train_X_det1.shape[1])#变成2维
    test_X_det1 = test_X_det1.reshape(test_X_det1.shape[0],test_X_det1.shape[1])

    classifier = LogisticRegression(random_state=5,solver='sag',C=10,)
    classifier.fit(train_X_det1, train_Y_det1.ravel())
    lg_predict = classifier.predict(test_X_det1)
    np.set_printoptions(threshold=100000000)
    # print("lg_predict:", lg_predict)
    LG_AC = accuracy_score(test_Y_det1, lg_predict)
    LG_f1 = f1_score(test_Y_det1, lg_predict, average='macro')
    print("LR_AC_64:", LG_AC)

    dtc = DecisionTreeClassifier(criterion="gini",max_features=6,min_samples_split=6)
    dtc.fit(train_X_det1, train_Y_det1.ravel())
    dt_pre = dtc.predict(test_X_det1)
    np.set_printoptions(threshold=100000000)
    # print("dt_pre:", dt_pre)
    DT_AC = accuracy_score(test_Y_det1, dt_pre)
    DT_f1 = f1_score(test_Y_det1, dt_pre, average='macro')
    print("DT_AC_64:", DT_AC)

    knn = KNeighborsClassifier(weights='distance',p=2,leaf_size=50)
    knn.fit(train_X_det1, train_Y_det1.ravel())
    knn_predict = knn.predict(test_X_det1)
    np.set_printoptions(threshold=100000000)
    # print("knn_predict:", knn_predict)
    KNN_AC = accuracy_score(test_Y_det1, knn_predict)
    KNN_f1 = f1_score(test_Y_det1, knn_predict, average='macro')
    print("KNN_AC_64:", KNN_AC)
    _ = projection(test_Y_det1, knn_predict,"KNN",6,64)
    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': 12,
        'gamma': 0.1,
        'max_depth': 12,
        'lambda': 2,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 1,
        'silent': 1,
        'eta': 0.3,
        'seed': 1000,
        'nthread': 4,
    }

    dtrain = xgb.DMatrix(train_X_det1, train_Y_det1.ravel())
    num_rounds = 20
    plst = list(params.items())
    xgb1 = xgb.train(plst, dtrain, num_rounds)
    dtest = xgb.DMatrix(test_X_det1)
    xgb1_p_temp = xgb1.predict(dtest)
    xgb1_p = xgb1_p_temp.reshape(-1, 1)
    xgb1_acc = accuracy_score(xgb1_p, test_Y_det1)
    print('xgb1_acc_64=', xgb1_acc)
    _ = projection(test_Y_det1, xgb1_p,"XGboost",6,64)
def ML_81():
    # start_time = time.time()#参数调优要花费1小时之久~~
    print("ML_81 starting")
    lev1_train,lev1_val,lev1_test,lev2_train,lev2_val,lev2_test,lev3_train,lev3_val,lev3_test,lev4_train,lev4_val,lev4_test = load_ML()
    train_X_det1, val_X_det1,test_X_det1,train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_8(lev1_train,lev1_val,lev1_test)
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    train_X_det1 = train_X_det1.reshape(train_X_det1.shape[0],train_X_det1.shape[1])#变成2维
    test_X_det1 = test_X_det1.reshape(test_X_det1.shape[0],test_X_det1.shape[1])

    classifier = LogisticRegression(random_state=5,solver='sag',C=10,)
    classifier.fit(train_X_det1, train_Y_det1.ravel())
    lg_predict = classifier.predict(test_X_det1)
    np.set_printoptions(threshold=100000000)
    # print("lg_predict:", lg_predict)
    LG_AC = accuracy_score(test_Y_det1, lg_predict)
    LG_f1 = f1_score(test_Y_det1, lg_predict, average='macro')
    print("LR_AC:", LG_AC)

    dtc = DecisionTreeClassifier(criterion="gini",max_features=6,min_samples_split=6)
    dtc.fit(train_X_det1, train_Y_det1.ravel())
    dt_pre = dtc.predict(test_X_det1)
    np.set_printoptions(threshold=100000000)
    # print("dt_pre:", dt_pre)
    DT_AC = accuracy_score(test_Y_det1, dt_pre)
    DT_f1 = f1_score(test_Y_det1, dt_pre, average='macro')
    print("DT_AC:", DT_AC)

    knn = KNeighborsClassifier(weights='distance',p=2,leaf_size=50)
    knn.fit(train_X_det1, train_Y_det1.ravel())
    knn_predict = knn.predict(test_X_det1)
    np.set_printoptions(threshold=100000000)
    # print("knn_predict:", knn_predict)
    KNN_AC = accuracy_score(test_Y_det1, knn_predict)
    KNN_f1 = f1_score(test_Y_det1, knn_predict, average='macro')
    print("KNN_AC:", KNN_AC)
    # _ = projection(test_Y_det1, knn_predict,"KNN",8,81)
    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': 10,
        'gamma': 0.1,
        'max_depth': 8,
        'lambda': 2,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 1,
        'silent': 1,
        'eta': 0.3,
        'seed': 1000,
        'nthread': 4,
    }

    dtrain = xgb.DMatrix(train_X_det1, train_Y_det1.ravel())
    num_rounds = 20
    plst = list(params.items())
    xgb1 = xgb.train(plst, dtrain, num_rounds)
    dtest = xgb.DMatrix(test_X_det1)
    xgb1_p_temp = xgb1.predict(dtest)
    xgb1_p = xgb1_p_temp.reshape(-1, 1)
    xgb1_acc = accuracy_score(xgb1_p, test_Y_det1)
    print('xgb1_acc=', xgb1_acc)
    # _ = projection(test_Y_det1, xgb1_p,"XGBoost",8,81)
def ML_82():
    # start_time = time.time()#参数调优要花费1小时之久~~
    print("ML_82 starting")
    lev1_train,lev1_val,lev1_test,lev2_train,lev2_val,lev2_test,lev3_train,lev3_val,lev3_test,lev4_train,lev4_val,lev4_test = load_ML()
    train_X_det1, val_X_det1,test_X_det1,train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_8(lev2_train,lev2_val,lev2_test)
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    train_X_det1 = train_X_det1.reshape(train_X_det1.shape[0],train_X_det1.shape[1])#变成2维
    test_X_det1 = test_X_det1.reshape(test_X_det1.shape[0],test_X_det1.shape[1])

    classifier = LogisticRegression(random_state=5,solver='sag',C=10,)
    classifier.fit(train_X_det1, train_Y_det1.ravel())
    lg_predict = classifier.predict(test_X_det1)
    np.set_printoptions(threshold=100000000)
    # print("lg_predict:", lg_predict)
    LG_AC = accuracy_score(test_Y_det1, lg_predict)
    LG_f1 = f1_score(test_Y_det1, lg_predict, average='macro')
    print("LR_AC:", LG_AC)

    dtc = DecisionTreeClassifier(criterion="gini",max_features=6,min_samples_split=6)
    dtc.fit(train_X_det1, train_Y_det1.ravel())
    dt_pre = dtc.predict(test_X_det1)
    np.set_printoptions(threshold=100000000)
    # print("dt_pre:", dt_pre)
    DT_AC = accuracy_score(test_Y_det1, dt_pre)
    DT_f1 = f1_score(test_Y_det1, dt_pre, average='macro')
    print("DT_AC:", DT_AC)

    knn = KNeighborsClassifier(weights='distance',p=2,leaf_size=50)
    knn.fit(train_X_det1, train_Y_det1.ravel())
    knn_predict = knn.predict(test_X_det1)
    np.set_printoptions(threshold=100000000)
    # print("knn_predict:", knn_predict)
    KNN_AC = accuracy_score(test_Y_det1, knn_predict)
    KNN_f1 = f1_score(test_Y_det1, knn_predict, average='macro')
    print("KNN_AC:", KNN_AC)
    # _ = projection(test_Y_det1, knn_predict,"KNN",8,82)
    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': 10,
        'gamma': 0.1,
        'max_depth': 8,
        'lambda': 2,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 1,
        'silent': 1,
        'eta': 0.3,
        'seed': 1000,
        'nthread': 4,
    }

    dtrain = xgb.DMatrix(train_X_det1, train_Y_det1.ravel())
    num_rounds = 20
    plst = list(params.items())
    xgb1 = xgb.train(plst, dtrain, num_rounds)
    dtest = xgb.DMatrix(test_X_det1)
    xgb1_p_temp = xgb1.predict(dtest)
    xgb1_p = xgb1_p_temp.reshape(-1, 1)
    xgb1_acc = accuracy_score(xgb1_p, test_Y_det1)
    print('xgb1_acc=', xgb1_acc)
    _ = projection(test_Y_det1, xgb1_p,"XGBoost",8,82)
def ML_83():
    # start_time = time.time()#参数调优要花费1小时之久~~
    print("ML_83 starting")
    lev1_train,lev1_val,lev1_test,lev2_train,lev2_val,lev2_test,lev3_train,lev3_val,lev3_test,lev4_train,lev4_val,lev4_test = load_ML()
    train_X_det1, val_X_det1,test_X_det1,train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_8(lev3_train,lev3_val,lev3_test)
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    train_X_det1 = train_X_det1.reshape(train_X_det1.shape[0],train_X_det1.shape[1])#变成2维
    test_X_det1 = test_X_det1.reshape(test_X_det1.shape[0],test_X_det1.shape[1])

    classifier = LogisticRegression(random_state=5,solver='sag',C=10,)
    classifier.fit(train_X_det1, train_Y_det1.ravel())
    lg_predict = classifier.predict(test_X_det1)
    np.set_printoptions(threshold=100000000)
    # print("lg_predict:", lg_predict)
    LG_AC = accuracy_score(test_Y_det1, lg_predict)
    LG_f1 = f1_score(test_Y_det1, lg_predict, average='macro')
    print("LR_AC:", LG_AC)

    dtc = DecisionTreeClassifier(criterion="gini",max_features=6,min_samples_split=6)
    dtc.fit(train_X_det1, train_Y_det1.ravel())
    dt_pre = dtc.predict(test_X_det1)
    np.set_printoptions(threshold=100000000)
    # print("dt_pre:", dt_pre)
    DT_AC = accuracy_score(test_Y_det1, dt_pre)
    DT_f1 = f1_score(test_Y_det1, dt_pre, average='macro')
    print("DT_AC:", DT_AC)

    knn = KNeighborsClassifier(weights='distance',p=2,leaf_size=50)
    knn.fit(train_X_det1, train_Y_det1.ravel())
    knn_predict = knn.predict(test_X_det1)
    np.set_printoptions(threshold=100000000)
    # print("knn_predict:", knn_predict)
    KNN_AC = accuracy_score(test_Y_det1, knn_predict)
    KNN_f1 = f1_score(test_Y_det1, knn_predict, average='macro')
    print("KNN_AC:", KNN_AC)
    # _ = projection(test_Y_det1, knn_predict,"KNN",8,83)
    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': 10,
        'gamma': 0.1,
        'max_depth': 8,
        'lambda': 2,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 1,
        'silent': 1,
        'eta': 0.3,
        'seed': 1000,
        'nthread': 4,
    }

    dtrain = xgb.DMatrix(train_X_det1, train_Y_det1.ravel())
    num_rounds = 20
    plst = list(params.items())
    xgb1 = xgb.train(plst, dtrain, num_rounds)
    dtest = xgb.DMatrix(test_X_det1)
    xgb1_p_temp = xgb1.predict(dtest)
    xgb1_p = xgb1_p_temp.reshape(-1, 1)
    xgb1_acc = accuracy_score(xgb1_p, test_Y_det1)
    print('xgb1_acc=', xgb1_acc)
    _ = projection(test_Y_det1, xgb1_p,"XGBoost",8,83)
def ML_84():
    # start_time = time.time()#参数调优要花费1小时之久~~
    print("ML_84 starting")
    lev1_train,lev1_val,lev1_test,lev2_train,lev2_val,lev2_test,lev3_train,lev3_val,lev3_test,lev4_train,lev4_val,lev4_test = load_ML()
    train_X_det1, val_X_det1,test_X_det1,train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_8(lev4_train,lev4_val,lev4_test)
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    train_X_det1 = train_X_det1.reshape(train_X_det1.shape[0],train_X_det1.shape[1])#变成2维
    test_X_det1 = test_X_det1.reshape(test_X_det1.shape[0],test_X_det1.shape[1])

    classifier = LogisticRegression(random_state=5,solver='sag',C=10,)
    classifier.fit(train_X_det1, train_Y_det1.ravel())
    lg_predict = classifier.predict(test_X_det1)
    np.set_printoptions(threshold=100000000)
    # print("lg_predict:", lg_predict)
    LG_AC = accuracy_score(test_Y_det1, lg_predict)
    LG_f1 = f1_score(test_Y_det1, lg_predict, average='macro')
    print("LR_AC:", LG_AC)

    dtc = DecisionTreeClassifier(criterion="gini",max_features=6,min_samples_split=6)
    dtc.fit(train_X_det1, train_Y_det1.ravel())
    dt_pre = dtc.predict(test_X_det1)
    np.set_printoptions(threshold=100000000)
    # print("dt_pre:", dt_pre)
    DT_AC = accuracy_score(test_Y_det1, dt_pre)
    DT_f1 = f1_score(test_Y_det1, dt_pre, average='macro')
    print("DT_AC:", DT_AC)

    knn = KNeighborsClassifier(weights='distance',p=2,leaf_size=50)
    knn.fit(train_X_det1, train_Y_det1.ravel())
    knn_predict = knn.predict(test_X_det1)
    np.set_printoptions(threshold=100000000)
    # print("knn_predict:", knn_predict)
    KNN_AC = accuracy_score(test_Y_det1, knn_predict)
    KNN_f1 = f1_score(test_Y_det1, knn_predict, average='macro')
    print("KNN_AC:", KNN_AC)
    # _ = projection(test_Y_det1, knn_predict,"KNN",8,84)
    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': 10,
        'gamma': 0.1,
        'max_depth': 8,
        'lambda': 2,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 1,
        'silent': 1,
        'eta': 0.3,
        'seed': 1000,
        'nthread': 4,
    }

    dtrain = xgb.DMatrix(train_X_det1, train_Y_det1.ravel())
    num_rounds = 20
    plst = list(params.items())
    xgb1 = xgb.train(plst, dtrain, num_rounds)
    dtest = xgb.DMatrix(test_X_det1)
    xgb1_p_temp = xgb1.predict(dtest)
    xgb1_p = xgb1_p_temp.reshape(-1, 1)
    xgb1_acc = accuracy_score(xgb1_p, test_Y_det1)
    print('xgb1_acc=', xgb1_acc)
    _ = projection(test_Y_det1, xgb1_p,"XGBoost",8,84)

def ML_101():
    # start_time = time.time()#参数调优要花费1小时之久~~
    print("ML_101 starting")
    lev1_train,lev1_val,lev1_test,lev2_train,lev2_val,lev2_test,lev3_train,lev3_val,lev3_test,lev4_train,lev4_val,lev4_test = load_ML()
    train_X_det1, val_X_det1,test_X_det1,train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_10(lev1_train,lev1_val,lev1_test)
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    train_X_det1 = train_X_det1.reshape(train_X_det1.shape[0],train_X_det1.shape[1])#变成2维
    test_X_det1 = test_X_det1.reshape(test_X_det1.shape[0],test_X_det1.shape[1])

    classifier = LogisticRegression(random_state=5,solver='sag',C=10,)
    classifier.fit(train_X_det1, train_Y_det1.ravel())
    lg_predict = classifier.predict(test_X_det1)
    np.set_printoptions(threshold=100000000)
    # print("lg_predict:", lg_predict)
    LG_AC = accuracy_score(test_Y_det1, lg_predict)
    LG_f1 = f1_score(test_Y_det1, lg_predict, average='macro')
    print("LR_AC:", LG_AC)


    dtc = DecisionTreeClassifier(criterion="gini",max_features=6,min_samples_split=6)
    dtc.fit(train_X_det1, train_Y_det1.ravel())
    dt_pre = dtc.predict(test_X_det1)
    np.set_printoptions(threshold=100000000)
    # print("dt_pre:", dt_pre)
    DT_AC = accuracy_score(test_Y_det1, dt_pre)
    DT_f1 = f1_score(test_Y_det1, dt_pre, average='macro')
    print("DT_AC:", DT_AC)

    knn = KNeighborsClassifier(weights='distance',p=2,leaf_size=50)
    knn.fit(train_X_det1, train_Y_det1.ravel())
    knn_predict = knn.predict(test_X_det1)
    np.set_printoptions(threshold=100000000)
    # print("knn_predict:", knn_predict)
    KNN_AC = accuracy_score(test_Y_det1, knn_predict)
    KNN_f1 = f1_score(test_Y_det1, knn_predict, average='macro')
    print("KNN_AC:", KNN_AC)
    _ = projection(test_Y_det1, knn_predict,"KNN",10,101)
    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': 10,
        'gamma': 0.1,
        'max_depth': 8,
        'lambda': 2,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 1,
        'silent': 1,
        'eta': 0.3,
        'seed': 1000,
        'nthread': 4,
    }

    dtrain = xgb.DMatrix(train_X_det1, train_Y_det1.ravel())
    num_rounds = 20
    plst = list(params.items())
    xgb1 = xgb.train(plst, dtrain, num_rounds)
    dtest = xgb.DMatrix(test_X_det1)
    xgb1_p_temp = xgb1.predict(dtest)
    xgb1_p = xgb1_p_temp.reshape(-1, 1)
    xgb1_acc = accuracy_score(xgb1_p, test_Y_det1)
    print('xgb1_acc=', xgb1_acc)
    _ = projection(test_Y_det1, xgb1_p,"XGboost",10,101)
def ML_102():
    # start_time = time.time()#参数调优要花费1小时之久~~
    print("ML_102 starting")
    lev1_train,lev1_val,lev1_test,lev2_train,lev2_val,lev2_test,lev3_train,lev3_val,lev3_test,lev4_train,lev4_val,lev4_test = load_ML()
    train_X_det1, val_X_det1,test_X_det1,train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_10(lev2_train,lev2_val,lev2_test)
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    train_X_det1 = train_X_det1.reshape(train_X_det1.shape[0],train_X_det1.shape[1])#变成2维
    test_X_det1 = test_X_det1.reshape(test_X_det1.shape[0],test_X_det1.shape[1])

    classifier = LogisticRegression(random_state=5,solver='sag',C=10,)
    classifier.fit(train_X_det1, train_Y_det1.ravel())
    lg_predict = classifier.predict(test_X_det1)
    np.set_printoptions(threshold=100000000)
    # print("lg_predict:", lg_predict)
    LG_AC = accuracy_score(test_Y_det1, lg_predict)
    LG_f1 = f1_score(test_Y_det1, lg_predict, average='macro')
    print("LR_AC:", LG_AC)


    dtc = DecisionTreeClassifier(criterion="gini",max_features=6,min_samples_split=6)
    dtc.fit(train_X_det1, train_Y_det1.ravel())
    dt_pre = dtc.predict(test_X_det1)
    np.set_printoptions(threshold=100000000)
    # print("dt_pre:", dt_pre)
    DT_AC = accuracy_score(test_Y_det1, dt_pre)
    DT_f1 = f1_score(test_Y_det1, dt_pre, average='macro')
    print("DT_AC:", DT_AC)

    knn = KNeighborsClassifier(weights='distance',p=2,leaf_size=50)
    knn.fit(train_X_det1, train_Y_det1.ravel())
    knn_predict = knn.predict(test_X_det1)
    np.set_printoptions(threshold=100000000)
    # print("knn_predict:", knn_predict)
    KNN_AC = accuracy_score(test_Y_det1, knn_predict)
    KNN_f1 = f1_score(test_Y_det1, knn_predict, average='macro')
    print("KNN_AC:", KNN_AC)
    _ = projection(test_Y_det1, knn_predict,"KNN",10,102)
    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': 10,
        'gamma': 0.1,
        'max_depth': 8,
        'lambda': 2,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 1,
        'silent': 1,
        'eta': 0.3,
        'seed': 1000,
        'nthread': 4,
    }

    dtrain = xgb.DMatrix(train_X_det1, train_Y_det1.ravel())
    num_rounds = 20
    plst = list(params.items())
    xgb1 = xgb.train(plst, dtrain, num_rounds)
    dtest = xgb.DMatrix(test_X_det1)
    xgb1_p_temp = xgb1.predict(dtest)
    xgb1_p = xgb1_p_temp.reshape(-1, 1)
    xgb1_acc = accuracy_score(xgb1_p, test_Y_det1)
    print('xgb1_acc=', xgb1_acc)
    _ = projection(test_Y_det1, xgb1_p,"XGboost",10,102)
def ML_103():
    # start_time = time.time()#参数调优要花费1小时之久~~
    print("ML_103 starting")
    lev1_train,lev1_val,lev1_test,lev2_train,lev2_val,lev2_test,lev3_train,lev3_val,lev3_test,lev4_train,lev4_val,lev4_test = load_ML()
    train_X_det1, val_X_det1,test_X_det1,train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_10(lev3_train,lev3_val,lev3_test)
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    train_X_det1 = train_X_det1.reshape(train_X_det1.shape[0],train_X_det1.shape[1])#变成2维
    test_X_det1 = test_X_det1.reshape(test_X_det1.shape[0],test_X_det1.shape[1])

    classifier = LogisticRegression(random_state=5,solver='sag',C=10,)
    classifier.fit(train_X_det1, train_Y_det1.ravel())
    lg_predict = classifier.predict(test_X_det1)
    np.set_printoptions(threshold=100000000)
    # print("lg_predict:", lg_predict)
    LG_AC = accuracy_score(test_Y_det1, lg_predict)
    LG_f1 = f1_score(test_Y_det1, lg_predict, average='macro')
    print("LR_AC:", LG_AC)


    dtc = DecisionTreeClassifier(criterion="gini",max_features=6,min_samples_split=6)
    dtc.fit(train_X_det1, train_Y_det1.ravel())
    dt_pre = dtc.predict(test_X_det1)
    np.set_printoptions(threshold=100000000)
    # print("dt_pre:", dt_pre)
    DT_AC = accuracy_score(test_Y_det1, dt_pre)
    DT_f1 = f1_score(test_Y_det1, dt_pre, average='macro')
    print("DT_AC:", DT_AC)

    knn = KNeighborsClassifier(weights='distance',p=2,leaf_size=50)
    knn.fit(train_X_det1, train_Y_det1.ravel())
    knn_predict = knn.predict(test_X_det1)
    np.set_printoptions(threshold=100000000)
    # print("knn_predict:", knn_predict)
    KNN_AC = accuracy_score(test_Y_det1, knn_predict)
    KNN_f1 = f1_score(test_Y_det1, knn_predict, average='macro')
    print("KNN_AC:", KNN_AC)
    _ = projection(test_Y_det1, knn_predict,"KNN",10,103)
    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': 10,
        'gamma': 0.1,
        'max_depth': 8,
        'lambda': 2,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 1,
        'silent': 1,
        'eta': 0.3,
        'seed': 1000,
        'nthread': 4,
    }

    dtrain = xgb.DMatrix(train_X_det1, train_Y_det1.ravel())
    num_rounds = 20
    plst = list(params.items())
    xgb1 = xgb.train(plst, dtrain, num_rounds)
    dtest = xgb.DMatrix(test_X_det1)
    xgb1_p_temp = xgb1.predict(dtest)
    xgb1_p = xgb1_p_temp.reshape(-1, 1)
    xgb1_acc = accuracy_score(xgb1_p, test_Y_det1)
    print('xgb1_acc=', xgb1_acc)
    _ = projection(test_Y_det1, xgb1_p,"XGboost",10,103)
def ML_104():
    # start_time = time.time()#参数调优要花费1小时之久~~
    print("ML_104 starting")
    lev1_train,lev1_val,lev1_test,lev2_train,lev2_val,lev2_test,lev3_train,lev3_val,lev3_test,lev4_train,lev4_val,lev4_test = load_ML()
    train_X_det1, val_X_det1,test_X_det1,train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_10(lev4_train,lev4_val,lev4_test)
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    train_X_det1 = train_X_det1.reshape(train_X_det1.shape[0],train_X_det1.shape[1])#变成2维
    test_X_det1 = test_X_det1.reshape(test_X_det1.shape[0],test_X_det1.shape[1])

    classifier = LogisticRegression(random_state=5,solver='sag',C=10,)
    classifier.fit(train_X_det1, train_Y_det1.ravel())
    lg_predict = classifier.predict(test_X_det1)
    np.set_printoptions(threshold=100000000)
    # print("lg_predict:", lg_predict)
    LG_AC = accuracy_score(test_Y_det1, lg_predict)
    LG_f1 = f1_score(test_Y_det1, lg_predict, average='macro')
    print("LR_AC:", LG_AC)


    dtc = DecisionTreeClassifier(criterion="gini",max_features=6,min_samples_split=6)
    dtc.fit(train_X_det1, train_Y_det1.ravel())
    dt_pre = dtc.predict(test_X_det1)
    np.set_printoptions(threshold=100000000)
    # print("dt_pre:", dt_pre)
    DT_AC = accuracy_score(test_Y_det1, dt_pre)
    DT_f1 = f1_score(test_Y_det1, dt_pre, average='macro')
    print("DT_AC:", DT_AC)

    knn = KNeighborsClassifier(weights='distance',p=2,leaf_size=50)
    knn.fit(train_X_det1, train_Y_det1.ravel())
    knn_predict = knn.predict(test_X_det1)
    np.set_printoptions(threshold=100000000)
    # print("knn_predict:", knn_predict)
    KNN_AC = accuracy_score(test_Y_det1, knn_predict)
    KNN_f1 = f1_score(test_Y_det1, knn_predict, average='macro')
    print("KNN_AC:", KNN_AC)
    _ = projection(test_Y_det1, knn_predict,"KNN",10,104)
    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': 10,
        'gamma': 0.1,
        'max_depth': 8,
        'lambda': 2,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 1,
        'silent': 1,
        'eta': 0.3,
        'seed': 1000,
        'nthread': 4,
    }

    dtrain = xgb.DMatrix(train_X_det1, train_Y_det1.ravel())
    num_rounds = 20
    plst = list(params.items())
    xgb1 = xgb.train(plst, dtrain, num_rounds)
    dtest = xgb.DMatrix(test_X_det1)
    xgb1_p_temp = xgb1.predict(dtest)
    xgb1_p = xgb1_p_temp.reshape(-1, 1)
    xgb1_acc = accuracy_score(xgb1_p, test_Y_det1)
    print('xgb1_acc=', xgb1_acc)
    _ = projection(test_Y_det1, xgb1_p,"XGboost",10,104)
def CNN_61():
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_6(lev1_train, lev1_val,
                                                                                                   lev1_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    model_cnn_det1 = tf.keras.Sequential()
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu',
                                              input_shape=(train_X_det1.shape[1], train_X_det1.shape[2])))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    # model_cnn_det1.add(tf.keras.layers.MaxPool1D(pool_size = 2))
    model_cnn_det1.add(tf.keras.layers.Flatten())
    # model_cnn_det1.add(tf.keras.layers.Dropout(0.1))
    model_cnn_det1.add(tf.keras.layers.Dense(128, activation='relu'))
    # model_cnn_det1.add(tf.keras.layers.Dropout(0.5))
    model_cnn_det1.add(tf.keras.layers.Dense(128, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(64, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(32, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(16, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(8, activation='softmax'))
    model_cnn_det1.compile(loss="sparse_categorical_crossentropy", optimizer='rmsprop', metrics=['accuracy'])
    history_cnn_det1 = model_cnn_det1.fit(train_X_det1, train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                          callbacks=callback1, batch_size=500, epochs=1000, verbose=2)
    det_cnn1 = model_cnn_det1.predict(test_X_det1)
    a_cnn_det1 = np.argmax(det_cnn1,axis=1)
    a_cnn_det1 = a_cnn_det1.reshape(-1,1)
    cnn_AC_det1 = accuracy_score(test_Y_det1, a_cnn_det1)
    print('cnn_AC_det1=', cnn_AC_det1)
    _ = projection(test_Y_det1, a_cnn_det1,"CNN",6,61)
def CNN_62():
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_6(lev2_train, lev2_val,
                                                                                                   lev2_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    model_cnn_det1 = tf.keras.Sequential()
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu',
                                              input_shape=(train_X_det1.shape[1], train_X_det1.shape[2])))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    # model_cnn_det1.add(tf.keras.layers.MaxPool1D(pool_size = 2))
    model_cnn_det1.add(tf.keras.layers.Flatten())
    # model_cnn_det1.add(tf.keras.layers.Dropout(0.1))
    model_cnn_det1.add(tf.keras.layers.Dense(128, activation='relu'))
    # model_cnn_det1.add(tf.keras.layers.Dropout(0.5))
    model_cnn_det1.add(tf.keras.layers.Dense(128, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(64, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(32, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(16, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(8, activation='softmax'))
    model_cnn_det1.compile(loss="sparse_categorical_crossentropy", optimizer='rmsprop', metrics=['accuracy'])
    history_cnn_det1 = model_cnn_det1.fit(train_X_det1, train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                          callbacks=callback1, batch_size=500, epochs=1000, verbose=2)
    det_cnn1 = model_cnn_det1.predict(test_X_det1)
    a_cnn_det1 = np.argmax(det_cnn1,axis=1)
    a_cnn_det1 = a_cnn_det1.reshape(-1,1)
    cnn_AC_det1 = accuracy_score(test_Y_det1, a_cnn_det1)
    print('cnn_AC_det1=', cnn_AC_det1)
    _ = projection(test_Y_det1, a_cnn_det1,"CNN",6,62)

def CNN_63():
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_6(lev3_train, lev3_val,
                                                                                                   lev3_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    model_cnn_det1 = tf.keras.Sequential()
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu',
                                              input_shape=(train_X_det1.shape[1], train_X_det1.shape[2])))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    # model_cnn_det1.add(tf.keras.layers.MaxPool1D(pool_size = 2))
    model_cnn_det1.add(tf.keras.layers.Flatten())
    # model_cnn_det1.add(tf.keras.layers.Dropout(0.1))
    model_cnn_det1.add(tf.keras.layers.Dense(128, activation='relu'))
    # model_cnn_det1.add(tf.keras.layers.Dropout(0.5))
    model_cnn_det1.add(tf.keras.layers.Dense(128, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(64, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(32, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(16, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(8, activation='softmax'))
    model_cnn_det1.compile(loss="sparse_categorical_crossentropy", optimizer='rmsprop', metrics=['accuracy'])
    history_cnn_det1 = model_cnn_det1.fit(train_X_det1, train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                          callbacks=callback1, batch_size=500, epochs=1000, verbose=2)
    det_cnn1 = model_cnn_det1.predict(test_X_det1)
    a_cnn_det1 = np.argmax(det_cnn1,axis=1)
    a_cnn_det1 = a_cnn_det1.reshape(-1,1)
    cnn_AC_det1 = accuracy_score(test_Y_det1, a_cnn_det1)
    print('cnn_AC_det1=', cnn_AC_det1)
    _ = projection(test_Y_det1, a_cnn_det1,"CNN",6,63)
def CNN_64():
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_6(lev4_train, lev4_val,
                                                                                                   lev4_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    model_cnn_det1 = tf.keras.Sequential()
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu',
                                              input_shape=(train_X_det1.shape[1], train_X_det1.shape[2])))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    # model_cnn_det1.add(tf.keras.layers.MaxPool1D(pool_size = 2))
    model_cnn_det1.add(tf.keras.layers.Flatten())
    # model_cnn_det1.add(tf.keras.layers.Dropout(0.1))
    model_cnn_det1.add(tf.keras.layers.Dense(128, activation='relu'))
    # model_cnn_det1.add(tf.keras.layers.Dropout(0.5))
    model_cnn_det1.add(tf.keras.layers.Dense(128, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(64, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(32, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(16, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(8, activation='softmax'))
    model_cnn_det1.compile(loss="sparse_categorical_crossentropy", optimizer='rmsprop', metrics=['accuracy'])
    history_cnn_det1 = model_cnn_det1.fit(train_X_det1, train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                          callbacks=callback1, batch_size=500, epochs=1000, verbose=2)
    det_cnn1 = model_cnn_det1.predict(test_X_det1)
    a_cnn_det1 = np.argmax(det_cnn1,axis=1)
    a_cnn_det1 = a_cnn_det1.reshape(-1,1)
    cnn_AC_det1 = accuracy_score(test_Y_det1, a_cnn_det1)
    print('cnn_AC_det1=', cnn_AC_det1)
    _ = projection(test_Y_det1, a_cnn_det1,"CNN",6,64)
def CNN_81():
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_8(lev1_train, lev1_val,
                                                                                                   lev1_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    model_cnn_det1 = tf.keras.Sequential()
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu',
                                              input_shape=(train_X_det1.shape[1], train_X_det1.shape[2])))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    # model_cnn_det1.add(tf.keras.layers.MaxPool1D(pool_size = 2))
    model_cnn_det1.add(tf.keras.layers.Flatten())
    # model_cnn_det1.add(tf.keras.layers.Dropout(0.1))
    model_cnn_det1.add(tf.keras.layers.Dense(128, activation='relu'))
    # model_cnn_det1.add(tf.keras.layers.Dropout(0.5))
    model_cnn_det1.add(tf.keras.layers.Dense(128, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(64, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(32, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(16, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(8, activation='softmax'))
    model_cnn_det1.compile(loss="sparse_categorical_crossentropy", optimizer='rmsprop', metrics=['accuracy'])
    history_cnn_det1 = model_cnn_det1.fit(train_X_det1, train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                          callbacks=callback1, batch_size=500, epochs=1000, verbose=2)
    det_cnn1 = model_cnn_det1.predict(test_X_det1)
    a_cnn_det1 = np.argmax(det_cnn1,axis=1)
    a_cnn_det1 = a_cnn_det1.reshape(-1,1)
    cnn_AC_det1 = accuracy_score(test_Y_det1, a_cnn_det1)
    print('cnn_AC_det1=', cnn_AC_det1)
    _ = projection(test_Y_det1, a_cnn_det1,"CNN",8,81)
def CNN_82():
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_8(lev2_train, lev2_val,
                                                                                                   lev2_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    model_cnn_det1 = tf.keras.Sequential()
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu',
                                              input_shape=(train_X_det1.shape[1], train_X_det1.shape[2])))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    # model_cnn_det1.add(tf.keras.layers.MaxPool1D(pool_size = 2))
    model_cnn_det1.add(tf.keras.layers.Flatten())
    # model_cnn_det1.add(tf.keras.layers.Dropout(0.1))
    model_cnn_det1.add(tf.keras.layers.Dense(128, activation='relu'))
    # model_cnn_det1.add(tf.keras.layers.Dropout(0.5))
    model_cnn_det1.add(tf.keras.layers.Dense(128, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(64, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(32, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(16, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(8, activation='softmax'))
    model_cnn_det1.compile(loss="sparse_categorical_crossentropy", optimizer='rmsprop', metrics=['accuracy'])
    history_cnn_det1 = model_cnn_det1.fit(train_X_det1, train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                          callbacks=callback1, batch_size=500, epochs=1000, verbose=2)
    det_cnn1 = model_cnn_det1.predict(test_X_det1)
    a_cnn_det1 = np.argmax(det_cnn1,axis=1)
    a_cnn_det1 = a_cnn_det1.reshape(-1,1)
    cnn_AC_det1 = accuracy_score(test_Y_det1, a_cnn_det1)
    print('cnn_AC_det1=', cnn_AC_det1)
    _ = projection(test_Y_det1, a_cnn_det1,"CNN",8,82)

def CNN_83():
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_8(lev3_train, lev3_val,
                                                                                                   lev3_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    model_cnn_det1 = tf.keras.Sequential()
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu',
                                              input_shape=(train_X_det1.shape[1], train_X_det1.shape[2])))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    # model_cnn_det1.add(tf.keras.layers.MaxPool1D(pool_size = 2))
    model_cnn_det1.add(tf.keras.layers.Flatten())
    # model_cnn_det1.add(tf.keras.layers.Dropout(0.1))
    model_cnn_det1.add(tf.keras.layers.Dense(128, activation='relu'))
    # model_cnn_det1.add(tf.keras.layers.Dropout(0.5))
    model_cnn_det1.add(tf.keras.layers.Dense(128, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(64, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(32, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(16, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(8, activation='softmax'))
    model_cnn_det1.compile(loss="sparse_categorical_crossentropy", optimizer='rmsprop', metrics=['accuracy'])
    history_cnn_det1 = model_cnn_det1.fit(train_X_det1, train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                          callbacks=callback1, batch_size=500, epochs=1000, verbose=2)
    det_cnn1 = model_cnn_det1.predict(test_X_det1)
    a_cnn_det1 = np.argmax(det_cnn1,axis=1)
    a_cnn_det1 = a_cnn_det1.reshape(-1,1)
    cnn_AC_det1 = accuracy_score(test_Y_det1, a_cnn_det1)
    print('cnn_AC_det1=', cnn_AC_det1)
    _ = projection(test_Y_det1, a_cnn_det1,"CNN",8,83)

def CNN_84():
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_8(lev4_train, lev4_val,
                                                                                                   lev4_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    model_cnn_det1 = tf.keras.Sequential()
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu',
                                              input_shape=(train_X_det1.shape[1], train_X_det1.shape[2])))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    # model_cnn_det1.add(tf.keras.layers.MaxPool1D(pool_size = 2))
    model_cnn_det1.add(tf.keras.layers.Flatten())
    # model_cnn_det1.add(tf.keras.layers.Dropout(0.1))
    model_cnn_det1.add(tf.keras.layers.Dense(128, activation='relu'))
    # model_cnn_det1.add(tf.keras.layers.Dropout(0.5))
    model_cnn_det1.add(tf.keras.layers.Dense(128, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(64, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(32, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(16, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(8, activation='softmax'))
    model_cnn_det1.compile(loss="sparse_categorical_crossentropy", optimizer='rmsprop', metrics=['accuracy'])
    history_cnn_det1 = model_cnn_det1.fit(train_X_det1, train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                          callbacks=callback1, batch_size=500, epochs=1000, verbose=2)
    det_cnn1 = model_cnn_det1.predict(test_X_det1)
    a_cnn_det1 = np.argmax(det_cnn1,axis=1)
    a_cnn_det1 = a_cnn_det1.reshape(-1,1)
    cnn_AC_det1 = accuracy_score(test_Y_det1, a_cnn_det1)
    print('cnn_AC_det1=', cnn_AC_det1)
    _ = projection(test_Y_det1, a_cnn_det1,"CNN",8,84)

def CNN_101():
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_10(lev1_train, lev1_val,
                                                                                                   lev1_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    model_cnn_det1 = tf.keras.Sequential()
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu',
                                              input_shape=(train_X_det1.shape[1], train_X_det1.shape[2])))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    # model_cnn_det1.add(tf.keras.layers.MaxPool1D(pool_size = 2))
    model_cnn_det1.add(tf.keras.layers.Flatten())
    # model_cnn_det1.add(tf.keras.layers.Dropout(0.1))
    model_cnn_det1.add(tf.keras.layers.Dense(128, activation='relu'))
    # model_cnn_det1.add(tf.keras.layers.Dropout(0.5))
    model_cnn_det1.add(tf.keras.layers.Dense(128, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(64, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(32, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(16, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(8, activation='softmax'))
    model_cnn_det1.compile(loss="sparse_categorical_crossentropy", optimizer='rmsprop', metrics=['accuracy'])
    history_cnn_det1 = model_cnn_det1.fit(train_X_det1, train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                          callbacks=callback1, batch_size=500, epochs=1000, verbose=2)
    det_cnn1 = model_cnn_det1.predict(test_X_det1)
    a_cnn_det1 = np.argmax(det_cnn1,axis=1)
    a_cnn_det1 = a_cnn_det1.reshape(-1,1)
    cnn_AC_det1 = accuracy_score(test_Y_det1, a_cnn_det1)
    print('cnn_AC_det1=', cnn_AC_det1)
    _ = projection(test_Y_det1, a_cnn_det1,"CNN",10,101)

def CNN_102():
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_10(lev2_train, lev2_val,
                                                                                                   lev2_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    model_cnn_det1 = tf.keras.Sequential()
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu',
                                              input_shape=(train_X_det1.shape[1], train_X_det1.shape[2])))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    # model_cnn_det1.add(tf.keras.layers.MaxPool1D(pool_size = 2))
    model_cnn_det1.add(tf.keras.layers.Flatten())
    # model_cnn_det1.add(tf.keras.layers.Dropout(0.1))
    model_cnn_det1.add(tf.keras.layers.Dense(128, activation='relu'))
    # model_cnn_det1.add(tf.keras.layers.Dropout(0.5))
    model_cnn_det1.add(tf.keras.layers.Dense(128, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(64, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(32, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(16, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(8, activation='softmax'))
    model_cnn_det1.compile(loss="sparse_categorical_crossentropy", optimizer='rmsprop', metrics=['accuracy'])
    history_cnn_det1 = model_cnn_det1.fit(train_X_det1, train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                          callbacks=callback1, batch_size=500, epochs=1000, verbose=2)
    det_cnn1 = model_cnn_det1.predict(test_X_det1)
    a_cnn_det1 = np.argmax(det_cnn1,axis=1)
    a_cnn_det1 = a_cnn_det1.reshape(-1,1)
    cnn_AC_det1 = accuracy_score(test_Y_det1, a_cnn_det1)
    print('cnn_AC_det1=', cnn_AC_det1)
    _ = projection(test_Y_det1, a_cnn_det1,"CNN",10,102)

def CNN_103():
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_10(lev3_train, lev3_val,
                                                                                                   lev3_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    model_cnn_det1 = tf.keras.Sequential()
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu',
                                              input_shape=(train_X_det1.shape[1], train_X_det1.shape[2])))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    # model_cnn_det1.add(tf.keras.layers.MaxPool1D(pool_size = 2))
    model_cnn_det1.add(tf.keras.layers.Flatten())
    # model_cnn_det1.add(tf.keras.layers.Dropout(0.1))
    model_cnn_det1.add(tf.keras.layers.Dense(128, activation='relu'))
    # model_cnn_det1.add(tf.keras.layers.Dropout(0.5))
    model_cnn_det1.add(tf.keras.layers.Dense(128, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(64, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(32, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(16, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(8, activation='softmax'))
    model_cnn_det1.compile(loss="sparse_categorical_crossentropy", optimizer='rmsprop', metrics=['accuracy'])
    history_cnn_det1 = model_cnn_det1.fit(train_X_det1, train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                          callbacks=callback1, batch_size=500, epochs=1000, verbose=2)
    det_cnn1 = model_cnn_det1.predict(test_X_det1)
    a_cnn_det1 = np.argmax(det_cnn1,axis=1)
    a_cnn_det1 = a_cnn_det1.reshape(-1,1)
    cnn_AC_det1 = accuracy_score(test_Y_det1, a_cnn_det1)
    print('cnn_AC_det1=', cnn_AC_det1)
    _ = projection(test_Y_det1, a_cnn_det1,"CNN",10,103)

def CNN_104():
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_10(lev4_train, lev4_val,
                                                                                                   lev4_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    model_cnn_det1 = tf.keras.Sequential()
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu',
                                              input_shape=(train_X_det1.shape[1], train_X_det1.shape[2])))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    # model_cnn_det1.add(tf.keras.layers.MaxPool1D(pool_size = 2))
    model_cnn_det1.add(tf.keras.layers.Flatten())
    # model_cnn_det1.add(tf.keras.layers.Dropout(0.1))
    model_cnn_det1.add(tf.keras.layers.Dense(128, activation='relu'))
    # model_cnn_det1.add(tf.keras.layers.Dropout(0.5))
    model_cnn_det1.add(tf.keras.layers.Dense(128, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(64, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(32, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(16, activation='relu'))
    model_cnn_det1.add(tf.keras.layers.Dense(8, activation='softmax'))
    model_cnn_det1.compile(loss="sparse_categorical_crossentropy", optimizer='rmsprop', metrics=['accuracy'])
    history_cnn_det1 = model_cnn_det1.fit(train_X_det1, train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                          callbacks=callback1, batch_size=500, epochs=1000, verbose=2)
    det_cnn1 = model_cnn_det1.predict(test_X_det1)
    a_cnn_det1 = np.argmax(det_cnn1,axis=1)
    a_cnn_det1 = a_cnn_det1.reshape(-1,1)
    cnn_AC_det1 = accuracy_score(test_Y_det1, a_cnn_det1)
    print('cnn_AC_det1=', cnn_AC_det1)
    _ = projection(test_Y_det1, a_cnn_det1,"CNN",10,104)

def WD61():
    print("WD61 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_6(lev1_train, lev1_val,
                                                                                                   lev1_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    input_wide = tf.keras.layers.Input(shape=(6, 1))
    hidden11 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(input_wide)
    hidden22 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden11)

    input_deep = tf.keras.layers.Input(shape=(6, 1))
    hidden1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_deep)
    hidden2 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(hidden1)
    # hidden3 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden2)
    hidden4 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden2)
    hidden5 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden4)
    # hidden6 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden5)
    hidden7 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden5)
    hidden8 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden7)
    hidden9 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden8)
    concat = tf.keras.layers.concatenate([hidden22, hidden9])
    # concat1 = attention_3d_block2(concat)
    hidden10 = tf.keras.layers.Flatten(name='out')(concat)
    hidden111 = tf.keras.layers.Dense(128, activation='relu')(hidden10)
    output = tf.keras.layers.Dense(8, activation='softmax')(hidden111)
    model_wd_12 = tf.keras.models.Model(inputs=[input_wide, input_deep], outputs=[output])
    model_wd_det1 = model_wd_12
    model_wd_det1.compile(loss="sparse_categorical_crossentropy", optimizer='Nadam', metrics=['sparse_categorical_accuracy'])
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    history = model_wd_det1.fit([train_X_det1, train_X_det1], train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                    callbacks=callback1, batch_size=500, epochs=500, verbose=2, validation_freq=1)
    det_wd1 = model_wd_det1.predict([test_X_det1, test_X_det1])
    a_wd_det1 = np.argmax(det_wd1, axis=1)
    a_wd_det1 = a_wd_det1.reshape(-1, 1)

######################################################################

    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    font2 = {'size': 12}
    fig = plt.figure(figsize=(9, 4))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel("Epoch", font2)
    plt.ylabel("Accuracy", font2)
    plt.legend()

    # plt.subplot(1, 2, 2)
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(loss, label='Training Loss')
    ax.plot(val_loss, label='Validation Loss')
    ax.set_xlabel("Epoch", font2)
    ax.set_ylabel("Loss", font2)
    # ax.yaxis.set_ticks_position('right')
    # ax.yaxis.set_label_position('right')
    plt.legend()
    plt.suptitle('Accuracy and loss of training set and verification set of SD')
    plt.show()
###########################################################################

    wd_AC_det1 = accuracy_score(test_Y_det1, a_wd_det1)
    print('wd_AC_det1=', wd_AC_det1)
    # model_wd_det1.save('model_sd_save_61.h5')
    _ = projection(test_Y_det1, a_wd_det1,"SD",6,61)


def WD62():
    print("WD62 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_6(lev2_train, lev2_val,
                                                                                                   lev2_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    input_wide = tf.keras.layers.Input(shape=(6, 1))
    hidden11 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(input_wide)
    hidden22 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden11)

    input_deep = tf.keras.layers.Input(shape=(6, 1))
    hidden1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_deep)
    hidden2 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(hidden1)
    # hidden3 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden2)
    hidden4 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden2)
    hidden5 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden4)
    # hidden6 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden5)
    hidden7 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden5)
    hidden8 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden7)
    hidden9 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden8)
    concat = tf.keras.layers.concatenate([hidden22, hidden9])
    hidden10 = tf.keras.layers.Flatten(name='out')(concat)
    hidden111 = tf.keras.layers.Dense(128, activation='relu')(hidden10)
    output = tf.keras.layers.Dense(8, activation='softmax')(hidden111)
    model_wd_12 = tf.keras.models.Model(inputs=[input_wide, input_deep], outputs=[output])
    model_wd_det1 = model_wd_12
    model_wd_det1.compile(loss="sparse_categorical_crossentropy", optimizer='Nadam', metrics=['accuracy'])
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    history_wd1 = model_wd_det1.fit([train_X_det1, train_X_det1], train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                    callbacks=callback1, batch_size=500, epochs=1000, verbose=2)
    det_wd1 = model_wd_det1.predict([test_X_det1, test_X_det1])
    a_wd_det1 = np.argmax(det_wd1, axis=1)
    a_wd_det1 = a_wd_det1.reshape(-1, 1)
    wd_AC_det1 = accuracy_score(test_Y_det1, a_wd_det1)
    print('wd_AC_det1=', wd_AC_det1)
    # model_wd_det1.save('model_sd_save_62.h5')
    _ = projection(test_Y_det1, a_wd_det1,"SD",6,62)

def WD63():
    print("WD63 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_6(lev3_train, lev3_val,
                                                                                                   lev3_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    input_wide = tf.keras.layers.Input(shape=(6, 1))
    hidden11 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(input_wide)
    hidden22 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden11)

    input_deep = tf.keras.layers.Input(shape=(6, 1))
    hidden1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_deep)
    hidden2 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(hidden1)
    # hidden3 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden2)
    hidden4 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden2)
    hidden5 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden4)
    # hidden6 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden5)
    hidden7 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden5)
    hidden8 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden7)
    hidden9 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden8)
    concat = tf.keras.layers.concatenate([hidden22, hidden9])
    hidden10 = tf.keras.layers.Flatten(name='out')(concat)
    hidden111 = tf.keras.layers.Dense(128, activation='relu')(hidden10)
    output = tf.keras.layers.Dense(8, activation='softmax')(hidden111)
    model_wd_12 = tf.keras.models.Model(inputs=[input_wide, input_deep], outputs=[output])
    model_wd_det1 = model_wd_12
    model_wd_det1.compile(loss="sparse_categorical_crossentropy", optimizer='Nadam', metrics=['accuracy'])
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    history_wd1 = model_wd_det1.fit([train_X_det1, train_X_det1], train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                    callbacks=callback1, batch_size=500, epochs=1000, verbose=2)
    det_wd1 = model_wd_det1.predict([test_X_det1, test_X_det1])
    a_wd_det1 = np.argmax(det_wd1, axis=1)
    a_wd_det1 = a_wd_det1.reshape(-1, 1)
    wd_AC_det1 = accuracy_score(test_Y_det1, a_wd_det1)
    print('wd_AC_det1=', wd_AC_det1)
    # model_wd_det1.save('model_sd_save_63.h5')
    _ = projection(test_Y_det1, a_wd_det1,"SD",6,63)

def WD64():
    print("WD64 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_6(lev4_train, lev4_val,
                                                                                                   lev4_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    input_wide = tf.keras.layers.Input(shape=(6, 1))
    hidden11 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(input_wide)
    hidden22 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden11)

    input_deep = tf.keras.layers.Input(shape=(6, 1))
    hidden1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_deep)
    hidden2 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(hidden1)
    # hidden3 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden2)
    hidden4 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden2)
    hidden5 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden4)
    # hidden6 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden5)
    hidden7 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden5)
    hidden8 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden7)
    hidden9 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden8)
    concat = tf.keras.layers.concatenate([hidden22, hidden9])
    hidden10 = tf.keras.layers.Flatten(name='out')(concat)
    hidden111 = tf.keras.layers.Dense(128, activation='relu')(hidden10)
    output = tf.keras.layers.Dense(8, activation='softmax')(hidden111)
    model_wd_12 = tf.keras.models.Model(inputs=[input_wide, input_deep], outputs=[output])
    model_wd_det1 = model_wd_12
    model_wd_det1.compile(loss="sparse_categorical_crossentropy", optimizer='Nadam', metrics=['accuracy'])
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    history_wd1 = model_wd_det1.fit([train_X_det1, train_X_det1], train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                    callbacks=callback1, batch_size=500, epochs=1000, verbose=2)
    det_wd1 = model_wd_det1.predict([test_X_det1, test_X_det1])
    a_wd_det1 = np.argmax(det_wd1, axis=1)
    a_wd_det1 = a_wd_det1.reshape(-1, 1)
    wd_AC_det1 = accuracy_score(test_Y_det1, a_wd_det1)
    print('wd_AC_det1=', wd_AC_det1)
    # model_wd_det1.save('model_sd_save_64.h5')
    _ = projection(test_Y_det1, a_wd_det1,"SD",6,64)

def WD81():
    print("WD81 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_8(lev1_train, lev1_val,
                                                                                                   lev1_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    input_wide = tf.keras.layers.Input(shape=(8, 1))
    hidden11 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(input_wide)
    hidden22 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden11)

    input_deep = tf.keras.layers.Input(shape=(8, 1))
    hidden1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_deep)
    hidden2 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(hidden1)
    # hidden3 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden2)
    hidden4 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden2)
    hidden5 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden4)
    # hidden6 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden5)
    hidden7 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden5)
    hidden8 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden7)
    hidden9 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden8)
    concat = tf.keras.layers.concatenate([hidden22, hidden9])
    # concat1 = attention_3d_block2(concat)
    hidden10 = tf.keras.layers.Flatten(name='out')(concat)
    hidden111 = tf.keras.layers.Dense(128, activation='relu')(hidden10)
    output = tf.keras.layers.Dense(8, activation='softmax')(hidden111)
    model_wd_12 = tf.keras.models.Model(inputs=[input_wide, input_deep], outputs=[output])
    model_wd_det1 = model_wd_12
    model_wd_det1.compile(loss="sparse_categorical_crossentropy", optimizer='Nadam', metrics=['accuracy'])
    # model_wd_det1.compile(loss="sparse_categorical_crossentropy", optimizer='Nadam', metrics=['sparse_categorical_accuracy'])

    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    history = model_wd_det1.fit([train_X_det1, train_X_det1], train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                    callbacks=callback1, batch_size=500, epochs=1000, verbose=2)
    det_wd1 = model_wd_det1.predict([test_X_det1, test_X_det1])
    a_wd_det1 = np.argmax(det_wd1, axis=1)
    a_wd_det1 = a_wd_det1.reshape(-1, 1)
    ######################################################################
    # #sparse_categorical_accuracy
    # acc = history.history['sparse_categorical_accuracy']
    # val_acc = history.history['val_sparse_categorical_accuracy']
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    #
    # font2 = {'size': 12}
    # fig = plt.figure(figsize=(9, 4))
    #
    # plt.subplot(1, 2, 1)
    # plt.plot(acc, label='Training Accuracy')
    # plt.plot(val_acc, label='Validation Accuracy')
    # plt.xlabel("Epoch", font2)
    # plt.ylabel("Accuracy", font2)
    # plt.legend()
    #
    # # plt.subplot(1, 2, 2)
    # ax = fig.add_subplot(1, 2, 2)
    # ax.plot(loss, label='Training Loss')
    # ax.plot(val_loss, label='Validation Loss')
    # ax.set_xlabel("Epoch", font2)
    # ax.set_ylabel("Loss", font2)
    # # ax.yaxis.set_ticks_position('right')
    # # ax.yaxis.set_label_position('right')
    # plt.legend()
    # plt.suptitle('Accuracy and loss of training set and verification set of SD')
    # # plt.show()
    # plt.savefig("C:\\Users\dong\\Desktop\\SD-XGboost"+"Accuracy and loss of 8_1")
    # plt.close(1)
    ###########################################################################


    wd_AC_det1 = accuracy_score(test_Y_det1, a_wd_det1)
    print('wd_AC_det1=', wd_AC_det1)
    # model_wd_det1.save('model_sd_save_81.h5')
    # _ = projection(test_Y_det1, a_wd_det1,"SD",8,81)

def WD82():
    print("WD82 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_8(lev2_train, lev2_val,
                                                                                                   lev2_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    input_wide = tf.keras.layers.Input(shape=(8, 1))
    hidden11 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(input_wide)
    hidden22 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden11)

    input_deep = tf.keras.layers.Input(shape=(8, 1))
    hidden1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_deep)
    hidden2 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(hidden1)
    # hidden3 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden2)
    hidden4 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden2)
    hidden5 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden4)
    # hidden6 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden5)
    hidden7 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden5)
    hidden8 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden7)
    hidden9 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden8)
    concat = tf.keras.layers.concatenate([hidden22, hidden9])
    hidden10 = tf.keras.layers.Flatten(name='out')(concat)
    hidden111 = tf.keras.layers.Dense(128, activation='relu')(hidden10)
    output = tf.keras.layers.Dense(8, activation='softmax')(hidden111)
    model_wd_12 = tf.keras.models.Model(inputs=[input_wide, input_deep], outputs=[output])
    model_wd_det1 = model_wd_12
    model_wd_det1.compile(loss="sparse_categorical_crossentropy", optimizer='Nadam', metrics=['accuracy'])
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    history = model_wd_det1.fit([train_X_det1, train_X_det1], train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                    callbacks=callback1, batch_size=500, epochs=500, verbose=2)
    det_wd1 = model_wd_det1.predict([test_X_det1, test_X_det1])
    a_wd_det1 = np.argmax(det_wd1, axis=1)
    a_wd_det1 = a_wd_det1.reshape(-1, 1)
    ######################################################################
    # # sparse_categorical_accuracy
    # acc = history.history['sparse_categorical_accuracy']
    # val_acc = history.history['val_sparse_categorical_accuracy']
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    #
    # font2 = {'size': 12}
    # fig = plt.figure(figsize=(9, 4))
    #
    # plt.subplot(1, 2, 1)
    # plt.plot(acc, label='Training Accuracy')
    # plt.plot(val_acc, label='Validation Accuracy')
    # plt.xlabel("Epoch", font2)
    # plt.ylabel("Accuracy", font2)
    # plt.legend()
    #
    # # plt.subplot(1, 2, 2)
    # ax = fig.add_subplot(1, 2, 2)
    # ax.plot(loss, label='Training Loss')
    # ax.plot(val_loss, label='Validation Loss')
    # ax.set_xlabel("Epoch", font2)
    # ax.set_ylabel("Loss", font2)
    # # ax.yaxis.set_ticks_position('right')
    # # ax.yaxis.set_label_position('right')
    # plt.legend()
    # plt.suptitle('Accuracy and loss of training set and verification set of SD')
    # # plt.show()
    # plt.savefig("C:\\Users\dong\\Desktop\\SD-XGboost" + "Accuracy and loss of 8_2")
    # plt.close(1)

    ###########################################################################
    wd_AC_det1 = accuracy_score(test_Y_det1, a_wd_det1)
    print('wd_AC_det1=', wd_AC_det1)
    # model_wd_det1.save('model_sd_save_82.h5')
    _ = projection(test_Y_det1, a_wd_det1,"SD",8,82)

def WD83():
    print("WD83 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_8(lev3_train, lev3_val,
                                                                                                   lev3_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    input_wide = tf.keras.layers.Input(shape=(8, 1))
    hidden11 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(input_wide)
    hidden22 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden11)

    input_deep = tf.keras.layers.Input(shape=(8, 1))
    hidden1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_deep)
    hidden2 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(hidden1)
    # hidden3 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden2)
    hidden4 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden2)
    hidden5 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden4)
    # hidden6 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden5)
    hidden7 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden5)
    hidden8 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden7)
    hidden9 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden8)
    concat = tf.keras.layers.concatenate([hidden22, hidden9])
    hidden10 = tf.keras.layers.Flatten(name='out')(concat)
    hidden111 = tf.keras.layers.Dense(128, activation='relu')(hidden10)
    output = tf.keras.layers.Dense(8, activation='softmax')(hidden111)
    model_wd_12 = tf.keras.models.Model(inputs=[input_wide, input_deep], outputs=[output])
    model_wd_det1 = model_wd_12
    model_wd_det1.compile(loss="sparse_categorical_crossentropy", optimizer='Nadam', metrics=['accuracy'])
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    history = model_wd_det1.fit([train_X_det1, train_X_det1], train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                    callbacks=callback1, batch_size=500, epochs=500, verbose=2)
    det_wd1 = model_wd_det1.predict([test_X_det1, test_X_det1])
    a_wd_det1 = np.argmax(det_wd1, axis=1)
    a_wd_det1 = a_wd_det1.reshape(-1, 1)
    ######################################################################
    # # sparse_categorical_accuracy
    # acc = history.history['sparse_categorical_accuracy']
    # val_acc = history.history['val_sparse_categorical_accuracy']
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    #
    # font2 = {'size': 12}
    # fig = plt.figure(figsize=(9, 4))
    #
    # plt.subplot(1, 2, 1)
    # plt.plot(acc, label='Training Accuracy')
    # plt.plot(val_acc, label='Validation Accuracy')
    # plt.xlabel("Epoch", font2)
    # plt.ylabel("Accuracy", font2)
    # plt.legend()
    #
    # # plt.subplot(1, 2, 2)
    # ax = fig.add_subplot(1, 2, 2)
    # ax.plot(loss, label='Training Loss')
    # ax.plot(val_loss, label='Validation Loss')
    # ax.set_xlabel("Epoch", font2)
    # ax.set_ylabel("Loss", font2)
    # # ax.yaxis.set_ticks_position('right')
    # # ax.yaxis.set_label_position('right')
    # plt.legend()
    # plt.suptitle('Accuracy and loss of training set and verification set of SD')
    # # plt.show()
    # plt.savefig("C:\\Users\dong\\Desktop\\SD-XGboost" + "Accuracy and loss of 8_3")
    # plt.close(1)
    #
    # ###########################################################################
    wd_AC_det1 = accuracy_score(test_Y_det1, a_wd_det1)
    print('wd_AC_det1=', wd_AC_det1)
    # model_wd_det1.save('model_sd_save_83.h5')
    _ = projection(test_Y_det1, a_wd_det1,"SD",8,83)

def WD84():
    print("WD84 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_8(lev4_train, lev4_val,
                                                                                                   lev4_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    input_wide = tf.keras.layers.Input(shape=(8, 1))
    hidden11 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(input_wide)
    hidden22 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden11)

    input_deep = tf.keras.layers.Input(shape=(8, 1))
    hidden1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_deep)
    hidden2 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(hidden1)
    # hidden3 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden2)
    hidden4 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden2)
    hidden5 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden4)
    # hidden6 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden5)
    hidden7 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden5)
    hidden8 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden7)
    hidden9 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden8)
    concat = tf.keras.layers.concatenate([hidden22, hidden9])
    hidden10 = tf.keras.layers.Flatten(name='out')(concat)
    hidden111 = tf.keras.layers.Dense(128, activation='relu')(hidden10)
    output = tf.keras.layers.Dense(8, activation='softmax')(hidden111)
    model_wd_12 = tf.keras.models.Model(inputs=[input_wide, input_deep], outputs=[output])
    model_wd_det1 = model_wd_12
    model_wd_det1.compile(loss="sparse_categorical_crossentropy", optimizer='Nadam', metrics=['accuracy'])
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    history = model_wd_det1.fit([train_X_det1, train_X_det1], train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                    callbacks=callback1, batch_size=500, epochs=500, verbose=2)
    det_wd1 = model_wd_det1.predict([test_X_det1, test_X_det1])
    a_wd_det1 = np.argmax(det_wd1, axis=1)
    a_wd_det1 = a_wd_det1.reshape(-1, 1)
    # ######################################################################
    # # sparse_categorical_accuracy
    # acc = history.history['sparse_categorical_accuracy']
    # val_acc = history.history['val_sparse_categorical_accuracy']
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    #
    # font2 = {'size': 12}
    # fig = plt.figure(figsize=(9, 4))
    #
    # plt.subplot(1, 2, 1)
    # plt.plot(acc, label='Training Accuracy')
    # plt.plot(val_acc, label='Validation Accuracy')
    # plt.xlabel("Epoch", font2)
    # plt.ylabel("Accuracy", font2)
    # plt.legend()
    #
    # # plt.subplot(1, 2, 2)
    # ax = fig.add_subplot(1, 2, 2)
    # ax.plot(loss, label='Training Loss')
    # ax.plot(val_loss, label='Validation Loss')
    # ax.set_xlabel("Epoch", font2)
    # ax.set_ylabel("Loss", font2)
    # # ax.yaxis.set_ticks_position('right')
    # # ax.yaxis.set_label_position('right')
    # plt.legend()
    # plt.suptitle('Accuracy and loss of training set and verification set of SD')
    # # plt.show()
    # plt.savefig("C:\\Users\dong\\Desktop\\SD-XGboost" + "Accuracy and loss of 8_4")
    # plt.close(1)
    #
    # ###########################################################################
    wd_AC_det1 = accuracy_score(test_Y_det1, a_wd_det1)
    print('wd_AC_det1=', wd_AC_det1)
    # model_wd_det1.save('model_sd_save_84.h5')
    _ = projection(test_Y_det1, a_wd_det1,"SD",8,84)

def WD101():
    print("WD101 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_10(lev1_train, lev1_val,
                                                                                                   lev1_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    input_wide = tf.keras.layers.Input(shape=(10, 1))
    hidden11 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(input_wide)
    hidden22 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden11)

    input_deep = tf.keras.layers.Input(shape=(10, 1))
    hidden1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_deep)
    hidden2 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(hidden1)
    # hidden3 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden2)
    hidden4 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden2)
    hidden5 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden4)
    # hidden6 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden5)
    hidden7 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden5)
    hidden8 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden7)
    hidden9 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden8)
    concat = tf.keras.layers.concatenate([hidden22, hidden9])
    hidden10 = tf.keras.layers.Flatten(name='out')(concat)
    hidden111 = tf.keras.layers.Dense(128, activation='relu')(hidden10)
    output = tf.keras.layers.Dense(8, activation='softmax')(hidden111)
    model_wd_12 = tf.keras.models.Model(inputs=[input_wide, input_deep], outputs=[output])
    model_wd_det1 = model_wd_12
    model_wd_det1.compile(loss="sparse_categorical_crossentropy", optimizer='Nadam', metrics=['accuracy'])
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    history_wd1 = model_wd_det1.fit([train_X_det1, train_X_det1], train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                    callbacks=callback1, batch_size=500, epochs=1000, verbose=2)
    det_wd1 = model_wd_det1.predict([test_X_det1, test_X_det1])
    a_wd_det1 = np.argmax(det_wd1, axis=1)
    a_wd_det1 = a_wd_det1.reshape(-1, 1)
    wd_AC_det1 = accuracy_score(test_Y_det1, a_wd_det1)
    print('wd_AC_det1=', wd_AC_det1)
    # model_wd_det1.save('model_sd_save_101.h5')
    _ = projection(test_Y_det1, a_wd_det1,"SD",10,101)

def WD102():
    print("WD102 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_10(lev2_train, lev2_val,
                                                                                                   lev2_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    input_wide = tf.keras.layers.Input(shape=(10, 1))
    hidden11 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(input_wide)
    hidden22 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden11)

    input_deep = tf.keras.layers.Input(shape=(10, 1))
    hidden1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_deep)
    hidden2 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(hidden1)
    # hidden3 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden2)
    hidden4 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden2)
    hidden5 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden4)
    # hidden6 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden5)
    hidden7 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden5)
    hidden8 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden7)
    hidden9 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden8)
    concat = tf.keras.layers.concatenate([hidden22, hidden9])
    hidden10 = tf.keras.layers.Flatten(name='out')(concat)
    hidden111 = tf.keras.layers.Dense(128, activation='relu')(hidden10)
    output = tf.keras.layers.Dense(8, activation='softmax')(hidden111)
    model_wd_12 = tf.keras.models.Model(inputs=[input_wide, input_deep], outputs=[output])
    model_wd_det1 = model_wd_12
    model_wd_det1.compile(loss="sparse_categorical_crossentropy", optimizer='Nadam', metrics=['accuracy'])
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    history_wd1 = model_wd_det1.fit([train_X_det1, train_X_det1], train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                    callbacks=callback1, batch_size=500, epochs=1000, verbose=2)
    det_wd1 = model_wd_det1.predict([test_X_det1, test_X_det1])
    a_wd_det1 = np.argmax(det_wd1, axis=1)
    a_wd_det1 = a_wd_det1.reshape(-1, 1)
    wd_AC_det1 = accuracy_score(test_Y_det1, a_wd_det1)
    print('wd_AC_det1=', wd_AC_det1)
    # model_wd_det1.save('model_sd_save_102.h5')
    _ = projection(test_Y_det1, a_wd_det1,"SD",10,102)

def WD103():
    print("WD103 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_10(lev3_train, lev3_val,
                                                                                                   lev3_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    input_wide = tf.keras.layers.Input(shape=(10, 1))
    hidden11 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(input_wide)
    hidden22 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden11)

    input_deep = tf.keras.layers.Input(shape=(10, 1))
    hidden1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_deep)
    hidden2 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(hidden1)
    # hidden3 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden2)
    hidden4 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden2)
    hidden5 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden4)
    # hidden6 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden5)
    hidden7 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden5)
    hidden8 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden7)
    hidden9 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden8)
    concat = tf.keras.layers.concatenate([hidden22, hidden9])
    hidden10 = tf.keras.layers.Flatten(name='out')(concat)
    hidden111 = tf.keras.layers.Dense(128, activation='relu')(hidden10)
    output = tf.keras.layers.Dense(8, activation='softmax')(hidden111)
    model_wd_12 = tf.keras.models.Model(inputs=[input_wide, input_deep], outputs=[output])
    model_wd_det1 = model_wd_12
    model_wd_det1.compile(loss="sparse_categorical_crossentropy", optimizer='Nadam', metrics=['accuracy'])
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    history_wd1 = model_wd_det1.fit([train_X_det1, train_X_det1], train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                    callbacks=callback1, batch_size=500, epochs=1000, verbose=2)
    det_wd1 = model_wd_det1.predict([test_X_det1, test_X_det1])
    a_wd_det1 = np.argmax(det_wd1, axis=1)
    a_wd_det1 = a_wd_det1.reshape(-1, 1)
    wd_AC_det1 = accuracy_score(test_Y_det1, a_wd_det1)
    print('wd_AC_det1=', wd_AC_det1)
    # model_wd_det1.save('model_sd_save_103.h5')
    _ = projection(test_Y_det1, a_wd_det1,"SD",10,103)

def WD104():
    print("WD104 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_10(lev4_train, lev4_val,
                                                                                                   lev4_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    input_wide = tf.keras.layers.Input(shape=(10, 1))
    hidden11 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(input_wide)
    hidden22 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden11)

    input_deep = tf.keras.layers.Input(shape=(10, 1))
    hidden1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_deep)
    hidden2 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(hidden1)
    # hidden3 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden2)
    hidden4 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden2)
    hidden5 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden4)
    # hidden6 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden5)
    hidden7 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden5)
    hidden8 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden7)
    hidden9 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden8)
    concat = tf.keras.layers.concatenate([hidden22, hidden9])
    hidden10 = tf.keras.layers.Flatten(name='out')(concat)
    hidden111 = tf.keras.layers.Dense(128, activation='relu')(hidden10)
    output = tf.keras.layers.Dense(8, activation='softmax')(hidden111)
    model_wd_12 = tf.keras.models.Model(inputs=[input_wide, input_deep], outputs=[output])
    model_wd_det1 = model_wd_12
    model_wd_det1.compile(loss="sparse_categorical_crossentropy", optimizer='Nadam', metrics=['accuracy'])
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    history_wd1 = model_wd_det1.fit([train_X_det1, train_X_det1], train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                    callbacks=callback1, batch_size=500, epochs=1000, verbose=2)
    det_wd1 = model_wd_det1.predict([test_X_det1, test_X_det1])
    a_wd_det1 = np.argmax(det_wd1, axis=1)
    a_wd_det1 = a_wd_det1.reshape(-1, 1)
    wd_AC_det1 = accuracy_score(test_Y_det1, a_wd_det1)
    print('wd_AC_det1=', wd_AC_det1)
    # model_wd_det1.save('model_sd_save_104.h5')
    _ = projection(test_Y_det1, a_wd_det1,"SD",10,104)

def WD121():
    print("WD121 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_12(lev1_train, lev1_val,
                                                                                                   lev1_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    input_wide = tf.keras.layers.Input(shape=(12, 1))
    hidden11 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(input_wide)
    hidden22 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden11)

    input_deep = tf.keras.layers.Input(shape=(12, 1))
    hidden1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_deep)
    hidden2 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(hidden1)
    # hidden3 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden2)
    hidden4 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden2)
    hidden5 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden4)
    # hidden6 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden5)
    hidden7 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden5)
    hidden8 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden7)
    hidden9 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden8)
    concat = tf.keras.layers.concatenate([hidden22, hidden9])
    hidden10 = tf.keras.layers.Flatten(name='out')(concat)
    hidden111 = tf.keras.layers.Dense(128, activation='relu')(hidden10)
    output = tf.keras.layers.Dense(8, activation='softmax')(hidden111)
    model_wd_12 = tf.keras.models.Model(inputs=[input_wide, input_deep], outputs=[output])
    model_wd_det1 = model_wd_12
    model_wd_det1.compile(loss="sparse_categorical_crossentropy", optimizer='Nadam', metrics=['accuracy'])
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    history_wd1 = model_wd_det1.fit([train_X_det1, train_X_det1], train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                    callbacks=callback1, batch_size=500, epochs=1000, verbose=2)
    det_wd1 = model_wd_det1.predict([test_X_det1, test_X_det1])
    a_wd_det1 = np.argmax(det_wd1, axis=1)
    a_wd_det1 = a_wd_det1.reshape(-1, 1)
    wd_AC_det1 = accuracy_score(test_Y_det1, a_wd_det1)
    print('wd_AC_det1=', wd_AC_det1)
    model_wd_det1.save('model_sd_save_121.h5')
    # _ = projection(test_Y_det1, a_wd_det1,"SD",12,121)

def WD122():
    print("WD122 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_12(lev2_train, lev2_val,
                                                                                                   lev2_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    input_wide = tf.keras.layers.Input(shape=(12, 1))
    hidden11 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(input_wide)
    hidden22 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden11)

    input_deep = tf.keras.layers.Input(shape=(12, 1))
    hidden1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_deep)
    hidden2 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(hidden1)
    # hidden3 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden2)
    hidden4 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden2)
    hidden5 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden4)
    # hidden6 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden5)
    hidden7 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden5)
    hidden8 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden7)
    hidden9 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden8)
    concat = tf.keras.layers.concatenate([hidden22, hidden9])
    hidden10 = tf.keras.layers.Flatten(name='out')(concat)
    hidden111 = tf.keras.layers.Dense(128, activation='relu')(hidden10)
    output = tf.keras.layers.Dense(8, activation='softmax')(hidden111)
    model_wd_12 = tf.keras.models.Model(inputs=[input_wide, input_deep], outputs=[output])
    model_wd_det1 = model_wd_12
    model_wd_det1.compile(loss="sparse_categorical_crossentropy", optimizer='Nadam', metrics=['accuracy'])
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    history_wd1 = model_wd_det1.fit([train_X_det1, train_X_det1], train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                    callbacks=callback1, batch_size=500, epochs=1000, verbose=2)
    det_wd1 = model_wd_det1.predict([test_X_det1, test_X_det1])
    a_wd_det1 = np.argmax(det_wd1, axis=1)
    a_wd_det1 = a_wd_det1.reshape(-1, 1)
    wd_AC_det1 = accuracy_score(test_Y_det1, a_wd_det1)
    print('wd_AC_det1=', wd_AC_det1)
    model_wd_det1.save('model_sd_save_122.h5')
    # _ = projection(test_Y_det1, a_wd_det1,"SD",12,122)

def WD123():
    print("WD123 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_12(lev3_train, lev3_val,
                                                                                                   lev3_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    input_wide = tf.keras.layers.Input(shape=(12, 1))
    hidden11 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(input_wide)
    hidden22 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden11)

    input_deep = tf.keras.layers.Input(shape=(12, 1))
    hidden1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_deep)
    hidden2 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(hidden1)
    # hidden3 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden2)
    hidden4 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden2)
    hidden5 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden4)
    # hidden6 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden5)
    hidden7 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden5)
    hidden8 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden7)
    hidden9 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden8)
    concat = tf.keras.layers.concatenate([hidden22, hidden9])
    hidden10 = tf.keras.layers.Flatten(name='out')(concat)
    hidden111 = tf.keras.layers.Dense(128, activation='relu')(hidden10)
    output = tf.keras.layers.Dense(8, activation='softmax')(hidden111)
    model_wd_12 = tf.keras.models.Model(inputs=[input_wide, input_deep], outputs=[output])
    model_wd_det1 = model_wd_12
    model_wd_det1.compile(loss="sparse_categorical_crossentropy", optimizer='Nadam', metrics=['accuracy'])
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    history_wd1 = model_wd_det1.fit([train_X_det1, train_X_det1], train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                    callbacks=callback1, batch_size=500, epochs=1000, verbose=2)
    det_wd1 = model_wd_det1.predict([test_X_det1, test_X_det1])
    a_wd_det1 = np.argmax(det_wd1, axis=1)
    a_wd_det1 = a_wd_det1.reshape(-1, 1)
    wd_AC_det1 = accuracy_score(test_Y_det1, a_wd_det1)
    print('wd_AC_det1=', wd_AC_det1)
    model_wd_det1.save('model_sd_save_123.h5')
    # _ = projection(test_Y_det1, a_wd_det1,"SD",12,123)

def WD124():
    print("WD124 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_12(lev4_train, lev4_val,
                                                                                                   lev4_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    input_wide = tf.keras.layers.Input(shape=(12, 1))
    hidden11 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(input_wide)
    hidden22 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden11)

    input_deep = tf.keras.layers.Input(shape=(12, 1))
    hidden1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_deep)
    hidden2 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(hidden1)
    # hidden3 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden2)
    hidden4 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden2)
    hidden5 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden4)
    # hidden6 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden5)
    hidden7 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden5)
    hidden8 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden7)
    hidden9 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden8)
    concat = tf.keras.layers.concatenate([hidden22, hidden9])
    hidden10 = tf.keras.layers.Flatten(name='out')(concat)
    hidden111 = tf.keras.layers.Dense(128, activation='relu')(hidden10)
    output = tf.keras.layers.Dense(8, activation='softmax')(hidden111)
    model_wd_12 = tf.keras.models.Model(inputs=[input_wide, input_deep], outputs=[output])
    model_wd_det1 = model_wd_12
    model_wd_det1.compile(loss="sparse_categorical_crossentropy", optimizer='Nadam', metrics=['accuracy'])
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    history_wd1 = model_wd_det1.fit([train_X_det1, train_X_det1], train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                    callbacks=callback1, batch_size=500, epochs=1000, verbose=2)
    det_wd1 = model_wd_det1.predict([test_X_det1, test_X_det1])
    a_wd_det1 = np.argmax(det_wd1, axis=1)
    a_wd_det1 = a_wd_det1.reshape(-1, 1)
    wd_AC_det1 = accuracy_score(test_Y_det1, a_wd_det1)
    print('wd_AC_det1=', wd_AC_det1)
    model_wd_det1.save('model_sd_save_124.h5')
    # _ = projection(test_Y_det1, a_wd_det1,"SD",12,124)


def AWD61():
    print("WD61 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_6(lev1_train, lev1_val,
                                                                                                   lev1_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    input_wide = tf.keras.layers.Input(shape=(6, 1))
    hidden11 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(input_wide)
    hidden22 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden11)

    input_deep = tf.keras.layers.Input(shape=(6, 1))
    hidden1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_deep)
    hidden2 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(hidden1)
    # hidden3 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden2)
    hidden4 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden2)
    hidden5 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden4)
    # hidden6 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden5)
    hidden7 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden5)
    hidden8 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden7)
    hidden9 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden8)
    concat = tf.keras.layers.concatenate([hidden22, hidden9])
    concat1 = attention_3d_block2(concat)
    hidden10 = tf.keras.layers.Flatten(name='out')(concat1)
    hidden111 = tf.keras.layers.Dense(128, activation='relu')(hidden10)
    output = tf.keras.layers.Dense(8, activation='softmax')(hidden111)
    model_wd_12 = tf.keras.models.Model(inputs=[input_wide, input_deep], outputs=[output])
    model_wd_det1 = model_wd_12
    model_wd_det1.compile(loss="sparse_categorical_crossentropy", optimizer='Nadam', metrics=['accuracy'])
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    history_wd1 = model_wd_det1.fit([train_X_det1, train_X_det1], train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                    callbacks=callback1, batch_size=500, epochs=1000, verbose=2)
    det_wd1 = model_wd_det1.predict([test_X_det1, test_X_det1])
    a_wd_det1 = np.argmax(det_wd1, axis=1)
    a_wd_det1 = a_wd_det1.reshape(-1, 1)
    wd_AC_det1 = accuracy_score(test_Y_det1, a_wd_det1)
    print('wd_AC_det1=', wd_AC_det1)
    model_wd_det1.save('model_asd_save_61.h5')
    _ = projection(test_Y_det1, a_wd_det1,"SD",6,61)
    ##############################################################################################

def AWD62():
    print("WD62 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_6(lev2_train, lev2_val,
                                                                                                   lev2_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    input_wide = tf.keras.layers.Input(shape=(6, 1))
    hidden11 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(input_wide)
    hidden22 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden11)

    input_deep = tf.keras.layers.Input(shape=(6, 1))
    hidden1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_deep)
    hidden2 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(hidden1)
    # hidden3 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden2)
    hidden4 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden2)
    hidden5 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden4)
    # hidden6 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden5)
    hidden7 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden5)
    hidden8 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden7)
    hidden9 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden8)
    concat = tf.keras.layers.concatenate([hidden22, hidden9])
    concat1 = attention_3d_block2(concat)
    hidden10 = tf.keras.layers.Flatten(name='out')(concat1)
    hidden111 = tf.keras.layers.Dense(128, activation='relu')(hidden10)
    output = tf.keras.layers.Dense(8, activation='softmax')(hidden111)
    model_wd_12 = tf.keras.models.Model(inputs=[input_wide, input_deep], outputs=[output])
    model_wd_det1 = model_wd_12
    model_wd_det1.compile(loss="sparse_categorical_crossentropy", optimizer='Nadam', metrics=['accuracy'])
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    history_wd1 = model_wd_det1.fit([train_X_det1, train_X_det1], train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                    callbacks=callback1, batch_size=500, epochs=1000, verbose=2)
    det_wd1 = model_wd_det1.predict([test_X_det1, test_X_det1])
    a_wd_det1 = np.argmax(det_wd1, axis=1)
    a_wd_det1 = a_wd_det1.reshape(-1, 1)
    wd_AC_det1 = accuracy_score(test_Y_det1, a_wd_det1)
    print('wd_AC_det1=', wd_AC_det1)
    model_wd_det1.save('model_asd_save_62.h5')
    _ = projection(test_Y_det1, a_wd_det1,"SD",6,62)

def AWD63():
    print("WD63 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_6(lev3_train, lev3_val,
                                                                                                   lev3_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    input_wide = tf.keras.layers.Input(shape=(6, 1))
    hidden11 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(input_wide)
    hidden22 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden11)

    input_deep = tf.keras.layers.Input(shape=(6, 1))
    hidden1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_deep)
    hidden2 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(hidden1)
    # hidden3 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden2)
    hidden4 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden2)
    hidden5 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden4)
    # hidden6 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden5)
    hidden7 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden5)
    hidden8 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden7)
    hidden9 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden8)
    concat = tf.keras.layers.concatenate([hidden22, hidden9])
    concat1 = attention_3d_block2(concat)
    hidden10 = tf.keras.layers.Flatten(name='out')(concat1)
    hidden111 = tf.keras.layers.Dense(128, activation='relu')(hidden10)
    output = tf.keras.layers.Dense(8, activation='softmax')(hidden111)
    model_wd_12 = tf.keras.models.Model(inputs=[input_wide, input_deep], outputs=[output])
    model_wd_det1 = model_wd_12
    model_wd_det1.compile(loss="sparse_categorical_crossentropy", optimizer='Nadam', metrics=['accuracy'])
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    history_wd1 = model_wd_det1.fit([train_X_det1, train_X_det1], train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                    callbacks=callback1, batch_size=500, epochs=1000, verbose=2)
    det_wd1 = model_wd_det1.predict([test_X_det1, test_X_det1])
    a_wd_det1 = np.argmax(det_wd1, axis=1)
    a_wd_det1 = a_wd_det1.reshape(-1, 1)
    wd_AC_det1 = accuracy_score(test_Y_det1, a_wd_det1)
    print('wd_AC_det1=', wd_AC_det1)
    model_wd_det1.save('model_asd_save_63.h5')
    _ = projection(test_Y_det1, a_wd_det1,"SD",6,63)

def AWD64():
    print("WD64 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_6(lev4_train, lev4_val,
                                                                                                   lev4_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    input_wide = tf.keras.layers.Input(shape=(6, 1))
    hidden11 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(input_wide)
    hidden22 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden11)

    input_deep = tf.keras.layers.Input(shape=(6, 1))
    hidden1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_deep)
    hidden2 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(hidden1)
    # hidden3 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden2)
    hidden4 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden2)
    hidden5 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden4)
    # hidden6 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden5)
    hidden7 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden5)
    hidden8 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden7)
    hidden9 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden8)
    concat = tf.keras.layers.concatenate([hidden22, hidden9])
    concat1 = attention_3d_block2(concat)
    hidden10 = tf.keras.layers.Flatten(name='out')(concat1)
    hidden111 = tf.keras.layers.Dense(128, activation='relu')(hidden10)
    output = tf.keras.layers.Dense(8, activation='softmax')(hidden111)
    model_wd_12 = tf.keras.models.Model(inputs=[input_wide, input_deep], outputs=[output])
    model_wd_det1 = model_wd_12
    model_wd_det1.compile(loss="sparse_categorical_crossentropy", optimizer='Nadam', metrics=['accuracy'])
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    history_wd1 = model_wd_det1.fit([train_X_det1, train_X_det1], train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                    callbacks=callback1, batch_size=500, epochs=1000, verbose=2)
    det_wd1 = model_wd_det1.predict([test_X_det1, test_X_det1])
    a_wd_det1 = np.argmax(det_wd1, axis=1)
    a_wd_det1 = a_wd_det1.reshape(-1, 1)
    wd_AC_det1 = accuracy_score(test_Y_det1, a_wd_det1)
    print('wd_AC_det1=', wd_AC_det1)
    model_wd_det1.save('model_asd_save_64.h5')
    _ = projection(test_Y_det1, a_wd_det1,"SD",6,64)

def AWD81():
    print("WD81 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_8(lev1_train, lev1_val,
                                                                                                   lev1_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    input_wide = tf.keras.layers.Input(shape=(8, 1))
    hidden11 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(input_wide)
    hidden22 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden11)

    input_deep = tf.keras.layers.Input(shape=(8, 1))
    hidden1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_deep)
    hidden2 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(hidden1)
    # hidden3 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden2)
    hidden4 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden2)
    hidden5 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden4)
    # hidden6 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden5)
    hidden7 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden5)
    hidden8 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden7)
    hidden9 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden8)
    concat = tf.keras.layers.concatenate([hidden22, hidden9])
    concat1 = attention_3d_block2(concat)
    hidden10 = tf.keras.layers.Flatten(name='out')(concat1)
    hidden111 = tf.keras.layers.Dense(128, activation='relu')(hidden10)
    output = tf.keras.layers.Dense(8, activation='softmax')(hidden111)
    model_wd_12 = tf.keras.models.Model(inputs=[input_wide, input_deep], outputs=[output])
    model_wd_det1 = model_wd_12
    model_wd_det1.compile(loss="sparse_categorical_crossentropy", optimizer='Nadam', metrics=['accuracy'])
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    history_wd1 = model_wd_det1.fit([train_X_det1, train_X_det1], train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                    callbacks=callback1, batch_size=500, epochs=1000, verbose=2)
    det_wd1 = model_wd_det1.predict([test_X_det1, test_X_det1])
    a_wd_det1 = np.argmax(det_wd1, axis=1)
    a_wd_det1 = a_wd_det1.reshape(-1, 1)
    wd_AC_det1 = accuracy_score(test_Y_det1, a_wd_det1)
    print('wd_AC_det1=', wd_AC_det1)
    model_wd_det1.save('model_asd_save_81.h5')
    _ = projection(test_Y_det1, a_wd_det1,"SD",8,81)

def AWD82():
    print("WD82 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_8(lev2_train, lev2_val,
                                                                                                   lev2_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    input_wide = tf.keras.layers.Input(shape=(8, 1))
    hidden11 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(input_wide)
    hidden22 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden11)

    input_deep = tf.keras.layers.Input(shape=(8, 1))
    hidden1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_deep)
    hidden2 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(hidden1)
    # hidden3 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden2)
    hidden4 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden2)
    hidden5 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden4)
    # hidden6 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden5)
    hidden7 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden5)
    hidden8 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden7)
    hidden9 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden8)
    concat = tf.keras.layers.concatenate([hidden22, hidden9])
    concat1 = attention_3d_block2(concat)
    hidden10 = tf.keras.layers.Flatten(name='out')(concat1)
    hidden111 = tf.keras.layers.Dense(128, activation='relu')(hidden10)
    output = tf.keras.layers.Dense(8, activation='softmax')(hidden111)
    model_wd_12 = tf.keras.models.Model(inputs=[input_wide, input_deep], outputs=[output])
    model_wd_det1 = model_wd_12
    model_wd_det1.compile(loss="sparse_categorical_crossentropy", optimizer='Nadam', metrics=['accuracy'])
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    history_wd1 = model_wd_det1.fit([train_X_det1, train_X_det1], train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                    callbacks=callback1, batch_size=500, epochs=1000, verbose=2)
    det_wd1 = model_wd_det1.predict([test_X_det1, test_X_det1])
    a_wd_det1 = np.argmax(det_wd1, axis=1)
    a_wd_det1 = a_wd_det1.reshape(-1, 1)
    wd_AC_det1 = accuracy_score(test_Y_det1, a_wd_det1)
    print('wd_AC_det1=', wd_AC_det1)
    model_wd_det1.save('model_asd_save_82.h5')
    _ = projection(test_Y_det1, a_wd_det1,"SD",8,82)

def AWD83():
    print("WD83 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_8(lev3_train, lev3_val,
                                                                                                   lev3_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    input_wide = tf.keras.layers.Input(shape=(8, 1))
    hidden11 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(input_wide)
    hidden22 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden11)

    input_deep = tf.keras.layers.Input(shape=(8, 1))
    hidden1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_deep)
    hidden2 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(hidden1)
    # hidden3 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden2)
    hidden4 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden2)
    hidden5 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden4)
    # hidden6 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden5)
    hidden7 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden5)
    hidden8 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden7)
    hidden9 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden8)
    concat = tf.keras.layers.concatenate([hidden22, hidden9])
    concat1 = attention_3d_block2(concat)
    hidden10 = tf.keras.layers.Flatten(name='out')(concat1)
    hidden111 = tf.keras.layers.Dense(128, activation='relu')(hidden10)
    output = tf.keras.layers.Dense(8, activation='softmax')(hidden111)
    model_wd_12 = tf.keras.models.Model(inputs=[input_wide, input_deep], outputs=[output])
    model_wd_det1 = model_wd_12
    model_wd_det1.compile(loss="sparse_categorical_crossentropy", optimizer='Nadam', metrics=['accuracy'])
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    history_wd1 = model_wd_det1.fit([train_X_det1, train_X_det1], train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                    callbacks=callback1, batch_size=500, epochs=1000, verbose=2)
    det_wd1 = model_wd_det1.predict([test_X_det1, test_X_det1])
    a_wd_det1 = np.argmax(det_wd1, axis=1)
    a_wd_det1 = a_wd_det1.reshape(-1, 1)
    wd_AC_det1 = accuracy_score(test_Y_det1, a_wd_det1)
    print('wd_AC_det1=', wd_AC_det1)
    model_wd_det1.save('model_asd_save_83.h5')
    _ = projection(test_Y_det1, a_wd_det1,"SD",8,83)

def AWD84():
    print("WD84 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_8(lev4_train, lev4_val,
                                                                                                   lev4_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    input_wide = tf.keras.layers.Input(shape=(8, 1))
    hidden11 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(input_wide)
    hidden22 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden11)

    input_deep = tf.keras.layers.Input(shape=(8, 1))
    hidden1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_deep)
    hidden2 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(hidden1)
    # hidden3 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden2)
    hidden4 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden2)
    hidden5 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden4)
    # hidden6 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden5)
    hidden7 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden5)
    hidden8 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden7)
    hidden9 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden8)
    concat = tf.keras.layers.concatenate([hidden22, hidden9])
    concat1 = attention_3d_block2(concat)
    hidden10 = tf.keras.layers.Flatten(name='out')(concat1)
    hidden111 = tf.keras.layers.Dense(128, activation='relu')(hidden10)
    output = tf.keras.layers.Dense(8, activation='softmax')(hidden111)
    model_wd_12 = tf.keras.models.Model(inputs=[input_wide, input_deep], outputs=[output])
    model_wd_det1 = model_wd_12
    model_wd_det1.compile(loss="sparse_categorical_crossentropy", optimizer='Nadam', metrics=['accuracy'])
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    history_wd1 = model_wd_det1.fit([train_X_det1, train_X_det1], train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                    callbacks=callback1, batch_size=500, epochs=1000, verbose=2)
    det_wd1 = model_wd_det1.predict([test_X_det1, test_X_det1])
    a_wd_det1 = np.argmax(det_wd1, axis=1)
    a_wd_det1 = a_wd_det1.reshape(-1, 1)
    wd_AC_det1 = accuracy_score(test_Y_det1, a_wd_det1)
    print('wd_AC_det1=', wd_AC_det1)
    model_wd_det1.save('model_asd_save_84.h5')
    _ = projection(test_Y_det1, a_wd_det1,"SD",8,84)

def AWD101():
    print("WD101 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_10(lev1_train, lev1_val,
                                                                                                   lev1_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    input_wide = tf.keras.layers.Input(shape=(10, 1))
    hidden11 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(input_wide)
    hidden22 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden11)

    input_deep = tf.keras.layers.Input(shape=(10, 1))
    hidden1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_deep)
    hidden2 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(hidden1)
    # hidden3 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden2)
    hidden4 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden2)
    hidden5 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden4)
    # hidden6 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden5)
    hidden7 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden5)
    hidden8 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden7)
    hidden9 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden8)
    concat = tf.keras.layers.concatenate([hidden22, hidden9])
    concat1 = attention_3d_block2(concat)
    hidden10 = tf.keras.layers.Flatten(name='out')(concat1)
    hidden111 = tf.keras.layers.Dense(128, activation='relu')(hidden10)
    output = tf.keras.layers.Dense(8, activation='softmax')(hidden111)
    model_wd_12 = tf.keras.models.Model(inputs=[input_wide, input_deep], outputs=[output])
    model_wd_det1 = model_wd_12
    model_wd_det1.compile(loss="sparse_categorical_crossentropy", optimizer='Nadam', metrics=['accuracy'])
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    history_wd1 = model_wd_det1.fit([train_X_det1, train_X_det1], train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                    callbacks=callback1, batch_size=500, epochs=1000, verbose=2)
    det_wd1 = model_wd_det1.predict([test_X_det1, test_X_det1])
    a_wd_det1 = np.argmax(det_wd1, axis=1)
    a_wd_det1 = a_wd_det1.reshape(-1, 1)
    wd_AC_det1 = accuracy_score(test_Y_det1, a_wd_det1)
    print('wd_AC_det1=', wd_AC_det1)
    model_wd_det1.save('model_asd_save_101.h5')
    _ = projection(test_Y_det1, a_wd_det1,"SD",10,101)

def AWD102():
    print("WD102 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_10(lev2_train, lev2_val,
                                                                                                   lev2_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    input_wide = tf.keras.layers.Input(shape=(10, 1))
    hidden11 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(input_wide)
    hidden22 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden11)

    input_deep = tf.keras.layers.Input(shape=(10, 1))
    hidden1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_deep)
    hidden2 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(hidden1)
    # hidden3 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden2)
    hidden4 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden2)
    hidden5 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden4)
    # hidden6 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden5)
    hidden7 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden5)
    hidden8 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden7)
    hidden9 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden8)
    concat = tf.keras.layers.concatenate([hidden22, hidden9])
    concat1 = attention_3d_block2(concat)
    hidden10 = tf.keras.layers.Flatten(name='out')(concat1)
    hidden111 = tf.keras.layers.Dense(128, activation='relu')(hidden10)
    output = tf.keras.layers.Dense(8, activation='softmax')(hidden111)
    model_wd_12 = tf.keras.models.Model(inputs=[input_wide, input_deep], outputs=[output])
    model_wd_det1 = model_wd_12
    model_wd_det1.compile(loss="sparse_categorical_crossentropy", optimizer='Nadam', metrics=['accuracy'])
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    history_wd1 = model_wd_det1.fit([train_X_det1, train_X_det1], train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                    callbacks=callback1, batch_size=500, epochs=1000, verbose=2)
    det_wd1 = model_wd_det1.predict([test_X_det1, test_X_det1])
    a_wd_det1 = np.argmax(det_wd1, axis=1)
    a_wd_det1 = a_wd_det1.reshape(-1, 1)
    wd_AC_det1 = accuracy_score(test_Y_det1, a_wd_det1)
    print('wd_AC_det1=', wd_AC_det1)
    model_wd_det1.save('model_asd_save_102.h5')
    _ = projection(test_Y_det1, a_wd_det1,"SD",10,102)

def AWD103():
    print("WD103 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_10(lev3_train, lev3_val,
                                                                                                   lev3_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    input_wide = tf.keras.layers.Input(shape=(10, 1))
    hidden11 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(input_wide)
    hidden22 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden11)

    input_deep = tf.keras.layers.Input(shape=(10, 1))
    hidden1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_deep)
    hidden2 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(hidden1)
    # hidden3 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden2)
    hidden4 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden2)
    hidden5 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden4)
    # hidden6 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden5)
    hidden7 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden5)
    hidden8 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden7)
    hidden9 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden8)
    concat = tf.keras.layers.concatenate([hidden22, hidden9])
    concat1 = attention_3d_block2(concat)
    hidden10 = tf.keras.layers.Flatten(name='out')(concat1)
    hidden111 = tf.keras.layers.Dense(128, activation='relu')(hidden10)
    output = tf.keras.layers.Dense(8, activation='softmax')(hidden111)
    model_wd_12 = tf.keras.models.Model(inputs=[input_wide, input_deep], outputs=[output])
    model_wd_det1 = model_wd_12
    model_wd_det1.compile(loss="sparse_categorical_crossentropy", optimizer='Nadam', metrics=['accuracy'])
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    history_wd1 = model_wd_det1.fit([train_X_det1, train_X_det1], train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                    callbacks=callback1, batch_size=500, epochs=1000, verbose=2)
    det_wd1 = model_wd_det1.predict([test_X_det1, test_X_det1])
    a_wd_det1 = np.argmax(det_wd1, axis=1)
    a_wd_det1 = a_wd_det1.reshape(-1, 1)
    wd_AC_det1 = accuracy_score(test_Y_det1, a_wd_det1)
    print('wd_AC_det1=', wd_AC_det1)
    model_wd_det1.save('model_asd_save_103.h5')
    _ = projection(test_Y_det1, a_wd_det1,"SD",10,103)

def AWD104():
    print("WD104 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_10(lev4_train, lev4_val,
                                                                                                   lev4_test)
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    print('train_X_det1.shape', train_X_det1.shape)
    print('train_Y_det1.shape:', train_Y_det1.shape)
    print('test_X_det1.shape', test_X_det1.shape)
    print('test_Y_det1.shape:', test_Y_det1.shape)
    input_wide = tf.keras.layers.Input(shape=(10, 1))
    hidden11 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(input_wide)
    hidden22 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden11)

    input_deep = tf.keras.layers.Input(shape=(10, 1))
    hidden1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_deep)
    hidden2 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(hidden1)
    # hidden3 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden2)
    hidden4 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden2)
    hidden5 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden4)
    # hidden6 = tf.keras.layers.MaxPool1D(pool_size = 2)(hidden5)
    hidden7 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden5)
    hidden8 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden7)
    hidden9 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden8)
    concat = tf.keras.layers.concatenate([hidden22, hidden9])
    concat1 = attention_3d_block2(concat)
    hidden10 = tf.keras.layers.Flatten(name='out')(concat1)
    hidden111 = tf.keras.layers.Dense(128, activation='relu')(hidden10)
    output = tf.keras.layers.Dense(8, activation='softmax')(hidden111)
    model_wd_12 = tf.keras.models.Model(inputs=[input_wide, input_deep], outputs=[output])
    model_wd_det1 = model_wd_12
    model_wd_det1.compile(loss="sparse_categorical_crossentropy", optimizer='Nadam', metrics=['accuracy'])
    callback1 = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]
    history_wd1 = model_wd_det1.fit([train_X_det1, train_X_det1], train_Y_det1,
                   validation_data = ([val_X_det1,val_X_det1],val_Y_det1),
                                    callbacks=callback1, batch_size=500, epochs=1000, verbose=2)
    det_wd1 = model_wd_det1.predict([test_X_det1, test_X_det1])
    a_wd_det1 = np.argmax(det_wd1, axis=1)
    a_wd_det1 = a_wd_det1.reshape(-1, 1)
    wd_AC_det1 = accuracy_score(test_Y_det1, a_wd_det1)
    print('wd_AC_det1=', wd_AC_det1)
    model_wd_det1.save('model_asd_save_104.h5')
    _ = projection(test_Y_det1, a_wd_det1,"SD",10,104)


def WD_Model_61():
    print("WD_Model_61 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_6(lev1_train, lev1_val, lev1_test)
    model_sd_save = tf.keras.models.load_model('model_sd_save_61.h5')
    model_feat = Model(inputs = model_sd_save.input,outputs=model_sd_save.get_layer(name='out').output)
    feat_train = model_feat.predict([train_X_det1,train_X_det1])
    feat_test = model_feat.predict([test_X_det1,test_X_det1])

    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': 10,
        'gamma': 0.1,
        'max_depth': 14,
        'lambda': 2,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'silent': 1,
        'eta': 0.5,
        'seed': 1000,
        'nthread': 4,
    }
    dtrain = xgb.DMatrix(feat_train, train_Y_det1.ravel())
    num_rounds = 20
    plst = list(params.items())
    xgb1 = xgb.train(plst, dtrain, num_rounds)
    dtest = xgb.DMatrix(feat_test)
    xgb1_p_temp = xgb1.predict(dtest)
    xgb1_p = xgb1_p_temp.reshape(-1, 1)
    xgb1_acc = accuracy_score(xgb1_p, test_Y_det1)
    print('sd_xgb_acc=', xgb1_acc)
    _ = projection(test_Y_det1, xgb1_p,"SD_XGboost",6,61)

    #DecentionTree
    sd_dec = DecisionTreeClassifier(criterion="gini",max_features=6,min_samples_split=6)
    sd_dec.fit(feat_train, train_Y_det1.ravel())
    sd_dec_p_temp = sd_dec.predict(feat_test)
    sd_dec_p = sd_dec_p_temp.reshape(-1, 1)
    sd_dec_acc = accuracy_score(sd_dec_p, test_Y_det1)
    print('sd_dec_acc=', sd_dec_acc)
    # #knn
    knn = KNeighborsClassifier(weights='distance',p=2,leaf_size=50)
    knn.fit(feat_train, train_Y_det1.ravel())
    knn_predict = knn.predict(feat_test)
    np.set_printoptions(threshold=100000000)
    KNN_AC = accuracy_score(test_Y_det1, knn_predict)
    print("sd_knn_acc:", KNN_AC)
    _ = projection(test_Y_det1, knn_predict,"SD_KNN",6,61)
    # #LR
    classifier = LogisticRegression(random_state=5,solver='sag',C=10,)
    classifier.fit(feat_train, train_Y_det1.ravel())
    lg_predict = classifier.predict(feat_test)
    LG_AC = accuracy_score(test_Y_det1, lg_predict)
    print("sd_lr_acc:", LG_AC)

def WD_Model_62():
    print("WD_Model_62 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_6(lev2_train, lev2_val, lev2_test)
    model_sd_save = tf.keras.models.load_model('model_sd_save_62.h5')
    model_feat = Model(inputs = model_sd_save.input,outputs=model_sd_save.get_layer(name='out').output)
    feat_train = model_feat.predict([train_X_det1,train_X_det1])
    feat_test = model_feat.predict([test_X_det1,test_X_det1])

    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': 10,
        'gamma': 0.1,
        'max_depth': 14,
        'lambda': 2,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'silent': 1,
        'eta': 0.5,
        'seed': 1000,
        'nthread': 4,
    }
    dtrain = xgb.DMatrix(feat_train, train_Y_det1.ravel())
    num_rounds = 20
    plst = list(params.items())
    xgb1 = xgb.train(plst, dtrain, num_rounds)
    dtest = xgb.DMatrix(feat_test)
    xgb1_p_temp = xgb1.predict(dtest)
    xgb1_p = xgb1_p_temp.reshape(-1, 1)
    xgb1_acc = accuracy_score(xgb1_p, test_Y_det1)
    print('sd_xgb_acc=', xgb1_acc)


    #DecentionTree
    sd_dec = DecisionTreeClassifier(criterion="gini",max_features=6,min_samples_split=6)
    sd_dec.fit(feat_train, train_Y_det1.ravel())
    sd_dec_p_temp = sd_dec.predict(feat_test)
    sd_dec_p = sd_dec_p_temp.reshape(-1, 1)
    sd_dec_acc = accuracy_score(sd_dec_p, test_Y_det1)
    print('sd_dec_acc=', sd_dec_acc)
    # #knn
    knn = KNeighborsClassifier(weights='distance',p=2,leaf_size=50)
    knn.fit(feat_train, train_Y_det1.ravel())
    knn_predict = knn.predict(feat_test)
    np.set_printoptions(threshold=100000000)
    KNN_AC = accuracy_score(test_Y_det1, knn_predict)
    print("sd_knn_acc:", KNN_AC)

    # #LR
    classifier = LogisticRegression(random_state=5,solver='sag',C=10,)
    classifier.fit(feat_train, train_Y_det1.ravel())
    lg_predict = classifier.predict(feat_test)
    LG_AC = accuracy_score(test_Y_det1, lg_predict)
    print("sd_lr_acc:", LG_AC)
    _ = projection(test_Y_det1, xgb1_p,"SD_XGboost",6,62)
    _ = projection(test_Y_det1, knn_predict,"SD_KNN",6,62)

def WD_Model_63():
    print("WD_Model_63 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_6(lev3_train, lev3_val, lev3_test)
    model_sd_save = tf.keras.models.load_model('model_sd_save_63.h5')
    model_feat = Model(inputs = model_sd_save.input,outputs=model_sd_save.get_layer(name='out').output)
    feat_train = model_feat.predict([train_X_det1,train_X_det1])
    feat_test = model_feat.predict([test_X_det1,test_X_det1])

    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': 10,
        'gamma': 0.1,
        'max_depth': 14,
        'lambda': 2,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'silent': 1,
        'eta': 0.5,
        'seed': 1000,
        'nthread': 4,
    }
    dtrain = xgb.DMatrix(feat_train, train_Y_det1.ravel())
    num_rounds = 20
    plst = list(params.items())
    xgb1 = xgb.train(plst, dtrain, num_rounds)
    dtest = xgb.DMatrix(feat_test)
    xgb1_p_temp = xgb1.predict(dtest)
    xgb1_p = xgb1_p_temp.reshape(-1, 1)
    xgb1_acc = accuracy_score(xgb1_p, test_Y_det1)
    print('sd_xgb_acc=', xgb1_acc)
    #DecentionTree
    sd_dec = DecisionTreeClassifier(criterion="gini",max_features=6,min_samples_split=6)
    sd_dec.fit(feat_train, train_Y_det1.ravel())
    sd_dec_p_temp = sd_dec.predict(feat_test)
    sd_dec_p = sd_dec_p_temp.reshape(-1, 1)
    sd_dec_acc = accuracy_score(sd_dec_p, test_Y_det1)
    print('sd_dec_acc=', sd_dec_acc)
    # #knn
    knn = KNeighborsClassifier(weights='distance',p=2,leaf_size=50)
    knn.fit(feat_train, train_Y_det1.ravel())
    knn_predict = knn.predict(feat_test)
    np.set_printoptions(threshold=100000000)
    KNN_AC = accuracy_score(test_Y_det1, knn_predict)
    print("sd_knn_acc:", KNN_AC)
    # #LR
    classifier = LogisticRegression(random_state=5,solver='sag',C=10,)
    classifier.fit(feat_train, train_Y_det1.ravel())
    lg_predict = classifier.predict(feat_test)
    LG_AC = accuracy_score(test_Y_det1, lg_predict)
    print("sd_lr_acc:", LG_AC)
    _ = projection(test_Y_det1, xgb1_p,"SD_XGboost",6,63)
    _ = projection(test_Y_det1, knn_predict,"SD_KNN",6,63)

def WD_Model_64():
    print("WD_Model_64 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_6(lev4_train, lev4_val, lev4_test)
    model_sd_save = tf.keras.models.load_model('model_sd_save_64.h5')
    model_feat = Model(inputs = model_sd_save.input,outputs=model_sd_save.get_layer(name='out').output)
    feat_train = model_feat.predict([train_X_det1,train_X_det1])
    feat_test = model_feat.predict([test_X_det1,test_X_det1])

    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': 10,
        'gamma': 0.1,
        'max_depth': 14,
        'lambda': 2,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'silent': 1,
        'eta': 0.5,
        'seed': 1000,
        'nthread': 4,
    }
    dtrain = xgb.DMatrix(feat_train, train_Y_det1.ravel())
    num_rounds = 20
    plst = list(params.items())
    xgb1 = xgb.train(plst, dtrain, num_rounds)
    dtest = xgb.DMatrix(feat_test)
    xgb1_p_temp = xgb1.predict(dtest)
    xgb1_p = xgb1_p_temp.reshape(-1, 1)
    xgb1_acc = accuracy_score(xgb1_p, test_Y_det1)
    print('sd_xgb_acc=', xgb1_acc)
    #DecentionTree
    sd_dec = DecisionTreeClassifier(criterion="gini",max_features=6,min_samples_split=6)
    sd_dec.fit(feat_train, train_Y_det1.ravel())
    sd_dec_p_temp = sd_dec.predict(feat_test)
    sd_dec_p = sd_dec_p_temp.reshape(-1, 1)
    sd_dec_acc = accuracy_score(sd_dec_p, test_Y_det1)
    print('sd_dec_acc=', sd_dec_acc)
    # #knn
    knn = KNeighborsClassifier(weights='distance',p=2,leaf_size=50)
    knn.fit(feat_train, train_Y_det1.ravel())
    knn_predict = knn.predict(feat_test)
    np.set_printoptions(threshold=100000000)
    KNN_AC = accuracy_score(test_Y_det1, knn_predict)
    print("sd_knn_acc:", KNN_AC)
    # #LR
    classifier = LogisticRegression(random_state=5,solver='sag',C=10,)
    classifier.fit(feat_train, train_Y_det1.ravel())
    lg_predict = classifier.predict(feat_test)
    LG_AC = accuracy_score(test_Y_det1, lg_predict)
    print("sd_lr_acc:", LG_AC)
    _ = projection(test_Y_det1, xgb1_p,"SD_XGboost",6,64)
    _ = projection(test_Y_det1, knn_predict,"SD_KNN",6,64)

def WD_Model_81():
    print("WD_Model_81 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_8(lev1_train, lev1_val, lev1_test)
    model_sd_save = tf.keras.models.load_model('model_sd_save_81.h5')
    model_feat = Model(inputs = model_sd_save.input,outputs=model_sd_save.get_layer(name='out').output)
    feat_train = model_feat.predict([train_X_det1,train_X_det1])
    feat_test = model_feat.predict([test_X_det1,test_X_det1])
    # _ = show1(feat_test)
    #xgboost
    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': 12,
        'gamma': 0.1,
        'max_depth': 12,
        'lambda': 2,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 1,
        'silent': 1,
        'eta': 0.5,
        'seed': 1000,
        'nthread': 5,
    }
    dtrain = xgb.DMatrix(feat_train, train_Y_det1.ravel())
    num_rounds = 20
    plst = list(params.items())
    xgb1 = xgb.train(plst, dtrain, num_rounds)
    dtest = xgb.DMatrix(feat_test)
    xgb1_p_temp = xgb1.predict(dtest)
    xgb1_p = xgb1_p_temp.reshape(-1, 1)
    xgb1_acc = accuracy_score(xgb1_p, test_Y_det1)
    print('sd_xgb_acc=', xgb1_acc)
    #DecentionTree
    sd_dec = DecisionTreeClassifier(criterion="gini",max_features=6,min_samples_split=6)
    sd_dec.fit(feat_train, train_Y_det1.ravel())
    sd_dec_p_temp = sd_dec.predict(feat_test)
    sd_dec_p = sd_dec_p_temp.reshape(-1, 1)
    sd_dec_acc = accuracy_score(sd_dec_p, test_Y_det1)
    print('sd_dec_acc=', sd_dec_acc)
    # #knn
    knn = KNeighborsClassifier(weights='distance',p=2,leaf_size=50)
    knn.fit(feat_train, train_Y_det1.ravel())
    knn_predict = knn.predict(feat_test)
    np.set_printoptions(threshold=100000000)
    KNN_AC = accuracy_score(test_Y_det1, knn_predict)
    print("sd_knn_acc:", KNN_AC)
    # #LR
    classifier = LogisticRegression(random_state=5,solver='sag',C=10,)
    classifier.fit(feat_train, train_Y_det1.ravel())
    lg_predict = classifier.predict(feat_test)
    LG_AC = accuracy_score(test_Y_det1, lg_predict)
    print("sd_lr_acc:", LG_AC)
    _ = projection(test_Y_det1, xgb1_p,"SD_XGBoost",8,81)
    _ = projection(test_Y_det1, knn_predict,"SD_KNN",8,81)

def WD_Model_82():
    print("WD_Model_82 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_8(lev2_train, lev2_val, lev2_test)
    model_sd_save = tf.keras.models.load_model('model_sd_save_82.h5')
    model_feat = Model(inputs = model_sd_save.input,outputs=model_sd_save.get_layer(name='out').output)
    feat_train = model_feat.predict([train_X_det1,train_X_det1])
    feat_test = model_feat.predict([test_X_det1,test_X_det1])

    #xgboost
    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': 12,
        'gamma': 0.1,
        'max_depth': 12,
        'lambda': 2,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 1,
        'silent': 1,
        'eta': 0.5,
        'seed': 1000,
        'nthread': 5,
    }
    dtrain = xgb.DMatrix(feat_train, train_Y_det1.ravel())
    num_rounds = 20
    plst = list(params.items())
    xgb1 = xgb.train(plst, dtrain, num_rounds)
    dtest = xgb.DMatrix(feat_test)
    xgb1_p_temp = xgb1.predict(dtest)
    xgb1_p = xgb1_p_temp.reshape(-1, 1)
    xgb1_acc = accuracy_score(xgb1_p, test_Y_det1)
    print('sd_xgb_acc=', xgb1_acc)
    #DecentionTree
    sd_dec = DecisionTreeClassifier(criterion="gini",max_features=6,min_samples_split=6)
    sd_dec.fit(feat_train, train_Y_det1.ravel())
    sd_dec_p_temp = sd_dec.predict(feat_test)
    sd_dec_p = sd_dec_p_temp.reshape(-1, 1)
    sd_dec_acc = accuracy_score(sd_dec_p, test_Y_det1)
    print('sd_dec_acc=', sd_dec_acc)
    # #knn
    knn = KNeighborsClassifier(weights='distance',p=2,leaf_size=50)
    knn.fit(feat_train, train_Y_det1.ravel())
    knn_predict = knn.predict(feat_test)
    np.set_printoptions(threshold=100000000)
    KNN_AC = accuracy_score(test_Y_det1, knn_predict)
    print("sd_knn_acc:", KNN_AC)
    # #LR
    classifier = LogisticRegression(random_state=5,solver='sag',C=10,)
    classifier.fit(feat_train, train_Y_det1.ravel())
    lg_predict = classifier.predict(feat_test)
    LG_AC = accuracy_score(test_Y_det1, lg_predict)
    print("sd_lr_acc:", LG_AC)
    _ = projection(test_Y_det1, xgb1_p,"SD_XGBoost",8,82)
    _ = projection(test_Y_det1, knn_predict,"SD_KNN",8,82)
def WD_Model_83():
    print("WD_Model_83 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_8(lev3_train, lev3_val, lev3_test)
    model_sd_save = tf.keras.models.load_model('model_sd_save_83.h5')
    model_feat = Model(inputs = model_sd_save.input,outputs=model_sd_save.get_layer(name='out').output)
    feat_train = model_feat.predict([train_X_det1,train_X_det1])
    feat_test = model_feat.predict([test_X_det1,test_X_det1])

    #xgboost
    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': 12,
        'gamma': 0.1,
        'max_depth': 12,
        'lambda': 2,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 1,
        'silent': 1,
        'eta': 0.5,
        'seed': 1000,
        'nthread': 5,
    }
    dtrain = xgb.DMatrix(feat_train, train_Y_det1.ravel())
    num_rounds = 20
    plst = list(params.items())
    xgb1 = xgb.train(plst, dtrain, num_rounds)
    dtest = xgb.DMatrix(feat_test)
    xgb1_p_temp = xgb1.predict(dtest)
    xgb1_p = xgb1_p_temp.reshape(-1, 1)
    xgb1_acc = accuracy_score(xgb1_p, test_Y_det1)
    print('sd_xgb_acc=', xgb1_acc)
    #DecentionTree
    sd_dec = DecisionTreeClassifier(criterion="gini",max_features=6,min_samples_split=6)
    sd_dec.fit(feat_train, train_Y_det1.ravel())
    sd_dec_p_temp = sd_dec.predict(feat_test)
    sd_dec_p = sd_dec_p_temp.reshape(-1, 1)
    sd_dec_acc = accuracy_score(sd_dec_p, test_Y_det1)
    print('sd_dec_acc=', sd_dec_acc)
    # #knn
    knn = KNeighborsClassifier(weights='distance',p=2,leaf_size=50)
    knn.fit(feat_train, train_Y_det1.ravel())
    knn_predict = knn.predict(feat_test)
    np.set_printoptions(threshold=100000000)
    KNN_AC = accuracy_score(test_Y_det1, knn_predict)
    print("sd_knn_acc:", KNN_AC)
    # #LR
    classifier = LogisticRegression(random_state=5,solver='sag',C=10,)
    classifier.fit(feat_train, train_Y_det1.ravel())
    lg_predict = classifier.predict(feat_test)
    LG_AC = accuracy_score(test_Y_det1, lg_predict)
    print("sd_lr_acc:", LG_AC)
    _ = projection(test_Y_det1, xgb1_p,"SD_XGBoost",8,83)
    _ = projection(test_Y_det1, knn_predict,"SD_KNN",8,83)

def WD_Model_84():
    print("WD_Model_84 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_8(lev4_train, lev4_val, lev4_test)
    model_sd_save = tf.keras.models.load_model('model_sd_save_84.h5')
    model_feat = Model(inputs = model_sd_save.input,outputs=model_sd_save.get_layer(name='out').output)
    feat_train = model_feat.predict([train_X_det1,train_X_det1])
    feat_test = model_feat.predict([test_X_det1,test_X_det1])

    #xgboost
    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': 12,
        'gamma': 0.1,
        'max_depth': 12,
        'lambda': 2,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 1,
        'silent': 1,
        'eta': 0.5,
        'seed': 1000,
        'nthread': 5,
    }
    dtrain = xgb.DMatrix(feat_train, train_Y_det1.ravel())
    num_rounds = 20
    plst = list(params.items())
    xgb1 = xgb.train(plst, dtrain, num_rounds)
    dtest = xgb.DMatrix(feat_test)
    xgb1_p_temp = xgb1.predict(dtest)
    xgb1_p = xgb1_p_temp.reshape(-1, 1)
    xgb1_acc = accuracy_score(xgb1_p, test_Y_det1)
    print('sd_xgb_acc=', xgb1_acc)
    #DecentionTree
    sd_dec = DecisionTreeClassifier(criterion="gini",max_features=6,min_samples_split=6)
    sd_dec.fit(feat_train, train_Y_det1.ravel())
    sd_dec_p_temp = sd_dec.predict(feat_test)
    sd_dec_p = sd_dec_p_temp.reshape(-1, 1)
    sd_dec_acc = accuracy_score(sd_dec_p, test_Y_det1)
    print('sd_dec_acc=', sd_dec_acc)
    # #knn
    knn = KNeighborsClassifier(weights='distance',p=2,leaf_size=50)
    knn.fit(feat_train, train_Y_det1.ravel())
    knn_predict = knn.predict(feat_test)
    np.set_printoptions(threshold=100000000)
    KNN_AC = accuracy_score(test_Y_det1, knn_predict)
    print("sd_knn_acc:", KNN_AC)
    # #LR
    classifier = LogisticRegression(random_state=5,solver='sag',C=10,)
    classifier.fit(feat_train, train_Y_det1.ravel())
    lg_predict = classifier.predict(feat_test)
    LG_AC = accuracy_score(test_Y_det1, lg_predict)
    print("sd_lr_acc:", LG_AC)
    _ = projection(test_Y_det1, xgb1_p,"SD_XGBoost",8,84)
    _ = projection(test_Y_det1, knn_predict,"SD_KNN",8,84)

def WD_Model_101():
    print("WD_Model_101 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_10(lev1_train, lev1_val, lev1_test)
    model_sd_save = tf.keras.models.load_model('model_sd_save_101.h5')
    model_feat = Model(inputs = model_sd_save.input,outputs=model_sd_save.get_layer(name='out').output)
    feat_train = model_feat.predict([train_X_det1,train_X_det1])
    feat_test = model_feat.predict([test_X_det1,test_X_det1])

    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': 14,
        'gamma': 0.1,
        'max_depth': 12,
        'lambda': 2,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 0.8,
        'silent': 1,
        'eta': 0.5,
        'seed': 1000,
        'nthread': 5,
    }
    dtrain = xgb.DMatrix(feat_train, train_Y_det1.ravel())
    num_rounds = 20
    plst = list(params.items())
    xgb1 = xgb.train(plst, dtrain, num_rounds)
    dtest = xgb.DMatrix(feat_test)
    xgb1_p_temp = xgb1.predict(dtest)
    xgb1_p = xgb1_p_temp.reshape(-1, 1)
    xgb1_acc = accuracy_score(xgb1_p, test_Y_det1)
    print('sd_xgb_acc=', xgb1_acc)
    #DecentionTree
    sd_dec = DecisionTreeClassifier(criterion="gini",max_features=6,min_samples_split=6)
    sd_dec.fit(feat_train, train_Y_det1.ravel())
    sd_dec_p_temp = sd_dec.predict(feat_test)
    sd_dec_p = sd_dec_p_temp.reshape(-1, 1)
    sd_dec_acc = accuracy_score(sd_dec_p, test_Y_det1)
    print('sd_dec_acc=', sd_dec_acc)
    # #knn
    knn = KNeighborsClassifier(weights='distance',p=2,leaf_size=50)
    knn.fit(feat_train, train_Y_det1.ravel())
    knn_predict = knn.predict(feat_test)
    np.set_printoptions(threshold=100000000)
    KNN_AC = accuracy_score(test_Y_det1, knn_predict)
    print("sd_knn_acc:", KNN_AC)
    # #LR
    classifier = LogisticRegression(random_state=5,solver='sag',C=10,)
    classifier.fit(feat_train, train_Y_det1.ravel())
    lg_predict = classifier.predict(feat_test)
    LG_AC = accuracy_score(test_Y_det1, lg_predict)
    print("sd_lr_acc:", LG_AC)
    _ = projection(test_Y_det1, xgb1_p,"SD_XGboost",10,101)
    _ = projection(test_Y_det1, knn_predict,"SD_KNN",10,101)

def WD_Model_102():
    print("WD_Model_102 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_10(lev2_train, lev2_val, lev2_test)
    model_sd_save = tf.keras.models.load_model('model_sd_save_102.h5')
    model_feat = Model(inputs = model_sd_save.input,outputs=model_sd_save.get_layer(name='out').output)
    feat_train = model_feat.predict([train_X_det1,train_X_det1])
    feat_test = model_feat.predict([test_X_det1,test_X_det1])

    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': 14,
        'gamma': 0.1,
        'max_depth': 12,
        'lambda': 2,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 0.8,
        'silent': 1,
        'eta': 0.5,
        'seed': 1000,
        'nthread': 5,
    }
    dtrain = xgb.DMatrix(feat_train, train_Y_det1.ravel())
    num_rounds = 20
    plst = list(params.items())
    xgb1 = xgb.train(plst, dtrain, num_rounds)
    dtest = xgb.DMatrix(feat_test)
    xgb1_p_temp = xgb1.predict(dtest)
    xgb1_p = xgb1_p_temp.reshape(-1, 1)
    xgb1_acc = accuracy_score(xgb1_p, test_Y_det1)
    print('sd_xgb_acc=', xgb1_acc)
    #DecentionTree
    sd_dec = DecisionTreeClassifier(criterion="gini",max_features=6,min_samples_split=6)
    sd_dec.fit(feat_train, train_Y_det1.ravel())
    sd_dec_p_temp = sd_dec.predict(feat_test)
    sd_dec_p = sd_dec_p_temp.reshape(-1, 1)
    sd_dec_acc = accuracy_score(sd_dec_p, test_Y_det1)
    print('sd_dec_acc=', sd_dec_acc)
    # #knn
    knn = KNeighborsClassifier(weights='distance',p=2,leaf_size=50)
    knn.fit(feat_train, train_Y_det1.ravel())
    knn_predict = knn.predict(feat_test)
    np.set_printoptions(threshold=100000000)
    KNN_AC = accuracy_score(test_Y_det1, knn_predict)
    print("sd_knn_acc:", KNN_AC)
    # #LR
    classifier = LogisticRegression(random_state=5,solver='sag',C=10,)
    classifier.fit(feat_train, train_Y_det1.ravel())
    lg_predict = classifier.predict(feat_test)
    LG_AC = accuracy_score(test_Y_det1, lg_predict)
    print("sd_lr_acc:", LG_AC)
    _ = projection(test_Y_det1, xgb1_p,"SD_XGboost",10,102)
    _ = projection(test_Y_det1, knn_predict,"SD_KNN",10,102)

def WD_Model_103():
    print("WD_Model_103 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_10(lev3_train, lev3_val, lev3_test)
    model_sd_save = tf.keras.models.load_model('model_sd_save_103.h5')
    model_feat = Model(inputs = model_sd_save.input,outputs=model_sd_save.get_layer(name='out').output)
    feat_train = model_feat.predict([train_X_det1,train_X_det1])
    feat_test = model_feat.predict([test_X_det1,test_X_det1])

    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': 14,
        'gamma': 0.1,
        'max_depth': 12,
        'lambda': 2,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 0.8,
        'silent': 1,
        'eta': 0.5,
        'seed': 1000,
        'nthread': 5,
    }
    dtrain = xgb.DMatrix(feat_train, train_Y_det1.ravel())
    num_rounds = 20
    plst = list(params.items())
    xgb1 = xgb.train(plst, dtrain, num_rounds)
    dtest = xgb.DMatrix(feat_test)
    xgb1_p_temp = xgb1.predict(dtest)
    xgb1_p = xgb1_p_temp.reshape(-1, 1)
    xgb1_acc = accuracy_score(xgb1_p, test_Y_det1)
    print('sd_xgb_acc=', xgb1_acc)
    #DecentionTree
    sd_dec = DecisionTreeClassifier(criterion="gini",max_features=6,min_samples_split=6)
    sd_dec.fit(feat_train, train_Y_det1.ravel())
    sd_dec_p_temp = sd_dec.predict(feat_test)
    sd_dec_p = sd_dec_p_temp.reshape(-1, 1)
    sd_dec_acc = accuracy_score(sd_dec_p, test_Y_det1)
    print('sd_dec_acc=', sd_dec_acc)
    # #knn
    knn = KNeighborsClassifier(weights='distance',p=2,leaf_size=50)
    knn.fit(feat_train, train_Y_det1.ravel())
    knn_predict = knn.predict(feat_test)
    np.set_printoptions(threshold=100000000)
    KNN_AC = accuracy_score(test_Y_det1, knn_predict)
    print("sd_knn_acc:", KNN_AC)
    # #LR
    classifier = LogisticRegression(random_state=5,solver='sag',C=10,)
    classifier.fit(feat_train, train_Y_det1.ravel())
    lg_predict = classifier.predict(feat_test)
    LG_AC = accuracy_score(test_Y_det1, lg_predict)
    print("sd_lr_acc:", LG_AC)
    _ = projection(test_Y_det1, xgb1_p,"SD_XGboost",10,103)
    _ = projection(test_Y_det1, knn_predict,"SD_KNN",10,103)

def WD_Model_104():
    print("WD_Model_104 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_10(lev4_train, lev4_val, lev4_test)
    model_sd_save = tf.keras.models.load_model('model_sd_save_104.h5')
    model_feat = Model(inputs = model_sd_save.input,outputs=model_sd_save.get_layer(name='out').output)
    feat_train = model_feat.predict([train_X_det1,train_X_det1])
    feat_test = model_feat.predict([test_X_det1,test_X_det1])

    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': 14,
        'gamma': 0.1,
        'max_depth': 12,
        'lambda': 2,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 0.8,
        'silent': 1,
        'eta': 0.5,
        'seed': 1000,
        'nthread': 5,
    }
    dtrain = xgb.DMatrix(feat_train, train_Y_det1.ravel())
    num_rounds = 20
    plst = list(params.items())
    xgb1 = xgb.train(plst, dtrain, num_rounds)
    dtest = xgb.DMatrix(feat_test)
    xgb1_p_temp = xgb1.predict(dtest)
    xgb1_p = xgb1_p_temp.reshape(-1, 1)
    xgb1_acc = accuracy_score(xgb1_p, test_Y_det1)
    print('sd_xgb_acc=', xgb1_acc)
    #DecentionTree
    sd_dec = DecisionTreeClassifier(criterion="gini",max_features=6,min_samples_split=6)
    sd_dec.fit(feat_train, train_Y_det1.ravel())
    sd_dec_p_temp = sd_dec.predict(feat_test)
    sd_dec_p = sd_dec_p_temp.reshape(-1, 1)
    sd_dec_acc = accuracy_score(sd_dec_p, test_Y_det1)
    print('sd_dec_acc=', sd_dec_acc)
    # #knn
    knn = KNeighborsClassifier(weights='distance',p=2,leaf_size=50)
    knn.fit(feat_train, train_Y_det1.ravel())
    knn_predict = knn.predict(feat_test)
    np.set_printoptions(threshold=100000000)
    KNN_AC = accuracy_score(test_Y_det1, knn_predict)
    print("sd_knn_acc:", KNN_AC)
    # #LR
    classifier = LogisticRegression(random_state=5,solver='sag',C=10,)
    classifier.fit(feat_train, train_Y_det1.ravel())
    lg_predict = classifier.predict(feat_test)
    LG_AC = accuracy_score(test_Y_det1, lg_predict)
    print("sd_lr_acc:", LG_AC)
    _ = projection(test_Y_det1, xgb1_p,"SD_XGboost",10,104)
    _ = projection(test_Y_det1, knn_predict,"SD_KNN",10,104)

def main():
    # _ = show()
    # _ = load_ML()
    # _ = load_ML1()
    _ = feature_select()
    # _ = ML_61()
    # _ = ML_62()
    # _ = ML_63()
    # _ = ML_64()
    # _ = ML_81()
    # _ = ML_82()
    # _ = ML_83()
    # _ = ML_84()
    # _ = ML_101()
    # _ = ML_102()
    # _ = ML_103()
    # _ = ML_104()
    # _ = CNN_61()
    # _ = CNN_62()
    # _ = CNN_63()
    # _ = CNN_64()
    # _ = CNN_81()
    # _ = CNN_82()
    # _ = CNN_83()
    # _ = CNN_84()
    # _ = CNN_101()
    # _ = CNN_102()
    # _ = CNN_103()
    # _ = CNN_104()
    # _ = WD61()
    # _ = WD62()
    # _ = WD63()
    # _ = WD64()
    # _ = WD81()
    _ = WD82()
    # _ = WD83()
    # _ = WD84()
    # _ = WD101()
    # _ = WD102()
    # _ = WD103()
    # _ = WD104()
    # _ = WD121()
    # _ = WD122()
    # _ = WD123()
    # _ = WD124()
    # _ = AWD61()
    # _ = AWD62()
    # _ = AWD63()
    # _ = AWD64()
    # _ = AWD81()
    # _ = AWD82()
    # _ = AWD83()
    # _ = AWD84()
    # _ = AWD101()
    # _ = AWD102()
    # _ = AWD103()
    # _ = AWD104()
    # _ = WD_Model_61()
    # _ = WD_Model_62()
    # _ = WD_Model_63()
    # _ = WD_Model_64()
    # _ = WD_Model_81()
    # _ = WD_Model_82()
    # _ = WD_Model_83()
    # _ = WD_Model_84()
    # _ = WD_Model_101()
    # _ = WD_Model_102()
    # _ = WD_Model_103()
    # _ = WD_Model_104()
    return None

if __name__ == '__main__':
    _ = main()