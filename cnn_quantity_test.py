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
import time
def load_ML():
    np.random.seed(2)
    lev1_lab0 = pd.read_csv(r'data_set0\\level1\\lev1_lab0.csv')
    lev1_lab1 = pd.read_csv(r'data_set0\\level1\\lev1_lab1.csv')
    lev1_lab2 = pd.read_csv(r'data_set0\\level1\\lev1_lab2.csv')
    lev1_lab3 = pd.read_csv(r'data_set0\\level1\\lev1_lab3.csv')
    lev1_lab4 = pd.read_csv(r'data_set0\\level1\\lev1_lab4.csv')
    lev1_lab5 = pd.read_csv(r'data_set0\\level1\\lev1_lab5.csv')
    lev1_lab6 = pd.read_csv(r'data_set0\\level1\\lev1_lab6.csv')
    lev1_lab7 = pd.read_csv(r'data_set0\\level1\\lev1_lab7.csv')

    lev1_lab0 = lev1_lab0.sample(n=5000)
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

def load_data_det_8(train_data,val_data,test_data):

    # train_X = train_data[['FWC', 'FWE', 'TCA', 'Tolerance%', 'PO_feed', 'PO_net', 'TWCD', 'VE']]
    # test_X = test_data[['FWC', 'FWE', 'TCA', 'Tolerance%', 'PO_feed', 'PO_net', 'TWCD', 'VE']]
    # train_X = train_data[['TCI', 'TEO', 'PO_feed', 'TCO','Evap Tons', 'TEI', 'TCA', 'TO_feed']]
    # test_X = test_data[['TCI', 'TEO', 'PO_feed', 'TCO','Evap Tons', 'TEI', 'TCA', 'TO_feed']]

    #[6, 4, 48, 8, 18, 2, 28, 46, 30, 61, 5, 25, 32, 26, 9, 33]
    # train_X = train_data[:, [6, 4, 48, 8, 18, 2]]
    # val_X = val_data[:, [6, 4, 48, 8, 18, 2]]
    # test_X = test_data[:, [ 6, 4,48, 8, 18, 2]]
    train_X = train_data[:, [6, 4, 48, 8, 18, 2, 28, 46]]
    val_X = val_data[:, [6, 4, 48, 8, 18, 2, 28, 46]]
    test_X = test_data[:, [ 6, 4,48, 8, 18, 2, 28, 46]]
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

def WD81():
    print("WD81 starting")
    t1 = time.time()
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
    hidden13 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(hidden11)
    hidden22 = tf.keras.layers.MaxPool1D(pool_size=2)(hidden13)

    input_deep = tf.keras.layers.Input(shape=(8, 1))
    hidden1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_deep)
    hidden2 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(hidden1)
    hidden4 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden2)
    hidden5 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(hidden4)
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
    t2 = time.time()
    t = t2-t1
    print("消耗用时:",t)
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
    model_wd_det1.save('test_model_sd_save_81.h5')

def WD_Model_81():
    print("WD_Model_81 starting")
    lev1_train, lev1_val, lev1_test, lev2_train, lev2_val, lev2_test, lev3_train, lev3_val, lev3_test, lev4_train, lev4_val, lev4_test = load_ML()
    train_X_det1, val_X_det1, test_X_det1, train_Y_det1, val_Y_det1, test_Y_det1 = load_data_det_8(lev1_train, lev1_val, lev1_test)
    model_sd_save = tf.keras.models.load_model('test_model_sd_save_81.h5')
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

def main():
    _ = WD81()
    _ = WD_Model_81()
    return None

if __name__ == '__main__':
    _ = main()