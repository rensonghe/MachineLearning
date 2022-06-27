#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import array
from sklearn.ensemble import RandomForestClassifier
# import tensorflow as tf
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

import math
import sklearn.metrics as skm
import datetime

#%%
data = pd.read_csv('test.csv')
#%%
data = data.iloc[:,1:]
data = data.drop(['time_y','future','contract_y','date','close_11_30'],axis=1)
#%%
data = data.set_index('datetime_y')
#%%
# from sklearn import preprocessing
#
# label = preprocessing.LabelEncoder()
# data['future'] = label.fit_transform(data['future'])
# data['contract_y'] = label.fit_transform(data['contract_y'])
#%%
def calcSpearman(data):

    ic_list = []
    data = data.copy()
    # target = data['target']
    for column in list(data.columns[0:33]):

        ic = data[column].rolling(50).corr(data['target'])
        ic_mean = np.mean(ic)
        print(ic_mean)
        ic_list.append(ic_mean)

        # print(ic_list)

    return ic_list

IC = calcSpearman(data)

IC = pd.DataFrame(IC)
columns = pd.DataFrame(data.columns)

IC_columns = pd.concat([IC, columns], axis=1)
col = ['value', 'variable']
IC_columns.columns = col

filter_value = 0.01
filter_value2 = -0.01
x_column = IC_columns.variable[IC_columns.value > filter_value]
y_column = IC_columns.variable[IC_columns.value < filter_value2]

x_column = x_column.tolist()
y_column = y_column.tolist()
final_col = x_column+y_column
#%%
data = data.reindex(columns=final_col)
#%%
def classify(y):

    if y < 0:
        return 0
    if y > 0:
        return 1
    else:
        return -1
data['target'] = data['target'].apply(lambda x:classify(x))
print(data['target'].value_counts())
#%%
data = data[~data['target'].isin([-1])]
#%%
cols = data.columns #所有列
train_col = [] # 选择测试集的列
for i in cols:
    if i != "target":
        train_col.append(i)

train = data[train_col]
target = data["target"] # 取前26列为训练数据，最后一列为target
#%%
train_set = train[11330:15078]
test_set = train[15078:]
train_target = target[11330:15078]
test_target = target[15078:]
#%%
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))  # 创建归一化模板
train_set_scaled = sc.fit_transform(train_set)# 数据归一
test_set_scaled = sc.transform(test_set)
train_target = np.array(train_target)
test_target = np.array(test_target)

X_train = train_set_scaled
X_train_target=train_target
X_test = test_set_scaled
X_test_target =test_target
X_test_target=test_target
#%%
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=2022)
X_train,X_train_target = sm.fit_resample(X_train,X_train_target)
#%%
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
def rf_cv(n_estimators, min_samples_split, max_features, max_depth):
    val = cross_val_score(
        RandomForestClassifier(n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=min(max_features, 0.999), # float
            max_depth=int(max_depth),
            random_state=2),
        X_train, X_train_target, scoring='f1', cv=5).mean()
    return val
rf_bo = BayesianOptimization(
    rf_cv,
    {'n_estimators': (10, 250),
    'min_samples_split': (2, 25),
    'max_features': (0.1, 0.999),
    'max_depth': (5, 15)}
)
rf_bo.maximize()
#%%
rf_bo.max['params']
#%%
# from sklearn.model_selection import GridSearchCV
# # Create the parameter grid based on the results of random search
# param_grid = {
#     'bootstrap': [True],
#     'max_depth': [80, 90, 100, 110],
#     'max_features': [2, 3],
#     'min_samples_leaf': [3, 4, 5],
#     'min_samples_split': [8, 10, 12],
#     'n_estimators': [100, 200, 300, 1000]
# }
# # Create a based model
# rf = RandomForestClassifier()
# # Instantiate the grid search model
# grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
#                           cv = 3, n_jobs = 5, verbose = 2)
#
# # Fit the random search model
# grid_search.fit(X_train,X_train_target)
# grid_search.best_params_
#%%
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()

#%% binary classification
model = RandomForestClassifier(n_estimators=32,
                               max_depth=14,
                               min_samples_split=10,
                               # min_samples_leaf=12,
                               max_features=0.1,
                               random_state=2
                               )

import time
start = time.time()

print(X_train.shape)
print(X_test.shape)
model.fit(X_train,X_train_target)
y_pred=model.predict(X_test)

end = time.time()
print('Total Time = %s'%(end-start))

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import metrics
print(accuracy_score(y_pred,X_test_target))
print("测试集表现：")
print(classification_report(y_pred,X_test_target))
print("评价指标-混淆矩阵：")
print(metrics.confusion_matrix(y_pred,X_test_target))
#%%
import graphviz
from sklearn.tree import export_graphviz
#%%
dot_data = export_graphviz(model.estimators_[1], out_file=None, feature_names=train_set.columns,
                           filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
#%%
graph.save('tree.dot')
#%%
graph.render('tree.gv')
#%%
import joblib
joblib.dump(model,'RandomForest_1.pkl')
#%%
data = data.reset_index()
#%%
test_data = data[70000:]
test_data = test_data.reset_index(drop=True)
predict = pd.DataFrame(y_pred,columns=list('P'))
predict['datetime'] = test_data['datetime']
predict['last_price'] = test_data['last_price']
predict['target'] = test_data['target']
#%%
predict.to_csv('predict_1_0_RF_2min_3month_ru_last_price.csv')

#%%
train_target_1 = model.predict(X_train)
print(accuracy_score(X_train_target,train_target_1))
print(classification_report(X_train_target,train_target_1))