#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm.sklearn import LGBMClassifier
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.metrics import classification_report
from sklearn import metrics
from statsmodels.tsa.holtwinters import ExponentialSmoothing,Holt
import matplotlib.pyplot as plt
from tqdm import tqdm
#%%
data = pd.read_csv('feat.csv')
#%%
data = data.iloc[:,1:]
#%%
data = data.drop(['time_y','future','contract_y','date','close_11_30'],axis=1)
#%%
data = data.set_index('datetime_y')
#%%
data = data.loc[(data['log_return_bu'] == data['log_return_bu'])]
#%%

data['target'] = data['log_return_bu']
#%%
from sklearn import preprocessing

label = preprocessing.LabelEncoder()
data['future'] = label.fit_transform(data['future'])
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
train_set = train[:5799]
test_set = train[5799:6088]
train_target = target[:5799]
test_target = target[5799:6088]
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
#%%
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=2022)
X_train, X_train_target = sm.fit_resample(X_train, X_train_target)
#%%
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn.model_selection import TimeSeriesSplit
from bayes_opt import BayesianOptimization

# kf = TimeSeriesSplit(n_splits=10)
def lgb_cv(colsample_bytree, learning_rate, min_child_samples, min_child_weight, n_estimators, num_leaves, subsample, max_depth, min_split_gain):
    model = LGBMClassifier(boosting_type='gbdt',objective='binary',
           colsample_bytree=float(colsample_bytree), learning_rate=float(learning_rate),
           min_child_samples=int(min_child_samples), min_child_weight=float(min_child_weight),
           n_estimators=int(n_estimators), n_jobs=10, num_leaves=int(num_leaves),
           random_state=None, reg_alpha=0.0, reg_lambda=0.0, max_depth=int(max_depth),
           subsample=float(subsample), min_split_gain=float(min_split_gain), )
    cv_score = cross_val_score(model, X_train, X_train_target, scoring='roc_auc', cv=5).mean()
    return cv_score
# 使用贝叶斯优化
lgb_bo = BayesianOptimization(
        lgb_cv,
        {'colsample_bytree': (0.7, 1),
         'learning_rate': (0.0001, 0.1),
         'min_child_samples': (2, 100),
         'min_child_weight':(0.0001, 0.1),
         'n_estimators': (500, 10000),
         'num_leaves': (5, 250),
         'subsample': (0.7, 1),
         'max_depth': (2, 100),
         'min_split_gain': (0.1, 1)
         }
    )
lgb_bo.maximize()
#%%
lgb_bo.max
#%%
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
import lightgbm as lgb


# kf = StratifiedKFold(n_splits=10,random_state=20,shuffle=True)
kf = TimeSeriesSplit(n_splits=10)
# kf = GapLeavePOut(p=35000, gap_before=11000, gap_after=24000)
y_pred = np.zeros(len(X_test_target))
y_pred_train = np.zeros(len(X_train_target))
for fold, (train_index, val_index) in enumerate(kf.split(X_train, X_train_target)):
    x_train, x_val = X_train[train_index], X_train[val_index]
    y_train, y_val = X_train_target[train_index], X_train_target[val_index]
    train_set = lgb.Dataset(x_train, y_train)
    val_set = lgb.Dataset(x_val, y_val)

    params = {
        'boosting_type': 'gbdt',
        'metric': {'cross_entropy', 'auc', 'average_precision', },
        'objective': 'cross_entropy',  # regression,binary,multiclass
        # 'num_class': 3,
        'seed': 666,
        'num_leaves': 94,
        'learning_rate': 0.1,
        'max_depth': 56,
        'n_estimators': 4091,
        # 'lambda_l1': 1,
        # 'lambda_l2': 1,
        # 'bagging_fraction': 1,
        # 'bagging_freq': 1,
        'colsample_bytree': 0.82,
        'subsample': 0.95,
        'min_child_samples': 7,
        'min_child_weight': 0.05,
        'min_split_gain': 0.89,
        'verbose': -1,
        # 'cross_entropy':'xentropy'
    }

    model = lgb.train(params, train_set, num_boost_round=5000, early_stopping_rounds=50,
                      valid_sets=[val_set], verbose_eval=100)

    y_pred += model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits
    y_pred_train += model.predict(X_train, num_iteration=model.best_iteration) / kf.n_splits
# lgb.plot_importance(model, max_num_features=20)
# plt.show()
#%%
feature_names = data.drop(['target'],axis=1).columns
feature_importance = pd.DataFrame({'column': feature_names,'importance': model.feature_importance()}).sort_values(by='importance', ascending=False, ignore_index=True)
feature_importance.plot.barh(x='column',figsize=(10,12))
plt.title('15:00 feature importance')
plt.savefig('feature_15_00.png')
#%%
from sklearn.metrics import roc_curve
from numpy import sqrt,argmax
fpr_train, tpr_train, thresholds_train = roc_curve(X_train_target, y_pred_train)
gmeans_train = sqrt(tpr_train * (1-fpr_train))
ix_train = argmax(gmeans_train)
print('Best train Threshold=%f, G-Mean=%.3f' % (thresholds_train[ix_train], gmeans_train[ix_train]))

thresholds_point_train = thresholds_train[ix_train]
yhat_train = [1 if y > thresholds_point_train else 0 for y in y_pred_train]
print("训练集表现：")
print(classification_report(yhat_train,X_train_target))
print(metrics.confusion_matrix(yhat_train, X_train_target))
#%% roccurve
from sklearn.metrics import roc_curve
from numpy import sqrt,argmax
fpr, tpr, thresholds = roc_curve(X_test_target, y_pred)
gmeans = sqrt(tpr * (1-fpr))
ix = argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
#%%
thresholds_point = 0.492841
yhat = [1 if y > thresholds_point else 0 for y in y_pred]
print("测试集表现：")
print(classification_report(yhat,X_test_target))
print(metrics.confusion_matrix(yhat, X_test_target))
#%%
import graphviz
from sklearn.tree import export_graphviz
dot_data = export_graphviz(lgb.estimators_[0, 0], out_file=None, feature_names=X_train.columns,
                           filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('tree1.gv')
#%%
import time
start = time.time()

gbm = LGBMClassifier(boosting_type='gbdt', objective='binary',
                     colsample_bytree=0.9667906439679592,
                     min_child_samples=100, min_child_weight=0.02795200706203385, min_split_gain=0.8498512300910855,
                     n_estimators=3702,
                     # feature_fraction=1,
                     subsample=0.9078280916321744,
                     learning_rate=0.06318390961915947, max_depth=57, num_leaves=9,
                     reg_alpha=0.0, reg_lambda=0.0
                     )
# cv_score = cross_val_score(gbm, X_train, X_train_target, scoring="roc_auc", cv=5).mean()
# y_pred_gbm = cv_score
gbm.fit(X_train,X_train_target)
y_pred_gbm = gbm.predict(X_test)
end = time.time()
print('Total Time = %s'%(end-start))

print(accuracy_score(y_pred_gbm,X_test_target))
print("测试集表现：")
print(classification_report(y_pred_gbm,X_test_target))
print("评价指标-混淆矩阵：")
print(metrics.confusion_matrix(y_pred_gbm,X_test_target))
#%%
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(X_test_target, y_pred)
plt.figure()
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.plot(recall, precision)
plt.show()
#%%
p = [1 if y > 0.6 else 0 for y in p]
print("测试集表现：")
print(classification_report(p,X_test_target))
print(metrics.confusion_matrix(p, X_test_target))
#%%
import joblib
joblib.dump(model,'lightGBM_ru.pkl')
#%%
features = data.columns
features = pd.DataFrame(features)
features.to_csv('features.csv')
#%%
data = data.reset_index()
#%%
test_data = data[15078:]
test_data = test_data.reset_index(drop=True)
predict = pd.DataFrame(yhat,columns=list('P'))
predict['datetime'] = test_data['datetime_y']
predict['contract'] = test_data['contract_y']
predict['future'] = test_data['future']
predict['close'] = test_data['close_y']
predict['close_11_30'] = test_data['close_11_30']
predict['target'] = test_data['target']
#%%
train_data = data[:15078]
train_data = train_data.reset_index(drop=True)
predict_train = pd.DataFrame(yhat_train,columns=list('P'))
predict_train['datetime'] = train_data['datetime_y']
predict_train['contract'] = train_data['contract_y']
predict_train['future'] = train_data['future']
predict_train['close'] = train_data['close_y']
predict_train['close_11_30'] = train_data['close_11_30']
predict_train['target'] = train_data['target']
#%%
model.save_model('lightGBM_ru.txt')
#%%
predict.to_csv('predict_1_0_GBDT_train_huagong.csv')
#%%
predict_train.to_csv('predict_1_0_GBDT_train_huagong.csv')
#%%
from scipy.stats import ks_2samp,kstest
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller
#%%
for i in range(39):
    print(i)
    print(adfuller(X_test[:,i]))
#%%
for i in range(39):
    print(i)
    print(ks_2samp(X_train[:,i],X_test[:,i]))
#%%
D_value = 1.36*np.sqrt((len(X_train_target)+len(X_test_target))/(len(X_train_target)*len(X_test_target)))
print(D_value)
#%%
for i in range(17):
    print(i)
    print(stats.jarque_bera(X_train[:,i]))
#%%
ks_2samp(X_train_target, X_test_target)