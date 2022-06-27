import time

import MySQLdb
import pandas as pd
import numpy as np
import itertools
import tqdm
import lightgbm as lgb
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
n_fold = 7
seed0 = 8586
use_supple_for_train = True

lags = [30,600,900]
#%%
# data_time = pd.read_csv('all_data_huagong_2019_2022_15min.csv',usecols=['epoch_time','future','close','log_return'])
# data_time = data_time.iloc[:,1:]
data_time = pd.read_csv('all_data_all_future_anotherday.csv',usecols=['datetime_y','future','close_y','target'])
#%%
data_time = data_time.dropna(axis=0)
#%%
data_time['log_return'] = data_time.log_return*100
#%%
# import time
# data_time['epoch_time'] = data_time['epoch_time']/1000000000
# data_time['datetime'] = data_time['epoch_time'].shift(1)
# data_time = data_time.drop(['epoch_time'], axis=1)
# data_time = data_time.dropna(axis=0)
#%%
conn = MySQLdb.connect(host="rm-bp1yvd1jr36kmmv834o.mysql.rds.aliyuncs.com",user="LJD",passwd="HZLJDcl123456", db="lgt", port=3306, charset='utf8' )
cursor = conn.cursor()
sql = "select distinct future from contractsize;"
cursor.execute(sql)
future = np.array(cursor.fetchall())
future_list = future.tolist()
future_list = list(itertools.chain.from_iterable(future))
# remove_list = ['IC', 'IF', 'IH', 'T', 'TF', 'TS','wr','WH','fb','LR','rr','RS','RI','JR','fu','PM']
index_list = [22,23,24,59,61,62,66,65,17,31,48,49,46,28,19,43]
future_list = [n for i, n in enumerate(future_list) if i not in index_list]
#%%
# future_list = ['bu','eb','eg','lu','MA','nr','pg','ru','SA','sc','sp','TA','UR']
#%%
train_merge = pd.DataFrame()
train_merge[data_time.columns] = 0
for i in future_list:
    # print(i)
    train_merge = train_merge.merge(data_time.loc[data_time['future'] == i,['datetime_y','close_y',
                    'target']].copy(),on='datetime_y',how='outer',suffixes=['',"_"+str(i)])
#%%
train_merge = train_merge.drop(data_time.columns.drop("datetime"), axis=1)
train_merge = train_merge.sort_values('datetime', ascending=True)

for i in future_list:
    train_merge[f'close_{i}'] = train_merge[f'close_{i}'].fillna(method='ffill',limit=100)

train_merge = train_merge.drop_duplicates('datetime', keep='last')
#%%
def get_features(df, train=True):
    # if train == True:
    #     totimestamp = lambda s: np.int32(time.mktime(datetime.datetime.strptime(s, "%d/%m/%Y").timetuple()))
    #     vaild_window = [totimestamp('01/01/2022')]
    #     df['train_flg'] = np.where(df['datetime']>=vaild_window[0], 0, 1)
    #     supple_start_window = [totimestamp('01/03/2022')]
    #     if use_supple_for_train:
    #         df['train_flg'] = np.where(df['datetime']>=supple_start_window[0], 1, df['train_flg'])
    for id in future_list:
        for lag in lags:
            df[f'log_close/mean_{lag}_{id}'] = np.log(np.array(df[f'close_{id}']) / np.roll(
                np.append(np.convolve(np.array(df[f'close_{id}']), np.ones(lag) / lag, mode="valid"), np.ones(lag - 1)),
                lag - 1))
            df[f'log_return_{lag}_{id}'] = np.log(
                np.array(df[f'close_{id}']) / np.roll(np.array(df[f'close_{id}']), lag))
    for lag in lags:
        df[f'mean_close/mean_{lag}'] = np.mean(df.iloc[:, df.columns.str.startswith(f'log_close/mean_{lag}')],
                                               axis=1)
        df[f'mean_log_returns_{lag}'] = np.mean(df.iloc[:, df.columns.str.startswith(f'log_return_{lag}')], axis=1)
        for id in future_list:
            df[f'log_close/mean_{lag}-mean_close/mean_{lag}_{id}'] = np.array(
                df[f'log_close/mean_{lag}_{id}']) - np.array(df[f'mean_close/mean_{lag}'])
            df[f'log_return_{lag}-mean_log_returns_{lag}_{id}'] = np.array(df[f'log_return_{lag}_{id}']) - np.array(
                df[f'mean_log_returns_{lag}'])

    # if train == True:
        # for id in future_list:
            # df = df.drop([f'close_{id}'], axis=1)
            # oldest_use_window = [totimestamp('01/01/2016')]
            # df = df[df['datetime']>=oldest_use_window[0]]

    return df
#%%
feat = get_features(train_merge)
feat = feat.reset_index(drop=True)

# not_use_features_train = ['datetime','train_flg']
not_use_features_train = ['datetime']
for i in future_list:
    not_use_features_train.append(f'log_return_{i}')
features = feat.columns
features = features.drop(not_use_features_train)
features = list(features)
#%%
def plot_importance(importances, features_names = features, PLOT_TOP_N = 20, figsize=(10, 10)):
    importance_df = pd.DataFrame(data=importances, columns=features)
    sorted_indices = importance_df.median(axis=0).sort_values(ascending=False).index
    sorted_importance_df = importance_df.loc[:, sorted_indices]
    plot_cols = sorted_importance_df.columns[:PLOT_TOP_N]
    _, ax = plt.subplots(figsize=figsize)
    ax.grid()
    ax.set_xscale('log')
    ax.set_ylabel('Feature')
    ax.set_xlabel('Importance')
    sns.boxplot(data=sorted_importance_df[plot_cols],
                orient='h',
                ax=ax)
    plt.show()
#%%
# def get_time_series_cross_val_splits(data, cv = n_fold, embargo = 3750):
#     all_train_timestamps = data['datetime'].unique()
#     len_split = len(all_train_timestamps) // cv
#     test_splits = [all_train_timestamps[i * len_split:(i + 1) * len_split] for i in range(cv)]
#     # print(test_splits)
#     # fix the last test split to have all the last timestamps, in case the number of timestamps wasn't divisible by cv
#     rem = len(all_train_timestamps) - len_split*cv
#     if rem>0:
#         test_splits[-1] = np.append(test_splits[-1], all_train_timestamps[-rem:])
#
#     train_splits = []
#     for test_split in test_splits:
#         # print(test_split)
#         test_split_max = float(np.max(test_split))
#         test_split_min = float(np.min(test_split))
#         print(test_split_min,'min')
#         print(test_split_max,'max')
#         # get all of the timestamps that aren't in the test split
#         train_split_not_embargoed = [e for e in all_train_timestamps if not (test_split_min <= int(e) <= test_split_max)]
#         # embargo the train split so we have no leakage. Note timestamps are expressed in seconds, so multiply by 60
#         embargo_sec = 60*embargo  #qihuo
#         train_split = [e for e in train_split_not_embargoed if
#                        abs(int(e) - test_split_max) > embargo_sec and abs(int(e) - test_split_min) > embargo_sec]
#         # print(len(train_split),'length')
#         train_splits.append(train_split)
#         # print(len(train_splits), 'length')
#
#
#     # convenient way to iterate over train and test splits
#     train_test_zip = zip(train_splits, test_splits)
#     return train_test_zip
#%%
def correlation(a, train_data):
    b = train_data.get_label()
    # print(b)
    a = np.ravel(a)
    b = np.ravel(b)

    len_data = len(a)
    mean_a = np.sum(a) / len_data
    mean_b = np.sum(b) / len_data
    var_a = np.sum(np.square(a - mean_a)) / len_data
    var_b = np.sum(np.square(b - mean_b)) / len_data

    cov = np.sum((a * b)) / len_data - mean_a * mean_b
    corr = cov / np.sqrt(var_a * var_b)

    return 'corr', corr, True

def corr_score(pred, valid):
    len_data = len(pred)
    mean_pred = np.sum(pred) / len_data
    mean_valid = np.sum(valid) / len_data
    var_pred = np.sum(np.square(pred - mean_pred)) / len_data
    var_valid = np.sum(np.square(valid - mean_valid)) / len_data

    cov = np.sum((pred * valid))/len_data - mean_pred*mean_valid
    corr = cov / np.sqrt(var_pred * var_valid)

    return corr

def wcorr_score(pred, valid, weight):
    len_data = len(pred)
    sum_w = np.sum(weight)
    mean_pred = np.sum(pred * weight) / sum_w
    mean_valid = np.sum(valid * weight) / sum_w
    var_pred = np.sum(weight * np.square(pred - mean_pred)) / sum_w
    var_valid = np.sum(weight * np.square(valid - mean_valid)) / sum_w

    cov = np.sum((pred * valid * weight)) / sum_w - mean_pred*mean_valid
    corr = cov / np.sqrt(var_pred * var_valid)

    return corr

def rmspe(y_true, y_pred):
    return  (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))


def feval_RMSPE(preds, train_data):
    labels = train_data.get_label()
    return 'RMSPE', round(rmspe(y_true = labels, y_pred = preds),10), False
#%%
# params = {
#     'early_stopping_rounds': 50,
#     'objective': 'regression',
#     'metric': 'rmse',
# #     'metric': 'None',
#     'boosting_type': 'gbdt',
#     'max_depth': 5,
#     'verbose': -1,
#     'max_bin':600,
#     'min_data_in_leaf':50,
#     'learning_rate': 0.03,
#     'subsample': 0.7,
#     'subsample_freq': 1,
#     'feature_fraction': 1,
#     'lambda_l1': 0.5,
#     'lambda_l2': 2,
#     'seed':seed0,
#     'feature_fraction_seed': seed0,
#     'bagging_fraction_seed': seed0,
#     'drop_seed': seed0,
#     'data_random_seed': seed0,
#     'extra_trees': True,
#     'extra_seed': seed0,
#     'zero_as_missing': True,
#     "first_metric_only": True
#          }
# not_use_overlap_to_train = False
# def get_Xy_and_model_for_asset(df_proc, future_list):
#     df_proc = df_proc.loc[(df_proc[f'log_return_{future_list}'] == df_proc[f'log_return_{future_list}'])]
#     # print(df_proc)
#     if not_use_overlap_to_train:
#         df_proc = df_proc.loc[(df_proc['train_flg'] == 1)]
#     train_test_zip = get_time_series_cross_val_splits(df_proc, cv=n_fold, embargo=3750)
#     print("entering time series cross validation loop")
#     importances = []
#     oof_pred = []
#     oof_valid = []
#     datetime = []
#     datetime_train = []
#
#     for split, train_test_split in enumerate(train_test_zip):
#         # gc.collect()
#
#         print(f"doing split {split + 1} out of {n_fold}")
#         train_split, test_split = train_test_split
#         train_split_index = df_proc['datetime'].isin(train_split)
#         test_split_index = df_proc['datetime'].isin(test_split)
#
#         train_dataset = lgb.Dataset(df_proc.loc[train_split_index, features],
#                                     df_proc.loc[train_split_index, f'log_return_{future_list}'].values,
#                                     feature_name=features,
#                                     )
#         val_dataset = lgb.Dataset(df_proc.loc[test_split_index, features],
#                                   df_proc.loc[test_split_index, f'log_return_{future_list}'].values,
#                                   feature_name=features,
#                                   )
#
#         print(f"number of train data: {len(df_proc.loc[train_split_index])}")
#         print(f"number of val data:   {len(df_proc.loc[test_split_index])}")
#
#         model = lgb.train(params=params,
#                           train_set=train_dataset,
#                           valid_sets=[train_dataset, val_dataset],
#                           valid_names=['tr', 'vl'],
#                           num_boost_round=5000,
#                           verbose_eval=100,
#                           feval=correlation,
#                           )
#         importances.append(model.feature_importance(importance_type='gain'))
#
#         # file = f'trained_model_id{future_list}_fold{split}.pkl'
#         # pickle.dump(model, open(file, 'wb'))
#         # print(f"Trained model was saved to 'trained_model_{future_list}_fold{split}.pkl'")
#         print("")
#         # print(df_proc.loc[test_split_index, features].iloc[:,0],'---------------------------')
#         oof_pred += list(model.predict(df_proc.loc[test_split_index, features]))
#         oof_valid += list(df_proc.loc[test_split_index, f'log_return_{future_list}'].values)
#         datetime += list(df_proc.loc[test_split_index, features].iloc[:,0])
#         # datetime_train += list(df_proc.loc[train_split_index, features].iloc[:,0])
#         print(df_proc.loc[test_split_index, features].iloc[:,0],'--------------')
#         # datetime_train_1 = pd.DataFrame(np.array(datetime_train),columns=['datetime'])
#         datetime_1 = pd.DataFrame(np.array(datetime), columns=['datetime'])
#         oof_pred_1 = pd.DataFrame(np.array(oof_pred),columns=['pred'])
#         oof_valid_1 = pd.DataFrame(np.array(oof_valid), columns=['vaild'])
#         final = pd.concat([datetime_1,oof_pred_1, oof_valid_1], axis=1)
#         # datetime_train_1.to_csv(f'train_{future_list}.csv')
#         final.to_csv(f'oof_pred_{future_list}.csv')
#
#     # plot_importance(np.array(importances), features, PLOT_TOP_N=20, figsize=(10, 5))
#
#     return oof_pred, oof_valid
#%%
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

# params = {
#     'early_stopping_rounds': 50,
#     'objective': 'cross_entropy',
#     'metric': {'cross_entropy', 'average_precision'},
# #     'metric': 'None',
#     'boosting_type': 'gbdt',
#     'max_depth': 5,
#     'verbose': -1,
#     'max_bin':600,
#     'min_data_in_leaf':50,
#     'learning_rate': 0.03,
#     'subsample': 0.7,
#     'subsample_freq': 1,
#     'feature_fraction': 1,
#     'lambda_l1': 0.5,
#     'lambda_l2': 2,
#     'seed':seed0,
#     'feature_fraction_seed': seed0,
#     'bagging_fraction_seed': seed0,
#     'drop_seed': seed0,
#     'data_random_seed': seed0,
#     'extra_trees': True,
#     'extra_seed': seed0,
#     'zero_as_missing': True,
#     "first_metric_only": True
#          }
# not_use_overlap_to_train = False
# def get_Xy_and_model_for_asset(df_proc, future_list):
#     df_proc = df_proc.loc[(df_proc[f'log_return_{future_list}'] == df_proc[f'log_return_{future_list}'])]
#     # print(df_proc)
#     if not_use_overlap_to_train:
#         df_proc = df_proc.loc[(df_proc['train_flg'] == 1)]
#     # train_test_zip = get_time_series_cross_val_splits(df_proc, cv=n_fold, embargo=3750)
#     df_proc['datetime'] = pd.to_datetime(df_proc['datetime'])
#
#     def classify(y):
#
#         if y < 0:
#             return 0
#         if y > 0:
#             return 1
#         else:
#             return -1
#     df_proc[f'log_return_{future_list}'] = df_proc[f'log_return_{future_list}'].apply(lambda x: classify(x))
#     print(df_proc[f'log_return_{future_list}'].value_counts())
#     df_proc= df_proc[~df_proc[f'log_return_{future_list}'].isin([-1])]
#
#     train = df_proc[df_proc.datetime<='2022-01-01']
#     test = df_proc[df_proc.datetime>='2022-01-04']
#     train = train.reset_index(drop=True)
#     test = test.reset_index(drop=True)
#     X_train = train.loc[:, features]
#     X_test = test.loc[:, features]
#     from sklearn.preprocessing import MinMaxScaler
#     sc = MinMaxScaler(feature_range=(0, 1))
#     X_train = sc.fit_transform(X_train)
#     X_test = sc.transform(X_test)
#     X_train = pd.DataFrame(X_train, columns=features)
#     X_test = pd.DataFrame(X_test, columns=features)
#     X_train_target = train.loc[:, [f'log_return_{future_list}']]
#     X_test_target = test.loc[:, [f'log_return_{future_list}']]
#
#     kf = TimeSeriesSplit(n_splits=10)
#     importances = []
#     oof_pred = np.zeros(len(X_test_target))
#     # oof_valid = X_test_target
#     # print(X_train)
#     final = pd.DataFrame()
#     for fold, (train_index, val_index) in enumerate(kf.split(X_train, X_train_target)):
#         # gc.collect()
#
#         x_train, x_val = X_train.loc[train_index,features], X_train.loc[val_index,features]
#         y_train, y_val = X_train_target.loc[train_index, f'log_return_{future_list}'].values, X_train_target.loc[val_index, f'log_return_{future_list}'].values
#         train_dataset = lgb.Dataset(x_train, y_train,feature_name=features)
#         val_dataset = lgb.Dataset(x_val, y_val, feature_name=features)
#
#         # train_dataset = lgb.Dataset(df_proc.loc[train_split_index, features],
#         #                             df_proc.loc[train_split_index, f'log_return_{future_list}'].values,
#         #                             feature_name=features,
#         #                             )
#         # val_dataset = lgb.Dataset(df_proc.loc[test_split_index, features],
#         #                           df_proc.loc[test_split_index, f'log_return_{future_list}'].values,
#         #                           feature_name=features,
#         #                           )
#
#         # print(f"number of train data: {len(df_proc.loc[train_split_index])}")
#         # print(f"number of val data:   {len(df_proc.loc[test_split_index])}")
#
#         model = lgb.train(params=params,
#                           train_set=train_dataset,
#                           valid_sets=[train_dataset,val_dataset],
#                           valid_names=['train', 'test'],
#                           num_boost_round=5000,
#                           verbose_eval=100,
#                           )
#         importances.append(model.feature_importance(importance_type='gain'))
#
#         # file = f'trained_model_id{future_list}_fold{split}.pkl'
#         # pickle.dump(model, open(file, 'wb'))
#         # print(f"Trained model was saved to 'trained_model_{future_list}_fold{split}.pkl'")
#         print("")
#         # print(df_proc.loc[test_split_index, features].iloc[:,0],'---------------------------')
#         oof_pred += model.predict(X_test, num_iteration=model.best_iteration)/kf.n_splits
#         # oof_valid += list(df_proc.loc[test_split_index, f'log_return_{future_list}'].values)
#         # datetime += list(df_proc.loc[test_split_index, features].iloc[:,0])
#         # datetime_train += list(df_proc.loc[train_split_index, features].iloc[:,0])
#         # print(df_proc.loc[test_split_index, features].iloc[:,0],'--------------')
#         from sklearn.metrics import roc_curve
#         from numpy import sqrt, argmax
#         fpr, tpr, thresholds = roc_curve(X_test_target, oof_pred)
#         gmeans = sqrt(tpr * (1 - fpr))
#         ix = argmax(gmeans)
#         print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
#         oof_pred = [1 if y > thresholds[ix] else 0 for y in oof_pred]
#
#         oof_pred_1 = pd.DataFrame(np.array(oof_pred),columns=['pred'])
#         oof_valid_1 = pd.DataFrame(np.array(X_test_target.loc[:, [f'log_return_{future_list}']]), columns=['valid'])
#         datetime = pd.DataFrame(np.array(test.iloc[:,0]), columns=['datetime'])
#         inverse = pd.DataFrame(sc.inverse_transform(X_test), columns=features)
#         close = pd.DataFrame(np.array(inverse.loc[:,[f'close_{future_list}']]), columns=['close'])
#         final = pd.concat([datetime,close,oof_pred_1, oof_valid_1], axis=1)
#         final.to_csv(f'oof_pred_15min_{future_list}_classification.csv')
#
#     # plot_importance(np.array(importances), features, PLOT_TOP_N=20, figsize=(10, 5))
#
#     return final
#%%
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

params = {
    'early_stopping_rounds': 50,
    'objective': 'regression',
    'metric': 'rmse',
#     'metric': 'None',
    'boosting_type': 'gbdt',
    'max_depth': 5,
    'verbose': -1,
    'max_bin':600,
    'min_data_in_leaf':50,
    'learning_rate': 0.03,
    'subsample': 0.7,
    'subsample_freq': 1,
    'feature_fraction': 1,
    'lambda_l1': 0.5,
    'lambda_l2': 2,
    'seed':seed0,
    'feature_fraction_seed': seed0,
    'bagging_fraction_seed': seed0,
    'drop_seed': seed0,
    'data_random_seed': seed0,
    'extra_trees': True,
    'extra_seed': seed0,
    'zero_as_missing': True,
    "first_metric_only": True
         }
not_use_overlap_to_train = False
def get_Xy_and_model_for_asset(df_proc, future_list):
    df_proc = df_proc.loc[(df_proc[f'log_return_{future_list}'] == df_proc[f'log_return_{future_list}'])]
    # print(df_proc)
    if not_use_overlap_to_train:
        df_proc = df_proc.loc[(df_proc['train_flg'] == 1)]
    # train_test_zip = get_time_series_cross_val_splits(df_proc, cv=n_fold, embargo=3750)
    df_proc['datetime'] = pd.to_datetime(df_proc['datetime'])

    train = df_proc[df_proc.datetime<='2022-01-01']
    test = df_proc[df_proc.datetime>='2022-01-04']
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    X_train = train.loc[:, features]
    X_test = test.loc[:, features]
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(0, 1))
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X_train = pd.DataFrame(X_train, columns=features)
    X_test = pd.DataFrame(X_test, columns=features)
    X_train_target = train.loc[:, [f'log_return_{future_list}']]
    X_test_target = test.loc[:, [f'log_return_{future_list}']]

    kf = TimeSeriesSplit(n_splits=10)
    importances = []
    oof_pred = np.zeros(len(X_test_target))
    # oof_valid = X_test_target
    # print(X_train)
    final = pd.DataFrame()
    for fold, (train_index, val_index) in enumerate(kf.split(X_train, X_train_target)):
        # gc.collect()

        x_train, x_val = X_train.loc[train_index,features], X_train.loc[val_index,features]
        y_train, y_val = X_train_target.loc[train_index, f'log_return_{future_list}'].values, X_train_target.loc[val_index, f'log_return_{future_list}'].values
        train_dataset = lgb.Dataset(x_train, y_train,feature_name=features)
        val_dataset = lgb.Dataset(x_val, y_val, feature_name=features)

        # train_dataset = lgb.Dataset(df_proc.loc[train_split_index, features],
        #                             df_proc.loc[train_split_index, f'log_return_{future_list}'].values,
        #                             feature_name=features,
        #                             )
        # val_dataset = lgb.Dataset(df_proc.loc[test_split_index, features],
        #                           df_proc.loc[test_split_index, f'log_return_{future_list}'].values,
        #                           feature_name=features,
        #                           )

        # print(f"number of train data: {len(df_proc.loc[train_split_index])}")
        # print(f"number of val data:   {len(df_proc.loc[test_split_index])}")

        model = lgb.train(params=params,
                          train_set=train_dataset,
                          valid_sets=[train_dataset,val_dataset],
                          valid_names=['train', 'test'],
                          num_boost_round=5000,
                          verbose_eval=100,
                          feval=correlation,
                          )
        importances.append(model.feature_importance(importance_type='gain'))

        # file = f'trained_model_id{future_list}_fold{split}.pkl'
        # pickle.dump(model, open(file, 'wb'))
        # print(f"Trained model was saved to 'trained_model_{future_list}_fold{split}.pkl'")
        print("")
        # print(df_proc.loc[test_split_index, features].iloc[:,0],'---------------------------')
        oof_pred += model.predict(X_test, num_iteration=model.best_iteration)/kf.n_splits
        # oof_valid += list(df_proc.loc[test_split_index, f'log_return_{future_list}'].values)
        # datetime += list(df_proc.loc[test_split_index, features].iloc[:,0])
        # datetime_train += list(df_proc.loc[train_split_index, features].iloc[:,0])
        # print(df_proc.loc[test_split_index, features].iloc[:,0],'--------------')

        oof_pred_1 = pd.DataFrame(np.array(oof_pred),columns=['pred'])
        oof_valid_1 = pd.DataFrame(np.array(X_test_target.loc[:, [f'log_return_{future_list}']]), columns=['valid'])
        datetime = pd.DataFrame(np.array(test.iloc[:,0]), columns=['datetime'])
        inverse = pd.DataFrame(sc.inverse_transform(X_test), columns=features)
        close = pd.DataFrame(np.array(inverse.loc[:,[f'close_{future_list}']]), columns=['close'])
        final = pd.concat([datetime,close,oof_pred_1, oof_valid_1], axis=1)
        final.to_csv(f'oof_pred_120min_{future_list}_minmax.csv')

    # plot_importance(np.array(importances), features, PLOT_TOP_N=20, figsize=(10, 5))

    return final
#%%
# oof = [[] for id in range(14)]
#
# all_oof_pred = []
# all_oof_valid = []
# all_oof_weight = []
#
# for asset_id, asset_name in enumerate(future_list):
#     print(f"Training model for {asset_name:<16} (ID={asset_id:<2})")
#
#     oof_pred, oof_valid = get_Xy_and_model_for_asset(feat, asset_name)
#
#     # weight_temp = float(df_asset_details.loc[df_asset_details['Asset_ID'] == asset_id, 'Weight'])
#
#     all_oof_pred += oof_pred
#     all_oof_valid += oof_valid
#     # all_oof_weight += [weight_temp] * len(oof_pred)
#
#     oof[asset_id] = corr_score(np.array(oof_pred), np.array(oof_valid))
#
#     print(f'OOF corr score of {asset_name} (ID={asset_id}) is {oof[asset_id]:.5f}')
#     print('')
#     print('')
#%%
from sklearn.metrics import classification_report,accuracy_score
oof = [[] for id in range(54)]

all_oof_pred = []
all_oof_valid = []
all_oof_weight = []

for asset_id, asset_name in enumerate(future_list):
    print(f"Training model for {asset_name:<16} (ID={asset_id:<2})")

    final = get_Xy_and_model_for_asset(feat, asset_name)

    # weight_temp = float(df_asset_details.loc[df_asset_details['Asset_ID'] == asset_id, 'Weight'])

    # all_oof_pred += oof_pred
    # all_oof_valid += oof_valid
    # all_oof_weight += [weight_temp] * len(oof_pred)

    oof[asset_id] = corr_score(final.pred, final.valid)
    # oof[asset_id] = accuracy_score(final.pred, final.valid)
    # oof[asset_id] = rmspe(final.pred, final.valid)
    # print(f'OOF classification of {asset_name} (ID={asset_id} is {classification_report(final.pred, final.valid)}')
    print(f'OOF corr score of {asset_name} (ID={asset_id}) is {oof[asset_id]:.5f}')
    print('')
    print('')
#%%
data_time['log_return'] = np.log(data_time.close/data_time.close.shift(1))
data_time['volatility'] = data_time.log_return.rolling(30).std(ddof=0)*np.sqrt(252)
data_time = data_time.fillna(method='bfill')
#%%
from scipy.stats import ks_2samp
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
for asset_id, asset_name in enumerate(future_list):
    print(asset_id)
    print(asset_name)
#%%
data = pd.read_csv('oof_pred_120min_a_minmax.csv')
#%%
data['datetime'] = pd.to_datetime(data['datetime'])
#%%
data = data.set_index('datetime').resample('2H').apply({'close':'last','pred':'last','valid':'last'})
#%%
data_1 = data.dropna(axis=0)
#%%
# test = len(bu[(bu.pred>0)&(bu.valid>0)|((bu.pred<0)&(bu.valid<0))])/len(bu)
#%%
data.to_csv('final_a_15min.csv')
#%%
final = pd.DataFrame()
for i in future_list:
    dirFuture = 'E:\\luojie\\MachineLearning\\oof_pred_120min_%s_minmax.csv' %(i)
    data = pd.read_csv(dirFuture)
    # columns = ['index', 'datetime', 'close', 'pred', 'valid']
    # data.columns = columns
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime').resample('2H').apply({'datetime':'last','close':'last', 'pred':'last', 'valid':'last'})
    data = data.dropna(axis=0)
    data = data.reset_index()
    data['future'] = i
    print(data)
    final = final.append(data)
    # print(final)
    # final = final.append(data)
#%%
merge = pd.DataFrame()
merge[final.columns] = 0
for i in future_list:
    # print(i)
    merge = merge.merge(final.loc[final['future'] == i,['datetime','close',
                    'pred', 'valid','future']].copy(), on='datetime',how='outer',suffixes=['',"_"+str(i)])
#%%
merge = merge.drop(final.columns.drop("datetime"), axis=1)
merge = merge.sort_values('datetime', ascending=True)
merge = merge.drop_duplicates('datetime', keep='last')
#%%
for i in future_list:
    merge = merge.drop([f'valid_{i}'], axis=1)
    merge[f'close_{i}'] = merge[f'close_{i}'].fillna(method='ffill',limit=100)
    merge[f'pred_{i}'] = merge[f'pred_{i}'].fillna(0)
    merge[f'future_{i}'] = merge[f'future_{i}'].fillna(method='ffill',limit=100)
#%%
merge.to_csv('all_future_hengjiemian_120min.csv')
#%%
merge = merge.drop(['close_bb','pred_bb','close_ni','pred_ni'],axis=1)
#%%
merge = merge.reset_index(drop=True)
#%%
# a = pd.DataFrame()
# for i in future_list:
#     b = merge.iloc[:, merge.columns.str.startswith(f'pred_{i}')]
#     print(b)
#     a = pd.concat([a,b], axis=1)
# #%%
# a = a.drop_duplicates().T.drop_duplicates().T
# #%%
# a['max_idx'] = a.idxmax(axis=1)

