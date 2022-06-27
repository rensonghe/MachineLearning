import MySQLdb
import pandas as pd
import numpy as np
import itertools
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
future_list = ['bu','eb','eg','lu','MA','nr','pg','ru','SA','sc','sp','TA','UR']
#%%
future_list = ['a','b','c','CJ','cs','m','OI','p','RM','y']
#%%
startdate = 20160101

df = pd.DataFrame()
for i, element in enumerate(future_list):
    print(element)

    conn = MySQLdb.connect(host="rm-bp1yvd1jr36kmmv834o.mysql.rds.aliyuncs.com", user="LJD", passwd="HZLJDcl123456",
                           db="datainfo", port=3306, charset='utf8')
    cursor = conn.cursor()
    sql = "select a.date,contract,open,close,openint,high,low,vol,volmoney from day_price a where contract like '%s____' and " \
          "openint=(select max(openint) from day_price where date=a.date  and contract like '%s____')  and a.date>='%s' group by date order by date;"%(element,element,startdate)
    cursor.execute(sql)
    open_price = np.array(cursor.fetchall())
    day_price = pd.DataFrame({'date': open_price[:, 0], 'contract': open_price[:, 1], 'open': open_price[:, 2], 'close': open_price[:, 3], 'openint': open_price[:, 4], 'high': open_price[:, 5], 'low': open_price[:, 6], 'vol': open_price[:, 7], 'volmoney': open_price[:, 8]})
    # day_price = pd.DataFrame(
        # {'date': open_price[:, 0], 'contract': open_price[:, 1], 'close': open_price[:, 3]})
    # i += 1
    df = df.append(day_price)
    print(df)
#%%
# conn = MySQLdb.connect(host="rm-bp1yvd1jr36kmmv834o.mysql.rds.aliyuncs.com", user="LJD", passwd="HZLJDcl123456",
#                        db="datainfo", port=3306, charset='utf8')
# cursor = conn.cursor()
# sql = 'select distinct contract from day_price a where contract like "a____" and openint=(select max(openint) from day_price where date=a.date  and contract like "a____")  and a.date>="20160101" group by date order by date;'
# cursor.execute(sql)
# main_list_a = np.array(cursor.fetchall())
# main_list_a = main_list_a.tolist()
# main_list_a = list(itertools.chain.from_iterable(main_list_a))
#%%
df.to_csv('day_price.csv')
#%%
df = pd.read_csv('day_price.csv')
df = df.iloc[:,1:]
#%%
import ta
import time
from datetime import datetime

all_data = pd.DataFrame()
for j in future_list:
    conn = MySQLdb.connect(host="rm-bp1yvd1jr36kmmv834o.mysql.rds.aliyuncs.com", user="LJD", passwd="HZLJDcl123456",
                           db="datainfo", port=3306, charset='utf8')
    cursor = conn.cursor()
    sql = "select distinct contract from day_price a where contract like '%s____' and openint=(select max(openint) from day_price where date=a.date  and contract like '%s____')  and a.date>='20160101' group by date order by date;"%(j,j)
    cursor.execute(sql)
    main_list_a = np.array(cursor.fetchall())
    main_list_a = main_list_a.tolist()
    main_list_a = list(itertools.chain.from_iterable(main_list_a))
    print(j,'------------------------------------j')
    for i in main_list_a:
        dirFuture = 'Z:\\Min_TQ\\%s\\%s.csv' %(j,i)
        fp = pd.read_csv(dirFuture)
        # print(fp.shape)
        fp.columns = ['datetime','epoch_time', 'open', 'high', 'low', 'close', 'volume', 'open_oi', 'close_oi']
        # fp['datetime'] = fp['datetime'].strftime('')
        date1 = df[df['contract'] == i]['date'].values[0] # 返回这个合约开始日期
        date2 = df[df['contract'] == i]['date'].values[-1]  # 返回这个合约开始日期
        future = fp[(fp.datetime >= str(date1))&(fp.datetime <= str(date2))]
        future['contract'] = i
        future['future'] = j
        future['datetime'] = pd.to_datetime(future['datetime'])
        future['close'] = future.close.astype(float)
        # future = future.set_index('datetime').resample('15min').apply({'epoch_time':'last','open':'last','high':'last','low':'last',
        #                                                                'close':'last','volume':'last','future':'last'})
        # future = future.dropna(axis=0)
        # future['log_return'] = np.log(future.close/future.close.shift(240))
        # future['log_return'] = future.log_return.shift(-240)
        # print(future)
        all_data = all_data.append(future)
        print(all_data)
#%%
all_data = all_data.dropna(axis=0)
#%%
data = df.set_index('date').sort_index()
#%%
min_data = pd.read_csv('Z:/Min_TQ/a/a1605.csv')
col = ['datetime','epoch_time', 'open', 'high', 'low', 'close', 'volume', 'open_oi', 'close_oi']
min_data.columns = col
data_time = min_data[(min_data.datetime>='2016-01-04')&(min_data.datetime<='2016-03-08')]
#%%
data_time = pd.read_csv('all_data_huagong.csv')
data_time = data_time.iloc[:,1:]
#%%
import ta
from ta.volume import ForceIndexIndicator, EaseOfMovementIndicator
from ta.volatility import BollingerBands, KeltnerChannel, DonchianChannel
from ta.trend import MACD, macd_diff, macd_signal, SMAIndicator
from ta.momentum import stochrsi, stochrsi_k, stochrsi_d
data_time = all_data.copy()
#%%
import talib
from talib import stream

#%%
data_time['open'] = data_time['open'].astype(float)
data_time['close'] = data_time['close'].astype(float)
data_time['open_oi'] = data_time['open_oi'].astype(float)
data_time['high'] = data_time['high'].astype(float)
data_time['low'] = data_time['low'].astype(float)
data_time['volume'] = data_time['volume'].astype(float)
data_time['close_oi'] = data_time['close_oi'].astype(float)
#%%
forceindex_30 = ForceIndexIndicator(close=data_time['close'], volume=data_time['volume'], window=30)
data_time['forceindex_30'] = forceindex_30.force_index()
easyofmove_30 = EaseOfMovementIndicator(high=data_time['high'], low=data_time['low'], volume=data_time['volume'], window=30)
data_time['easyofmove_30'] = easyofmove_30.ease_of_movement()
# easyofmove_60 = EaseOfMovementIndicator(high=data_time['high'], low=data_time['low'], volume=data_time['volume'], window=60)
# data_time['easyofmove_60'] = easyofmove_60.ease_of_movement()
bollingband_30 = BollingerBands(close=data_time['close'], window=30, window_dev=30)
data_time['bollingerhband_30'] = bollingband_30.bollinger_hband()
data_time['bollingerlband_30'] = bollingband_30.bollinger_lband()
data_time['bollingermband_30'] = bollingband_30.bollinger_mavg()
data_time['bollingerpband_30'] = bollingband_30.bollinger_pband()
data_time['bollingerwband_30'] = bollingband_30.bollinger_wband()
# bollingband_60 = BollingerBands(close=data_time['close'], window=60, window_dev=60)
# data_time['bollingerhband_60'] = bollingband_60.bollinger_hband()
# data_time['bollingerlband_60'] = bollingband_60.bollinger_lband()
# data_time['bollingermband_60'] = bollingband_60.bollinger_mavg()
# data_time['bollingerpband_60'] = bollingband_60.bollinger_pband()
# data_time['bollingerwband_60'] = bollingband_60.bollinger_wband()
keltnerchannel_30 = KeltnerChannel(high=data_time['high'], low=data_time['low'], close=data_time['close'], window=30)
data_time['keltnerhband_30'] = keltnerchannel_30.keltner_channel_hband()
data_time['keltnerlband_30'] = keltnerchannel_30.keltner_channel_lband()
data_time['keltnerwband_30'] = keltnerchannel_30.keltner_channel_wband()
data_time['keltnerpband_30'] = keltnerchannel_30.keltner_channel_pband()
# keltnerchannel_60 = KeltnerChannel(high=data_time['high'], low=data_time['low'], close=data_time['close'], window=60)
# data_time['keltnerhband_60'] = keltnerchannel_60.keltner_channel_hband()
# data_time['keltnerlband_60'] = keltnerchannel_60.keltner_channel_lband()
# data_time['keltnerwband_60'] = keltnerchannel_60.keltner_channel_wband()
# data_time['keltnerpband_60'] = keltnerchannel_60.keltner_channel_pband()
donchichannel_30 = DonchianChannel(high=data_time['high'], low=data_time['low'], close=data_time['close'],window=30)
data_time['donchimband_30'] = donchichannel_30.donchian_channel_mband()
data_time['donchilband_30'] = donchichannel_30.donchian_channel_lband()
data_time['donchipband_30'] = donchichannel_30.donchian_channel_pband()
data_time['donchiwband_30'] = donchichannel_30.donchian_channel_wband()
# donchichannel_60 = DonchianChannel(high=data_time['high'], low=data_time['low'], close=data_time['close'],window=60)
# data_time['donchimband_60'] = donchichannel_60.donchian_channel_mband()
# data_time['donchilband_60'] = donchichannel_60.donchian_channel_lband()
# data_time['donchipband_60'] = donchichannel_60.donchian_channel_pband()
# data_time['donchiwband_60'] = donchichannel_60.donchian_channel_wband()
macd_30 = MACD(close=data_time['close'],window_fast=30, window_slow=60)
data_time['macd_30'] = macd_30.macd()
# macd_60 = MACD(close=data_time['close'],window_fast=60, window_slow=120)
# data_time['macd_60'] = macd_60.macd()
data_time['macdsignal_30'] = macd_signal(close=data_time['close'],window_fast=30,window_slow=60)
# data_time['macdsignal_60'] = macd_signal(close=data_time['close'],window_fast=60,window_slow=120)
data_time['macddiff_30'] = macd_diff(close=data_time['close'],window_fast=30, window_slow=60)
# data_time['macddiff_60'] = macd_diff(close=data_time['close'],window_fast=60, window_slow=120)
smafast_30 = SMAIndicator(close=data_time['close'],window=30)
data_time['smafast_30'] = smafast_30.sma_indicator()
# smafast_60 = SMAIndicator(close=data_time['close'],window=60)
# data_time['smafast_60'] = smafast_60.sma_indicator()
# smaslow_120 = SMAIndicator(close=data_time['close'],window=120)
# data_time['smaslow_120'] = smaslow_120.sma_indicator()
data_time['stochrsi_30'] = stochrsi(close=data_time['close'],window=30, smooth1=30, smooth2=15)
data_time['stochrsi_k_30'] = stochrsi_k(close=data_time['close'],window=30, smooth1=30, smooth2=15)
data_time['stochrsi_d_30'] = stochrsi_d(close=data_time['close'],window=30, smooth1=30, smooth2=15)
# data_time['stochrsi_60'] = stochrsi(close=data_time['close'],window=60, smooth1=30, smooth2=15)
# data_time['stochrsi_k_60'] = stochrsi_k(close=data_time['close'],window=60, smooth1=30, smooth2=15)
# data_time['stochrsi_d_60'] = stochrsi_d(close=data_time['close'],window=60, smooth1=30, smooth2=15)
#%%
data_time['CDL2CROWS'] = talib.CDL2CROWS(data_time['open'], data_time['high'],data_time['low'],data_time['close'])
data_time['CDL3INSIDE'] = talib.CDL3INSIDE(data_time['open'], data_time['high'],data_time['low'],data_time['close'])
data_time['CDL3OUTSIDE'] = talib.CDL3OUTSIDE(data_time['open'], data_time['high'],data_time['low'],data_time['close'])
data_time['CDL3STARSINSOUTH'] = talib.CDL3STARSINSOUTH(data_time['open'], data_time['high'],data_time['low'],data_time['close'])
data_time['CDLADVANCEBLOCK'] = talib.CDLADVANCEBLOCK(data_time['open'], data_time['high'],data_time['low'],data_time['close'])
#%%
data_time = data_time.fillna(method='bfill')
data_time = data_time.replace(np.inf, 1)
data_time = data_time.replace(-np.inf, -1)
#%%
data_time['log_forceindex_5'] = np.log(np.array(data_time['log_forceindex_5'])/np.roll(np.array(data_time['log_forceindex_5']),5))
data_time['log_forceindex_15'] = np.log(np.array(data_time['log_forceindex_15'])/np.roll(np.array(data_time['log_forceindex_15']),15))
data_time['log_forceindex_20'] = np.log(np.array(data_time['log_forceindex_20'])/np.roll(np.array(data_time['log_forceindex_20']),20))

#%%
from sklearn import preprocessing

label = preprocessing.LabelEncoder()
data_time['future'] = label.fit_transform(data_time['future'])
data_time = data_time.drop(['contract'], axis=1)
#%%
import time
from datetime import datetime
data_time['datetime'] = pd.to_datetime(data_time['datetime'])
data_time['time'] = data_time['datetime'].dt.strftime('%H:%M:%S')
#%%
data_time['close_mean_5'] = np.log(np.array(data_time['close'])/np.roll(np.append(np.convolve(np.array(data_time['close']),
                            np.ones(5)/5, mode='vaild'),np.ones(5-1)),5-1))
data_time['close_mean_15'] = np.log(np.array(data_time['close'])/np.roll(np.append(np.convolve(np.array(data_time['close']),
                            np.ones(15)/15, mode='vaild'),np.ones(15-1)),15-1))
data_time['close_mean_20'] = np.log(np.array(data_time['close'])/np.roll(np.append(np.convolve(np.array(data_time['close']),
                            np.ones(20)/20, mode='vaild'),np.ones(20-1)),20-1))
#%%
data_time['log_return_5'] = np.log(np.array(data_time['close'])/np.roll(np.array(data_time['close']),5))
data_time['log_return_15'] = np.log(np.array(data_time['close'])/np.roll(np.array(data_time['close']),15))
data_time['log_return_20'] = np.log(np.array(data_time['close'])/np.roll(np.array(data_time['close']),20))
#%%
data_time['mean_log_5'] = data_time.iloc[:,data_time.columns.str.startswith('close_mean_5')].rolling(5).mean()
data_time['mean_log_15'] = data_time.iloc[:,data_time.columns.str.startswith('close_mean_5')].rolling(15).mean()
data_time['mean_log_20'] = data_time.iloc[:,data_time.columns.str.startswith('close_mean_5')].rolling(20).mean()
#%%
data_time_9_00 = data_time[data_time['time'].isin(['09:00:00'])]
# data_time_9_15 = data_time[data_time['time'].isin(['09:15:00'])]   #开仓15分钟
data_time_9_30 = data_time[data_time['time'].isin(['09:30:00'])]
# data_time_10_00 = data_time[data_time['time'].isin(['10:00:00'])]  #开仓60分钟
data_time_11_30 = data_time[data_time['time'].isin(['11:29:00'])]
# data_time_13_30 = data_time[data_time['time'].isin(['13:30:00'])]
# data_time_15_00 = data_time[data_time['time'].isin(['14:59:00'])]
data_time_21_30 = data_time[data_time['time'].isin(['21:30:00'])]
# data_time_21_15 = data_time[data_time['time'].isin(['21:15:00'])]   #开仓15分钟
# data_time_22_00 = data_time[data_time['time'].isin(['22:00:00'])]   #开仓60分钟
# data_time_9_00 = data_time[data_time['time'].isin(['09:00:00'])]
# data_time_2_00 = data_time[data_time['time'].isin(['01:59:00'])]
#%%
data_time_9_00['date'] = data_time_9_00['datetime'].dt.strftime('%Y-%m-%d')
# data_time_9_15['date'] = data_time_9_15['datetime'].dt.strftime('%Y-%m-%d')
# data_time_10_00['date'] = data_time_10_00['datetime'].dt.strftime('%Y-%m-%d')
data_time_9_30['date'] = data_time_9_30['datetime'].dt.strftime('%Y-%m-%d')
data_time_11_30['date'] = data_time_11_30['datetime'].dt.strftime('%Y-%m-%d')
# data_time_13_30['date'] = data_time_13_30['datetime'].dt.strftime('%Y-%m-%d')
# data_time_15_00['date'] = data_time_15_00['datetime'].dt.strftime('%Y-%m-%d')
data_time_21_30['date'] = data_time_21_30['datetime'].dt.strftime('%Y-%m-%d')
# data_time_21_15['date'] = data_time_21_15['datetime'].dt.strftime('%Y-%m-%d')
# data_time_22_00['date'] = data_time_22_00['datetime'].dt.strftime('%Y-%m-%d')
# data_time_9_00['date'] = data_time_9_00['datetime'].dt.strftime('%Y-%m-%d')
# data_time_2_00['date'] = data_time_2_00['datetime'].dt.strftime('%Y-%m-%d')
#%%
# data_time_9_15 = data_time_9_15.reset_index(drop=True)
# data_time_10_00 = data_time_10_00.reset_index(drop=True)
data_time_9_30 = data_time_9_30.reset_index(drop=True)
data_time_11_30 = data_time_11_30.reset_index(drop=True)
# data_time_13_30 = data_time_13_30.reset_index(drop=True)
# data_time_15_00 = data_time_15_00.reset_index(drop=True)
data_time_21_30 = data_time_21_30.reset_index(drop=True)
# data_time_21_15 = data_time_21_15.reset_index(drop=True)
# data_time_22_00 = data_time_22_00.reset_index(drop=True)
#%%
data_time_9_00 = data_time_9_00.reset_index(drop=True)
# data_time_2_00 = data_time_2_00.reset_index(drop=True)
#%%
final_data_9_00 = data_time_9_00[['date','close','future']]
final_data_9_30 = data_time_9_30[['date','close','future']]
final_data_11_30 = data_time_11_30[['date','close','future']]
# final_data_13_30 = data_time_13_30[['date','close','future']]
# final_data_15_00 = data_time_15_00[['date','close','future']]
# final_data_2_00 = data_time_2_00[['date','close','future']]
final_data_21_30 = data_time_21_30[['date','close','future']]
#%%
a = pd.merge(data_time_9_30, data_time_21_30, on=['date', 'future'], how='left')
# a = pd.merge(data_time_9_15, data_time_21_15, on=['date', 'future'], how='left')
# a = pd.merge(data_time_10_00, data_time_22_00, on=['date', 'future'], how='left')
#%%
# a['future_y'] = a['future']
#%%
a = a.set_index(['date', 'future'])
#%%
df1 = a[a.isna().any(axis=1)] #have nan
df2 = a[~a.isna().any(axis=1)] # do not have nan
#%%
col_9_30 = df1.iloc[:,0:11]
col_21_30 = df2.iloc[:,11:]
col = col_21_30.columns
col_9_30.columns = col
#%%
# col_21_15 = col_21_15.reset_index()
# c_21_15_11_30 = pd.merge(col_21_15, final_data_11_30, on=['date','future'], how='left')
# c_21_15_11_30['close_11_30'] = c_21_15_11_30['close'].shift(-1)
# c_21_15_11_30 = c_21_15_11_30.dropna(axis=0)
# c_21_15_11_30 = c_21_15_11_30.drop(['close'],axis=1)
#
# col_9_15 = col_9_15.reset_index()
# c_9_15_11_30 = pd.merge(col_9_15, final_data_11_30,on=['date','future'], how='left')
# c_9_15_11_30['close_11_30'] = c_9_15_11_30['close']
# c_9_15_11_30 = c_9_15_11_30.drop(['close'],axis=1)
#
# c_final_data = pd.concat([c_21_15_11_30,c_9_15_11_30])
# c_final_data = c_final_data.set_index(['date','future']).sort_index()
# c_final_data['target'] = np.log(c_final_data['close_11_30']/c_final_data['close_y'])*100
# c_final_data = c_final_data.reset_index()
# c_final_data = c_final_data[~c_final_data['date'].isin(['2022-05-13'])]
#%%
# col_22_00 = col_22_00.reset_index()
# c_22_00_11_30 = pd.merge(col_22_00, final_data_11_30, on=['date','future'], how='left')
# c_22_00_11_30['close_11_30'] = c_22_00_11_30['close'].shift(-1)
# c_22_00_11_30 = c_22_00_11_30.dropna(axis=0)
# c_22_00_11_30 = c_22_00_11_30.drop(['close'],axis=1)
#
# col_10_00 = col_10_00.reset_index()
# c_10_00_11_30 = pd.merge(col_10_00, final_data_11_30,on=['date','future'], how='left')
# c_10_00_11_30['close_11_30'] = c_10_00_11_30['close']
# c_10_00_11_30 =c_10_00_11_30.drop(['close'],axis=1)
#
# c_final_data = pd.concat([c_22_00_11_30,c_10_00_11_30])
# c_final_data = c_final_data.set_index(['date','future']).sort_index()
# c_final_data['target'] = np.log(c_final_data['close_11_30']/c_final_data['close_y'])*100
# c_final_data = c_final_data.reset_index()
# c_final_data = c_final_data[~c_final_data['date'].isin(['2022-05-13'])]
#%% 夜盘品种夜盘收盘平仓 白天盘品种转天9点平仓
# col_21_15 = col_21_15.reset_index()
# c_21_15_2_00 = pd.merge(col_21_15, final_data_2_00, on=['date','future'], how='left')
# c_21_15_2_00['close_x'] = c_21_15_2_00['close'].shift(-1)
# c_21_15_2_00 = c_21_15_2_00.dropna(axis=0)
# c_21_15_2_00 = c_21_15_2_00.drop(['close'],axis=1)

# col_9_15 = col_9_15.reset_index()
# c_9_15_9_00 = pd.merge(col_9_15, final_data_9_00,on=['date','future'], how='left')
# c_9_15_9_00['close_x'] = c_9_15_9_00['close'].shift(-2)
# c_9_15_9_00 = c_9_15_9_00.dropna(axis=0)
# c_9_15_9_00 = c_9_15_9_00.drop(['close'],axis=1)

# c_final_data = pd.concat([c_21_15_2_00,c_9_15_9_00])
# c_final_data = c_final_data.set_index(['date','future']).sort_index()
# c_final_data['target'] = np.log(c_final_data['close_x']/c_final_data['close_y'])*100
# c_final_data = c_final_data.reset_index()
# c_final_data = c_final_data[~c_final_data['date'].isin(['2022-05-13'])]
#%%
# col_21_30 = col_21_30.reset_index()
# c_21_30_11_30 = pd.merge(col_21_30, final_data_11_30, on=['date','future'], how='left')
# c_21_30_11_30['close_11_30'] = c_21_30_11_30['close'].shift(-1)
# c_21_30_11_30 = c_21_30_11_30.dropna(axis=0)
# c_21_30_11_30 = c_21_30_11_30.drop(['close'],axis=1)
#
# col_9_30 = col_9_30.reset_index()
# c_9_30_11_30 = pd.merge(col_9_30, final_data_11_30,on=['date','future'], how='left')
# c_9_30_11_30['close_11_30'] = c_9_30_11_30['close']
# c_9_30_11_30 = c_9_30_11_30.drop(['close'],axis=1)
#
# c_final_data = pd.concat([c_21_30_11_30,c_9_30_11_30])
# c_final_data = c_final_data.set_index(['date','future']).sort_index()
# c_final_data['target'] = np.log(c_final_data['close_11_30']/c_final_data['close_y'])*100
# c_final_data = c_final_data.reset_index()
# c_final_data = c_final_data[~c_final_data['date'].isin(['2022-06-01'])]
#%%
# c_21_30_13_30 = pd.merge(col_21_30, final_data_13_30, on=['date','future'], how='left')
# c_21_30_13_30['close_13_30'] = c_21_30_13_30['close'].shift(-1)
# c_21_30_13_30 = c_21_30_13_30.dropna(axis=0)
# c_21_30_13_30 = c_21_30_13_30.drop(['close'],axis=1)
#
# c_9_30_13_30 = pd.merge(col_9_30, final_data_13_30,on=['date','future'], how='left')
# c_9_30_13_30['close_13_30'] = c_9_30_13_30['close']
# c_9_30_13_30 = c_9_30_13_30.drop(['close'],axis=1)
#
# c_final_data_2 = pd.concat([c_21_30_13_30,c_9_30_13_30])
# c_final_data_2 = c_final_data_2.set_index(['date','future']).sort_index()
# c_final_data_2['target'] = np.log(c_final_data_2['close_13_30']/c_final_data_2['close_y'])*100
#
# c_final_data_2 = c_final_data_2.reset_index()
# c_final_data_2 = c_final_data_2[~c_final_data_2['date'].isin(['2022-05-13'])]
#%%
# col_21_15 = col_21_15.reset_index()
# c_21_15_15_00 = pd.merge(col_21_15, final_data_15_00, on=['date','future'], how='left')
# c_21_15_15_00['close_15_00'] = c_21_15_15_00['close'].shift(-1)
# c_21_15_15_00 = c_21_15_15_00.dropna(axis=0)
# c_21_15_15_00 = c_21_15_15_00.drop(['close'],axis=1)
#
# # col_9_15 = col_9_15.reset_index()
# c_9_15_15_00 = pd.merge(col_9_15, final_data_15_00,on=['date','future'], how='left')
# c_9_15_15_00['close_15_00'] = c_9_15_15_00['close']
# c_9_15_15_00 = c_9_15_15_00.drop(['close'],axis=1)
#
# c_final_data_2 = pd.concat([c_21_15_15_00,c_9_15_15_00])
# c_final_data_2 = c_final_data_2.set_index(['date','future']).sort_index()
# c_final_data_2['target'] = np.log(c_final_data_2['close_15_00']/c_final_data_2['close_y'])*100
# c_final_data_2 = c_final_data_2.reset_index()
# c_final_data_2 = c_final_data_2[~c_final_data_2['date'].isin(['2022-05-13'])]
#%%
# col_21_15 = col_21_15.reset_index()
# c_22_00_15_00 = pd.merge(col_22_00, final_data_15_00, on=['date','future'], how='left')
# c_22_00_15_00['close_15_00'] = c_22_00_15_00['close'].shift(-1)
# c_22_00_15_00 = c_22_00_15_00.dropna(axis=0)
# c_22_00_15_00 = c_22_00_15_00.drop(['close'],axis=1)
#
# # col_9_15 = col_9_15.reset_index()
# c_10_00_15_00 = pd.merge(col_10_00, final_data_15_00,on=['date','future'], how='left')
# c_10_00_15_00['close_15_00'] = c_10_00_15_00['close']
# c_10_00_15_00 = c_10_00_15_00.drop(['close'],axis=1)
#
# c_final_data_2 = pd.concat([c_22_00_15_00,c_10_00_15_00])
# c_final_data_2 = c_final_data_2.set_index(['date','future']).sort_index()
# c_final_data_2['target'] = np.log(c_final_data_2['close_15_00']/c_final_data_2['close_y'])*100
# c_final_data_2 = c_final_data_2.reset_index()
# c_final_data_2 = c_final_data_2[~c_final_data_2['date'].isin(['2022-05-13'])]
#%% 晚上9点半开仓，转天21点半平仓
col_21_30 = col_21_30.reset_index()
c_21_30_21_30 = pd.merge(col_21_30, final_data_21_30, on=['date','future'], how='left')
c_21_30_21_30['close_21_30'] = c_21_30_21_30['close'].shift(-1)
c_21_30_21_30 = c_21_30_21_30.dropna(axis=0)
c_21_30_21_30 = c_21_30_21_30.drop(['close'],axis=1)
c_21_30_21_30['target'] = np.log(c_21_30_21_30['close_21_30']/c_21_30_21_30['close_y'])
#%%
col_9_30 = col_9_30.reset_index()
c_9_30_9_30 = pd.merge(col_9_30, final_data_9_30,on=['date','future'], how='left')
c_9_30_9_30['close_9_30'] = c_9_30_9_30['close']
c_9_30_9_30 = c_9_30_9_30.drop(['close'],axis=1)
c_9_30_9_30['target'] = np.log(c_9_30_9_30['close_9_30']/c_9_30_9_30['close_y'])
#%%
c_final_data_3 = pd.concat([c_21_30_21_30,c_9_30_9_30])
c_final_data_3 = c_final_data_3.set_index(['date','future']).sort_index()
#%%
c_final_data_3 = c_final_data_3.drop(['close_9_30','close_21_30'], axis=1)
# c_final_data_3['target_21_30'] = np.log(c_final_data_3['close_21_30']/c_final_data_3['close_y'])*100
c_final_data_3 = c_final_data_3.reset_index()
#%%
c_final_data_3 = c_final_data_3[~c_final_data_3['date'].isin(['2022-06-22'])]
#%%0.498982
c_final_data.to_csv('all_contract_min_data_22_00_10_00_11_30_close_60var.csv')
#%%
c_final_data_2.to_csv('all_contract_min_data_22_00_10_00_15_00_close_60var.csv')
#%%
c_final_data_3.to_csv('all_data_all_future_anotherday.csv')
#%%
c_final_data.to_csv('test_51var.csv')