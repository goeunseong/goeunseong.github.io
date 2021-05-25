```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image
from sklearn.preprocessing import LabelEncoder

file_path = "mobileTrendDataset.csv"
df = pd.read_csv(file_path, sep='\t')
```

### 파일 불러오기
- yearID: 연도 구분 (Y1 다음년도가 Y2)
- weekIndex: 주차 1~52주
- rank: 해당 주차의 앱 순위
- appID: 앱 구분자
- uniqueInstall: 앱 설치자 수(중복 불포함)
- UU: 주간 이용자 수
- useRate: 설치자중 이용자 비율
- totalDuration: 총 이용시간(분)
- avgDuration: 이용자당 평균 이용시간(분)
- dayCounts: 이용자의 평균 접속일수
- genreID: 장르 구분


```python
df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>yearID</th>
      <th>weekIndex</th>
      <th>rank</th>
      <th>appID</th>
      <th>uniqueInstall</th>
      <th>UU</th>
      <th>useRate</th>
      <th>totalDuration</th>
      <th>avgDuration</th>
      <th>dayCounts</th>
      <th>genreID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Y1</td>
      <td>1</td>
      <td>1</td>
      <td>N0001</td>
      <td>7027982</td>
      <td>6690166</td>
      <td>95.19</td>
      <td>1314483816</td>
      <td>196.48</td>
      <td>4.10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Y1</td>
      <td>1</td>
      <td>2</td>
      <td>N0002</td>
      <td>8979953</td>
      <td>6012020</td>
      <td>66.95</td>
      <td>598676952</td>
      <td>99.58</td>
      <td>4.09</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Y1</td>
      <td>1</td>
      <td>3</td>
      <td>N0003</td>
      <td>8527132</td>
      <td>5475405</td>
      <td>64.21</td>
      <td>818737310</td>
      <td>149.53</td>
      <td>4.33</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Y1</td>
      <td>1</td>
      <td>4</td>
      <td>N0004</td>
      <td>5527188</td>
      <td>2617074</td>
      <td>47.35</td>
      <td>116433622</td>
      <td>44.49</td>
      <td>3.02</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Y1</td>
      <td>1</td>
      <td>5</td>
      <td>N0005</td>
      <td>2733015</td>
      <td>1920640</td>
      <td>70.28</td>
      <td>175892211</td>
      <td>91.58</td>
      <td>3.44</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



## 데이터 정보, 기초 통계량 출력


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10400 entries, 0 to 10399
    Data columns (total 11 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   yearID         10400 non-null  object 
     1   weekIndex      10400 non-null  int64  
     2   rank           10400 non-null  int64  
     3   appID          10398 non-null  object 
     4   uniqueInstall  10400 non-null  int64  
     5   UU             10400 non-null  int64  
     6   useRate        10400 non-null  float64
     7   totalDuration  10400 non-null  int64  
     8   avgDuration    10400 non-null  float64
     9   dayCounts      10400 non-null  float64
     10  genreID        10400 non-null  object 
    dtypes: float64(3), int64(5), object(3)
    memory usage: 893.9+ KB
    


```python
df.describe()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weekIndex</th>
      <th>rank</th>
      <th>uniqueInstall</th>
      <th>UU</th>
      <th>useRate</th>
      <th>totalDuration</th>
      <th>avgDuration</th>
      <th>dayCounts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10400.000000</td>
      <td>10400.000000</td>
      <td>1.040000e+04</td>
      <td>1.040000e+04</td>
      <td>10400.000000</td>
      <td>1.040000e+04</td>
      <td>10400.000000</td>
      <td>10400.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>26.500000</td>
      <td>50.499904</td>
      <td>7.335554e+05</td>
      <td>4.121797e+05</td>
      <td>65.433333</td>
      <td>6.263803e+07</td>
      <td>134.394693</td>
      <td>3.236485</td>
    </tr>
    <tr>
      <th>std</th>
      <td>15.009053</td>
      <td>28.867311</td>
      <td>1.600245e+06</td>
      <td>6.155132e+05</td>
      <td>19.033467</td>
      <td>1.367896e+08</td>
      <td>161.114699</td>
      <td>1.371650</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>7.908500e+04</td>
      <td>7.689300e+04</td>
      <td>1.040000</td>
      <td>2.076700e+04</td>
      <td>0.230000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>13.750000</td>
      <td>25.750000</td>
      <td>2.203168e+05</td>
      <td>1.311460e+05</td>
      <td>52.967500</td>
      <td>5.146609e+06</td>
      <td>28.340000</td>
      <td>2.070000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>26.500000</td>
      <td>50.500000</td>
      <td>3.450420e+05</td>
      <td>1.982480e+05</td>
      <td>66.480000</td>
      <td>1.848760e+07</td>
      <td>81.345000</td>
      <td>3.170000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>39.250000</td>
      <td>75.250000</td>
      <td>5.936968e+05</td>
      <td>3.805282e+05</td>
      <td>79.830000</td>
      <td>5.555300e+07</td>
      <td>174.505000</td>
      <td>4.300000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>52.000000</td>
      <td>100.000000</td>
      <td>2.431204e+07</td>
      <td>6.690166e+06</td>
      <td>100.000000</td>
      <td>1.763252e+09</td>
      <td>1437.490000</td>
      <td>7.000000</td>
    </tr>
  </tbody>
</table>
</div>



## LabelEncoder를 통해 범주형 genreID 변환


```python
le = LabelEncoder()
le_gen = le.fit(df['genreID'])
df['genreID'] = le_gen.transform(df['genreID'])
```

## Copy 데이터셋 생성, 수치형 Feature들만 필터링 하여 정규화


```python
X = df.copy()
```


```python
# df 컬럼 중 수치형 피처들만 필터링
df_features = df[['rank', 'uniqueInstall', 'UU', 'useRate', 'totalDuration', 'avgDuration', 'dayCounts', 'genreID']]
cols = df_features.columns.tolist()

# 피처 스케일링
pd.options.mode.chained_assignment = None

def standard_scaling(df, scale_columns):
    for col in scale_columns:
        series_mean = df[col].mean()
        series_std = df[col].std()
        df[col] = df[col].apply(lambda x : (x-series_mean)/series_std)
    return df
scaled_df = standard_scaling(df_features, cols)
X[['rank', 'uniqueInstall', 'UU', 'useRate', 'totalDuration', 'avgDuration', 'dayCounts', 'genreID']] = scaled_df
```

## Feature별 히스토그램 시각화


```python
def plot_hist_each_column(df):
    plt.rcParams['figure.figsize'] = [20,20]
    fig = plt.figure()
    for i in range(len(df.columns)):
        ax = fig.add_subplot(5, 5, i+1)
        plt.hist(df[df.columns[i]], bins=50)
        ax.set_title(df.columns[i])
    plt.show()

plot_hist_each_column(scaled_df)
```


    
![output_12_0](https://user-images.githubusercontent.com/69621732/119501423-d07b6a00-bda3-11eb-8ac3-d7c4107a4300.png)
    


## Feature 간의 히트맵과 산점도 그래프 출력

→ heatmap, pairplot을 활용하여 상관관계 분석
- totalDuration 피처는 대체적으로 모든 피처와 양의 상관관계를 띄는 것으로 보임
- 특히 totalDuration과 UU는 상관성이 높은 것으로 나타남
- 앱 순위에는 UU, totalDuration 순으로 상관성이 있는 것으로 나타남
- 나머지 Feature 사이에는 뚜렷한 상관성을 정의하기 어려움


```python
# Feature간 상관관계 도출
cols = scaled_df.columns.tolist()
corr = scaled_df.corr(method = 'pearson') 

# 산점도 그래프 출력
sns.set(font_scale = 1.5)
hm = sns.heatmap(corr.values, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws ={'size' : 15}, 
                 yticklabels = cols, xticklabels = cols )

sns.set(style = 'whitegrid', context = 'notebook')
sns.pairplot(df_features, height = 2.5)
plt.show()
```


    
![output_14_0](https://user-images.githubusercontent.com/69621732/119501431-d1ac9700-bda3-11eb-8751-ef8e88e20c14.png)
    



    
![output_14_1](https://user-images.githubusercontent.com/69621732/119501435-d2452d80-bda3-11eb-9b20-25db05422da3.png)
    


# LSTM 모델링 


```python
from datetime import datetime
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
from tensorflow.keras import layers, activations
from keras.layers import TimeDistributed
from keras.layers import concatenate
import tensorflow as tf
from math import sqrt
```

### 전체 기간 동안 앱별 Top100에 집계된 횟수 평균, Y2 기간 횟수 평균을 구한 후 평균 이상인 앱만 추출  


```python
rank_mean = X['appID'].value_counts().mean()
df_y2 = X.loc[X['yearID'] == 'Y2']
ft_mean = df_y2['appID'].value_counts().mean()
# # Y1-Y2 기간 앱 별 100위권 내 집계 평균
# # 앱 순위 100위 내에 10번 이상 든 앱만 추출
ft = df_y2['appID'].value_counts()
ft = ft[ft > 9].index.tolist()
```

###  - globals 함수를 이용해 appID별 데이터프레임 생성 후, 학습 데이터와 검증 데이터로 분류

## - appID별 rank에 대한 Y3년도 8주 현황 예측 모델 구축


```python
for i in ft:
    globals()['df_{}'.format(i)] = X[X['appID'] == i]
    globals()['df_{}'.format(i)]= globals()['df_{}'.format(i)].drop(['yearID', 'weekIndex', 
                                                                 'appID'], axis=1)
    globals()['target_{}'.format(i)] = globals()['df_{}'.format(i)]['rank'].values
    globals()['df_x_{}'.format(i)] = globals()['df_{}'.format(i)].drop(['rank'], axis=1)
    globals()['dataset_{}'.format(i)] = globals()['df_x_{}'.format(i)].values
    # train set은 전체 데이터의 0.7, 나머지는 val set으로 분류
    globals()['train_split_{}'.format(i)] = int(len(globals()['df_{}'.format(i)])*0.7)
    globals()['val_split_{}'.format(i)] = int(len(globals()['df_{}'.format(i)])-globals()['train_split_{}'.format(i)])
```

 ### - 8주간의 데이터를 통해 8주간의 미래 현황을 예측하는 lstm 모델 구축한 후, 3차원 numpy 형태로 변환

### - 변환된 appID별 데이터셋 병합


```python
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []
    
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for x in range(start_index, end_index):
        indices = range(x-history_size, x, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[x+target_size])
        else:
            labels.append(target[x:x+target_size])

    return np.array(data), np.array(labels)
```


```python
past_history = 8
future_target = 8
STEP = 1
```


```python
# train_multi = globals()['dataset_{}'.format(i)], globals()['target_{}'.format(i)], 0, 
# globals()['train_split_{}'.format(i)], past_history, future_target, STEP
for i in ft:
    globals()['x_train_multi_{}'.format(i)], globals()['y_train_multi_{}'.format(i)] = multivariate_data(globals()['dataset_{}'.format(i)], globals()['target_{}'.format(i)], 0, 
    globals()['train_split_{}'.format(i)], past_history, future_target, STEP)
    globals()['x_val_multi_{}'.format(i)], globals()['y_val_multi_{}'.format(i)] = multivariate_data(
    globals()['dataset_{}'.format(i)], globals()['target_{}'.format(i)], globals()['train_split_{}'.format(i)],
    None, past_history, future_target, STEP)
```


```python
correct = []
for i in ft:
    if globals()['x_train_multi_{}'.format(i)].shape == (0,):
        del globals()['x_train_multi_{}'.format(i)]
    if globals()['y_train_multi_{}'.format(i)].shape == (0,):
        del globals()['y_train_multi_{}'.format(i)]
    if globals()['x_val_multi_{}'.format(i)].shape == (0,):
        del globals()['x_val_multi_{}'.format(i)]
    if globals()['y_val_multi_{}'.format(i)].shape == (0,):
        print('{}는 오류'.format(i))
        del globals()['y_val_multi_{}'.format(i)]
    else:
        print('x_train_multi_{}'.format(i), sep='\n')
        correct.append(i)
```

    
    


```python
# appID별로 생성된 3차원 배열을 np.concatenate를 통해 병합

x_train_multi = np.concatenate([x_train_multi_N0214, x_train_multi_N0649, x_train_multi_N0580, x_train_multi_N0003, x_train_multi_N0261, x_train_multi_N0006, x_train_multi_N0572, x_train_multi_N0379, x_train_multi_N0012, x_train_multi_N0486, x_train_multi_N0159, x_train_multi_N0443, x_train_multi_N0569, x_train_multi_N0001, x_train_multi_N0034, x_train_multi_N0141, x_train_multi_N0628, x_train_multi_N0547, x_train_multi_N0542, x_train_multi_N0188, x_train_multi_N0099, x_train_multi_N0126, x_train_multi_N0018, x_train_multi_N0054, x_train_multi_N0481, x_train_multi_N0038, x_train_multi_N0413, x_train_multi_N0013, x_train_multi_N0177, x_train_multi_N0297, x_train_multi_N0161, x_train_multi_N0113, x_train_multi_N0037, 
x_train_multi_N0017])

y_train_multi = np.concatenate([y_train_multi_N0214, y_train_multi_N0649, y_train_multi_N0580, y_train_multi_N0003, y_train_multi_N0261, y_train_multi_N0006, y_train_multi_N0572, y_train_multi_N0379, y_train_multi_N0012, y_train_multi_N0486, y_train_multi_N0159, y_train_multi_N0443, y_train_multi_N0569, y_train_multi_N0001, y_train_multi_N0034, y_train_multi_N0141, y_train_multi_N0628, y_train_multi_N0547, y_train_multi_N0542, y_train_multi_N0188, y_train_multi_N0099, y_train_multi_N0126, y_train_multi_N0018, y_train_multi_N0054, y_train_multi_N0481, y_train_multi_N0038, y_train_multi_N0413, y_train_multi_N0013, y_train_multi_N0177, y_train_multi_N0297, y_train_multi_N0161, y_train_multi_N0113, y_train_multi_N0037, 
y_train_multi_N0017])

x_val_multi = np.concatenate([x_val_multi_N0214, x_val_multi_N0649, x_val_multi_N0580, x_val_multi_N0003, x_val_multi_N0261, x_val_multi_N0006, x_val_multi_N0572, x_val_multi_N0379, x_val_multi_N0012, x_val_multi_N0486, x_val_multi_N0159, x_val_multi_N0443, x_val_multi_N0569, x_val_multi_N0001, x_val_multi_N0034, x_val_multi_N0141, x_val_multi_N0628, x_val_multi_N0547, x_val_multi_N0542, x_val_multi_N0188, x_val_multi_N0099, x_val_multi_N0126, x_val_multi_N0018, x_val_multi_N0054, x_val_multi_N0481, x_val_multi_N0038, x_val_multi_N0413, x_val_multi_N0013, x_val_multi_N0177, x_val_multi_N0297, x_val_multi_N0161, x_val_multi_N0113, x_val_multi_N0037, 
x_val_multi_N0017])

y_val_multi = np.concatenate([y_val_multi_N0214, y_val_multi_N0649, y_val_multi_N0580, y_val_multi_N0003, y_val_multi_N0261, y_val_multi_N0006, y_val_multi_N0572, y_val_multi_N0379, y_val_multi_N0012, y_val_multi_N0486, y_val_multi_N0159, y_val_multi_N0443, y_val_multi_N0569, y_val_multi_N0001, y_val_multi_N0034, y_val_multi_N0141, y_val_multi_N0628, y_val_multi_N0547, y_val_multi_N0542, y_val_multi_N0188, y_val_multi_N0099, y_val_multi_N0126, y_val_multi_N0018, y_val_multi_N0054, y_val_multi_N0481, y_val_multi_N0038, y_val_multi_N0413, y_val_multi_N0013, y_val_multi_N0177, y_val_multi_N0297, y_val_multi_N0161, y_val_multi_N0113, y_val_multi_N0037, 
y_val_multi_N0017])
```

## 학습데이터와 검증데이터셋은 다음과 같이 구성


```python
y_train_multi = y_train_multi.reshape(1658,8,1)
y_val_multi = y_val_multi.reshape(308,8,1)

print('Train Set: ',x_train_multi.shape,y_train_multi.shape)
print('Validation Set: ',x_val_multi.shape,y_val_multi.shape)
```

    Train Set:  (1658, 8, 7) (1658, 8, 1)
    Validation Set:  (308, 8, 7) (308, 8, 1)
    


```python
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
from tensorflow.keras import layers, activations
from keras.layers import TimeDistributed
```

# Lstm 모델 실행


```python
batch_size = len(x_train_multi)//128
Epoch = 200

model=Sequential()
model.add(tf.keras.layers.LSTM(256, return_sequences=True, input_shape=x_train_multi.shape[-2:]))
model.add(tf.keras.layers.LSTM(512, activation='relu', return_sequences=True))
model.add(tf.keras.layers.LSTM(512, activation='relu', return_sequences=False))
model.add(Dense(256))
model.add(Dense(256))
model.add(Dense(8))
opt = tf.keras.optimizers.Adam(learning_rate = 0.001)
model.compile(optimizer=opt, loss='mse')
earlystopper = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode='min',patience=10, verbose=1)
multi_step_history = model.fit(x_train_multi,y_train_multi,
                               validation_data=(x_val_multi,y_val_multi),
                               epochs=Epoch, batch_size=batch_size, callbacks=[earlystopper])
loss = model.evaluate(x_val_multi,y_val_multi)
```

    Epoch 1/200
    139/139 [==============================] - 46s 329ms/step - loss: 0.1906 - val_loss: 0.2232
    Epoch 2/200
    139/139 [==============================] - 37s 269ms/step - loss: 0.1248 - val_loss: 0.2876
    Epoch 3/200
    139/139 [==============================] - 32s 227ms/step - loss: 0.1214 - val_loss: 0.2216
    Epoch 4/200
    139/139 [==============================] - 31s 226ms/step - loss: 0.1206 - val_loss: 0.2070
    Epoch 5/200
    139/139 [==============================] - 32s 228ms/step - loss: 0.1012 - val_loss: 0.2750
    Epoch 6/200
    139/139 [==============================] - 32s 227ms/step - loss: 0.1082 - val_loss: 0.2186
    Epoch 7/200
    139/139 [==============================] - 46s 330ms/step - loss: 0.1033 - val_loss: 0.2132
    Epoch 8/200
    139/139 [==============================] - 46s 330ms/step - loss: 0.1002 - val_loss: 0.2276
    Epoch 9/200
    139/139 [==============================] - 46s 329ms/step - loss: 0.0957 - val_loss: 0.2564
    Epoch 10/200
    139/139 [==============================] - 46s 331ms/step - loss: 0.0872 - val_loss: 0.2417
    Epoch 11/200
    139/139 [==============================] - 45s 325ms/step - loss: 0.0881 - val_loss: 0.2039
    Epoch 12/200
    139/139 [==============================] - 47s 339ms/step - loss: 0.0816 - val_loss: 0.2112
    Epoch 13/200
    139/139 [==============================] - 45s 325ms/step - loss: 0.0793 - val_loss: 0.2303
    Epoch 14/200
    139/139 [==============================] - 46s 331ms/step - loss: 0.0758 - val_loss: 0.2506
    Epoch 15/200
    139/139 [==============================] - 45s 324ms/step - loss: 0.0785 - val_loss: 0.2291
    Epoch 16/200
    139/139 [==============================] - 45s 325ms/step - loss: 0.0746 - val_loss: 0.1980
    Epoch 17/200
    139/139 [==============================] - 46s 330ms/step - loss: 0.0771 - val_loss: 0.2258
    Epoch 18/200
    139/139 [==============================] - 47s 341ms/step - loss: 0.0742 - val_loss: 0.2484
    Epoch 19/200
    139/139 [==============================] - 46s 331ms/step - loss: 0.0706 - val_loss: 0.2870
    Epoch 20/200
    139/139 [==============================] - 50s 360ms/step - loss: 0.0767 - val_loss: 0.2481
    Epoch 21/200
    139/139 [==============================] - 46s 334ms/step - loss: 0.0706 - val_loss: 0.2544
    Epoch 22/200
    139/139 [==============================] - 46s 334ms/step - loss: 0.0708 - val_loss: 0.2322
    Epoch 23/200
    139/139 [==============================] - 48s 344ms/step - loss: 0.0710 - val_loss: 0.2460
    Epoch 24/200
    139/139 [==============================] - 46s 331ms/step - loss: 0.0658 - val_loss: 0.1990
    Epoch 25/200
    139/139 [==============================] - 48s 345ms/step - loss: 0.0635 - val_loss: 0.2366
    Epoch 26/200
    139/139 [==============================] - 46s 334ms/step - loss: 0.0673 - val_loss: 0.2263
    Epoch 00026: early stopping
    10/10 [==============================] - 1s 119ms/step - loss: 0.2263
    


```python
 model.summary()
```

    Model: "sequential_4"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lstm_9 (LSTM)                (None, 8, 256)            270336    
    _________________________________________________________________
    lstm_10 (LSTM)               (None, 8, 512)            1574912   
    _________________________________________________________________
    lstm_11 (LSTM)               (None, 512)               2099200   
    _________________________________________________________________
    dense_9 (Dense)              (None, 256)               131328    
    _________________________________________________________________
    dense_10 (Dense)             (None, 256)               65792     
    _________________________________________________________________
    dense_11 (Dense)             (None, 8)                 2056      
    =================================================================
    Total params: 4,143,624
    Trainable params: 4,143,624
    Non-trainable params: 0
    _________________________________________________________________
    


```python
def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()

def pred_vis(name, y_test_vis, y_pred_vis):
    y_test_m_vis = y_test_vis
    plt.figure(figsize=(14,9))
    plt.title('%s Prediction' %name)
    plt.plot(y_test_m_vis, c='steelblue', alpha=1, lw=2, marker="o", ms=2, mec='steelblue', mew=5)
    plt.plot(y_pred_vis, c='darkorange', alpha=2, lw=2, marker="o", ms=2, mec='darkorange', mew=5)
    legend_list = ['y_test', 'y_pred']
    plt.xlabel('number of index', fontsize='12')
    plt.ylabel('e', fontsize='12')
    plt.legend(legend_list, loc=1, fontsize='12')
    plt.grid(True)
    plt.show()
```

## 모델 Training loss와 Validation loss 시각화


```python
plt.rcParams['figure.figsize'] = (10, 7)
plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')
```


    
![output_35_0](https://user-images.githubusercontent.com/69621732/119501441-d3765a80-bda3-11eb-8abe-11eb03c40f25.png)
    


## 모델 예측값과 실제값 시각화


```python
series_mean = X.mean()
series_std = X.std()


pred = model.predict(x_val_multi)
pred = (pred*series_std[0])+series_mean[0]
pred = np.round(pred, 0)

y_val_multi_vis = np.reshape(y_val_multi, (y_val_multi.shape[0],8), order='C')
y_val_multi_vis = (y_val_multi_vis*series_std[0])+series_mean[0]
pred_vis('LSTM', y_val_multi_vis[5], pred[5])
```


    
![output_37_0](https://user-images.githubusercontent.com/69621732/119501444-d3765a80-bda3-11eb-9954-54fc69140121.png)
    


# 랜덤샘플링 후 예측값과 실제값 비교


```python
BATCH_SIZE = 256
BUFFER_SIZE = 10000

vis_data = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
vis_data = vis_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
```


```python
def create_time_steps(length):
    return list(range(-length, 0))

def multi_step_plot(history, true_future, prediction):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

#   plt.plot(num_in, np.array(history[:, 5]), label='History')
    plt.plot(num_in, np.array(history[:, 0]), 'bo',
           label='History')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
           label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
             label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()
```


```python
for x, y in vis_data.take(5):
    multi_step_plot((x[0][:8]*series_std[0])+series_mean[0], (y[0]*series_std[0])+series_mean[0], 
                    (model.predict(x)[0]*series_std[0])+series_mean[0])
```


    
![output_41_0](https://user-images.githubusercontent.com/69621732/119501445-d40ef100-bda3-11eb-8564-f564157249ef.png)






![output_41_1](https://user-images.githubusercontent.com/69621732/119501449-d4a78780-bda3-11eb-9555-8f7dc4f8457a.png)






![output_41_2](https://user-images.githubusercontent.com/69621732/119501453-d4a78780-bda3-11eb-9acd-797c882cb742.png)






![output_41_3](https://user-images.githubusercontent.com/69621732/119501454-d5401e00-bda3-11eb-93cb-699ff1db364a.png)






![output_41_4](https://user-images.githubusercontent.com/69621732/119501455-d5d8b480-bda3-11eb-8b35-114accf289c4.png)
    


# appID별 rank에 대한 Y3년도 8주 Dataset


```python
pred_set = []
for i in correct:
    dataset = globals()['df_x_{}'.format(i)].iloc[-8:].values
    dataset = np.reshape(dataset, (1,dataset.shape[0],7), order='C')
    pred = model.predict(dataset)
    pred = (pred*series_std[0])+series_mean[0]
    pred = np.round(pred, 0)
    pred_set.append(pred)
```


```python
Y3_rank_df = pd.DataFrame(index=range(0,8), columns=[col for col in correct])    
for i in range(len(pred_set)):
    Y3_rank_df.iloc[:,i] = pred_set[i][0]
```


```python
Y3_rank_df
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>N0569</th>
      <th>N0547</th>
      <th>N0572</th>
      <th>N0649</th>
      <th>N0003</th>
      <th>N0443</th>
      <th>N0379</th>
      <th>N0012</th>
      <th>N0214</th>
      <th>N0261</th>
      <th>...</th>
      <th>N0038</th>
      <th>N0481</th>
      <th>N0413</th>
      <th>N0013</th>
      <th>N0177</th>
      <th>N0297</th>
      <th>N0161</th>
      <th>N0113</th>
      <th>N0037</th>
      <th>N0017</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>25.0</td>
      <td>19.0</td>
      <td>36.0</td>
      <td>8.0</td>
      <td>3.0</td>
      <td>14.0</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>25.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>42.0</td>
      <td>38.0</td>
      <td>31.0</td>
      <td>41.0</td>
      <td>31.0</td>
      <td>38.0</td>
      <td>28.0</td>
      <td>35.0</td>
      <td>38.0</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25.0</td>
      <td>20.0</td>
      <td>36.0</td>
      <td>8.0</td>
      <td>4.0</td>
      <td>14.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>25.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>42.0</td>
      <td>38.0</td>
      <td>30.0</td>
      <td>41.0</td>
      <td>31.0</td>
      <td>38.0</td>
      <td>27.0</td>
      <td>34.0</td>
      <td>37.0</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25.0</td>
      <td>19.0</td>
      <td>36.0</td>
      <td>8.0</td>
      <td>3.0</td>
      <td>14.0</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>26.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>42.0</td>
      <td>38.0</td>
      <td>29.0</td>
      <td>41.0</td>
      <td>30.0</td>
      <td>38.0</td>
      <td>26.0</td>
      <td>33.0</td>
      <td>36.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>25.0</td>
      <td>19.0</td>
      <td>37.0</td>
      <td>9.0</td>
      <td>3.0</td>
      <td>14.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>27.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>42.0</td>
      <td>39.0</td>
      <td>28.0</td>
      <td>41.0</td>
      <td>30.0</td>
      <td>38.0</td>
      <td>25.0</td>
      <td>33.0</td>
      <td>35.0</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>25.0</td>
      <td>19.0</td>
      <td>37.0</td>
      <td>8.0</td>
      <td>3.0</td>
      <td>14.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>28.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>41.0</td>
      <td>39.0</td>
      <td>26.0</td>
      <td>40.0</td>
      <td>30.0</td>
      <td>38.0</td>
      <td>23.0</td>
      <td>31.0</td>
      <td>33.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>25.0</td>
      <td>18.0</td>
      <td>37.0</td>
      <td>8.0</td>
      <td>2.0</td>
      <td>14.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>28.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>41.0</td>
      <td>40.0</td>
      <td>26.0</td>
      <td>40.0</td>
      <td>30.0</td>
      <td>38.0</td>
      <td>23.0</td>
      <td>31.0</td>
      <td>33.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>25.0</td>
      <td>18.0</td>
      <td>37.0</td>
      <td>9.0</td>
      <td>3.0</td>
      <td>14.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>29.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>40.0</td>
      <td>39.0</td>
      <td>26.0</td>
      <td>39.0</td>
      <td>30.0</td>
      <td>37.0</td>
      <td>23.0</td>
      <td>31.0</td>
      <td>32.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>25.0</td>
      <td>18.0</td>
      <td>36.0</td>
      <td>9.0</td>
      <td>3.0</td>
      <td>14.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>29.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>39.0</td>
      <td>39.0</td>
      <td>25.0</td>
      <td>38.0</td>
      <td>30.0</td>
      <td>36.0</td>
      <td>22.0</td>
      <td>30.0</td>
      <td>31.0</td>
      <td>9.0</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 34 columns</p>
</div>



# appID별 UU Y3년도 8주 현황 예측


```python
UU_df = df.copy()
UU_mean = df['UU'].mean()
UU_std = df['UU'].std()
```


```python
UU_df[['rank', 'uniqueInstall', 'UU', 'useRate', 'totalDuration', 'avgDuration', 'dayCounts', 'genreID']] = scaled_df
UU_df = UU_df[['yearID', 'weekIndex', 'UU', 'rank', 'appID', 'uniqueInstall', 'useRate', 'totalDuration',
              'avgDuration', 'dayCounts', 'genreID']]
```


```python
for i in ft:
    globals()['UU_df_{}'.format(i)] = UU_df[UU_df['appID'] == i]
    globals()['UU_df_{}'.format(i)]= globals()['UU_df_{}'.format(i)].drop(['yearID', 'weekIndex', 
                                                                 'appID'], axis=1)
    globals()['UU_target_{}'.format(i)] = globals()['UU_df_{}'.format(i)]['UU'].values
    globals()['UU_df_x_{}'.format(i)] = globals()['UU_df_{}'.format(i)].drop(['UU'], axis=1)
    globals()['UU_dataset_{}'.format(i)] = globals()['UU_df_x_{}'.format(i)].values
    
    globals()['UU_train_split_{}'.format(i)] = int(len(globals()['UU_df_{}'.format(i)])*0.7)
    globals()['UU_val_split_{}'.format(i)] = int(len(globals()['UU_df_{}'.format(i)])-globals()['UU_train_split_{}'.format(i)])
```


```python
for i in ft:
    globals()['UU_x_train_multi_{}'.format(i)], globals()['UU_y_train_multi_{}'.format(i)] = multivariate_data(globals()['UU_dataset_{}'.format(i)], globals()['UU_target_{}'.format(i)], 0, 
    globals()['UU_train_split_{}'.format(i)], past_history, future_target, STEP)
    globals()['UU_x_val_multi_{}'.format(i)], globals()['UU_y_val_multi_{}'.format(i)] = multivariate_data(
    globals()['UU_dataset_{}'.format(i)], globals()['UU_target_{}'.format(i)], globals()['UU_train_split_{}'.format(i)],
    None, past_history, future_target, STEP)
```


```python
UU_correct = []
for i in ft:
    if globals()['UU_x_train_multi_{}'.format(i)].shape == (0,):
        del globals()['UU_x_train_multi_{}'.format(i)]
    if globals()['UU_y_train_multi_{}'.format(i)].shape == (0,):
        del globals()['UU_y_train_multi_{}'.format(i)]
    if globals()['UU_x_val_multi_{}'.format(i)].shape == (0,):
        del globals()['UU_x_val_multi_{}'.format(i)]
    if globals()['UU_y_val_multi_{}'.format(i)].shape == (0,):
        print('{}는 오류'.format(i))
        del globals()['UU_y_val_multi_{}'.format(i)]
    else:
        UU_correct.append(i)
```

    
    


```python
UU_x_train_multi = np.concatenate([UU_x_train_multi_N0214, UU_x_train_multi_N0649, UU_x_train_multi_N0580, UU_x_train_multi_N0003, UU_x_train_multi_N0261, UU_x_train_multi_N0006, UU_x_train_multi_N0572, UU_x_train_multi_N0379, UU_x_train_multi_N0012, UU_x_train_multi_N0486, UU_x_train_multi_N0159, UU_x_train_multi_N0443, UU_x_train_multi_N0569, UU_x_train_multi_N0001, UU_x_train_multi_N0034, UU_x_train_multi_N0141, UU_x_train_multi_N0628, UU_x_train_multi_N0547, UU_x_train_multi_N0542, UU_x_train_multi_N0188, UU_x_train_multi_N0099, UU_x_train_multi_N0126, UU_x_train_multi_N0018, UU_x_train_multi_N0054, UU_x_train_multi_N0481, UU_x_train_multi_N0038, UU_x_train_multi_N0413, UU_x_train_multi_N0013, UU_x_train_multi_N0177, UU_x_train_multi_N0297, UU_x_train_multi_N0161, 
                                   UU_x_train_multi_N0113, UU_x_train_multi_N0037, UU_x_train_multi_N0017])

UU_y_train_multi = np.concatenate([UU_y_train_multi_N0214, UU_y_train_multi_N0649, UU_y_train_multi_N0580, UU_y_train_multi_N0003, UU_y_train_multi_N0261, UU_y_train_multi_N0006, UU_y_train_multi_N0572, UU_y_train_multi_N0379, UU_y_train_multi_N0012, UU_y_train_multi_N0486, UU_y_train_multi_N0159, UU_y_train_multi_N0443, UU_y_train_multi_N0569, UU_y_train_multi_N0001, UU_y_train_multi_N0034, UU_y_train_multi_N0141, UU_y_train_multi_N0628, UU_y_train_multi_N0547, UU_y_train_multi_N0542, UU_y_train_multi_N0188, UU_y_train_multi_N0099, UU_y_train_multi_N0126, UU_y_train_multi_N0018, UU_y_train_multi_N0054, UU_y_train_multi_N0481, UU_y_train_multi_N0038, UU_y_train_multi_N0413, UU_y_train_multi_N0013, UU_y_train_multi_N0177, UU_y_train_multi_N0297, 
                                UU_y_train_multi_N0161, UU_y_train_multi_N0113, UU_y_train_multi_N0037, UU_y_train_multi_N0017])

UU_x_val_multi = np.concatenate([UU_x_val_multi_N0214, UU_x_val_multi_N0649, UU_x_val_multi_N0580, UU_x_val_multi_N0003, UU_x_val_multi_N0261, UU_x_val_multi_N0006, UU_x_val_multi_N0572, UU_x_val_multi_N0379, UU_x_val_multi_N0012, UU_x_val_multi_N0486, UU_x_val_multi_N0159, UU_x_val_multi_N0443, UU_x_val_multi_N0569, UU_x_val_multi_N0001, UU_x_val_multi_N0034, UU_x_val_multi_N0141, UU_x_val_multi_N0628, UU_x_val_multi_N0547, UU_x_val_multi_N0542, UU_x_val_multi_N0188, UU_x_val_multi_N0099, UU_x_val_multi_N0126, UU_x_val_multi_N0018, UU_x_val_multi_N0054, UU_x_val_multi_N0481, UU_x_val_multi_N0038, UU_x_val_multi_N0413, UU_x_val_multi_N0013, UU_x_val_multi_N0177, UU_x_val_multi_N0297, 
                              UU_x_val_multi_N0161, UU_x_val_multi_N0113, UU_x_val_multi_N0037, UU_x_val_multi_N0017])

UU_y_val_multi = np.concatenate([UU_y_val_multi_N0214, UU_y_val_multi_N0649, UU_y_val_multi_N0580, UU_y_val_multi_N0003, UU_y_val_multi_N0261, UU_y_val_multi_N0006, UU_y_val_multi_N0572, UU_y_val_multi_N0379, UU_y_val_multi_N0012, UU_y_val_multi_N0486, UU_y_val_multi_N0159, UU_y_val_multi_N0443, UU_y_val_multi_N0569, UU_y_val_multi_N0001, UU_y_val_multi_N0034, UU_y_val_multi_N0141, UU_y_val_multi_N0628, UU_y_val_multi_N0547, UU_y_val_multi_N0542, UU_y_val_multi_N0188, UU_y_val_multi_N0099, UU_y_val_multi_N0126, UU_y_val_multi_N0018, UU_y_val_multi_N0054, UU_y_val_multi_N0481, UU_y_val_multi_N0038, UU_y_val_multi_N0413, UU_y_val_multi_N0013, UU_y_val_multi_N0177, UU_y_val_multi_N0297, 
                              UU_y_val_multi_N0161, UU_y_val_multi_N0113, UU_y_val_multi_N0037, UU_y_val_multi_N0017])
```


```python
batch_size = len(UU_x_train_multi)//128
Epoch = 200

model=Sequential()
model.add(tf.keras.layers.LSTM(256, return_sequences=True, input_shape=UU_x_train_multi.shape[-2:]))
model.add(tf.keras.layers.LSTM(512, activation='relu', return_sequences=True))
model.add(tf.keras.layers.LSTM(512, activation='relu', return_sequences=False))
model.add(Dense(256))
model.add(Dense(256))
model.add(Dense(8))
opt = tf.keras.optimizers.Adam(learning_rate = 0.001)
model.compile(optimizer=opt, loss='mse')
earlystopper = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode='min',patience=10, verbose=1)
multi_step_history = model.fit(UU_x_train_multi,UU_y_train_multi,
                               validation_data=(UU_x_val_multi,UU_y_val_multi),
                               epochs=Epoch, batch_size=batch_size, callbacks=[earlystopper])
loss = model.evaluate(UU_x_val_multi,UU_y_val_multi)
```

    Epoch 1/200
    139/139 [==============================] - 49s 356ms/step - loss: 0.3847 - val_loss: 0.0587
    Epoch 2/200
    139/139 [==============================] - 47s 335ms/step - loss: 0.1403 - val_loss: 0.0540
    Epoch 3/200
    139/139 [==============================] - 48s 342ms/step - loss: 0.1262 - val_loss: 0.0461
    Epoch 4/200
    139/139 [==============================] - 47s 338ms/step - loss: 0.1297 - val_loss: 0.0589
    Epoch 5/200
    139/139 [==============================] - 50s 360ms/step - loss: 0.0868 - val_loss: 0.0572
    Epoch 6/200
    139/139 [==============================] - 47s 339ms/step - loss: 0.0833 - val_loss: 0.0543
    Epoch 7/200
    139/139 [==============================] - 48s 342ms/step - loss: 0.0917 - val_loss: 0.0680
    Epoch 8/200
    139/139 [==============================] - 49s 354ms/step - loss: 0.0683 - val_loss: 0.0440
    Epoch 9/200
    139/139 [==============================] - 48s 348ms/step - loss: 0.0592 - val_loss: 0.0436
    Epoch 10/200
    139/139 [==============================] - 51s 368ms/step - loss: 0.0707 - val_loss: 0.0523
    Epoch 11/200
    139/139 [==============================] - 48s 345ms/step - loss: 0.0737 - val_loss: 0.0388
    Epoch 12/200
    139/139 [==============================] - 47s 339ms/step - loss: 0.0676 - val_loss: 0.0392
    Epoch 13/200
    139/139 [==============================] - 47s 341ms/step - loss: 0.0607 - val_loss: 0.0466
    Epoch 14/200
    139/139 [==============================] - 48s 343ms/step - loss: 0.0483 - val_loss: 0.0523
    Epoch 15/200
    139/139 [==============================] - 52s 376ms/step - loss: 0.0718 - val_loss: 0.0662
    Epoch 16/200
    139/139 [==============================] - 49s 354ms/step - loss: 0.0528 - val_loss: 0.0438
    Epoch 17/200
    139/139 [==============================] - 48s 347ms/step - loss: 0.0456 - val_loss: 0.0501
    Epoch 18/200
    139/139 [==============================] - 61s 439ms/step - loss: 0.0472 - val_loss: 0.0500
    Epoch 19/200
    139/139 [==============================] - 50s 359ms/step - loss: 0.0524 - val_loss: 0.0430
    Epoch 20/200
    139/139 [==============================] - 50s 359ms/step - loss: 0.0528 - val_loss: 0.0490
    Epoch 21/200
    139/139 [==============================] - 57s 409ms/step - loss: 0.0568 - val_loss: 0.0465
    Epoch 00021: early stopping
    10/10 [==============================] - 2s 155ms/step - loss: 0.0465
    


```python
plt.rcParams['figure.figsize'] = (10, 7)
plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')
```


    
![output_54_0](https://user-images.githubusercontent.com/69621732/119501459-d5d8b480-bda3-11eb-9a20-79291f3582d8.png)
    



```python
pred = model.predict(UU_x_val_multi)
pred = (pred*UU_std)+UU_mean
pred = np.round(pred, 0)

UU_y_val_multi_vis = np.reshape(UU_y_val_multi, (UU_y_val_multi.shape[0],8), order='C')
UU_y_val_multi_vis = (UU_y_val_multi_vis*UU_std)+UU_mean
pred_vis('LSTM', UU_y_val_multi_vis[5], pred[5])
```


    
![output_55_0](https://user-images.githubusercontent.com/69621732/119501460-d6714b00-bda3-11eb-9592-ec2cb230331d.png)
    



```python
BATCH_SIZE = 256
BUFFER_SIZE = 10000

vis_data = tf.data.Dataset.from_tensor_slices((UU_x_train_multi, UU_y_train_multi))
vis_data = vis_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
```


```python
for x, y in vis_data.take(5):
    multi_step_plot((x[0][:8]*UU_std)+UU_mean, (y[0]*UU_std)+UU_mean, 
                    (model.predict(x)[0]*UU_std)+UU_mean)
```


    
![output_57_0](https://user-images.githubusercontent.com/69621732/119501461-d6714b00-bda3-11eb-9fb9-e2ca395e6863.png)






![output_57_1](https://user-images.githubusercontent.com/69621732/119501462-d709e180-bda3-11eb-8e80-432f137396ad.png)






![output_57_2](https://user-images.githubusercontent.com/69621732/119501464-d7a27800-bda3-11eb-990e-44266b546c9e.png)






![output_57_3](https://user-images.githubusercontent.com/69621732/119501466-d7a27800-bda3-11eb-8418-125469dd0d11.png)






![output_57_4](https://user-images.githubusercontent.com/69621732/119501467-d83b0e80-bda3-11eb-92e5-79587e130adb.png)
    


## appID별 UU에 대한 Y3년도 8주 Dataset


```python
pred_set = []
for i in UU_correct:
    dataset = globals()['UU_df_x_{}'.format(i)].iloc[-8:].values
    dataset = np.reshape(dataset, (1,dataset.shape[0],7), order='C')
    pred = model.predict(dataset)
    pred = (pred*UU_std)+UU_mean
    pred = np.round(pred, 0)
    pred_set.append(pred)
```


```python
Y3_UU_df = pd.DataFrame(index=range(0,8), columns=[col for col in UU_correct])    
for i in range(len(pred_set)):
    Y3_UU_df.iloc[:,i] = pred_set[i][0]
```


```python
Y3_UU_df
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>N0569</th>
      <th>N0547</th>
      <th>N0572</th>
      <th>N0649</th>
      <th>N0003</th>
      <th>N0443</th>
      <th>N0379</th>
      <th>N0012</th>
      <th>N0214</th>
      <th>N0261</th>
      <th>...</th>
      <th>N0038</th>
      <th>N0481</th>
      <th>N0413</th>
      <th>N0013</th>
      <th>N0177</th>
      <th>N0297</th>
      <th>N0161</th>
      <th>N0113</th>
      <th>N0037</th>
      <th>N0017</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>271839.0</td>
      <td>214894.0</td>
      <td>190540.0</td>
      <td>636534.0</td>
      <td>1364894.0</td>
      <td>409094.0</td>
      <td>1731513.0</td>
      <td>906645.0</td>
      <td>160932.0</td>
      <td>1846356.0</td>
      <td>...</td>
      <td>157965.0</td>
      <td>122178.0</td>
      <td>209764.0</td>
      <td>189503.0</td>
      <td>161923.0</td>
      <td>168627.0</td>
      <td>147517.0</td>
      <td>174980.0</td>
      <td>114687.0</td>
      <td>327723.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>258477.0</td>
      <td>213845.0</td>
      <td>178032.0</td>
      <td>654679.0</td>
      <td>1358869.0</td>
      <td>390050.0</td>
      <td>1675011.0</td>
      <td>907585.0</td>
      <td>152679.0</td>
      <td>1843540.0</td>
      <td>...</td>
      <td>150528.0</td>
      <td>111634.0</td>
      <td>197797.0</td>
      <td>179941.0</td>
      <td>151698.0</td>
      <td>153381.0</td>
      <td>143560.0</td>
      <td>168478.0</td>
      <td>108907.0</td>
      <td>434830.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>255465.0</td>
      <td>180942.0</td>
      <td>170310.0</td>
      <td>709691.0</td>
      <td>1399350.0</td>
      <td>384062.0</td>
      <td>1674996.0</td>
      <td>935916.0</td>
      <td>151204.0</td>
      <td>1904383.0</td>
      <td>...</td>
      <td>146848.0</td>
      <td>103835.0</td>
      <td>192414.0</td>
      <td>174843.0</td>
      <td>145128.0</td>
      <td>143870.0</td>
      <td>142134.0</td>
      <td>164978.0</td>
      <td>104888.0</td>
      <td>568188.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>255953.0</td>
      <td>191971.0</td>
      <td>174860.0</td>
      <td>724078.0</td>
      <td>1379844.0</td>
      <td>374562.0</td>
      <td>1597229.0</td>
      <td>933584.0</td>
      <td>173250.0</td>
      <td>1891286.0</td>
      <td>...</td>
      <td>157302.0</td>
      <td>112907.0</td>
      <td>197038.0</td>
      <td>181570.0</td>
      <td>153025.0</td>
      <td>145456.0</td>
      <td>157050.0</td>
      <td>176748.0</td>
      <td>119201.0</td>
      <td>700164.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>245531.0</td>
      <td>151323.0</td>
      <td>163078.0</td>
      <td>729496.0</td>
      <td>1377520.0</td>
      <td>362232.0</td>
      <td>1573757.0</td>
      <td>925116.0</td>
      <td>164480.0</td>
      <td>1889028.0</td>
      <td>...</td>
      <td>149255.0</td>
      <td>101996.0</td>
      <td>185800.0</td>
      <td>172505.0</td>
      <td>143485.0</td>
      <td>133253.0</td>
      <td>149929.0</td>
      <td>167863.0</td>
      <td>110640.0</td>
      <td>753376.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>240477.0</td>
      <td>140813.0</td>
      <td>157918.0</td>
      <td>753230.0</td>
      <td>1354332.0</td>
      <td>353126.0</td>
      <td>1509683.0</td>
      <td>920447.0</td>
      <td>174107.0</td>
      <td>1865907.0</td>
      <td>...</td>
      <td>149970.0</td>
      <td>103996.0</td>
      <td>183188.0</td>
      <td>168434.0</td>
      <td>140226.0</td>
      <td>130816.0</td>
      <td>155122.0</td>
      <td>168540.0</td>
      <td>116940.0</td>
      <td>789007.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>232327.0</td>
      <td>118511.0</td>
      <td>149757.0</td>
      <td>769565.0</td>
      <td>1348074.0</td>
      <td>339235.0</td>
      <td>1470823.0</td>
      <td>916005.0</td>
      <td>172279.0</td>
      <td>1863531.0</td>
      <td>...</td>
      <td>143388.0</td>
      <td>95237.0</td>
      <td>175258.0</td>
      <td>161251.0</td>
      <td>131512.0</td>
      <td>118423.0</td>
      <td>150908.0</td>
      <td>163323.0</td>
      <td>111994.0</td>
      <td>816174.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>234144.0</td>
      <td>142839.0</td>
      <td>149212.0</td>
      <td>789297.0</td>
      <td>1367016.0</td>
      <td>339286.0</td>
      <td>1467568.0</td>
      <td>931500.0</td>
      <td>180168.0</td>
      <td>1895833.0</td>
      <td>...</td>
      <td>147439.0</td>
      <td>98841.0</td>
      <td>176268.0</td>
      <td>161350.0</td>
      <td>132950.0</td>
      <td>118010.0</td>
      <td>157935.0</td>
      <td>167208.0</td>
      <td>118724.0</td>
      <td>835554.0</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 34 columns</p>
</div>



# appID별 avgDuration에 대한 Y3년도 8주 현황 예측


```python
avg_df = df.copy()
avg_mean = df['avgDuration'].mean()
avg_std = df['avgDuration'].std()

avg_df[['rank', 'uniqueInstall', 'UU', 'useRate', 'totalDuration', 'avgDuration', 'dayCounts', 'genreID']] = scaled_df
avg_df = UU_df[['yearID', 'weekIndex', 'avgDuration', 'rank', 'appID', 'uniqueInstall', 'UU', 'useRate', 'totalDuration',
              'dayCounts', 'genreID']]

for i in ft:
    globals()['avg_df_{}'.format(i)] = avg_df[avg_df['appID'] == i]
    globals()['avg_df_{}'.format(i)]= globals()['avg_df_{}'.format(i)].drop(['yearID', 'weekIndex', 
                                                                 'appID'], axis=1)
    globals()['avg_target_{}'.format(i)] = globals()['avg_df_{}'.format(i)]['avgDuration'].values
    globals()['avg_df_x_{}'.format(i)] = globals()['avg_df_{}'.format(i)].drop(['avgDuration'], axis=1)
    globals()['avg_dataset_{}'.format(i)] = globals()['avg_df_x_{}'.format(i)].values
    
    globals()['avg_train_split_{}'.format(i)] = int(len(globals()['avg_df_{}'.format(i)])*0.7)
    globals()['avg_val_split_{}'.format(i)] = int(len(globals()['avg_df_{}'.format(i)])-globals()['avg_train_split_{}'.format(i)])
```


```python
for i in ft:
    globals()['avg_x_train_multi_{}'.format(i)], globals()['avg_y_train_multi_{}'.format(i)] = multivariate_data(globals()['avg_dataset_{}'.format(i)], globals()['avg_target_{}'.format(i)], 0, 
    globals()['avg_train_split_{}'.format(i)], past_history, future_target, STEP)
    globals()['avg_x_val_multi_{}'.format(i)], globals()['avg_y_val_multi_{}'.format(i)] = multivariate_data(
    globals()['avg_dataset_{}'.format(i)], globals()['avg_target_{}'.format(i)], globals()['avg_train_split_{}'.format(i)],
    None, past_history, future_target, STEP)

avg_correct = []
for i in ft:
    if globals()['avg_x_train_multi_{}'.format(i)].shape == (0,):
        del globals()['avg_x_train_multi_{}'.format(i)]
    if globals()['avg_y_train_multi_{}'.format(i)].shape == (0,):
        del globals()['avg_y_train_multi_{}'.format(i)]
    if globals()['avg_x_val_multi_{}'.format(i)].shape == (0,):
        del globals()['avg_x_val_multi_{}'.format(i)]
    if globals()['avg_y_val_multi_{}'.format(i)].shape == (0,):
        print('{}는 오류'.format(i))
        del globals()['avg_y_val_multi_{}'.format(i)]
    else:
        avg_correct.append(i)

```

    
    


```python
avg_x_train_multi = np.concatenate([avg_x_train_multi_N0214, avg_x_train_multi_N0649, avg_x_train_multi_N0580, avg_x_train_multi_N0003, avg_x_train_multi_N0261, avg_x_train_multi_N0006, avg_x_train_multi_N0572, avg_x_train_multi_N0379, avg_x_train_multi_N0012, avg_x_train_multi_N0486, avg_x_train_multi_N0159, avg_x_train_multi_N0443, avg_x_train_multi_N0569, avg_x_train_multi_N0001, avg_x_train_multi_N0034, avg_x_train_multi_N0141, avg_x_train_multi_N0628, avg_x_train_multi_N0547, avg_x_train_multi_N0542, avg_x_train_multi_N0188, avg_x_train_multi_N0099, avg_x_train_multi_N0126, avg_x_train_multi_N0018, avg_x_train_multi_N0054, avg_x_train_multi_N0481, avg_x_train_multi_N0038, avg_x_train_multi_N0413, avg_x_train_multi_N0013, avg_x_train_multi_N0177, avg_x_train_multi_N0297, avg_x_train_multi_N0161, 
                                   avg_x_train_multi_N0113, avg_x_train_multi_N0037, avg_x_train_multi_N0017])

avg_y_train_multi = np.concatenate([avg_y_train_multi_N0214, avg_y_train_multi_N0649, avg_y_train_multi_N0580, avg_y_train_multi_N0003, avg_y_train_multi_N0261, avg_y_train_multi_N0006, avg_y_train_multi_N0572, avg_y_train_multi_N0379, avg_y_train_multi_N0012, avg_y_train_multi_N0486, avg_y_train_multi_N0159, avg_y_train_multi_N0443, avg_y_train_multi_N0569, avg_y_train_multi_N0001, avg_y_train_multi_N0034, avg_y_train_multi_N0141, avg_y_train_multi_N0628, avg_y_train_multi_N0547, avg_y_train_multi_N0542, avg_y_train_multi_N0188, avg_y_train_multi_N0099, avg_y_train_multi_N0126, avg_y_train_multi_N0018, avg_y_train_multi_N0054, avg_y_train_multi_N0481, avg_y_train_multi_N0038, avg_y_train_multi_N0413, avg_y_train_multi_N0013, avg_y_train_multi_N0177, avg_y_train_multi_N0297, 
                                avg_y_train_multi_N0161, avg_y_train_multi_N0113, avg_y_train_multi_N0037, avg_y_train_multi_N0017])

avg_x_val_multi = np.concatenate([avg_x_val_multi_N0214, avg_x_val_multi_N0649, avg_x_val_multi_N0580, avg_x_val_multi_N0003, avg_x_val_multi_N0261, avg_x_val_multi_N0006, avg_x_val_multi_N0572, avg_x_val_multi_N0379, avg_x_val_multi_N0012, avg_x_val_multi_N0486, avg_x_val_multi_N0159, avg_x_val_multi_N0443, avg_x_val_multi_N0569, avg_x_val_multi_N0001, avg_x_val_multi_N0034, avg_x_val_multi_N0141, avg_x_val_multi_N0628, avg_x_val_multi_N0547, avg_x_val_multi_N0542, avg_x_val_multi_N0188, avg_x_val_multi_N0099, avg_x_val_multi_N0126, avg_x_val_multi_N0018, avg_x_val_multi_N0054, avg_x_val_multi_N0481, avg_x_val_multi_N0038, avg_x_val_multi_N0413, avg_x_val_multi_N0013, avg_x_val_multi_N0177, avg_x_val_multi_N0297, 
                              avg_x_val_multi_N0161, avg_x_val_multi_N0113, avg_x_val_multi_N0037, avg_x_val_multi_N0017])

avg_y_val_multi = np.concatenate([avg_y_val_multi_N0214, avg_y_val_multi_N0649, avg_y_val_multi_N0580, avg_y_val_multi_N0003, avg_y_val_multi_N0261, avg_y_val_multi_N0006, avg_y_val_multi_N0572, avg_y_val_multi_N0379, avg_y_val_multi_N0012, avg_y_val_multi_N0486, avg_y_val_multi_N0159, avg_y_val_multi_N0443, avg_y_val_multi_N0569, avg_y_val_multi_N0001, avg_y_val_multi_N0034, avg_y_val_multi_N0141, avg_y_val_multi_N0628, avg_y_val_multi_N0547, avg_y_val_multi_N0542, avg_y_val_multi_N0188, avg_y_val_multi_N0099, avg_y_val_multi_N0126, avg_y_val_multi_N0018, avg_y_val_multi_N0054, avg_y_val_multi_N0481, avg_y_val_multi_N0038, avg_y_val_multi_N0413, avg_y_val_multi_N0013, avg_y_val_multi_N0177, avg_y_val_multi_N0297, 
                              avg_y_val_multi_N0161, avg_y_val_multi_N0113, avg_y_val_multi_N0037, avg_y_val_multi_N0017])
```


```python
batch_size = len(avg_x_train_multi)//128
Epoch = 20

model=Sequential()
model.add(tf.keras.layers.LSTM(256, return_sequences=True, input_shape=avg_x_train_multi.shape[-2:]))
model.add(tf.keras.layers.LSTM(512, activation='relu', return_sequences=True))
model.add(tf.keras.layers.LSTM(512, activation='relu', return_sequences=False))
model.add(Dense(256))
model.add(Dense(256))
model.add(Dense(8))
opt = tf.keras.optimizers.Adam(learning_rate = 0.001)
model.compile(optimizer=opt, loss='mse')
earlystopper = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode='min',patience=10, verbose=1)
multi_step_history = model.fit(avg_x_train_multi,avg_y_train_multi,
                               validation_data=(avg_x_val_multi,avg_y_val_multi),
                               epochs=Epoch, batch_size=batch_size, callbacks=[earlystopper])
loss = model.evaluate(avg_x_val_multi,avg_y_val_multi)
```

    Epoch 1/20
    139/139 [==============================] - 50s 360ms/step - loss: 0.3711 - val_loss: 0.2112
    Epoch 2/20
    139/139 [==============================] - 48s 346ms/step - loss: 0.1466 - val_loss: 0.1427
    Epoch 3/20
    139/139 [==============================] - 52s 376ms/step - loss: 0.1332 - val_loss: 0.1372
    Epoch 4/20
    139/139 [==============================] - 47s 339ms/step - loss: 0.1230 - val_loss: 0.1356
    Epoch 5/20
    139/139 [==============================] - 51s 370ms/step - loss: 0.1067 - val_loss: 0.1568
    Epoch 6/20
    139/139 [==============================] - 49s 350ms/step - loss: 0.1124 - val_loss: 0.1868
    Epoch 7/20
    139/139 [==============================] - 51s 364ms/step - loss: 0.0948 - val_loss: 0.1539
    Epoch 8/20
    139/139 [==============================] - 47s 340ms/step - loss: 0.0966 - val_loss: 0.1848
    Epoch 9/20
    139/139 [==============================] - 49s 349ms/step - loss: 0.0896 - val_loss: 0.1164
    Epoch 10/20
    139/139 [==============================] - 48s 343ms/step - loss: 0.0928 - val_loss: 0.2452
    Epoch 11/20
    139/139 [==============================] - 47s 339ms/step - loss: 0.0920 - val_loss: 0.1309
    Epoch 12/20
    139/139 [==============================] - 46s 331ms/step - loss: 0.0835 - val_loss: 0.1791
    Epoch 13/20
    139/139 [==============================] - 50s 358ms/step - loss: 0.0901 - val_loss: 0.1437
    Epoch 14/20
    139/139 [==============================] - 53s 381ms/step - loss: 0.0826 - val_loss: 0.1105
    Epoch 15/20
    139/139 [==============================] - 49s 352ms/step - loss: 0.0817 - val_loss: 0.1356
    Epoch 16/20
    139/139 [==============================] - 48s 349ms/step - loss: 0.0878 - val_loss: 0.1796
    Epoch 17/20
    139/139 [==============================] - 53s 379ms/step - loss: 0.0956 - val_loss: 0.1778
    Epoch 18/20
    139/139 [==============================] - 49s 353ms/step - loss: 0.0796 - val_loss: 0.1514
    Epoch 19/20
    139/139 [==============================] - 50s 359ms/step - loss: 0.0788 - val_loss: 0.1942
    Epoch 20/20
    139/139 [==============================] - 44s 318ms/step - loss: 0.0788 - val_loss: 0.1391
    10/10 [==============================] - 1s 142ms/step - loss: 0.1391
    


```python
plt.rcParams['figure.figsize'] = (10, 7)
plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')
```


    
![output_67_0](https://user-images.githubusercontent.com/69621732/119501469-d83b0e80-bda3-11eb-999d-6e090f6443a6.png)
    



```python
pred = model.predict(avg_x_val_multi)
pred = (pred*avg_std)+avg_mean
pred = np.round(pred, 0)

avg_y_val_multi_vis = np.reshape(avg_y_val_multi, (avg_y_val_multi.shape[0],8), order='C')
avg_y_val_multi_vis = (avg_y_val_multi_vis*avg_std)+avg_mean
pred_vis('LSTM', avg_y_val_multi_vis[5], pred[5])
```


    
![output_68_0](https://user-images.githubusercontent.com/69621732/119501471-d8d3a500-bda3-11eb-834a-4413736fc449.png)
    



```python
BATCH_SIZE = 256
BUFFER_SIZE = 10000

vis_data = tf.data.Dataset.from_tensor_slices((avg_x_train_multi, avg_y_train_multi))
vis_data = vis_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

for x, y in vis_data.take(5):
    multi_step_plot((x[0][:8]*avg_std)+avg_mean, (y[0]*avg_std)+avg_mean, 
                    (model.predict(x)[0]*avg_std)+avg_mean)
```


    
![output_69_0](https://user-images.githubusercontent.com/69621732/119501472-d8d3a500-bda3-11eb-9c5b-e82ab86dc351.png)






![output_69_1](https://user-images.githubusercontent.com/69621732/119501473-d96c3b80-bda3-11eb-85bd-df2c3264bdfa.png)






![output_69_2](https://user-images.githubusercontent.com/69621732/119501475-da04d200-bda3-11eb-9bc0-4ea18ae37b08.png)






![output_69_3](https://user-images.githubusercontent.com/69621732/119501477-da04d200-bda3-11eb-9069-f1775abd7caa.png)






![output_69_4](https://user-images.githubusercontent.com/69621732/119501479-da9d6880-bda3-11eb-8a2e-f3fdf89059ed.png)
    


# appID별 avgDuration에 대한 Y3년도 8주 Dataset


```python
pred_set = []
for i in avg_correct:
    dataset = globals()['avg_df_x_{}'.format(i)].iloc[-8:].values
    dataset = np.reshape(dataset, (1,dataset.shape[0],7), order='C')
    pred = model.predict(dataset)
    pred = (pred*avg_std)+avg_mean
    pred = np.round(pred, 0)
    pred_set.append(pred)
    
Y3_avg_df = pd.DataFrame(index=range(0,8), columns=[col for col in avg_correct])    
for i in range(len(pred_set)):
    Y3_avg_df.iloc[:,i] = pred_set[i][0]
```


```python
Y3_avg_df
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>N0569</th>
      <th>N0547</th>
      <th>N0572</th>
      <th>N0649</th>
      <th>N0003</th>
      <th>N0443</th>
      <th>N0379</th>
      <th>N0012</th>
      <th>N0214</th>
      <th>N0261</th>
      <th>...</th>
      <th>N0038</th>
      <th>N0481</th>
      <th>N0413</th>
      <th>N0013</th>
      <th>N0177</th>
      <th>N0297</th>
      <th>N0161</th>
      <th>N0113</th>
      <th>N0037</th>
      <th>N0017</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16.0</td>
      <td>-2.0</td>
      <td>38.0</td>
      <td>94.0</td>
      <td>114.0</td>
      <td>246.0</td>
      <td>147.0</td>
      <td>111.0</td>
      <td>440.0</td>
      <td>124.0</td>
      <td>...</td>
      <td>138.0</td>
      <td>243.0</td>
      <td>30.0</td>
      <td>51.0</td>
      <td>115.0</td>
      <td>127.0</td>
      <td>145.0</td>
      <td>132.0</td>
      <td>156.0</td>
      <td>485.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22.0</td>
      <td>5.0</td>
      <td>43.0</td>
      <td>97.0</td>
      <td>117.0</td>
      <td>241.0</td>
      <td>147.0</td>
      <td>114.0</td>
      <td>429.0</td>
      <td>125.0</td>
      <td>...</td>
      <td>140.0</td>
      <td>240.0</td>
      <td>35.0</td>
      <td>55.0</td>
      <td>118.0</td>
      <td>130.0</td>
      <td>147.0</td>
      <td>135.0</td>
      <td>157.0</td>
      <td>473.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15.0</td>
      <td>-1.0</td>
      <td>38.0</td>
      <td>95.0</td>
      <td>116.0</td>
      <td>246.0</td>
      <td>146.0</td>
      <td>114.0</td>
      <td>443.0</td>
      <td>123.0</td>
      <td>...</td>
      <td>143.0</td>
      <td>243.0</td>
      <td>28.0</td>
      <td>50.0</td>
      <td>119.0</td>
      <td>129.0</td>
      <td>148.0</td>
      <td>136.0</td>
      <td>158.0</td>
      <td>489.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.0</td>
      <td>-4.0</td>
      <td>34.0</td>
      <td>95.0</td>
      <td>116.0</td>
      <td>248.0</td>
      <td>145.0</td>
      <td>116.0</td>
      <td>450.0</td>
      <td>122.0</td>
      <td>...</td>
      <td>143.0</td>
      <td>244.0</td>
      <td>23.0</td>
      <td>46.0</td>
      <td>119.0</td>
      <td>129.0</td>
      <td>148.0</td>
      <td>137.0</td>
      <td>159.0</td>
      <td>496.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19.0</td>
      <td>5.0</td>
      <td>42.0</td>
      <td>102.0</td>
      <td>122.0</td>
      <td>245.0</td>
      <td>148.0</td>
      <td>123.0</td>
      <td>438.0</td>
      <td>127.0</td>
      <td>...</td>
      <td>144.0</td>
      <td>240.0</td>
      <td>31.0</td>
      <td>53.0</td>
      <td>122.0</td>
      <td>132.0</td>
      <td>149.0</td>
      <td>140.0</td>
      <td>161.0</td>
      <td>481.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>14.0</td>
      <td>-0.0</td>
      <td>37.0</td>
      <td>100.0</td>
      <td>121.0</td>
      <td>248.0</td>
      <td>148.0</td>
      <td>123.0</td>
      <td>447.0</td>
      <td>125.0</td>
      <td>...</td>
      <td>144.0</td>
      <td>242.0</td>
      <td>25.0</td>
      <td>48.0</td>
      <td>121.0</td>
      <td>131.0</td>
      <td>148.0</td>
      <td>140.0</td>
      <td>161.0</td>
      <td>492.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>15.0</td>
      <td>2.0</td>
      <td>38.0</td>
      <td>100.0</td>
      <td>121.0</td>
      <td>243.0</td>
      <td>145.0</td>
      <td>123.0</td>
      <td>439.0</td>
      <td>124.0</td>
      <td>...</td>
      <td>141.0</td>
      <td>238.0</td>
      <td>27.0</td>
      <td>49.0</td>
      <td>119.0</td>
      <td>129.0</td>
      <td>146.0</td>
      <td>138.0</td>
      <td>158.0</td>
      <td>483.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>12.0</td>
      <td>-3.0</td>
      <td>34.0</td>
      <td>99.0</td>
      <td>119.0</td>
      <td>243.0</td>
      <td>144.0</td>
      <td>122.0</td>
      <td>440.0</td>
      <td>122.0</td>
      <td>...</td>
      <td>139.0</td>
      <td>236.0</td>
      <td>23.0</td>
      <td>45.0</td>
      <td>116.0</td>
      <td>127.0</td>
      <td>143.0</td>
      <td>137.0</td>
      <td>155.0</td>
      <td>484.0</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 34 columns</p>
</div>



# appID별 dayCounts에 대한 Y3년도 8주 현황 예측


```python
day_df = df.copy()
day_mean = df['dayCounts'].mean()
day_std = df['dayCounts'].std()

day_df[['rank', 'uniqueInstall', 'UU', 'useRate', 'totalDuration', 'avgDuration', 'dayCounts', 'genreID']] = scaled_df
day_df = UU_df[['yearID', 'weekIndex', 'dayCounts', 'rank', 'appID', 'uniqueInstall', 'UU', 'useRate', 'totalDuration',
              'avgDuration', 'genreID']]

for i in ft:
    globals()['day_df_{}'.format(i)] = day_df[day_df['appID'] == i]
    globals()['day_df_{}'.format(i)]= globals()['day_df_{}'.format(i)].drop(['yearID', 'weekIndex', 
                                                                 'appID'], axis=1)
    globals()['day_target_{}'.format(i)] = globals()['day_df_{}'.format(i)]['dayCounts'].values
    globals()['day_df_x_{}'.format(i)] = globals()['day_df_{}'.format(i)].drop(['dayCounts'], axis=1)
    globals()['day_dataset_{}'.format(i)] = globals()['day_df_x_{}'.format(i)].values
    
    globals()['day_train_split_{}'.format(i)] = int(len(globals()['day_df_{}'.format(i)])*0.7)
    globals()['day_val_split_{}'.format(i)] = int(len(globals()['day_df_{}'.format(i)])-globals()['day_train_split_{}'.format(i)])
```


```python
for i in ft:
    globals()['day_x_train_multi_{}'.format(i)], globals()['day_y_train_multi_{}'.format(i)] = multivariate_data(globals()['day_dataset_{}'.format(i)], globals()['day_target_{}'.format(i)], 0, 
    globals()['day_train_split_{}'.format(i)], past_history, future_target, STEP)
    globals()['day_x_val_multi_{}'.format(i)], globals()['day_y_val_multi_{}'.format(i)] = multivariate_data(
    globals()['day_dataset_{}'.format(i)], globals()['day_target_{}'.format(i)], globals()['day_train_split_{}'.format(i)],
    None, past_history, future_target, STEP)

day_correct = []
for i in ft:
    if globals()['day_x_train_multi_{}'.format(i)].shape == (0,):
        del globals()['day_x_train_multi_{}'.format(i)]
    if globals()['day_y_train_multi_{}'.format(i)].shape == (0,):
        del globals()['day_y_train_multi_{}'.format(i)]
    if globals()['day_x_val_multi_{}'.format(i)].shape == (0,):
        del globals()['day_x_val_multi_{}'.format(i)]
    if globals()['day_y_val_multi_{}'.format(i)].shape == (0,):
        print('{}는 오류'.format(i))
        del globals()['day_y_val_multi_{}'.format(i)]
    else:
        day_correct.append(i)
```

    
    


```python
day_x_train_multi = np.concatenate([day_x_train_multi_N0214, day_x_train_multi_N0649, day_x_train_multi_N0580, day_x_train_multi_N0003, day_x_train_multi_N0261, day_x_train_multi_N0006, day_x_train_multi_N0572, day_x_train_multi_N0379, day_x_train_multi_N0012, day_x_train_multi_N0486, day_x_train_multi_N0159, day_x_train_multi_N0443, day_x_train_multi_N0569, day_x_train_multi_N0001, day_x_train_multi_N0034, day_x_train_multi_N0141, day_x_train_multi_N0628, day_x_train_multi_N0547, day_x_train_multi_N0542, day_x_train_multi_N0188, day_x_train_multi_N0099, day_x_train_multi_N0126, day_x_train_multi_N0018, day_x_train_multi_N0054, day_x_train_multi_N0481, day_x_train_multi_N0038, day_x_train_multi_N0413, day_x_train_multi_N0013, day_x_train_multi_N0177, day_x_train_multi_N0297, day_x_train_multi_N0161, 
                                   day_x_train_multi_N0113, day_x_train_multi_N0037, day_x_train_multi_N0017])

day_y_train_multi = np.concatenate([day_y_train_multi_N0214, day_y_train_multi_N0649, day_y_train_multi_N0580, day_y_train_multi_N0003, day_y_train_multi_N0261, day_y_train_multi_N0006, day_y_train_multi_N0572, day_y_train_multi_N0379, day_y_train_multi_N0012, day_y_train_multi_N0486, day_y_train_multi_N0159, day_y_train_multi_N0443, day_y_train_multi_N0569, day_y_train_multi_N0001, day_y_train_multi_N0034, day_y_train_multi_N0141, day_y_train_multi_N0628, day_y_train_multi_N0547, day_y_train_multi_N0542, day_y_train_multi_N0188, day_y_train_multi_N0099, day_y_train_multi_N0126, day_y_train_multi_N0018, day_y_train_multi_N0054, day_y_train_multi_N0481, day_y_train_multi_N0038, day_y_train_multi_N0413, day_y_train_multi_N0013, day_y_train_multi_N0177, day_y_train_multi_N0297, 
                                day_y_train_multi_N0161, day_y_train_multi_N0113, day_y_train_multi_N0037, day_y_train_multi_N0017])

day_x_val_multi = np.concatenate([day_x_val_multi_N0214, day_x_val_multi_N0649, day_x_val_multi_N0580, day_x_val_multi_N0003, day_x_val_multi_N0261, day_x_val_multi_N0006, day_x_val_multi_N0572, day_x_val_multi_N0379, day_x_val_multi_N0012, day_x_val_multi_N0486, day_x_val_multi_N0159, day_x_val_multi_N0443, day_x_val_multi_N0569, day_x_val_multi_N0001, day_x_val_multi_N0034, day_x_val_multi_N0141, day_x_val_multi_N0628, day_x_val_multi_N0547, day_x_val_multi_N0542, day_x_val_multi_N0188, day_x_val_multi_N0099, day_x_val_multi_N0126, day_x_val_multi_N0018, day_x_val_multi_N0054, day_x_val_multi_N0481, day_x_val_multi_N0038, day_x_val_multi_N0413, day_x_val_multi_N0013, day_x_val_multi_N0177, day_x_val_multi_N0297, 
                              day_x_val_multi_N0161, day_x_val_multi_N0113, day_x_val_multi_N0037, day_x_val_multi_N0017])

day_y_val_multi = np.concatenate([day_y_val_multi_N0214, day_y_val_multi_N0649, day_y_val_multi_N0580, day_y_val_multi_N0003, day_y_val_multi_N0261, day_y_val_multi_N0006, day_y_val_multi_N0572, day_y_val_multi_N0379, day_y_val_multi_N0012, day_y_val_multi_N0486, day_y_val_multi_N0159, day_y_val_multi_N0443, day_y_val_multi_N0569, day_y_val_multi_N0001, day_y_val_multi_N0034, day_y_val_multi_N0141, day_y_val_multi_N0628, day_y_val_multi_N0547, day_y_val_multi_N0542, day_y_val_multi_N0188, day_y_val_multi_N0099, day_y_val_multi_N0126, day_y_val_multi_N0018, day_y_val_multi_N0054, day_y_val_multi_N0481, day_y_val_multi_N0038, day_y_val_multi_N0413, day_y_val_multi_N0013, day_y_val_multi_N0177, day_y_val_multi_N0297, 
                              day_y_val_multi_N0161, day_y_val_multi_N0113, day_y_val_multi_N0037, day_y_val_multi_N0017])
```


```python
batch_size = len(day_x_train_multi)//128
Epoch = 20

model=Sequential()
model.add(tf.keras.layers.LSTM(256, return_sequences=True, input_shape=day_x_train_multi.shape[-2:]))
model.add(tf.keras.layers.LSTM(512, activation='relu', return_sequences=True))
model.add(tf.keras.layers.LSTM(512, activation='relu', return_sequences=False))
model.add(Dense(256))
model.add(Dense(256))
model.add(Dense(8))
opt = tf.keras.optimizers.Adam(learning_rate = 0.001)
model.compile(optimizer=opt, loss='mse')
earlystopper = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode='min',patience=10, verbose=1)
multi_step_history = model.fit(day_x_train_multi,day_y_train_multi,
                               validation_data=(day_x_val_multi,day_y_val_multi),
                               epochs=Epoch, batch_size=batch_size, callbacks=[earlystopper])
loss = model.evaluate(day_x_val_multi,day_y_val_multi)
```

    Epoch 1/20
    139/139 [==============================] - 44s 317ms/step - loss: 0.3605 - val_loss: 0.3084
    Epoch 2/20
    139/139 [==============================] - 43s 312ms/step - loss: 0.2055 - val_loss: 0.3514
    Epoch 3/20
    139/139 [==============================] - 44s 318ms/step - loss: 0.1883 - val_loss: 0.3186
    Epoch 4/20
    139/139 [==============================] - 44s 317ms/step - loss: 0.1633 - val_loss: 0.2908
    Epoch 5/20
    139/139 [==============================] - 45s 321ms/step - loss: 0.1549 - val_loss: 0.2904
    Epoch 6/20
    139/139 [==============================] - 45s 321ms/step - loss: 0.1431 - val_loss: 0.3066
    Epoch 7/20
    139/139 [==============================] - 47s 337ms/step - loss: 0.1436 - val_loss: 0.2591
    Epoch 8/20
    139/139 [==============================] - 47s 337ms/step - loss: 0.1372 - val_loss: 0.2511
    Epoch 9/20
    139/139 [==============================] - 45s 321ms/step - loss: 0.1307 - val_loss: 0.2974
    Epoch 10/20
    139/139 [==============================] - 45s 321ms/step - loss: 0.1276 - val_loss: 0.2898
    Epoch 11/20
    139/139 [==============================] - 45s 324ms/step - loss: 0.1220 - val_loss: 0.2762
    Epoch 12/20
    139/139 [==============================] - 46s 329ms/step - loss: 0.1211 - val_loss: 0.2932
    Epoch 13/20
    139/139 [==============================] - 45s 326ms/step - loss: 0.1192 - val_loss: 0.3051
    Epoch 14/20
    139/139 [==============================] - 44s 318ms/step - loss: 0.1137 - val_loss: 0.2975
    Epoch 15/20
    139/139 [==============================] - 50s 358ms/step - loss: 0.1184 - val_loss: 0.3317
    Epoch 16/20
    139/139 [==============================] - 46s 331ms/step - loss: 0.1301 - val_loss: 0.2894
    Epoch 17/20
    139/139 [==============================] - 44s 318ms/step - loss: 0.1203 - val_loss: 0.3156
    Epoch 18/20
    139/139 [==============================] - 47s 337ms/step - loss: 0.1193 - val_loss: 0.2892
    Epoch 00018: early stopping
    10/10 [==============================] - 1s 140ms/step - loss: 0.2892
    


```python
plt.rcParams['figure.figsize'] = (10, 7)
plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')
```


    
![output_78_0](https://user-images.githubusercontent.com/69621732/119501483-da9d6880-bda3-11eb-8244-fb06f53bd159.png)
    



```python
pred = model.predict(avg_x_val_multi)
pred = (pred*avg_std)+avg_mean
pred = np.round(pred, 0)

avg_y_val_multi_vis = np.reshape(avg_y_val_multi, (avg_y_val_multi.shape[0],8), order='C')
avg_y_val_multi_vis = (avg_y_val_multi_vis*avg_std)+avg_mean
pred_vis('LSTM', avg_y_val_multi_vis[5], pred[5])
```


    
![output_79_0](https://user-images.githubusercontent.com/69621732/119501484-db35ff00-bda3-11eb-9d93-489853faeaf9.png)
    



```python
BATCH_SIZE = 256
BUFFER_SIZE = 10000

vis_data = tf.data.Dataset.from_tensor_slices((day_x_train_multi, day_y_train_multi))
vis_data = vis_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

for x, y in vis_data.take(5):
    multi_step_plot((x[0][:8]*day_std)+day_mean, (y[0]*day_std)+day_mean, 
                    (model.predict(x)[0]*day_std)+day_mean)
```


    
![output_80_0](https://user-images.githubusercontent.com/69621732/119501486-dbce9580-bda3-11eb-829f-c79d2b13d624.png)






![output_80_1](https://user-images.githubusercontent.com/69621732/119501490-dbce9580-bda3-11eb-9b60-08d41f614f05.png)







![output_80_2](https://user-images.githubusercontent.com/69621732/119501492-dc672c00-bda3-11eb-88fe-baa37f0687da.png)






![output_80_3](https://user-images.githubusercontent.com/69621732/119501493-dc672c00-bda3-11eb-9543-dd65cd21b717.png)






![output_80_4](https://user-images.githubusercontent.com/69621732/119501494-dcffc280-bda3-11eb-8773-53b6ada43f13.png)
    


# appID별 dayCounts에 대한 Y3년도 8주 Dataset


```python
pred_set = []
for i in day_correct:
    dataset = globals()['day_df_x_{}'.format(i)].iloc[-8:].values
    dataset = np.reshape(dataset, (1,dataset.shape[0],7), order='C')
    pred = model.predict(dataset)
    pred = (pred*day_std)+day_mean
    pred = np.round(pred, 0)
    pred_set.append(pred)
    
Y3_day_df = pd.DataFrame(index=range(0,8), columns=[col for col in day_correct])    
for i in range(len(pred_set)):
    Y3_day_df.iloc[:,i] = pred_set[i][0]
```


```python
Y3_day_df
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>N0569</th>
      <th>N0547</th>
      <th>N0572</th>
      <th>N0649</th>
      <th>N0003</th>
      <th>N0443</th>
      <th>N0379</th>
      <th>N0012</th>
      <th>N0214</th>
      <th>N0261</th>
      <th>...</th>
      <th>N0038</th>
      <th>N0481</th>
      <th>N0413</th>
      <th>N0013</th>
      <th>N0177</th>
      <th>N0297</th>
      <th>N0161</th>
      <th>N0113</th>
      <th>N0037</th>
      <th>N0017</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>5.0</td>
      <td>...</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>5.0</td>
      <td>...</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>5.0</td>
      <td>...</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>5.0</td>
      <td>...</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>5.0</td>
      <td>...</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>5.0</td>
      <td>...</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>5.0</td>
      <td>...</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>5.0</td>
      <td>...</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 34 columns</p>
</div>




```python
Y3_rank_df.to_csv('Y3_rank_df.csv')
Y3_day_df.to_csv('Y3_day_df.csv')
Y3_UU_df.to_csv('Y3_UU_df.csv')
Y3_avg_df.to_csv('Y3_avg_df.csv')
```
