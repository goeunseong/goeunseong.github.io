---
layout: single
title:  "통계1일차(기술통계)"
categories: jupyter notebook
tag: [python, blog, jupyter, statistic]
toc: true
author_profile: false
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


## [실습1] 자동차 연비 Data Set에서 기술통계치 구하기

- (dataset : 'mycars.csv')



```python
# 필요 모듈 임포트
import pandas as pd
import numpy as np
```


```python
# 데이터 읽기
df = pd.read_csv('./mycars.csv', engine='python')
df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>manufacturer</th>
      <th>model</th>
      <th>displacement</th>
      <th>year</th>
      <th>cylinder</th>
      <th>automatic</th>
      <th>driving</th>
      <th>mpg</th>
      <th>highway_mileage</th>
      <th>fuel</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>audi</td>
      <td>a4</td>
      <td>1.8</td>
      <td>1999</td>
      <td>4</td>
      <td>auto</td>
      <td>f</td>
      <td>18</td>
      <td>29</td>
      <td>p</td>
      <td>compact</td>
    </tr>
    <tr>
      <th>1</th>
      <td>audi</td>
      <td>a4</td>
      <td>1.8</td>
      <td>1999</td>
      <td>4</td>
      <td>manual</td>
      <td>f</td>
      <td>21</td>
      <td>29</td>
      <td>p</td>
      <td>compact</td>
    </tr>
    <tr>
      <th>2</th>
      <td>audi</td>
      <td>a4</td>
      <td>2.0</td>
      <td>2008</td>
      <td>4</td>
      <td>manual</td>
      <td>f</td>
      <td>20</td>
      <td>31</td>
      <td>p</td>
      <td>compact</td>
    </tr>
    <tr>
      <th>3</th>
      <td>audi</td>
      <td>a4</td>
      <td>2.0</td>
      <td>2008</td>
      <td>4</td>
      <td>auto</td>
      <td>f</td>
      <td>21</td>
      <td>30</td>
      <td>p</td>
      <td>compact</td>
    </tr>
    <tr>
      <th>4</th>
      <td>audi</td>
      <td>a4</td>
      <td>2.8</td>
      <td>1999</td>
      <td>6</td>
      <td>auto</td>
      <td>f</td>
      <td>16</td>
      <td>26</td>
      <td>p</td>
      <td>compact</td>
    </tr>
  </tbody>
</table>
</div>



```python
# 시내에서 연비 구동방식 별 통계치 구하기
df_1 = df[['mpg', 'driving']]
df_1.groupby('driving').describe(include='all')
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">mpg</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>driving</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>103.0</td>
      <td>14.330097</td>
      <td>2.874459</td>
      <td>9.0</td>
      <td>13.0</td>
      <td>14.0</td>
      <td>16.0</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>f</th>
      <td>106.0</td>
      <td>19.971698</td>
      <td>3.626510</td>
      <td>11.0</td>
      <td>18.0</td>
      <td>19.0</td>
      <td>21.0</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>r</th>
      <td>25.0</td>
      <td>14.080000</td>
      <td>2.215852</td>
      <td>11.0</td>
      <td>12.0</td>
      <td>15.0</td>
      <td>15.0</td>
      <td>18.0</td>
    </tr>
  </tbody>
</table>
</div>


### 결과 해석



#### driving(구동방식)

- __4: 사륜구동__

- __f: 전륜구동__

- __r: 후륜구동__

  

__위 데이터 값을 바탕으로, 구동방식 별 기술통계량을 각각 해석하면 다음과 같다.__

- 시내 연비 데이터에서 각각 사륜구동은 103개, 전륜구동은 106개, 후륜구동은 25개의 관측치를 가지고 있다.

- 시내 연비의 평균은 각각 사륜구동방식 차량: 약 14.3, 전륜구동방식 차량: 약 20, 후륜구동방식 차량: 약 14이다.

- 시내 연비의 표준편차는 각각 사륜구동방식 차량: 약 2.87, 전륜구동방식 차량: 약 3.63, 후륜구동방식 차량: 약 2.21이다.

    - 전륜구동방식 차량의 시내에서 연비는 표준편차가 상대적으로 높으며, 데이터의 변동성이 높다는 것을 알 수 있다(데이터 별 차이가 크다).

- 이때 평균과 표준편차를 이용해 이상치를 판별해내는 기준을 수립할 수도 있다. ('평균 + =3x표준편차')

<br/>



- 시내 연비의 최소값은 사륜구동방식 차량: 9, 전륜구동방식 차량: 11, 후륜구동방식 차량: 11이다.

    - 사륜구동방식 차량의 시내에서 연비 최소값이 가장 작으며, 표준편차(데이터의 분포)까지 고려하였을 때 __상대적으로 다른 구동방식의 차량보다 연비가 적다는 결론을 도출할 수 있다.

- 시내 연비의 사분위 수를 바탕으로 데이터의 분포 구간을 알아낼 수 있다.

    - 사륜구동방식 차량의 연비는 13~16 구간에 가장 많이 분포되어 있다.

    - 전륜구동방식 차량의 연비는 18~21 구간에 가장 많이 분포되어 있다.

    - 후륜구동방식 차량의 연비는 12~15 구간에 가장 많이 분포되어 있다.

        - 결론적으로 전륜구동방식 차량의 연비가 상대적으로 높다는 결론을 도출할 수 있다.

- 시내 연비의 최대값은 전륜구동방식 차량의 연비가 35로 가장 높으며, 사륜구동방식 차량은 21, 후륜구동방식 차량은 18이다.



#### 종합결론

- 구동방식 별로 시내 연비 데이터를 분석하였을 때, 전륜구동방식 차량은 다른 구동방식의 차량보다 더 많은 연비 값을 보였다. 이때 표준편차 값도 상대적으로 높아 데이터 간의 값의 차이가 다른 두 방식보다 크다는 결론을 지을 수 있다.



```python
# 고속도로에서 연비 구동방식 별 통계치 구하기
df_2 = df[['highway_mileage', 'driving']]
df_2.groupby('driving').describe(include='all')
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">highway_mileage</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>driving</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>103.0</td>
      <td>19.174757</td>
      <td>4.078704</td>
      <td>12.0</td>
      <td>17.0</td>
      <td>18.0</td>
      <td>22.0</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>f</th>
      <td>106.0</td>
      <td>28.160377</td>
      <td>4.206881</td>
      <td>17.0</td>
      <td>26.0</td>
      <td>28.0</td>
      <td>29.0</td>
      <td>44.0</td>
    </tr>
    <tr>
      <th>r</th>
      <td>25.0</td>
      <td>21.000000</td>
      <td>3.662877</td>
      <td>15.0</td>
      <td>17.0</td>
      <td>21.0</td>
      <td>24.0</td>
      <td>26.0</td>
    </tr>
  </tbody>
</table>
</div>


### 결과 해석



#### driving(구동방식)

- __4: 사륜구동__

- __f: 전륜구동__

- __r: 후륜구동__

  

#### __위 데이터 값을 바탕으로, 구동방식 별 기술통계량을 각각 해석하면 다음과 같다.__

- 고속도로 연비 데이터에서 각각 __사륜구동은 103개, 전륜구동은 106개, 후륜구동은 25개의 관측치__를 가지고 있다.

- 고속도로 연비의 __평균은 각각 사륜구동방식 차량: 약 19, 전륜구동방식 차량: 약 28, 후륜구동방식 차량: 약 21__이다.

    - __전륜구동방식 차량__의 평균값이 높은 것으로 보아, 값이 유독 높은 이상치가 있거나, 해당 데이터 내의 값이 전체적으로 높다는 예상이 가능하다.

- 고속도로 연비의 __표준편차는 각각 사륜구동방식 차량: 약 4.08, 전륜구동방식 차량: 약 4.2, 후륜구동방식 차량: 약 3.66__이다.

    - __전륜구동방식 차량__의 시내에서 연비는 표준편차가 상대적으로 높으며, __데이터의 변동성이 높다__는 것을 알 수 있다(데이터 별 차이가 크다).

- 마찬가지로 평균과 표준편차를 이용해 이상치를 판별해내는 기준을 수립할 수도 있다. ('평균 + =3x표준편차')

<br/>



- 고속도로 연비의 __최소값은 사륜구동방식 차량: 12, 전륜구동방식 차량: 17, 후륜구동방식 차량: 15__이다.

    - __사륜구동방식 차량__의 시내에서 연비 최소값이 가장 작으며, 표준편차(데이터의 분포)까지 고려하였을 때 __상대적으로 다른 구동방식의 차량보다 연비가 적다__는 결론을 도출할 수 있다.

- 고속도로 연비의 __사분위 수를 바탕으로 데이터의 분포 구간__을 알아낼 수 있다.

    - 사륜구동방식 차량의 연비는 17~22 구간에 가장 많이 분포되어 있다.

    - 전륜구동방식 차량의 연비는 26~29 구간에 가장 많이 분포되어 있다.

    - 후륜구동방식 차량의 연비는 17~24 구간에 가장 많이 분포되어 있다.

        - 결론적으로 __전륜구동방식 차량의 연비는 전체적으로 높다__는 결론을 도출할 수 있다.

- 고속도로 연비의 __최대값은 전륜구동방식 차량의 연비가 44로 가장 높으며, 사륜구동방식 차량은 28, 후륜구동방식 차량은 26__이다.



#### 종합결론

- 구동방식 별로 고속도로 연비 데이터를 분석하였을 때, __전륜구동방식 차량은 다른 구동방식의 차량보다 더 많은 연비 값을 보였다. 이때 표준편차 값도 상대적으로 높아 데이터 간의 값의 차이가 다른 두 방식보다 크다__는 결론을 지을 수 있다.


------


## [실습2] 범주형 데이터에 대한 counts, percents, cumulative counts, cumulative percents 계산

- (dataset: 'mycars.csv')



```python
df = pd.read_csv('./mycars.csv', engine='python')
df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>manufacturer</th>
      <th>model</th>
      <th>displacement</th>
      <th>year</th>
      <th>cylinder</th>
      <th>automatic</th>
      <th>driving</th>
      <th>mpg</th>
      <th>highway_mileage</th>
      <th>fuel</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>audi</td>
      <td>a4</td>
      <td>1.8</td>
      <td>1999</td>
      <td>4</td>
      <td>auto</td>
      <td>f</td>
      <td>18</td>
      <td>29</td>
      <td>p</td>
      <td>compact</td>
    </tr>
    <tr>
      <th>1</th>
      <td>audi</td>
      <td>a4</td>
      <td>1.8</td>
      <td>1999</td>
      <td>4</td>
      <td>manual</td>
      <td>f</td>
      <td>21</td>
      <td>29</td>
      <td>p</td>
      <td>compact</td>
    </tr>
    <tr>
      <th>2</th>
      <td>audi</td>
      <td>a4</td>
      <td>2.0</td>
      <td>2008</td>
      <td>4</td>
      <td>manual</td>
      <td>f</td>
      <td>20</td>
      <td>31</td>
      <td>p</td>
      <td>compact</td>
    </tr>
    <tr>
      <th>3</th>
      <td>audi</td>
      <td>a4</td>
      <td>2.0</td>
      <td>2008</td>
      <td>4</td>
      <td>auto</td>
      <td>f</td>
      <td>21</td>
      <td>30</td>
      <td>p</td>
      <td>compact</td>
    </tr>
    <tr>
      <th>4</th>
      <td>audi</td>
      <td>a4</td>
      <td>2.8</td>
      <td>1999</td>
      <td>6</td>
      <td>auto</td>
      <td>f</td>
      <td>16</td>
      <td>26</td>
      <td>p</td>
      <td>compact</td>
    </tr>
  </tbody>
</table>
</div>



```python
# 칼럼 별 데이터 타입 확인
df.info()
```

<pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 234 entries, 0 to 233
Data columns (total 11 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   manufacturer     234 non-null    object 
 1   model            234 non-null    object 
 2   displacement     234 non-null    float64
 3   year             234 non-null    int64  
 4   cylinder         234 non-null    int64  
 5   automatic        234 non-null    object 
 6   driving          234 non-null    object 
 7   mpg              234 non-null    int64  
 8   highway_mileage  234 non-null    int64  
 9   fuel             234 non-null    object 
 10  class            234 non-null    object 
dtypes: float64(1), int64(4), object(6)
memory usage: 20.2+ KB
</pre>

```python
# 범주형 변수들만을 골라 리스트에 받아 저장
object_col = [i for i in df.columns if df[i].dtypes == 'object']
df_obj = df[object_col]
df_obj
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>manufacturer</th>
      <th>model</th>
      <th>automatic</th>
      <th>driving</th>
      <th>fuel</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>audi</td>
      <td>a4</td>
      <td>auto</td>
      <td>f</td>
      <td>p</td>
      <td>compact</td>
    </tr>
    <tr>
      <th>1</th>
      <td>audi</td>
      <td>a4</td>
      <td>manual</td>
      <td>f</td>
      <td>p</td>
      <td>compact</td>
    </tr>
    <tr>
      <th>2</th>
      <td>audi</td>
      <td>a4</td>
      <td>manual</td>
      <td>f</td>
      <td>p</td>
      <td>compact</td>
    </tr>
    <tr>
      <th>3</th>
      <td>audi</td>
      <td>a4</td>
      <td>auto</td>
      <td>f</td>
      <td>p</td>
      <td>compact</td>
    </tr>
    <tr>
      <th>4</th>
      <td>audi</td>
      <td>a4</td>
      <td>auto</td>
      <td>f</td>
      <td>p</td>
      <td>compact</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>229</th>
      <td>volkswagen</td>
      <td>passat</td>
      <td>auto</td>
      <td>f</td>
      <td>p</td>
      <td>midsize</td>
    </tr>
    <tr>
      <th>230</th>
      <td>volkswagen</td>
      <td>passat</td>
      <td>manual</td>
      <td>f</td>
      <td>p</td>
      <td>midsize</td>
    </tr>
    <tr>
      <th>231</th>
      <td>volkswagen</td>
      <td>passat</td>
      <td>auto</td>
      <td>f</td>
      <td>p</td>
      <td>midsize</td>
    </tr>
    <tr>
      <th>232</th>
      <td>volkswagen</td>
      <td>passat</td>
      <td>manual</td>
      <td>f</td>
      <td>p</td>
      <td>midsize</td>
    </tr>
    <tr>
      <th>233</th>
      <td>volkswagen</td>
      <td>passat</td>
      <td>auto</td>
      <td>f</td>
      <td>p</td>
      <td>midsize</td>
    </tr>
  </tbody>
</table>
<p>234 rows × 6 columns</p>
</div>



```python
# count_data를 보여주는 함수 구현

def show_count_data(col):
    count = df[col].value_counts().sort_index()
    cumcnt = np.cumsum(count)
    percent = count / sum(count) * 100
    cumpct = np.cumsum(percent)
    
    count_data = pd.DataFrame({'Count': count, 'CumCnt': cumcnt, 'Percent': percent, 'CumPct': cumpct})
    count_data.columns.name = col
    print(count_data)
```


```python
# 함수를 통해 범주형 변수에 관한 데이터 결과 출력
for col in object_col:
    show_count_data(col)
```

<pre>
manufacturer  Count  CumCnt    Percent      CumPct
audi             18      18   7.692308    7.692308
chevrolet        19      37   8.119658   15.811966
dodge            37      74  15.811966   31.623932
ford             25      99  10.683761   42.307692
honda             9     108   3.846154   46.153846
hyundai          14     122   5.982906   52.136752
jeep              8     130   3.418803   55.555556
land rover        4     134   1.709402   57.264957
lincoln           3     137   1.282051   58.547009
mercury           4     141   1.709402   60.256410
nissan           13     154   5.555556   65.811966
pontiac           5     159   2.136752   67.948718
subaru           14     173   5.982906   73.931624
toyota           34     207  14.529915   88.461538
volkswagen       27     234  11.538462  100.000000
model                   Count  CumCnt   Percent      CumPct
4runner 4wd                 6       6  2.564103    2.564103
a4                          7      13  2.991453    5.555556
a4 quattro                  8      21  3.418803    8.974359
a6 quattro                  3      24  1.282051   10.256410
altima                      6      30  2.564103   12.820513
c1500 suburban 2wd          5      35  2.136752   14.957265
camry                       7      42  2.991453   17.948718
camry solara                7      49  2.991453   20.940171
caravan 2wd                11      60  4.700855   25.641026
civic                       9      69  3.846154   29.487179
corolla                     5      74  2.136752   31.623932
corvette                    5      79  2.136752   33.760684
dakota pickup 4wd           9      88  3.846154   37.606838
durango 4wd                 7      95  2.991453   40.598291
expedition 2wd              3      98  1.282051   41.880342
explorer 4wd                6     104  2.564103   44.444444
f150 pickup 4wd             7     111  2.991453   47.435897
forester awd                6     117  2.564103   50.000000
grand cherokee 4wd          8     125  3.418803   53.418803
grand prix                  5     130  2.136752   55.555556
gti                         5     135  2.136752   57.692308
impreza awd                 8     143  3.418803   61.111111
jetta                       9     152  3.846154   64.957265
k1500 tahoe 4wd             4     156  1.709402   66.666667
land cruiser wagon 4wd      2     158  0.854701   67.521368
malibu                      5     163  2.136752   69.658120
maxima                      3     166  1.282051   70.940171
mountaineer 4wd             4     170  1.709402   72.649573
mustang                     9     179  3.846154   76.495726
navigator 2wd               3     182  1.282051   77.777778
new beetle                  6     188  2.564103   80.341880
passat                      7     195  2.991453   83.333333
pathfinder 4wd              4     199  1.709402   85.042735
ram 1500 pickup 4wd        10     209  4.273504   89.316239
range rover                 4     213  1.709402   91.025641
sonata                      7     220  2.991453   94.017094
tiburon                     7     227  2.991453   97.008547
toyota tacoma 4wd           7     234  2.991453  100.000000
automatic  Count  CumCnt    Percent      CumPct
auto         157     157  67.094017   67.094017
manual        77     234  32.905983  100.000000
driving  Count  CumCnt    Percent      CumPct
4          103     103  44.017094   44.017094
f          106     209  45.299145   89.316239
r           25     234  10.683761  100.000000
fuel  Count  CumCnt    Percent      CumPct
c         1       1   0.427350    0.427350
d         5       6   2.136752    2.564103
e         8      14   3.418803    5.982906
p        52      66  22.222222   28.205128
r       168     234  71.794872  100.000000
class       Count  CumCnt    Percent      CumPct
2seater         5       5   2.136752    2.136752
compact        47      52  20.085470   22.222222
midsize        41      93  17.521368   39.743590
minivan        11     104   4.700855   44.444444
pickup         33     137  14.102564   58.547009
subcompact     35     172  14.957265   73.504274
suv            62     234  26.495726  100.000000
</pre>
### 결과 해석



#### manufacturer

- 'manufacturer'에는 총 15개의 제조사가 있으며, value_count()를 통해 가장 많이 관측된 제조사를 도출한 후 percent를 계산함

    - 해당 데이터셋에서는 'dodge'사에서 제조한 차량에 대한 관측 빈도수가 가장 높으며, 관측치의 15.8% 가량을 차지한다.



#### model

- 차량의 종류에 관한 'model'데이터에서, value_count()를 통해 가장 많이 관측된 모델의 종류를 도출한 후 percent를 계산함

    - 해당 데이터셋에서는 'caravan 2wd' 모델에 대한 관측 빈도수가 가장 높으며, 이는 관측치의 4.7% 가량을 차지한다.



#### automatic

- 수동/자동 여부를 가르는 'automatic' 데이터에서, value_count()를 통해 보다 많이 관측된 모델의 종류를 도출한 후 percent를 계산함

    - 해당 데이터셋에서는 'auto'에 해당하는 차량 정보가 더 많이 담겨 있으며, 이는 관측치의 67% 가량을 차지한다.

    - 해당 데이터셋에 담긴 차량에 대해서, 수동과 자동의 비율은 약 7:3이다.

    

#### driving

- 구동방식에 대한 'driving' 데이터에서, value_counts()를 통해 보다 많이 관측된 차량의 구동방식을 도출한 후 percent를 계산함

    - 해당 데이터셋에서는 전륜구동 차량의 관측치가 106개로 가장 많았으며, 이는 관측치의 45.3% 가량을 차지하는 값이다.

    - 반면 후륜구동 차량의 관측치는 25개로 가장 적었으며, 관측치의 10% 가량만을 차지한다.

    

#### class

- 차량의 형태에 대한 'class' 데이터에서, value_counts()를 통해 보다 많이 관측된 형태를 도출한 후 percent를 계산함

    - 해당 데이터셋에서는 SUV 차량의 관측치가 62개로 가장 많았으며, 이는 관측치의 26.5% 가량을 차지하는 값이다.

    - 반면 2seater 차량의 관측치는 5개로 가장 적었으며, 이는 관측치의 2% 가량만을 차지한다.

