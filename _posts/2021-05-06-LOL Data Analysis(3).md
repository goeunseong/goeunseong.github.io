# League Of Legends Data Analysis

지난 범주형 변수에 대한 EDA에 이어 앙상블 기법을 통한 승/패 예측과,  
각 변수들이 승/패에 얼마만큼 영향을 미치는지 로지스틱 회귀분석을 통해서 알아보도록 하겠습니다.

- [<Step3. 머신러닝 분석>](#Step3.-머신러닝-분석)
    - [데이터 호출]
    - [데이터 전처리]
    - [앙상블 분석]
    - [로지스틱 회귀분석]

# Step3. 머신러닝 분석

### 데이터 호출


```python
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings(action='ignore')
```


```python
blue_team = pd.read_csv('blue_team')
```


```python
blue_team.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gameId</th>
      <th>gameDuration</th>
      <th>Wins</th>
      <th>FirstBlood</th>
      <th>FirstTower</th>
      <th>FirstBaron</th>
      <th>FirstDragon</th>
      <th>FirstInhibitor</th>
      <th>DragonKills</th>
      <th>BaronKills</th>
      <th>TowerKills</th>
      <th>InhibitorKills</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4241678498</td>
      <td>2098</td>
      <td>Lose</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4233222221</td>
      <td>1686</td>
      <td>Lose</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4233113995</td>
      <td>1588</td>
      <td>Win</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>10</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4229230455</td>
      <td>1126</td>
      <td>Win</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4228244819</td>
      <td>1262</td>
      <td>Win</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### 데이터 전처리

예측 모델링의 경우, 범주형 데이터값을 입력값으로 처리하지 않습니다. 따라서 범주형 데이터를 수치형 데이터로 변환해줘야 합니다.  
이전 포스팅에서 승리/패배 컬럼을 제외한 범주형 컬럼에 대해 변환을 해주었으므로, 종속변수인 승/패 컬럼만 인코딩을 진행해줍니다.


```python
# 종속변수 라벨링
winner = {'Win' : 1, 'Lose' : 0}
blue_team['Wins'] = blue_team['Wins'].map(winner)

blue_team[['Wins', 'FirstBlood', 'FirstTower',
       'FirstBaron', 'FirstDragon', 'FirstInhibitor', 'DragonKills',
       'BaronKills', 'TowerKills', 'InhibitorKills']].astype(float)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Wins</th>
      <th>FirstBlood</th>
      <th>FirstTower</th>
      <th>FirstBaron</th>
      <th>FirstDragon</th>
      <th>FirstInhibitor</th>
      <th>DragonKills</th>
      <th>BaronKills</th>
      <th>TowerKills</th>
      <th>InhibitorKills</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>10.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>65891</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>65892</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>65893</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>65894</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>65895</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>65896 rows × 10 columns</p>
</div>



다음은 수치형 설명변수에 대한 히스토그램입니다.  
최소값과 최대값의 차이가 크지 않기 때문에, 이상값 제거를 거치지 않고 분석을 진행하도록 하겠습니다.


```python
df = blue_team[['DragonKills','BaronKills', 'TowerKills', 'InhibitorKills']]

def plot_hist_each_column(df):
    plt.rcParams['figure.figsize'] = [20,20]
    fig = plt.figure()
    for i in range(len(df.columns)):
        ax = fig.add_subplot(5, 5, i+1)
        plt.hist(df[df.columns[i]], bins=50)
        ax.set_title(df.columns[i])
    plt.show()

plot_hist_each_column(df)
```


    
![output_12_0](https://user-images.githubusercontent.com/69621732/117421194-4769d380-af59-11eb-8a9e-002a40d6d444.png)
    


### 앙상블 분석

앙상블 기법은 주어진 자료로부터 여러 개의 예측모형들을 만든 후, 이를 조합하여 하나의 최종 예측 모형을 만드는 방법입니다.  
앙상블에는 배깅, 부스팅, 랜덤포레스트 기법이 있습니다. 저는 랜덤포레스트와 부스팅 기법을 사용하여 승리/패배 예측 모델링을 만들어보겠습니다.

---
먼저 랜덤포레스트를 통해 학습을 진행하였습니다. Train, Test셋은 0.75, 0.25 비율로 나누었습니다.  
목표변수 : 승리/패배 여부( = blue_team['Wins'])  
설명변수 : 게임ID와 게임시간, 승리/패배 여부를 제외한 데이터 


```python
from sklearn.model_selection import train_test_split
X = blue_team[['FirstBlood', 'FirstTower',
       'FirstBaron', 'FirstDragon', 'FirstInhibitor', 'DragonKills',
       'BaronKills', 'TowerKills', 'InhibitorKills']]
y = blue_team['Wins']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25, stratify=y, 
                                                    random_state=123456)
```


```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)
rf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

predicted = rf.predict(X_test)
accuracy = accuracy_score(y_test, predicted)

#oob_score = out of bag score로써 예측이 얼마나 정확한가에 대한 추정치입니다.
print(f'Out-of-bag score estimate: {rf.oob_score_:.3}')
print(f'Mean accuracy score: {accuracy:.3}')
```




    RandomForestClassifier(oob_score=True, random_state=123456)



    Out-of-bag score estimate: 0.871
    Mean accuracy score: 0.865
    

**랜덤포레스트의 모델 예측 결과, 정확도는 약 87%가 나온 것을 알 수 있습니다.**


```python
from sklearn.ensemble import GradientBoostingClassifier
clf_gbc = GradientBoostingClassifier()
clf_gbc.fit(X_train,y_train)

y_pred = clf_gbc.predict(X_test)

print('테스트 정확도 = ' + str(accuracy_score(y_test,y_pred)))
```




    GradientBoostingClassifier()



    테스트 정확도 = 0.8663955323540123
    

**부스팅 모델 중 하나인 GradientBoosting의 경우, 랜덤포레스트와 비슷하게 87% 정도의 정확도가 나왔습니다.**


```python
from xgboost import XGBClassifier
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot
```


```python
xgb_model = XGBClassifier()
xgb_model.fit(X_train,y_train)
y_pred = xgb_model.predict(X_test)

print('테스트 정확도 = ' + str(accuracy_score(y_test,y_pred)))#정확도 계산
```

    [20:36:52] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    




    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
                  importance_type='gain', interaction_constraints='',
                  learning_rate=0.300000012, max_delta_step=0, max_depth=6,
                  min_child_weight=1, missing=nan, monotone_constraints='()',
                  n_estimators=100, n_jobs=8, num_parallel_tree=1, random_state=0,
                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
                  tree_method='exact', validate_parameters=1, verbosity=None)



    테스트 정확도 = 0.8708874590263446
    

**마찬가지로 부스팅 모델인 xgboost 또한, 약 87% 정확도를 보이고 있습니다.**

---
그렇다면, 앞서 3개의 모델이 어떻게 데이터를 학습시켜서 높은 설명력을 가지게 되었는지 **변수중요도**를 통해서 알아보겠습니다.


```python
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# 랜덤포레스트 변수 중요도
plt.figure(figsize=(10,5))

features_label = X.columns
rf_importances = rf.feature_importances_
indices = np.argsort(rf_importances)[::-1]
for i in range(X.shape[1]):
     print('%2d)%-*s%f'%(i+1, 30, features_label[i], rf_importances[indices[i]]))
plt.title('랜덤포레스트 Feature Importances')
plt.bar(range(X.shape[1]), rf_importances[indices], color='deepskyblue', align='center')
plt.xticks(range(X.shape[1]), features_label, rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()

# Gradient Boosting 변수 중요도
plt.figure(figsize=(10,5))

gb_importances = clf_gbc.feature_importances_
gb_importances = pd.Series(gb_importances, index = X_train.columns)

plt.title('Gradient Boosting Feature importances')
sns.barplot(x=gb_importances, y=X_train.columns)
plt.show()

# xgboost 변수 중요도

fig, ax = plt.subplots(figsize=(10, 5))
plot_importance(xgb_model, title='xgboost Feature importances', ax=ax)
```




    <Figure size 720x360 with 0 Axes>




    
![output_25_6](https://user-images.githubusercontent.com/69621732/117421197-48026a00-af59-11eb-8243-608b19e8bdc9.png)
    







    
![output_25_10](https://user-images.githubusercontent.com/69621732/117421198-489b0080-af59-11eb-8719-a80eb517778f.png)
    





    <AxesSubplot:title={'center':'xgboost Feature importances'}, xlabel='F score', ylabel='Features'>




    
![output_25_12](https://user-images.githubusercontent.com/69621732/117421202-489b0080-af59-11eb-8615-f75b53fa4a3b.png)
    


랜덤포레스트의 경우 퍼블, 두 부스팅 모델의 경우 타워, 억제기 철거 수가 높은 변수 중요도를 가진 것으로 나타납니다.  
퍼블은 잘 납득이 가지 않지만, 타워, 억제기는 많이 깰수록 승리에 가까워지기 때문에 충분히 설득력이 있는 것 같습니다.

그렇다면 게임 승패에 상관성이 있는 각 변수들은 과연 승패에 얼마만큼의 영향을 주는걸까요?  
이번에는 로지스틱 회귀분석을 활용하여 오브젝트 변수들이 승패에 미치는 영향을 분석해보려고 합니다. 

### 로지스틱 회귀분석

로지스틱 회귀분석은 odds라는 승산을 이용하여 0, 1값과 같은 이산형 데이터에 대한 회귀분석을 가능하게 합니다.  
따라서, 데이터셋에 있는 범주형, 수치형 데이터를 모두 입력값으로 받아 보다 정확한 예측이 가능합니다.

먼저, 목표변수와 설명변수의 상관관계를 확인해보겠습니다.


```python
blue_team.drop(['gameId', 'gameDuration'], axis=1).corr()[['Wins']]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Wins</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Wins</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>FirstBlood</th>
      <td>0.202711</td>
    </tr>
    <tr>
      <th>FirstTower</th>
      <td>0.459949</td>
    </tr>
    <tr>
      <th>FirstBaron</th>
      <td>0.361664</td>
    </tr>
    <tr>
      <th>FirstDragon</th>
      <td>0.264384</td>
    </tr>
    <tr>
      <th>FirstInhibitor</th>
      <td>0.650728</td>
    </tr>
    <tr>
      <th>DragonKills</th>
      <td>0.449474</td>
    </tr>
    <tr>
      <th>BaronKills</th>
      <td>0.348685</td>
    </tr>
    <tr>
      <th>TowerKills</th>
      <td>0.712080</td>
    </tr>
    <tr>
      <th>InhibitorKills</th>
      <td>0.567115</td>
    </tr>
  </tbody>
</table>
</div>



앞서 진행했던 부스팅 모델의 변수중요도와 유사하게, 타워와 억제기 관련 변수가 높은 상관계수를 나타내고 있습니다.  
그렇다면 본격적으로 승/패와 관련없는 게임ID, 게임시간 변수를 제거하고, 로지스틱 회귀분석을 진행해보겠습니다.


```python
# 게임ID, 게임시간 컬럼 제거
reg_df = blue_team.drop(columns=['gameId','gameDuration'])
reg_df = reg_df.dropna()
```


```python
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

X = reg_df[reg_df.columns.difference(['Wins'])]
y = reg_df['Wins']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=19)
model = sm.OLS(y_train, X_train).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>Wins</td>       <th>  R-squared (uncentered):</th>      <td>   0.776</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared (uncentered):</th> <td>   0.776</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>          <td>1.902e+04</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 07 May 2021</td> <th>  Prob (F-statistic):</th>           <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>16:20:38</td>     <th>  Log-Likelihood:    </th>          <td> -15781.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td> 49422</td>      <th>  AIC:               </th>          <td>3.158e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 49413</td>      <th>  BIC:               </th>          <td>3.166e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     9</td>      <th>                     </th>              <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>              <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
         <td></td>           <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>BaronKills</th>     <td>   -0.0300</td> <td>    0.006</td> <td>   -4.688</td> <td> 0.000</td> <td>   -0.043</td> <td>   -0.017</td>
</tr>
<tr>
  <th>DragonKills</th>    <td>    0.0010</td> <td>    0.002</td> <td>    0.538</td> <td> 0.591</td> <td>   -0.003</td> <td>    0.005</td>
</tr>
<tr>
  <th>FirstBaron</th>     <td>    0.0383</td> <td>    0.008</td> <td>    4.925</td> <td> 0.000</td> <td>    0.023</td> <td>    0.054</td>
</tr>
<tr>
  <th>FirstBlood</th>     <td>    0.0437</td> <td>    0.003</td> <td>   15.132</td> <td> 0.000</td> <td>    0.038</td> <td>    0.049</td>
</tr>
<tr>
  <th>FirstDragon</th>    <td>    0.0681</td> <td>    0.004</td> <td>   17.498</td> <td> 0.000</td> <td>    0.060</td> <td>    0.076</td>
</tr>
<tr>
  <th>FirstInhibitor</th> <td>    0.2751</td> <td>    0.006</td> <td>   49.080</td> <td> 0.000</td> <td>    0.264</td> <td>    0.286</td>
</tr>
<tr>
  <th>FirstTower</th>     <td>    0.1357</td> <td>    0.003</td> <td>   38.840</td> <td> 0.000</td> <td>    0.129</td> <td>    0.143</td>
</tr>
<tr>
  <th>InhibitorKills</th> <td>   -0.0133</td> <td>    0.003</td> <td>   -4.457</td> <td> 0.000</td> <td>   -0.019</td> <td>   -0.007</td>
</tr>
<tr>
  <th>TowerKills</th>     <td>    0.0643</td> <td>    0.001</td> <td>   67.805</td> <td> 0.000</td> <td>    0.062</td> <td>    0.066</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>997.596</td> <th>  Durbin-Watson:     </th> <td>   2.009</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>1351.854</td> 
</tr>
<tr>
  <th>Skew:</th>          <td> 0.253</td>  <th>  Prob(JB):          </th> <td>2.81e-294</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.633</td>  <th>  Cond. No.          </th> <td>    38.2</td> 
</tr>
</table><br/><br/>Notes:<br/>[1] R² is computed without centering (uncentered) since the model does not contain a constant.<br/>[2] Standard Errors assume that the covariance matrix of the errors is correctly specified.





실행 결과, 결정계수(R-squared)는 0.776이 나왔습니다.  
이 점수는 회귀분석이 얼마나 잘 되었는지 평가하는 지표이며, '추정한 모델이 주어진 데이터를 얼마나 잘 설명하는가?'에 대한 점수입니다.  
1에 가까울수록 데이터를 잘 설명하는데, 0.776이 나왔으므로 나름 설명력이 있다고 볼 수 있습니다.  
다음으로 F 통계량에 대한 p-value인 Prob(F-statistic) 수치를 보면 0.00으로 0.05 이하임을 알 수 있습니다.  
일반적으로 p-value가 0.05 이하면 회귀분석이 유의미한 결과를 가진다고 보셔도 될 것 같습니다.  

다음은 각 설명변수에 대한 p-value를 살펴보겠습니다.  
P>[t]값을 보면 'DragonKills' 변수를 제외하고는 모두 0.000으로 유의미한 피처라고 볼 수 있습니다.

학습한 coef(계수)값을 시각화하여 어떤 변수가 가장 영향력이 큰지 확인해보겠습니다.


```python
coefs = model.params.tolist()
coefs_series = pd.Series(coefs)

plt.rcParams['figure.figsize'] = [10,5]

x_labels = model.params.index.tolist()

ax = coefs_series.plot(kind='bar')
ax.set_title('feature_coef_graph')
ax.set_xlabel('X_features')
ax.set_ylabel('coef')
ax.set_xticklabels(x_labels)
```




    




    
![output_34_4](https://user-images.githubusercontent.com/69621732/117421204-49339700-af59-11eb-9331-8f8ed7fcac40.png)
    


FirstInhibitor(첫 억제기), FirstTower(첫 타워), TowerKills(타워 철거 수) 순으로 영향력이 큰 것으로 보입니다.  

특이한 점은 BaronKills(바론 처치 수)가 오히려 승리에 부정적 영향을 미치는 것으로 나타났습니다.  
아마도 변수간 높은 상관성으로 인해 다중공선성이 발생한 것으로 생각됩니다.  

이를 알아보기 위해 Heatmap 방식의 시각화와 VIF 척도를 통해 변수간의 상관관계를 알아보겠습니다.


```python
corr = reg_df[X.columns].corr(method='pearson')
sns.set(font_scale=1.5)
hm = sns.heatmap(corr.values, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws ={'size' : 15}, 
                 yticklabels = X.columns, xticklabels = X.columns )

sns.set(style = 'whitegrid', context = 'notebook')
plt.show()
```


    
![output_36_0](https://user-images.githubusercontent.com/69621732/117421205-49339700-af59-11eb-8c49-9a908d6ec1ac.png)
    


FirstBaron과 BaronKills에서 0.88이라는 높은 상관성을 발견할 수 있습니다.  


```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif['VIF Factor'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['features'] = X.columns
vif.round()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VIF Factor</th>
      <th>features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.0</td>
      <td>BaronKills</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.0</td>
      <td>DragonKills</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6.0</td>
      <td>FirstBaron</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
      <td>FirstBlood</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.0</td>
      <td>FirstDragon</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5.0</td>
      <td>FirstInhibitor</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3.0</td>
      <td>FirstTower</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5.0</td>
      <td>InhibitorKills</td>
    </tr>
    <tr>
      <th>8</th>
      <td>12.0</td>
      <td>TowerKills</td>
    </tr>
  </tbody>
</table>
</div>



VIF는 다중 회귀 모델에서 독립 변수간 상관 관계가 있는지 측정하는 척도입니다.  
VIF가 10이 넘으면 다중공선성 있다고 판단하며 5가 넘으면 주의할 필요가 있는 것으로 봅니다.  

TowerKills의 VIF가 12이고, BaronKills의 경우 FirstBaron과 상관성이 높고 VIF 또한 7이기 때문에 두 변수를 제거해주겠습니다.


```python
# 두 변수를 제거하고 회귀분석 수행

X2 = reg_df[reg_df.columns.difference(['Wins', 'TowerKills', 'BaronKills'])]
y2 = reg_df['Wins']

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.25, random_state=19)
model = sm.OLS(y_train2, X_train2).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>Wins</td>       <th>  R-squared (uncentered):</th>      <td>   0.755</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared (uncentered):</th> <td>   0.755</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>          <td>2.177e+04</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 07 May 2021</td> <th>  Prob (F-statistic):</th>           <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>16:38:08</td>     <th>  Log-Likelihood:    </th>          <td> -17980.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td> 49422</td>      <th>  AIC:               </th>          <td>3.597e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 49415</td>      <th>  BIC:               </th>          <td>3.604e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     7</td>      <th>                     </th>              <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>              <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
         <td></td>           <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>DragonKills</th>    <td>    0.0566</td> <td>    0.002</td> <td>   32.098</td> <td> 0.000</td> <td>    0.053</td> <td>    0.060</td>
</tr>
<tr>
  <th>FirstBaron</th>     <td>    0.0830</td> <td>    0.004</td> <td>   19.130</td> <td> 0.000</td> <td>    0.074</td> <td>    0.091</td>
</tr>
<tr>
  <th>FirstBlood</th>     <td>    0.0812</td> <td>    0.003</td> <td>   27.413</td> <td> 0.000</td> <td>    0.075</td> <td>    0.087</td>
</tr>
<tr>
  <th>FirstDragon</th>    <td>    0.0671</td> <td>    0.004</td> <td>   16.572</td> <td> 0.000</td> <td>    0.059</td> <td>    0.075</td>
</tr>
<tr>
  <th>FirstInhibitor</th> <td>    0.4088</td> <td>    0.005</td> <td>   75.396</td> <td> 0.000</td> <td>    0.398</td> <td>    0.419</td>
</tr>
<tr>
  <th>FirstTower</th>     <td>    0.2308</td> <td>    0.003</td> <td>   69.137</td> <td> 0.000</td> <td>    0.224</td> <td>    0.237</td>
</tr>
<tr>
  <th>InhibitorKills</th> <td>    0.0612</td> <td>    0.003</td> <td>   21.590</td> <td> 0.000</td> <td>    0.056</td> <td>    0.067</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>1137.275</td> <th>  Durbin-Watson:     </th> <td>   2.009</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>1969.691</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 0.200</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 3.892</td>  <th>  Cond. No.          </th> <td>    8.45</td>
</tr>
</table><br/><br/>Notes:<br/>[1] R² is computed without centering (uncentered) since the model does not contain a constant.<br/>[2] Standard Errors assume that the covariance matrix of the errors is correctly specified.



지수함수를 통해 분석 결과를 직관적으로 해석해보겠습니다.


```python
for i in range(len(model.params)):
    print('다른 변수가 고정되어 있으며, {} 이 한단위 상승할 때 승리할 확률이 {} 배 증가한다.\n'.format(model.params.keys()[i],np.exp(model.params.values[i])))
```

    다른 변수가 고정되어 있으며, DragonKills 이 한단위 상승할 때 승리할 확률이 1.0582131938024417 배 증가한다.
    
    다른 변수가 고정되어 있으며, FirstBaron 이 한단위 상승할 때 승리할 확률이 1.0865299658421483 배 증가한다.
    
    다른 변수가 고정되어 있으며, FirstBlood 이 한단위 상승할 때 승리할 확률이 1.084617434256652 배 증가한다.
    
    다른 변수가 고정되어 있으며, FirstDragon 이 한단위 상승할 때 승리할 확률이 1.0693691306250497 배 증가한다.
    
    다른 변수가 고정되어 있으며, FirstInhibitor 이 한단위 상승할 때 승리할 확률이 1.5050327542454751 배 증가한다.
    
    다른 변수가 고정되어 있으며, FirstTower 이 한단위 상승할 때 승리할 확률이 1.259585066764061 배 증가한다.
    
    다른 변수가 고정되어 있으며, InhibitorKills 이 한단위 상승할 때 승리할 확률이 1.0631612573632667 배 증가한다.
    
    

변수 제거 후 회귀분석 결과, 결정계수는 약간 감소하였지만 각 변수들에 대한 결과는 알맞게 나온 것 같습니다.  
p-value 또한 모두 0.00으로 변수들 모두 유의미함을 알 수 있습니다. 이제 VIF를 다시 구해보겠습니다.


```python
vif = pd.DataFrame()
vif['VIF Factor'] = [variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])]
vif['features'] = X2.columns
vif.round()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VIF Factor</th>
      <th>features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.0</td>
      <td>BaronKills</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>DragonKills</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6.0</td>
      <td>FirstBaron</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
      <td>FirstBlood</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.0</td>
      <td>FirstDragon</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5.0</td>
      <td>FirstInhibitor</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2.0</td>
      <td>FirstTower</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4.0</td>
      <td>InhibitorKills</td>
    </tr>
  </tbody>
</table>
</div>



모든 VIF가 10 미만으로 다중공선성 문제가 해결되었습니다.  

---
이번 포스팅에서는 머신러닝 기법을 활용해서 리그 오브 레전드 데이터를 좀 더 디테일하게 분석해보았습니다.  
팀 전체 데이터만을 활용해서 승리/패배 예측 모델링을 진행해보았는데, 플레이어 데이터를 추가한다면 보다 더 정교한 분석이 이루어질 것 같습니다.  

확실히 롤을 매우 즐겨하다보니 분석 과정이나 결과에서 해석에 대한 어려움은 없이 즐기면서 분석을 할 수 있었던 것 같습니다.  
다음 포스팅에서는 플레이어 데이터를 활용해서 다양한 인사이트를 발굴해내는 시간을 갖도록 하겠습니다.  

비록 단순한 개인 프로젝트이지만, 지속적으로 분석 아키텍처를 공부하면서 보다 유용한 결과를 유저들에게 제공한다면 정말 좋을 것 같습니다:)


```python

```
