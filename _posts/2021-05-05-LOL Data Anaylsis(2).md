# League Of Legends Data Analysis
---

이전에 포스팅한 롤 게임 데이터 수집에 이어, 게임 데이터를 활용해 EDA를 진행해보도록 하겠습니다.  
승/패에 영향을 미치는 변수 확인 및 세분화를 통해 다양한 게임 요소를 분석해보겠습니다.

**이전에 수집했던 본인의 소환사 데이터의 경우 표본이 부족하기 때문에, 그랜드 마스터 티어 유저들의 데이터를 수집하여 분석을 진행하였습니다.**

- [<Step2. 리그 오브 레전드 데이터 EDA>](#Step1.-Data-Processing)
    - [데이터 불러오기]
    - [데이터셋 기본정보 확인]
    - [데이터 전처리]

# Step2. 리그 오브 레전드 데이터 EDA

### [데이터 불러오기]


```python
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from pandas.io.json import json_normalize
import seaborn as sns
import warnings
warnings.filterwarnings(action='ignore')
```


```python
lol_df = pd.read_csv('GrandMaster_Games.csv')
lol_df.head()
```




<div>

</style>
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
      <th>...</th>
      <th>Wins.1</th>
      <th>FirstBlood.1</th>
      <th>FirstTower.1</th>
      <th>FirstBaron.1</th>
      <th>FirstDragon.1</th>
      <th>FirstInhibitor.1</th>
      <th>DragonKills.1</th>
      <th>BaronKills.1</th>
      <th>TowerKills.1</th>
      <th>InhibitorKills.1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4241678498</td>
      <td>2098</td>
      <td>Lose</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>Win</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4233222221</td>
      <td>1686</td>
      <td>Lose</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>Win</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>4</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4233113995</td>
      <td>1588</td>
      <td>Win</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>Lose</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4229230455</td>
      <td>1126</td>
      <td>Win</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>Lose</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4228244819</td>
      <td>1262</td>
      <td>Win</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>Lose</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



### [데이터셋 기본정보 확인]

데이터셋에 대한 기본 속성을 확인해보았습니다.  
null값은 존재하지 않으며, 블루팀과 레드팀 데이터가 하나의 데이터셋에 있음을 알 수 있습니다.  


```python
# 데이터 유형 및 결측값 확인
lol_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 65896 entries, 0 to 65895
    Data columns (total 22 columns):
     #   Column            Non-Null Count  Dtype 
    ---  ------            --------------  ----- 
     0   gameId            65896 non-null  int64 
     1   gameDuration      65896 non-null  int64 
     2   Wins              65896 non-null  object
     3   FirstBlood        65896 non-null  bool  
     4   FirstTower        65896 non-null  bool  
     5   FirstBaron        65896 non-null  bool  
     6   FirstDragon       65896 non-null  bool  
     7   FirstInhibitor    65896 non-null  bool  
     8   DragonKills       65896 non-null  int64 
     9   BaronKills        65896 non-null  int64 
     10  TowerKills        65896 non-null  int64 
     11  InhibitorKills    65896 non-null  int64 
     12  Wins.1            65896 non-null  object
     13  FirstBlood.1      65896 non-null  bool  
     14  FirstTower.1      65896 non-null  bool  
     15  FirstBaron.1      65896 non-null  bool  
     16  FirstDragon.1     65896 non-null  bool  
     17  FirstInhibitor.1  65896 non-null  bool  
     18  DragonKills.1     65896 non-null  int64 
     19  BaronKills.1      65896 non-null  int64 
     20  TowerKills.1      65896 non-null  int64 
     21  InhibitorKills.1  65896 non-null  int64 
    dtypes: bool(10), int64(10), object(2)
    memory usage: 6.7+ MB
    

라이엇 개발자 페이지에 따른 컬럼별 명세는 다음과 같습니다

- gameId - 경기의 고유 Id입니다.
- gameDuration - 경기 시간, 초라고 생각하시면 됩니다.(연속형변수)
- Wins - 승 / 패 , target_variable로 사용할 변수입니다. (W/F)
- FirstBlood - 가장 먼저 상대팀의 챔피언을 킬했는지 여부. (T/F)
- FirstTower - 가장 먼저 상대팀의 타워를 깻는지 여부. (T/F)
- FirstBaron - 가장 먼저 바론을 먹었는지 여부. (T/F)
- FirstDragon - 가장 먼저 드래곤을 먹었는지 여부. (T/F)
- Firstinhibitor - 가장 먼저 상대팀의 억제기를 깻는지 여부. (T/F)
- DragonKills - 처치한 드래곤의 수(연속형변수)
- BaronKills - 처치한 바론의 수(연속형변수)
- TowerKills - 깬 타워의 수(연속형변수)
- InhibitorKills - 깬 억제기의 수(연속형변수)

다음은 승리팀과 패배팀의 연속형 변수 통계치입니다.  
상위 티어의 경우, 스노우볼링 능력이 뛰어나 한 번의 격차가 승패를 판가름 짓는 경우가 많습니다.  
그 때문인지 승리한 팀이 타워, 억제기, 바론, 드래곤의 처치 횟수가 평균적으로 훨씬 높은것을 확인할 수 있습니다.  
제 예상이지만 하위 티어에서는 반전이 많이 일어나기 때문에, 그랜드마스터 게임과는 다른 통계치가 나올 것 같습니다. 


```python
# 승리팀 통계치

lol_df[lol_df['Wins'] == 'Win'].describe()[['TowerKills','InhibitorKills','BaronKills','DragonKills']]
```




<div>

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TowerKills</th>
      <th>InhibitorKills</th>
      <th>BaronKills</th>
      <th>DragonKills</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>32659.000000</td>
      <td>32659.000000</td>
      <td>32659.000000</td>
      <td>32659.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6.731284</td>
      <td>1.166294</td>
      <td>0.482991</td>
      <td>1.907223</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.670333</td>
      <td>0.949420</td>
      <td>0.615926</td>
      <td>1.238739</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>9.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>11.000000</td>
      <td>9.000000</td>
      <td>4.000000</td>
      <td>7.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 패배팀 통계치

lol_df[lol_df['Wins'] == 'Lose'].describe()[['TowerKills','InhibitorKills','BaronKills','DragonKills']]
```




<div>

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TowerKills</th>
      <th>InhibitorKills</th>
      <th>BaronKills</th>
      <th>DragonKills</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>33237.000000</td>
      <td>33237.000000</td>
      <td>33237.000000</td>
      <td>33237.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.949755</td>
      <td>0.134218</td>
      <td>0.110509</td>
      <td>0.780726</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.002361</td>
      <td>0.477318</td>
      <td>0.352215</td>
      <td>0.988165</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>11.000000</td>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>6.000000</td>
    </tr>
  </tbody>
</table>
</div>



### [데이터 전처리]

다음은 분석의 용이성을 위해 T/F 범주형 데이터를 1,0으로 인코딩 해줍니다.  
또한, 블루팀 데이터만으로 승/패에 대한 특성을 분석할 수 있기 때문에 블루팀 데이터만 추출합니다.


```python
#분석의 용이성을 위해서 타겟 데이터를 제외한 범주형 데이터를 인코딩
'''
True : 1
False : 0
'''
tf_mapping = {True:1,False:0}
bool_column = lol_df.select_dtypes('bool').columns.tolist()

for i in bool_column:
    lol_df[i] = lol_df[i].map(tf_mapping)
```


```python
# 블루팀 관련 컬럼까지만 슬라이싱

blue_team = lol_df.iloc[:,0:12]

blue_team.head()
```




<div>

</style>
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



### [데이터 EDA]


```python
# 여러 개의 결과값을 한 셀에 출력할 수 있는 라이브러리
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

plt.rcParams['font.family'] = 'KoPubDotum Medium'
```

다음은 crosstab 함수를 통해 범주형 데이터별로 승/패 여부에 대한 교차분석을 진행하였습니다.  
승리한 팀의 경우 실제로 패배한 팀에 비해 퍼블, 첫 타워 철거·첫 오브젝트 처치 비율이 높음을 알 수 있습니다. 


```python
# 범주형 데이터는 'FirstBlood', 'FirstTower', 'FirstBaron', 'FirstDragon', FirstInhibitor

cols = ['FirstBlood', 'FirstTower', 'FirstBaron', 'FirstDragon', 'FirstInhibitor']
for i in cols:
    pd.crosstab(blue_team[i], blue_team['Wins'], 
    margins=True).style.background_gradient(cmap='Blues')
```

![output_22_1](https://user-images.githubusercontent.com/69621732/117247253-e7036500-ae78-11eb-9f3a-d3f24f26d77e.png)



좀 더 한눈에 알아볼 수 있도록 데이터값에 따른 승/패 누적 그래프로 나타내보았습니다.  
첫 오브젝트를 차지할수록 승리할 확률이 높음을 알 수 있습니다.  
특히, 억제기의 경우 승리팀이 처음 억제기를 깬 비율이 압도적입니다.


```python
plt.figure(figsize=(20,5))
f, ax = plt.subplots(1,5)
plt.suptitle('변수값에 따른 승/패 Count', fontsize='x-large')

plot = blue_team.groupby(['FirstBlood', 'Wins'])['Wins'].count().unstack('Wins')
plot.plot(kind='bar', stacked=True, ax=ax[0])
ax[0].get_legend().set_visible(False)

plot2 = blue_team.groupby(['FirstTower', 'Wins'])['Wins'].count().unstack('Wins')
plot2.plot(kind='bar', stacked=True, ax=ax[1])
ax[1].get_legend().set_visible(False)

plot3 = blue_team.groupby(['FirstBaron', 'Wins'])['Wins'].count().unstack('Wins')
plot3.plot(kind='bar', stacked=True, ax=ax[2])
ax[2].get_legend().set_visible(False)

plot4 = blue_team.groupby(['FirstDragon', 'Wins'])['Wins'].count().unstack('Wins')
plot4.plot(kind='bar', stacked=True, ax=ax[3])
ax[3].get_legend().set_visible(False)

plot5 = blue_team.groupby(['FirstInhibitor', 'Wins'])['Wins'].count().unstack('Wins')
plot5.plot(kind='bar', stacked=True, ax=ax[4])

f.tight_layout()
plt.legend(bbox_to_anchor=(1.05, 1))
plt.show()
```




    
![output_22_9](https://user-images.githubusercontent.com/69621732/117247259-e9fe5580-ae78-11eb-8e81-4c1ee5e1c856.png)
    


### 경기 시간 segment별 EDA

분석을 진행하다보니 문득 게임 시간대별 통계가 궁금했습니다. 흔히 롤은 멘탈 게임이라고도 하죠.  
게임 후반으로 갈수록 오브젝트의 영향이 줄어들고, 챔피언 포텐·상황 판단 등의 정성적 요소가 게임 판도를 뒤집기도 합니다.  
"과연 이러한 게임의 양상들을 데이터도 설명하고 있을까?" 라는 생각이 들어서 게임시간별로 데이터를 segment화하여 분석해보았습니다.


```python
#n_tile로 게임시간을 분위수로 파악

blue_team['game_time'] = blue_team['gameDuration']/60

game_part1 = blue_team[blue_team['game_time']<20].sort_values('Wins')
game_part2 = blue_team[blue_team['game_time']<30].sort_values('Wins')
game_part3 = blue_team[blue_team['game_time']>=30].sort_values('Wins')
game_part4 = blue_team[blue_team['game_time']>=40].sort_values('Wins')
```


```python
game_part = []
for i in range(len(blue_team)):
    if blue_team['game_time'][i]>=30 and blue_team['game_time'][i] <40:
        game_part.append('30분 이상')
    elif blue_team['game_time'][i]<30 and blue_team['game_time'][i]>=20:
        game_part.append('30분 미만')
    elif blue_team['game_time'][i]<20:
        game_part.append('20분 미만')
    else:
        game_part.append('40분 이상')
        
blue_team['game_part'] = game_part
```


```python
# factorplot을 활용하여 각 변수값에 대한 승리확률 분석


ax1 = fig.add_subplot(2,3,1)
ax2 = fig.add_subplot(2,3,2)
ax3 = fig.add_subplot(2,3,3)
ax4 = fig.add_subplot(2,3,4)
ax5 = fig.add_subplot(2,3,5)

sns.factorplot('Wins', 'FirstBlood', hue='game_part', data=blue_team, ax=ax1)
sns.factorplot('Wins', 'FirstTower', hue='game_part', data=blue_team, ax=ax2)
sns.factorplot('Wins', 'FirstBaron', hue='game_part', data=blue_team, ax=ax3)
sns.factorplot('Wins', 'FirstDragon', hue='game_part', data=blue_team, ax=ax4)
sns.factorplot('Wins', 'FirstInhibitor', hue='game_part', data=blue_team, ax=ax5)

plt.show()
```



    
![output_27_5](https://user-images.githubusercontent.com/69621732/117247269-eec30980-ae78-11eb-9278-381008c276c5.png)  ![output_27_6](https://user-images.githubusercontent.com/69621732/117247271-ef5ba000-ae78-11eb-826b-dab3800e9a04.png)
   
    

    
![output_27_7](https://user-images.githubusercontent.com/69621732/117247273-eff43680-ae78-11eb-8fa8-bb416ba98517.png)  ![output_27_8](https://user-images.githubusercontent.com/69621732/117247276-eff43680-ae78-11eb-9b50-b988ae11b54a.png)
    



![output_27_9](https://user-images.githubusercontent.com/69621732/117247278-f08ccd00-ae78-11eb-8d33-cce033873fc2.png)
    


분석 결과, 게임 시간이 길어질수록 오브젝트에 대한 영향이 확연히 줄어드는 것을 알 수 있습니다.  
퍼블의 경우, 40분 이후에는 오히려 퍼블을 따낸 팀이 패배할 확률이 더 높은 것으로 나타났습니다.  
반면, 20분 미만의 게임에서는 오브젝트의 영향도가 크게 나타납니다.  
스노우볼링, 멘탈, 집중력 등을 복합적으로 보여주는 재밌는 결과인 것 같습니다.

---

이번 포스팅에서는 승리/패배한 팀에 대한 범주형 변수 분석을 진행하였습니다.  
요즘 롤을 플레이 하다보면 '퍼블 따는 팀은 지게 되어있다'라는 말이 자주 나옵니다.  
재미로 나온 말이지만, 데이터의 결과처럼 실제로 퍼블의 영향은 실제로 크지 않은 것 같습니다.  
데이터 결과가 실제 제가 느꼈던 게임 양상과 비슷한 것을 보면, 알다가도 신기한 기분이 듭니다.

다음 포스팅에서는, 변수들을 통한 승/패 예측과 각 변수들이 승/패에 얼마만큼 영향을 미치는지  
Logistic Regression을 통해서 알아보도록 하겠습니다.

```python

```
