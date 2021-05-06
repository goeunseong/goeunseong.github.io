---
jupyter:
  jupytext:
    formats: ipynb,py:light,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

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

<!-- #region heading_collapsed=true -->

### [데이터 불러오기]
<!-- #endregion -->

```python hidden=true
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

```python hidden=true
lol_df = pd.read_csv('GrandMaster_Games.csv')
lol_df.head()
```

### [데이터셋 기본정보 확인]


데이터셋에 대한 기본 속성을 확인해보았습니다.  
null값은 존재하지 않으며, 블루팀과 레드팀 데이터가 하나의 데이터셋에 있음을 알 수 있습니다.  

```python
# 데이터 유형 및 결측값 확인
lol_df.info()
```

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

```python
# 패배팀 통계치

lol_df[lol_df['Wins'] == 'Lose'].describe()[['TowerKills','InhibitorKills','BaronKills','DragonKills']]
```

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
# 범주형 데이터는 'FirstBlood', 'FirstTower', 'FirstBaron', 'FirstDragon'

cols = ['FirstBlood', 'FirstTower', 'FirstBaron', 'FirstDragon']
for i in cols:
    pd.crosstab(blue_team[i], blue_team['Wins'], 
    margins=True).style.background_gradient(cmap='Blues')
```

![image-20210506134902071](C:\Users\goeunseong\AppData\Roaming\Typora\typora-user-images\image-20210506134902071.png)

좀 더 한눈에 알아볼 수 있도록 데이터값에 따른 승/패 누적 그래프로 나타내보았습니다.  
첫 오브젝트를 차지할수록 승리할 확률이 높음을 알 수 있습니다.

```python
plt.figure(figsize=(20,5))
f, ax = plt.subplots(1,4)
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

f.tight_layout()
plt.legend(bbox_to_anchor=(1.05, 1))
plt.show()
```

![output_22_8](https://user-images.githubusercontent.com/69621732/117244624-588ce480-ae74-11eb-9430-8f6344166479.png)

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

ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
sns.factorplot('Wins', 'FirstBlood', hue='game_part', data=blue_team, ax=ax1)
sns.factorplot('Wins', 'FirstTower', hue='game_part', data=blue_team, ax=ax2)
sns.factorplot('Wins', 'FirstBaron', hue='game_part', data=blue_team, ax=ax3)
sns.factorplot('Wins', 'FirstDragon', hue='game_part', data=blue_team, ax=ax4)

plt.show()
```

https://user-images.githubusercontent.com/69621732/117244635-5cb90200-ae74-11eb-9ee2-55345a73ede4.png

https://user-images.githubusercontent.com/69621732/117244648-617db600-ae74-11eb-90c5-8a5a32be4a75.png

https://user-images.githubusercontent.com/69621732/117244652-62aee300-ae74-11eb-88ee-07a1e03fbd67.png

https://user-images.githubusercontent.com/69621732/117244653-63477980-ae74-11eb-8801-47a345ccfb8f.png


분석 결과, 게임 시간이 길어질수록 오브젝트에 대한 영향이 확연히 줄어드는 것을 알 수 있습니다.  
퍼블의 경우, 40분 이후에는 오히려 퍼블을 따낸 팀이 패배할 확률이 더 높은 것으로 나타났습니다.  
반면, 20분 미만의 게임에서는 오브젝트의 영향도가 크게 나타납니다.  
스노우볼링, 멘탈, 집중력 등을 복합적으로 보여주는 재밌는 결과인 것 같습니다.

```python

```

