# League Of Legends Data Analysis
---

목표 : League Of Legends에서 제공하는 API 파싱을 통해 OP.GG와 같은 전적 검색 커뮤니티에서 제공하는   
통계적 분석과 승/패 예측 등의 모델링을 수행해보고자 합니다.


롤에 대한 애정이 남다른 만큼, 직접 플레이한 게임 데이터를 통해 분석을 진행해보겠습니다.


- [<Step1. API 파싱> 리그 오브 레전드 데이터 살펴보기]
    - [플레이어 정보 불러오기]
    - [전적 정보 불러오기]

# <Step1. API 파싱> 리그 오브 레전드 데이터 살펴보기

### [플레이어 정보 불러오기]

OP.GG, 포우와 같은 게임 데이터 플랫폼의 경우 Riot api를 이용하여 데이터를 수집해 유저에게 유용한 정보를 제공하고 있습니다.  이러한 플랫폼만큼은 아니지만, 직접 api를 수집해 나름의 유용한 인사이트를 찾아보도록 하겠습니다.  

먼저 Riot API를 이용하기 위해 API Key를 발급받습니다. 라이엇 개발자 포털의 SUMMONER-V4라는 API에서는 소환사 이름으로 id를 포함한 여러 정보를 호출할 수 있습니다. 요청 헤더와 URL의 name 영역에 각각 제 API Key와 소환사 이름을 입력한 결과, 해당 닉네임에 대한 여러 정보를 얻을 수 있습니다.  

아래는 소환사 이름을 통해 소환사 정보를 가져오는 간단한 코드 및 결과입니다.


```python
import json
import requests
import pandas as pd
import time


api_key = "RGAPI-4a174133-5c53-4bea-b1ab-84b9766d5cbf"
selectnum = "1"

def get_rankinfo(api, name):
    
    print('플레이어를 검색합니다')
    if selectnum == "1":
        URL = "https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-name/"+name
        re = requests.get(URL, headers={"X-Riot-Token": api_key})
        if re.status_code == 200:
            #코드가 200일때
            resobj = json.loads(re.text)
            URL = "https://kr.api.riotgames.com/lol/league/v4/entries/by-summoner/"+resobj["id"]
            res = requests.get(URL, headers={"X-Riot-Token": api_key})
            rankinfo = json.loads(res.text)
            print("소환사명: "+name)
            for i in rankinfo:
                if i["queueType"] == "RANKED_SOLO_5x5":
                    #솔랭과 자랭중 솔랭
                    print("솔로랭크:")
                    print(f'티어: {i["tier"]} {i["rank"]}')
                    print(f'승: {i["wins"]}판, 패: {i["losses"]}판')
                else:
                    # 솔랭과 자랭중 자랭
                    print("자유랭크:")
                    print(f'티어: {i["tier"]} {i["rank"]}')
                    print(f'승: {i["wins"]}판, 패: {i["losses"]}판')
        else:
            # 코드가 200이 아닐때(즉 찾는 닉네임이 없을때)
            print("소환사가 존재하지 않습니다")
```

저의 소환사 닉네임을 입력한 결과입니다. 본캐를 포함해 총 3개의 계정을 갖고 있습니다.   
과거 다이아라는 명예를 회복하기 위해 고군분투 중이지만, 플레티넘과 골드에서 허덕이는 제 전적이 보입니다.  


```python
get_rankinfo(api_key, 'sylass')
print('\n')
get_rankinfo(api_key, '마스터 무야호')
print('\n')
get_rankinfo(api_key, '뭐시 중헌데')
```

    플레이어를 검색합니다
    소환사명: sylass
    솔로랭크:
    티어: GOLD II
    승: 61판, 패: 60판
    
    
    플레이어를 검색합니다
    소환사명: 마스터 무야호
    솔로랭크:
    티어: PLATINUM III
    승: 152판, 패: 153판
    자유랭크:
    티어: SILVER III
    승: 4판, 패: 6판
    
    
    플레이어를 검색합니다
    소환사명: 뭐시 중헌데
    

### [전적 정보 불러오기]

다음은 소환사 닉네임을 입력해 전적 정보를 수집해보았습니다.  
이를 통해, 매치(게임) 고유 ID와 챔피언, 라인, 게임 시간 등을 알 수 있습니다.  

특히 매치 고유 ID의 경우, 게임 내부 데이터를 수집에 활용되는 파라미터로 활용됩니다. 


```python
def get_match_info(api, name):
    URL = "https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-name/"+name
    res = requests.get(URL, headers={"X-Riot-Token": api_key})
    if res.status_code == 200:
        #코드가 200일때
        resobj = json.loads(res.text)
        URL = "https://kr.api.riotgames.com/lol/match/v4/matchlists/by-account/"+resobj["accountId"]
        res = requests.get(URL, headers={"X-Riot-Token": api_key})
        match_info = json.loads(res.text)
    else:
        print("소환사가 존재하지 않습니다")
    
    return match_info
```


```python
match_info_1 = get_match_info(api_key, 'sylass')
match_info_2 = get_match_info(api_key, '마스터 무야호')
match_info_3 = get_match_info(api_key, '뭐시 중헌데')

match_info_df1 = pd.DataFrame(match_info_1['matches'])
match_info_df2 = pd.DataFrame(match_info_2['matches'])
match_info_df3 = pd.DataFrame(match_info_3['matches'])

match_df = pd.concat([match_info_df1, match_info_df2, match_info_df3])
match_df.reset_index(inplace=True)

match_df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>platformId</th>
      <th>gameId</th>
      <th>champion</th>
      <th>queue</th>
      <th>season</th>
      <th>timestamp</th>
      <th>role</th>
      <th>lane</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>KR</td>
      <td>5159458443</td>
      <td>92</td>
      <td>450</td>
      <td>13</td>
      <td>1619666971374</td>
      <td>DUO</td>
      <td>NONE</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>KR</td>
      <td>5159515828</td>
      <td>84</td>
      <td>450</td>
      <td>13</td>
      <td>1619665611647</td>
      <td>DUO_SUPPORT</td>
      <td>MID</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>KR</td>
      <td>5159202258</td>
      <td>22</td>
      <td>450</td>
      <td>13</td>
      <td>1619626184879</td>
      <td>DUO_SUPPORT</td>
      <td>NONE</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>KR</td>
      <td>5159059636</td>
      <td>164</td>
      <td>450</td>
      <td>13</td>
      <td>1619624848006</td>
      <td>DUO</td>
      <td>NONE</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>KR</td>
      <td>5159115216</td>
      <td>75</td>
      <td>450</td>
      <td>13</td>
      <td>1619623349856</td>
      <td>DUO_CARRY</td>
      <td>MID</td>
    </tr>
  </tbody>
</table>
</div>



### [매치ID를 이용한 최종 게임 데이터 불러오기]

위에서 수집한 매치ID(gameId)를 이용하여 게임 내부의 데이터를 본격적으로 수집해보도록 하겠습니다.  
라이엇측에서 개발자 api를 이용하는데 이용량의 한계를 걸어놨기 때문에 지속적으로 수집이 불가능합니다.  

따라서 API Cost 제한에 따른 429error, 503error가 나왔을 때, time 모듈을 활용하여 항상 시간텀을  
무한루프로 돌게끔 만들어서 수집하도록 코드를 작성하였습니다.


```python
match_data = pd.DataFrame()

for i in range(len(match_df)):
    match_url = 'https://kr.api.riotgames.com/lol/match/v4/matches/' + str(match_df['gameId'][i])
    res = requests.get(match_url, headers={"X-Riot-Token": api_key})
    
    if res.status_code == 200: # response가 정상이면 바로 맨 밑으로 이동하여 정상적으로 코드 실행
        pass
    elif res.status_code == 429:
        print('api cost full : infinite loop start')
        print('loop location : ',i)
        start_time = time.time()

        while True: # 429error가 끝날 때까지 무한 루프
            if res.status_code == 429:

                print('try 10 second wait time')
                time.sleep(10)
                
                res = requests.get(match_url, headers={"X-Riot-Token": api_key})
                print(res.status_code)
                
            elif res.status_code == 200: #다시 response 200이면 loop escape
                print('total wait time : ', time.time() - start_time)
                print('recovery api cost')
                break
    elif res.status_code == 503: # 잠시 서비스를 이용하지 못하는 에러
        print('service available error')
        start_time = time.time()

        while True:
            if res.status_code == 503 or res.status_code == 429:

                print('try 10 second wait time')
                time.sleep(10)

                res = requests.get(match_url, headers={"X-Riot-Token": api_key})
                print(res.status_code)

            elif res.status_code == 200: # 똑같이 response가 정상이면 loop escape
                print('total error wait time : ', time.time() - start_time)
                print('recovery api cost')
                break            
    mat = pd.DataFrame(list(res.json().values()), index=list(res.json().keys())).T
    match_data = pd.concat([match_data,mat])
```

    api cost full : infinite loop start
    loop location :  88
    try 10 second wait time
    429
    try 10 second wait time
    429
    try 10 second wait time
    429
    try 10 second wait time
    429
    try 10 second wait time
    429
    try 10 second wait time
    429
    try 10 second wait time
    200
    total wait time :  71.53895282745361
    recovery api cost
    api cost full : infinite loop start
    loop location :  188
    try 10 second wait time
    429
    try 10 second wait time
    429
    try 10 second wait time
    429
    try 10 second wait time
    429
    try 10 second wait time
    429
    try 10 second wait time
    429
    try 10 second wait time
    429
    try 10 second wait time
    429
    try 10 second wait time
    200
    total wait time :  92.027419090271
    recovery api cost
    api cost full : infinite loop start
    loop location :  288
    try 10 second wait time
    429
    try 10 second wait time
    429
    try 10 second wait time
    429
    try 10 second wait time
    429
    try 10 second wait time
    429
    try 10 second wait time
    429
    try 10 second wait time
    429
    try 10 second wait time
    429
    try 10 second wait time
    200
    total wait time :  91.93531656265259
    recovery api cost
    


```python
# 수집한 게임 데이터 결과
match_data.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gameId</th>
      <th>platformId</th>
      <th>gameCreation</th>
      <th>gameDuration</th>
      <th>queueId</th>
      <th>mapId</th>
      <th>seasonId</th>
      <th>gameVersion</th>
      <th>gameMode</th>
      <th>gameType</th>
      <th>teams</th>
      <th>participants</th>
      <th>participantIdentities</th>
      <th>status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5159458443</td>
      <td>KR</td>
      <td>1619666971374</td>
      <td>1123</td>
      <td>450</td>
      <td>12</td>
      <td>13</td>
      <td>11.9.372.2066</td>
      <td>ARAM</td>
      <td>MATCHED_GAME</td>
      <td>[{'teamId': 100, 'win': 'Fail', 'firstBlood': ...</td>
      <td>[{'participantId': 1, 'teamId': 100, 'champion...</td>
      <td>[{'participantId': 1, 'player': {'platformId':...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>0</th>
      <td>5159515828</td>
      <td>KR</td>
      <td>1619665611647</td>
      <td>1211</td>
      <td>450</td>
      <td>12</td>
      <td>13</td>
      <td>11.9.372.2066</td>
      <td>ARAM</td>
      <td>MATCHED_GAME</td>
      <td>[{'teamId': 100, 'win': 'Win', 'firstBlood': T...</td>
      <td>[{'participantId': 1, 'teamId': 100, 'champion...</td>
      <td>[{'participantId': 1, 'player': {'platformId':...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>0</th>
      <td>5159202258</td>
      <td>KR</td>
      <td>1619626184879</td>
      <td>1084</td>
      <td>450</td>
      <td>12</td>
      <td>13</td>
      <td>11.9.372.2066</td>
      <td>ARAM</td>
      <td>MATCHED_GAME</td>
      <td>[{'teamId': 100, 'win': 'Fail', 'firstBlood': ...</td>
      <td>[{'participantId': 1, 'teamId': 100, 'champion...</td>
      <td>[{'participantId': 1, 'player': {'platformId':...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>0</th>
      <td>5159059636</td>
      <td>KR</td>
      <td>1619624848006</td>
      <td>1154</td>
      <td>450</td>
      <td>12</td>
      <td>13</td>
      <td>11.9.372.2066</td>
      <td>ARAM</td>
      <td>MATCHED_GAME</td>
      <td>[{'teamId': 100, 'win': 'Fail', 'firstBlood': ...</td>
      <td>[{'participantId': 1, 'teamId': 100, 'champion...</td>
      <td>[{'participantId': 1, 'player': {'platformId':...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>0</th>
      <td>5159115216</td>
      <td>KR</td>
      <td>1619623349856</td>
      <td>1385</td>
      <td>450</td>
      <td>12</td>
      <td>13</td>
      <td>11.9.372.2066</td>
      <td>ARAM</td>
      <td>MATCHED_GAME</td>
      <td>[{'teamId': 100, 'win': 'Win', 'firstBlood': T...</td>
      <td>[{'participantId': 1, 'teamId': 100, 'champion...</td>
      <td>[{'participantId': 1, 'player': {'platformId':...</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



**수집한 게임 데이터를 살펴보면 게임 시간(gameDuration), 팀별 세부스탯(teams), 플레이어 세부스탯(participants) 등의 수치데이터가 있음을 알 수 있습니다.
'teams', 'participants' 컬럼 데이터는 리스트 안에 딕셔너리 구조로 되어 있습니다. 따라서, 딕셔너리 데이터를 컬럼으로 풀어주는 작업을 진행해줍니다.**

**이후 각 데이터를 병합하여 하나의 데이터프레임으로 만들어줍니다.**


```python
teams = list(match_data['teams'])
#team1
team1_df = pd.DataFrame()
for i in range(len(teams)):
    try:
        teams[i][0].pop('bans',None)
        team1 = pd.DataFrame(list(teams[i][0].values()),index = list(teams[i][0].keys())).T
        team1_df = team1_df.append(team1)
    except:
        pass
    
team1_df.index = range(len(team1_df))

#team2
team2_df = pd.DataFrame()
for i in range(len(teams)):
    try:
        teams[i][1].pop('bans',None)
        team2 = pd.DataFrame(list(teams[i][1].values()),index = list(teams[i][1].keys())).T
        team2_df = team2_df.append(team2)
    except:
        pass
    
team2_df.index = range(len(team2_df))
```


```python
gameDuration = pd.DataFrame(list(match_data['gameDuration']))
team_data = pd.concat([team1_df, team2_df], axis=1)
team_data['gameDuration'] = gameDuration
```


```python
team_data
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>teamId</th>
      <th>win</th>
      <th>firstBlood</th>
      <th>firstTower</th>
      <th>firstInhibitor</th>
      <th>firstBaron</th>
      <th>firstDragon</th>
      <th>firstRiftHerald</th>
      <th>towerKills</th>
      <th>inhibitorKills</th>
      <th>...</th>
      <th>firstDragon</th>
      <th>firstRiftHerald</th>
      <th>towerKills</th>
      <th>inhibitorKills</th>
      <th>baronKills</th>
      <th>dragonKills</th>
      <th>vilemawKills</th>
      <th>riftHeraldKills</th>
      <th>dominionVictoryScore</th>
      <th>gameDuration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100</td>
      <td>Fail</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1123.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100</td>
      <td>Win</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>4</td>
      <td>1</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1211.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100</td>
      <td>Fail</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1084.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100</td>
      <td>Fail</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1154.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100</td>
      <td>Win</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>4</td>
      <td>2</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1385.0</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>293</th>
      <td>100</td>
      <td>Fail</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>True</td>
      <td>True</td>
      <td>11</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1420.0</td>
    </tr>
    <tr>
      <th>294</th>
      <td>100</td>
      <td>Win</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>7</td>
      <td>1</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1778.0</td>
    </tr>
    <tr>
      <th>295</th>
      <td>100</td>
      <td>Fail</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>True</td>
      <td>True</td>
      <td>11</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1855.0</td>
    </tr>
    <tr>
      <th>296</th>
      <td>100</td>
      <td>Fail</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>True</td>
      <td>True</td>
      <td>11</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>297</th>
      <td>100</td>
      <td>Win</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>9</td>
      <td>1</td>
      <td>...</td>
      <td>True</td>
      <td>True</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>298 rows × 31 columns</p>
</div>



Riot API를 활용하여 데이터 분석에 필요한 최종 게임 데이터 수집을 완료하였습니다. 롤에 대한 애정이 남다른만큼, 데이터를 수집하고 확인하는 시간이 마치 게임을 하는 것만 같은 느낌이었습니다.또한, 예상했던 것보다 라이엇에서 다양한 게임 데이터를 제공하는 것을 알 수 있었습니다. 

이를 활용해 자신 뿐만 아니라 프로게이머, 상위랭커들의 데이터를 분석하고 유용한 인사이트를 발굴하면 좋을 것 같다는 생각이 들었습니다. 다음 세션에서는 EDA, 머신러닝 기법을 통해 LOL 게임 데이터가 어떻게 활용될 수 있는지 분석해보겠습니다.
