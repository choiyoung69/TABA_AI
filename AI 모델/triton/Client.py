#AWS DB 연동 및 Input 데이터 가공
import pymysql
import pandas as pd
from konlpy.tag import Okt
import json
import requests

#AWS DB와 연동 및 dataframe화하기
def DB_connect() :
    #mySQL과 connect
    conn = pymysql.connect(host='',
                       user = '',
                       password='',
                       db='',
                       charset='')
    
    news_df = pd.read_sql("SELECT ID, TITLE, CONTENT FROM NEWS_ARTICLES;",
                          conn,
                          index = 'index')
    return news_df

def dummy() :
    news_df = pd.DataFrame({'ID' : [1, 2, 3],
                            'TITLE' : ["90개 지역행사와 연계한 ‘황금녘 동행 축제’, 30일부터 한 달간 열린다",
                                       "한자리 모인 벤처인들…'벤처썸머포럼' 전주서 열려",
                                       "방치된 동부화물터미널, 물류·여가·주거 복합공간으로 거듭난다"]})
    return news_df

#형태소 분석 함수
def morphological_analysis(title):
    nouns = []
    okt = Okt()
    nouns = okt.nouns(title)
    nouns_text = ' '.join(nouns)
    return nouns_text

#keywords에 맞는 Dataframe 만들기
def keywords_model() :
    news_df = dummy()
    for i in range(len(news_df)) :
        news_df.loc[i, 'TITLE'] = morphological_analysis(news_df.loc[i, 'TITLE'])
    return news_df

#input 데이터 형식 만들기
def input_data() :
    news_df = keywords_model()
    input_data = []
    for i in range(len(news_df)):
        input_item = {
            "name" : f"input_name_{i}",
            "shape" : [1],
            "datatype" : "STRING",
            "data" : [str(news_df.loc[i, 'TITLE'])]
        }
    input_data.append(input_item)

    data = {
        "input" : input_data
    }
    return data

#client에서 서버 연결
def triton_server_client(data):
    url = "http://127.0.0.1:8000"
    response = requests.post(url, json=input_data)

    if response.status_code == 200:
        result = response.json()
        print("Triton server 응답: ", result)
    else :
        print("triton server 요청 실패:" , response.status_code, response.txt)

data = input_data()
triton_server_client(data)