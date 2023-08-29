#AWS DB 연동 및 Input 데이터 가공
import pymysql
import pandas as pd
from konlpy.tag import Okt
from transformers import BertModel
from keybert import KeyBERT
from sqlalchemy import create_engine
import itertools
import kss
from konlpy.tag import Okt
from textrank import KeysentenceSummarizer
from konlpy.tag import Komoran
from tqdm import tqdm
import time
import schedule

#AWS DB와 연동 및 dataframe화하기
def DB_connect() :
    #mySQL과 connect
    conn = pymysql.connect(host='tissue-app-backend-database.cyzh2s69rj9f.us-east-2.rds.amazonaws.com',
                       user = 'tissue',
                       password='tissue1234',
                       db='tissue_db')
    
    #NEWS_KEYWORDS이 NULL일 떄 오류 handling
    try:
        news_df = pd.read_sql("SELECT NEWS_ARTICLES.ID, NEWS_ARTICLES.TITLE, NEWS_ARTICLES.CONTENT \
                          FROM NEWS_ARTICLES, NEWS_KEYWORDS \
                          LEFT JOIN NEWS_KEYWORDS ON NEWS_ARTICLES.ID = NEWS_KEYWORDS.ID \
                          WHERE (NEWS_ARTICLES.ID IS NULL) and (NEWS_ARTICLES.DATE >= '2023-08-01');", conn)
    except pd.errors.DatabaseError : 
        news_df = pd.read_sql("SELECT ID, TITLE, CONTENT \
                          FROM NEWS_ARTICLES \
                         WHERE (NEWS_ARTICLES.DATE >= '2023-08-01');", conn)
    return news_df

#형태소 분석 함수_keywords 추출 함수에서 사용
def morphological_analysis(title):
    nouns = []
    okt = Okt()
    nouns = okt.nouns(title)
    nouns_text = ' '.join(nouns)
    return nouns_text

#keywords에 맞는 Dataframe 만들기
def keywords_model():
    news_df = DB_connect()
    news_df = news_df.drop(columns=['CONTENT'])
    for i in range(len(news_df)) :
        news_df.loc[i, 'TITLE'] = morphological_analysis(news_df.loc[i, 'TITLE'])
    return news_df

#형태소 분석 함수_summary함수에서 사용
komoran = Komoran()    
def komoran_tokenizer(sent):
    words = komoran.pos(sent, join=True)
    words = [w for w in words if ('/NN' in w or '/XR' in w or '/VA' in w or '/VV' in w)]
    return words

#summary model에 맞는 Dataframe 만들기
def summary_model() :
    news_df = DB_connect()
    news_df = news_df.drop(columns=['TITLE'])
    return news_df

#요약 함수
def getsummarize(txt):
    sents = kss.split_sentences(txt)
    summarizer = KeysentenceSummarizer(
        tokenize = komoran_tokenizer,
        min_sim = 0.5,
        verbose = True
        )
    keysents = summarizer.summarize(sents, topk=1)
    keysents.sort(key = lambda x : x[0])
    sentences = list(itertools.chain(*keysents))[2::3]
    return ' '.join(sentences) 

def job():
    #dataframe 불러오기
    data = keywords_model()
    sdata = summary_model()

    #KeyBert를 활용한 Keywords 추출
    model = BertModel.from_pretrained('skt/kobert-base-v1')
    kw_model = KeyBERT(model)

    new_keyword = pd.DataFrame(columns = ['ID', 'KEYWORD'])
    count = 0
    for i in tqdm(range(len(data))):
        time.sleep(0.1)
        keyword = kw_model.extract_keywords(data.loc[i, 'TITLE'], keyphrase_ngram_range=(1, 2), stop_words=None, top_n=10)
        for item in keyword:
            word = item[0]
            ID = data.loc[i, 'ID']
            new_keyword.loc[count] = [ID, word]
            count += 1

    #summary 추출
    new_summary= pd.DataFrame(columns = ['ID', 'SUMMARY'])

    for i in tqdm(range(len(sdata))):
        time.sleep(0.1)
        summary = getsummarize(sdata.loc[i, 'CONTENT'])
        if summary.find('\n'):
            index = summary.find('\n')
            summary = summary[:index]
        new_summary.loc[sdata.loc[i, 'ID']] = [sdata.loc[i, 'ID'], summary]

        #DB에 PUT
    db_connect_str = 'mysql+pymysql://tissue:tissue1234@tissue-app-backend-database.cyzh2s69rj9f.us-east-2.rds.amazonaws.com/tissue_db'
    db_connection = create_engine(db_connect_str)
    new_keyword.to_sql('NEWS_KEYWORDS', con=db_connection, if_exists='append', index=False)
    new_summary.to_sql('NEWS_SUMMARIES', con=db_connection, if_exists='replacee', index=False)

schedule.every().hour.do(job)

while True:
    schedule.run_pending()
    time.sleep(1)