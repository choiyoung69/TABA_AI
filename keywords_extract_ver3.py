######Ver.3
#Bert model : KoBert Model
#Token : Konlpy 형태소 분석기
#키워드 추출: KeyBert Model

from konlpy.tag import Okt
from keybert import KeyBERT
from transformers import BertModel

#형태소 분석
def noun_extract(text):
    nouns = []
    okt = Okt()
    nouns = okt.nouns(text)
    nouns_text = ' '.join(nouns)
    return nouns_text


#INPUT - 뉴스 title
text = """
6월 은행 대출 연체율 0.35%로 하락…분기말에 연체 채권 정리 증가
"""
nouns_text = noun_extract(text)

#KeyBert를 활용한 Keywords 추출
model = BertModel.from_pretrained('skt/kobert-base-v1')
kw_model = KeyBERT(model)
keywords = kw_model.extract_keywords(nouns_text, keyphrase_ngram_range=(1, 2), stop_words=None, top_n=len(nouns))
keywords