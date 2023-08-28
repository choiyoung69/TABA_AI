######Ver.2
#Bert Model : KoBert Model
#Token : Kobert
#키워드 추출: KeyBert Model

from keybert import KeyBERT
from transformers import BertModel, BertTokenizer

#Input - 뉴스 title
text = """
6월 은행 대출 연체율 0.35%로 하락…분기말에 연체 채권 정리 증가
"""

#kobert를 통한 token화
tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
token =  tokenizer(text)

#keybert model을 통한 keywords 추출
model = BertModel.from_pretrained('skt/kobert-base-v1')
KB_model = KeyBERT(model)
keywords = KB_model.extract_keywords(token, keyphrase_ngram_range=(1, 2), stop_words=None, top_n=10)
keywords