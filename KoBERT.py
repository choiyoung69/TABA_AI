######Ver.1
#Bert Model : KoBert Model
#Token : Kiwi 형태소 분석기
#키워드 추출: KeyBert Model

from kiwipiepy import Kiwi
from keybert import KeyBERT
from transformers import BertModel

#형태소 분석
def noun_extract(text):
    nouns = []
    token = kiwi.analyze(text)
    for word, tag, start, _ in token[0][0]:
        if (tag.startswith('N') or tag.startswith('SL')):
            nouns.append(word)
    nouns_text = ' '.join(nouns)
    return nouns_text

#INPUT - 뉴스 title
text = """
6월 은행 대출 연체율 0.35%로 하락…분기말에 연체 채권 정리 증가
"""

#형태소 분석
kiwi = Kiwi()
nouns_text = noun_extract(text)

#키워드 추출
model = BertModel.from_pretrained('skt/kobert-base-v1')
KB_model = KeyBERT(model)
keywords = KB_model.extract_keywords(nouns_text, keyphrase_ngram_range=(1, 2), stop_words=None, top_n=20)
keywords