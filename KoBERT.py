from kiwipiepy import Kiwi
from keybert import KeyBERT
from transformers import BertModel

kiwi = Kiwi()

def noun_extract(text):
    nouns = []
    token = kiwi.analyze(text)
    for word, tag, start, _ in token[0][0]:
        if (tag.startswith('N') or tag.startswith('SL')):
            nouns.append(word)
    return nouns

text = """
6월 은행 대출 연체율 0.35%로 하락…분기말에 연체 채권 정리 증가
"""

#kiwi형태소 분석기 + keyword 모델
model = BertModel.from_pretrained('skt/kobert-base-v1')
kw_model = KeyBERT(model)

nouns = noun_extract(text)
nouns_text = ' '.join(nouns)
keywords = kw_model.extract_keywords(nouns_text, keyphrase_ngram_range=(1, 2), stop_words=None, top_n=20)
keywords