#요약 모델

import kss
from konlpy.tag import Okt
from textrank import KeysentenceSummarizer
from konlpy.tag import Komoran

komoran = Komoran()    
def komoran_tokenizer(sent):
    words = komoran.pos(sent, join=True)
    words = [w for w in words if ('/NN' in w or '/XR' in w or '/VA' in w or '/VV' in w)]
    return words

def getsummarize(txt):
    sents = kss.split_sentences(txt)
    summarizer = KeysentenceSummarizer(
        tokenize = komoran_tokenizer,
        min_sim = 0.5,
        verbose = True
        )
    keysents = summarizer.summarize(sents, topk=3)
    keysents.sort(key = lambda x : x[0])
    sentences = list(itertools.chain(*keysents))[2::3]
    return ' '.join(sentences)


#INPUT - 뉴스 
text = '''A 장학재단은 공익사업을 이유로 정부로부터 세제 혜택을 받는 공익법인이다.

이 재단은 이사장이 대표인 B 회사로부터 받은 기부금의 예금이자와 부동산 임대 수입 등으로 장학사업을 운영했다. 정관에 '장학금 수혜자의 출생지·출신학교·근무처 등에 의해 공익 수혜의 차별을 두지 않는다'는 조항을 두고 장학사업의 공정성과 투명성을 강조해왔다.

하지만 국세청 분석 결과 이사장이 대표를 맡고 있는 회사와 계열사의 임직원 자녀에게만 장학금을 지급해온 것으로 드러났다. 국세청은 A 재단을 상대로 정밀 검증을 벌여 필요하면 세금 추징, 세무조사 의뢰 등 조처를 할 방침이다.

국세청이 23일 공개한 공익법인의 세법 위반 혐의 사례에는 부당 내부거래, 출연재산의 사적 유용, 변칙 회계처리 등 다양한 꼼수가 포함됐다.

특히 공익법인 자금을 이사장 가족의 생활비 등으로 사적 유용한 법인이 상당수 정밀 검증 대상에 포함됐다.

C 공익법인은 법인 자금을 빼돌려 해외에서 살고 있는 이사장 손녀의 학교 등록금으로 사용했다. 법인 명의의 신용카드로 해외에 거주하는 자녀의 생활비와 항공료를 결제하고, 해외에 사는 자녀와 배우자를 국내 법인에서 일한 것처럼 서류를 꾸며 급여 명목으로 자금을 빼돌린 사실도 확인됐다.

D 공익법인은 기부금을 고가 골프 회원권을 다수 매입해 주무 관청에 '임직원 복리증진용'으로 신고했지만 실제로는 이사장 등 특정인만 사용했다. 이 법인은 결산서류에 골프회원권을 공익법인 재산으로 공시하지도 않았다.

퇴직한 뒤에도 법인카드로 귀금속, 고가 한복, 상품권 등을 구입한 한 전직 이사장도 국세청에 덜미를 잡혔다.

이사장 가족에게 공익법인 명의의 집을 공짜로 빌려주는 등 공익법인 재산을 특수관계자를 위해 부당하게 사용한 사례도 다수 확인됐다.

E 공익법인의 이사장은 출연받은 체육시설을 자녀가 소유한 법인에 시세보다 현저하게 낮은 가격으로 임대했다가 들통이 났다.

F 공익법인은 이사장 일가가 출자한 법인에 건물관리 업무를 모두 위탁한 뒤 고액의 관리 수수료를 지급하는 수법으로 재산을 빼돌린 것으로 분석됐다. G 공익법인은 이사장의 장모가 살고 있는 아파트를 공익법인 자금으로 매입한 뒤 장모로부터 월세나 전세를 받지 않았다.

특수관계자 법인으로부터 돈을 빌린 뒤 시중 금리보다 더 많은 이자를 지급하거나 공익목적으로 출연한 토지에 공익법인 돈으로 사주 일가를 위한 고액의 개인 시설을 지은 사례도 있었다.

국세청 관계자는 "공익법인은 영리법인과 달리 사업구조가 간단해서 세무조사를 하지 않아도 사후 검증으로도 사실관계를 확인할 수 있다"며 "검증만으로 사실관계 확인이 어려울 경우 세무조사를 의뢰할 수 있다"고 말했다.'''
sents = kss.split_sentences(text)
summary = getsummarize(text)
summary