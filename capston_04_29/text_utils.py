# 키워드 추출, 문단 분리 함수

import nltk
from konlpy.tag import Okt

# nltk 초기화
nltk.download('punkt')
okt = Okt()

def extract_keywords(text, top_n=30):
    words = okt.nouns(text)  # 명사만 추출
    freq_dist = nltk.FreqDist(words)
    keywords = [word for word, freq in freq_dist.most_common(top_n)]


    return keywords


#줄 바꿈을 기준으로 나누고 30자 이후 문장은 제외해서 의미 있는 문단만 추리려고 함
def split_into_paragraphs(text):
    paragraphs = text.split('\n')
    paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 30]
    return paragraphs
