import nltk
from konlpy.tag import Okt

nltk.download('punkt')
okt = Okt()

def extract_keywords(text, top_n=30):
    """
    명사 기반 키워드 추출
    """
    words = okt.nouns(text)
    freq_dist = nltk.FreqDist(words)
    return [word for word, freq in freq_dist.most_common(top_n)]

def split_into_paragraphs(text):
    """
    줄 바꿈 기준으로 문단 분리 + 30자 이하 제외
    """
    paragraphs = text.split('\n\n')  # 🔄 두 줄 개행 기준
    return [p.strip() for p in paragraphs if len(p.strip()) > 30]
