import pprint 
import re
import sys
import pandas as pd
import numpy  as np
from konlpy.tag import Okt

# ----- 추가 라이브러리 -----
sys = sys.path.append("lib/KnuSentiLex") #   KNU(케이앤유) 한국어 감성사전 추가를 위해 소스코드 경로 추가
import knusl #  KNU(케이앤유) 한국어 감성사전

# ----- 필요한 파일데이터를 읽어옵니다. -----
# 리뷰데이터
yogiyo_data = pd.read_csv("data/RAWDATA.txt", sep="\t")
# 테스트용으로 길이를 줄임
yogiyo_data = yogiyo_data

# ----- 데이터 전처리를 위한 도구를 준비합니다. -----
# 1. 불용어 사전을 만들고 불용어 제거 함수 만들기
stopword_list = []
with open("data/stopwords.txt", encoding="utf-8") as stopwords_file:
    stopword_list = stopwords_file.read().split("\n") # 불용어 사전 데이터 불러오기

def remove_stopwords(text_list):
    temp = [item for item in text_list if item not in stopword_list]
    if type(text_list) == str:
        return "".join(temp)
    else:
        return temp

# 2. 정규식으로 한글데이터만 추출하는 함수 만들기
def get_korean_words(text):
    hangul = re.compile("[^ ㄱ-ㅣ 가-힣]") # 추출규칙은 띄어쓰기(1 개)를 포함한 한글!!
    result = hangul.sub("", text) # hangul에서 정의한 패턴규칙을 입력받은 text 파라메터에 적용
    return result

print("* 불용어사전(샘플출력) : ")
print("\t",stopword_list[10],"\n")

print("* 불용어 제거 함수 (샘플출력) : ")
print("\t처리 전: ", ["의해", "강아지"])
print("\t처리 후: ", remove_stopwords(["의해", "강아지"]), "\n")
print("* 리뷰데이터 수 : ", len(yogiyo_data))
print("* 리뷰데이터(샘플출력) : ")
print("\t", yogiyo_data["document"][0], "\n")

sample = yogiyo_data["document"] # 10개만 테스트로 수행

okt = Okt()
sample_rating = []
l1 = len(yogiyo_data)
l2 = 0
for doc in sample:
    ddoc = get_korean_words(doc)
    sum = 0
    count = 0
    for token in okt.pos(ddoc, norm=True, stem=True): # 정규화(NORM), 근어(STEM)
        word = token[0] # 단어를 감성사전에 검색
        point_str = knusl.KnuSL.data_list(word)[1] # 어근 말고 극성만 취득
        if(point_str != "None"): # 극성을 알 수 없는 경우는 제외
            point = int(point_str) # 취득한 극성은 숫자로 변환
            sum += point # 극성의 평균을 구하기 위해 우선 더함
            count += 1
        
    total = 0
    if(sum>0):
        total = round(sum/count)
    
    # -2 ~ 2 척도를 별점 1 ~ 5점 척도로 전환
    rating = 0
    if total == -2:
       rating = 1
    elif total == -1:
       rating = 2
    elif total == 0:
       rating = 3
    elif total == 1:
       rating = 4
    elif total == 2:
       rating = 5

    sample_rating.append(total)
    l2 += 1
    print("[ ", l2, "/", l1, " ] 태깅완료")

# 감성사전으로 통계 낸 점수 추가
yogiyo_data["sample_rating"] = sample_rating


yogiyo_data.to_csv("task2_result.csv")
yogiyo_data.to_pickle("task2_result.pkl")

x0 = 0 # 없음
x1 = 0 # 1점 차이
x2 = 0 # 2점 차이
x3 = 0 # 3점 차이
x4 = 0 # 4점 차이
# 오차 측정
for index, row in yogiyo_data.iterrows():
    m = abs(row['sample_rating']-row['total'])
    if(m == 4):
        x4 += 1
    elif(m == 3):
        x3 += 1
    elif(m == 2):
        x2 += 1
    elif(m == 1):
        x1 += 1
    else:
        x0 += 1

print("일치 : ", x0)
print("1점차이 : ", x1)
print("2점차이 : ", x2)
print("3점차이 : ", x3)
print("4점차이 : ", x4)
