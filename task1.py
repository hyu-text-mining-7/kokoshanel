import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.classify import NaiveBayesClassifier
from sklearn.metrics import classification_report
import re

# 리뷰데이터
yogiyo_data = pd.read_csv('data/RAWDATA.txt', sep='\t')

#================================================================================
# 전체 평점 데이터 분포
#================================================================================
rates = yogiyo_data.total.tolist()
rates = pd.DataFrame(rates).value_counts()
agg_rates_df = rates.reset_index(name='count').sort_values(by=0).set_index(0)
agg_rates = agg_rates_df.T.values[0]
props = [f"{x:.1f}%" for x in agg_rates/agg_rates.sum()*100]

plt.figure(figsize=(15, 6))
ax = plt.gca()
agg_rates_df.plot.bar(ax=ax)
plt.ylim(0, 50000)

for idx, prob in enumerate(props) :
    plt.text(x=idx-0.1, y=agg_rates_df.iloc[idx,0] + 700, s=prob, fontsize=20)
plt.show()

f, ax = plt.subplots(1, 4, figsize=(20,5))
sns.scatterplot(x=yogiyo_agg_df.cnt, y=yogiyo_agg_df.review_len, ax=ax[0])
sns.scatterplot(x=yogiyo_agg_df.cnt, y=yogiyo_agg_df.rate_avg, ax=ax[1])
sns.scatterplot(x=yogiyo_agg_df.cnt, y=yogiyo_agg_df.rate_std, ax=ax[2])
sns.scatterplot(x=yogiyo_agg_df.rate_avg, y=yogiyo_agg_df.rate_std, ax=ax[3])

#================================================================================
# 요기요 리뷰수 (문장길이, 평점평균, 평점표준편차) 산포도 분석을 통한 어뷰징 가능성 추정
#================================================================================
yogiyo_data_copy = yogiyo_data.copy()
yogiyo_data_copy['review_len'] = yogiyo_data.document.apply(len)

yogiyo_agg_df = yogiyo_data_copy.groupby('id').agg(
    cnt=('id', np.size),
    review_len=('review_len', np.mean),
    rate_avg=('total', np.mean),
    rate_std=('total', np.std),
)



#================================================================================
# @ 요기요 리뷰 텍스트 감성분석
#  - 한글 LEXICON 감성 사전 데이터로 모델 훈련
#================================================================================
df_dic = pd.read_csv("data/lexicon/polarity.csv", encoding='utf-8')
df = df_dic[df_dic['max.value'].notnull()]
df = df[['ngram', 'max.value']]

# 한글과 영문이 섞여있는 ngram에서, 가장 앞에 있는 한글단어만 추출하는 정규표현식
p = r'^[가-힣]+'

# KOSAC으로 부터 긍정(POS), 부정(NEG), 중립(NEU)의 사전을 생성
pos_dic = []
neg_dic = []
neu_dic = []

for i, row in df.iterrows():
    if row['max.value'] ==  'POS':
        pos_dic.extend(re.findall(p, row['ngram']))
    elif row['max.value'] ==  'NEG':
        neg_dic.extend(re.findall(p, row['ngram']))
    elif row['max.value'] ==  'NEUT':
        neu_dic.extend(re.findall(p, row['ngram']))

# 중복 제거
positive_vocab = list(set(pos_dic)) #총 1830개 단어
negative_vocab = list(set(neg_dic)) #총 1623개 단어
neutral_vocab = list(set(neu_dic)) #총 340개 단어

def word_feats(words):
    return dict((word, True) for word in words)

# 사전의 긍정, 부정, 중립단어를 navie bayes에 학습시킬 준비를 한다
positive_features = [(word_feats(pos), 'pos') for pos in positive_vocab]
negative_features = [(word_feats(neg), 'neg') for neg in negative_vocab]
neutral_features = [(word_feats(neu), 'neu') for neu in neutral_vocab]

# 트레인 데이터셋 생성 완료! naive bayes에 학습 시킨다
train_set = negative_features + positive_features + neutral_features
classifier = NaiveBayesClassifier.train(train_set)


#================================================================================
# 감성모델을 통한 긍부정 3등급제 평가 예측 및 검증
#================================================================================
mapping = {'neg':0, 'neu': 1, 'pos': 2}
def classify(df) :
    sentence = df['document']
    pred = classifier.classify(word_feats(sentence))
    return mapping.get(pred)

yogiyo_data['total_3grade'] = yogiyo_data['total'].apply(lambda x : 2 if x >= 4 else 0 if x < 3 else 1)
yogiyo_data['total_pred_3grade'] = yogiyo_data.apply(classify, axis=1)

print(classification_report(yogiyo_data.total_3grade, yogiyo_data.total_pred_3grade))

#yogiyo_data.loc[yogiyo_data['total_3grade'] == 1, ['document', 'total_3grade', 'total_pred_3grade']].head(30)


#================================================================================
# 감성모델을 통한 긍부정 2등급제 평가 예측 및 검증
#================================================================================
mapping = {'neg':0, 'neu': 0, 'pos': 1}
def classify(df) :
    sentence = df['document']
    if re.findall(r'ㅠ|ㅜ', sentence) :
        pred = 'neg'
    else :
        pred = classifier.classify(word_feats(sentence))
    return mapping.get(pred)

yogiyo_data['total_pred_2grade'] = yogiyo_data.apply(classify, axis=1)
yogiyo_data['total_2grade'] = yogiyo_data['total'].apply(lambda x : 1 if x >= 4 else 0)

print(classification_report(yogiyo_data.total_2grade, yogiyo_data.total_pred_2grade))

# 거짓긍정문장 확인
#yogiyo_data.loc[(yogiyo_data['total_2grade'] == 0) & (yogiyo_data['total_pred_2grade'] == 1), ['document', 'total_2grade', 'total_pred_2grade']].head(50)