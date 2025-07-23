#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

# 한글 폰트 설정 (matplotlib 시각화용)
matplotlib.rc('font', family='Malgun Gothic')

# 데이터 불러오기
df = pd.read_csv(r'C:\Users\백나무\Desktop\airbnb_project\2025_Airbnb_NYC_listings.csv')

# 1. 리뷰 유무
df['has_reviews'] = df['number_of_reviews'] > 0  # 리뷰 있는 숙소 True

# 2. reviews_per_month 결측치 처리
# 리뷰가 없으면 한 달 리뷰 수도 없으므로 → 0으로 대체
df['reviews_per_month'].fillna(0, inplace=True)

# 3. 평점 결측치 처리 (review_scores_rating, review_scores_cleanliness, review_scores_communication)
# 리뷰 없으면 평점도 없음 → -1로 대체 + 결측 플래그 생성

# review_scores_rating
df['review_scores_rating_missing'] = df['review_scores_rating'].isnull().astype(int)
df['review_scores_rating'].fillna(-1, inplace=True)

# review_scores_cleanliness
df['review_scores_cleanliness_missing'] = df['review_scores_cleanliness'].isnull().astype(int)
df['review_scores_cleanliness'].fillna(-1, inplace=True)

# review_scores_communication
df['review_scores_communication_missing'] = df['review_scores_communication'].isnull().astype(int)
df['review_scores_communication'].fillna(-1, inplace=True)


# 4. 리뷰 날짜 처리(first_review / last_review ) 날짜형으로 나오긴하는데 그래도 혹시나 해서
df['first_review'] = pd.to_datetime(df['first_review'], errors='coerce')
df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')


# 5. 날짜 기반 파생 변수 생성
# 첫 리뷰~마지막 리뷰까지 활동 기간 (일수)
df['review_active_days'] = (df['last_review'] - df['first_review']).dt.days
# 2025년 3월 3일로부터 마지막 리뷰까지 경과일 계산
df['days_since_last_review'] = (pd.to_datetime('2025-03-03') - df['last_review']).dt.days


# 6. 리뷰 날짜 자체가 전혀 없는 숙소 여부
df['no_review_dates'] = df['first_review'].isnull() & df['last_review'].isnull()
# 두 날짜가 모두 결측 → 리뷰 자체가 아예 없는 숙소
# 이 숙소들은 신규일 수 있고, 평점도 없음 → 모델에 큰 신호가 될 수 있음
# first_review & last_review가 결측인 6798개 숙소는 no_review_dates == True일 것임.

# host_listings_count(호스트의 전체 등록 숙소 수) - 결측치 제거, 로그 변환 후 이상치 제거
df = df.dropna(subset=['host_listings_count'])
df['log_host_listings'] = np.log1p(df['host_listings_count'])

q1 = df['log_host_listings'].quantile(0.25)
q3 = df['log_host_listings'].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

df = df[(df['log_host_listings'] >= lower) & (df['log_host_listings'] <= upper)]

# host_is_superhost(슈퍼호스트 여부) - 결측치 제거, 이진코딩
df['host_is_superhost'] = df['host_is_superhost'].astype(str).str.strip()
df['host_is_superhost'] = df['host_is_superhost'].replace('nan', np.nan)
df['host_is_superhost'] = df['host_is_superhost'].map({'t': 1, 'f': 0})
df = df.dropna(subset=['host_is_superhost'])

# host_identity_verified(신원인증 여부) - 결측치 제거, 이진코딩
df['host_identity_verified'] = df['host_identity_verified'].astype(str).str.strip()
df['host_identity_verified'] = df['host_identity_verified'].replace('nan', np.nan)
df['host_identity_verified'] = df['host_identity_verified'].map({'t': 1, 'f': 0})
df = df.dropna(subset=['host_identity_verified'])

### host_response_rate & host_acceptance_rate - 문자열로 변환 후 '%' 제거 -> float -> 결측치 채우기
df['host_response_rate'] = df['host_response_rate'].astype(str).str.rstrip('%').replace('nan', np.nan).astype(float)
df['host_acceptance_rate'] = df['host_acceptance_rate'].astype(str).str.rstrip('%').replace('nan', np.nan).astype(float)

df['host_response_rate'].fillna(df['host_response_rate'].median(), inplace=True)
df['host_acceptance_rate'].fillna(df['host_acceptance_rate'].median(), inplace=True)


# host_response_time(호스트가 메시지에 응답하는 데 걸리는 평균 시간) - 원핫인코딩 
df['host_response_time'] = df['host_response_time'].fillna('missing')
df = pd.get_dummies(df, columns=['host_response_time'], prefix='response_time')

# room_type (Entire home/apt, Private room, Shared room) - 원-핫 인코딩
# 기존 room_type 컬럼 유지하면서 더미컬럼을 df에 추가
# prefix='room_type'은 함수에서 생성되는 더미변수 이름 앞에 붙일 접두사 room_type_privae room이렇게 앞에 붙음
# 랜덤포레스트회귀는 다중공선성 문제 없으므로 drop_first=False해서 room_type의 모든 범주가 더미변수로 생성되게 하기

dummies = pd.get_dummies(df['room_type'], prefix='room_type', drop_first=False)
df = pd.concat([df, dummies], axis=1)

# bathrooms(욕실 수)
#값이 10.5인 이상치 데이터의 인덱스 확인해서 index_to_drop에 담기
index_to_drop = df[df['bathrooms'] == 10.5].index
#인덱스를 기준으로 행 제거하기, 원본에 바로 반영
df.drop(index=index_to_drop, inplace=True)

#ast.literal_eval() 문자열로 된 리시트, 딕셔너리, 숫자, 튜플 등을 실제 파이썬 객체로 안전하게 변환하고 싶을 때 사용
#ast : abstract syntax tree 모듈 -> 파이썬 코드 자체를 문법구조로 분석하는 내장 모듈
#literal_eval : 문자 그대로 평가하는 뜻
#문법을 보고 문자 그대로 평가해서 객체로 바꿔
#apply 각 행에 대해 람다 함수를 적용해
import ast

df['amenities_count'] = df['amenities'].apply(lambda x : len(ast.literal_eval(x))) 
# 즉 '[]'로 된 문자열 자체를 평가해서 리스트로 변환하고 리스트의 요소를 세는 함수를 각 행에 적용해서 반환

#25년도 7월 기준 에어비앤비 홈페이지 뉴욕시 어메니티 인기 키워드 6개
popular_amenities = [
    'TV',
    'Air conditioning',
    'Wifi',
    'Washer',
    'Free parking',
    'Dryer'
]
#인기 키워드 6종을 모두 포함한 숙소를 1, 아닌 숙소를 0으로 분류
def contains_all_popular_amenity(amenities_str):
    return int(all(keyword in amenities_str for keyword in popular_amenities))

df['has_all_popular_amenity'] = df['amenities'].apply(contains_all_popular_amenity)

# neighbourhood_group_cleansed 원 핫 인코딩 
# encoding = pd.get_dummies(df, columns = ['neighbourhood_group_cleansed'], drop_first = True)
df = pd.get_dummies(df, columns = ['neighbourhood_group_cleansed'])
df[df.select_dtypes(include = 'bool').columns] = df.select_dtypes(include = 'bool').astype(int)

# instant_bookable (즉시 예약 가능 여부)
df['instant_bookable'] = (df['instant_bookable'] == 't').astype(int)

# maximum_nights
Q1 = df['maximum_nights'].quantile(0.25)
Q3 = df['maximum_nights'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df['maximum_nights'] >= lower_bound) & (df['maximum_nights'] <= upper_bound)]
df = df[df['minimum_nights'] < 365]

# Price 전처리: $, , 제거 → float형으로 변환
df['price'] = df['price'].replace('[$,]', '', regex=True).astype(float)
# 로그 변환
df['log_price'] = np.log1p(df['price'])
# IQR 기반 이상치 제거 (log_price 기준)
Q1 = df['log_price'].quantile(0.25)
Q3 = df['log_price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 이상치 제거된 데이터 추출
df = df[(df['log_price'] >= lower_bound) & (df['log_price'] <= upper_bound)]


# In[4]:


df.shape


# In[5]:


# 규제 시행일
regulation_date = pd.to_datetime('2023-09-05')
# 규제 이후 생존 컬럼 생성
df['is_survived'] = df['last_review'] >= regulation_date
# 규제일 이후에 라스트 리뷰가 있으면 생존, 그렇지 않으면 중단으로 마킹
df['survival_label'] = df['is_survived'].map({True: '생존', False: '중단'})
# 처음과 마지막 리뷰 날짜가 모두 없는데 중단이라고 마킹 된 것 변수 mask에 담음
mask = (df['first_review'].isna()) & (df['last_review'].isna()) & (df['survival_label'] == '중단')
# 처음, 마지막 리뷰 날짜가 모두 없는 것은 보류라고 마킹
# 이유 : 가급적 규제의 영향으로 중단된 숙소를 필터링 하고 싶은데, 리뷰 자체가 없었던 데이터는 그것을 알아내기 어려워서
df.loc[mask, 'survival_label'] = '보류'


# In[6]:


print(f" 생존/보류/중단 수 : {df['survival_label'].value_counts()}\n")
print(f" 생존/보류/중단 비율 : {df['survival_label'].value_counts(normalize=True)}")


# In[7]:


df = df[df['survival_label'] != '보류'].copy()
print(f"생존/중단 수: \n{df['survival_label'].value_counts()}\n")
print(f"생존/중단 비율: \n{df['survival_label'].value_counts(normalize=True)}")


# 1. 설명변수에 변수 모두 다 넣어서 모델돌리기

# In[8]:


# # 첫번째로 피처임포턴스 모두 다 돌리기
features1 = [
    # 숫자형
    'host_id','host_response_rate', 'host_acceptance_rate','host_listings_count','log_host_listings',
    'accommodates', 'bathrooms', 'bedrooms', 'beds','amenities_count',
    'review_scores_rating', 'review_scores_cleanliness', 'review_scores_communication',
    'reviews_per_month', 'review_active_days', 'days_since_last_review',

    # 이진 변수 및 파생 변수
    'host_is_superhost','host_identity_verified',
    'has_reviews', 'no_review_dates',
    'review_scores_rating_missing', 'review_scores_cleanliness_missing', 'review_scores_communication_missing',
    'availability_365',

    # host_response_time dummy
    'response_time_within an hour', 'response_time_within a few hours',
    'response_time_within a day', 'response_time_a few days or more',
    'response_time_missing',

    # 기타 이진 변수
    'has_all_popular_amenity', 'instant_bookable',

    # 지역 dummy
    'neighbourhood_group_cleansed_Bronx','neighbourhood_group_cleansed_Brooklyn',
    'neighbourhood_group_cleansed_Manhattan','neighbourhood_group_cleansed_Queens','neighbourhood_group_cleansed_Staten Island',

    # 방 종류 dummy
    'room_type_Entire home/apt', 'room_type_Private room', 'room_type_Shared room',
]


# In[9]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. 데이터 로드
X = df[features1]
y = df['survival_label']

# 2. 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. 랜덤 포레스트 모델 생성
# n_estimators는 사용할 트리의 개수, max_depth는 각 트리의 최대 깊이를 의미하며
# 위 2개의 값을 높일 수록 시간과 연산량은 늘어나지만 더욱 복잡한 특징을 잡을 수 있음
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42
)

# 4. 모델 학습
rf_model.fit(X_train, y_train)

# 5. 예측
y_pred = rf_model.predict(X_test)

# 6. 성능 평가
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {acc:.4f}")
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", report)


# 7. 피처 중요도 추출
importances = pd.Series(rf_model.feature_importances_, index=X.columns)

# 8. 내림차순 정렬 후 상위 20개 시각화
top_n = 20
top_features = importances.sort_values(ascending=False).head(top_n)

plt.figure(figsize=(10, 6))
top_features.sort_values().plot(kind='barh', color='teal')
plt.title(f'랜덤 포레스트 - 상위 {top_n} 중요 변수')
plt.xlabel('Feature Importance')
plt.tight_layout()
plt.show()


# 내림차순 정렬된 리스트로 변환
top_features
feature_list = top_features.index.tolist() #시리즈-> 리스트

# 출력 예시
print("🎯 중요도 순 피처 리스트:")
for i, col in enumerate(feature_list, 1): #피처리스트 항목들을 1부터 번호 매김
        print(f"{i:>2}. {col} ({top_features[col]:.4f})") #i라는 숫자를 오른쪽으로 정렬(>)해서 전체자리수는 2칸으로 출력


# In[10]:


top_features.sort_values()


# 2. 상위 10개만 추려서 랜덤포레스트분류에 돌리기 -> 생존:중단이 8:2로 심한 클래스 불균형이라고함... 
# 클래스 불균형 대응 -> 클래스별 샘플 수에 반비례하는 가중치를 자동으로 부여
# 중단 클래스에 더 많은 비중을 줘서 재현율을 높이는 효과가 기대됨. 하지만 효과가 미미했음.

# In[11]:


# 설명변수 줄이기
# 전체 변수를 돌렸을 때 중요도 탑 10중 last_reivew관련된 변수만 제거
#  1. days_since_last_review (0.6704)  -> 제거(날짜와 관련된 변수니까)
#  2. reviews_per_month (0.1061) -> 유지(한달의 평균 리뷰수)
#  3. response_time_missing (0.0440) -> 제거(호스트 응답 기록이 없는 데이터)   ---> 지우기
#  4. host_acceptance_rate (0.0360) -> 유지(호스트 응답률)
#  5. response_time_within an hour (0.0201) -> 유지 (한시간 이내 응답)
#  6. review_active_days (0.0163) -> 유지(첫 리뷰~마지막 리뷰까지 활동 기간)
#  7. availability_365 (0.0155) -> 유지(1년 중 예약가능일수)
#  8. host_id (0.0118) -> 유지(호스트 id) ---> 지우기(호스트 아이디는 고유값이라 생존/중단의 특성을 알알내기 어려워서 )
#  9. amenities_count (0.0110) -> 유지(어메니티 개수)
# 10. host_is_superhost (0.0089) -> 유지 (슈퍼호스트 여부)
# 11. review_scores_cleanliness (0.0066) -> 유지(청결도)
# 12. review_scores_rating (0.0066) -> 유지(리뷰점수)
# 13 host_response_rate --> 유지 (호스트응답율)
features2 = ['reviews_per_month', 
           'host_acceptance_rate',
           'response_time_within an hour',
           'review_active_days',
           'availability_365',
           'amenities_count',
           'host_is_superhost',
           'review_scores_cleanliness',
           'review_scores_rating',
           'host_response_rate']


# 3 다른 방법 SMOTE
# '중단'(소수 클래스)에 대한 예측 성능이 명확히 개선되었다고 함.
# 재현율 (63 -> 74)
# 정확도는 소폭 감소 90.1 -> 87.4
# f1-score은 유지
# 

# In[12]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# 1. 데이터 로드
X = df[features2]
y = df['survival_label']

# 2. 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. SMOTE 적용 (훈련 데이터에만)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 4. 랜덤 포레스트 모델 생성
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42
)

# 5. 모델 학습
rf_model.fit(X_train_res, y_train_res)

# 6. 예측
y_pred = rf_model.predict(X_test)

# 7. 성능 평가
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {acc:.4f}")
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", report)

# 8. 피처 중요도 추출
importances = pd.Series(rf_model.feature_importances_, index=X.columns)

# 9. 내림차순 정렬 후 상위 20개 시각화
top_n = 20
top_features = importances.sort_values(ascending=False).head(top_n)

plt.figure(figsize=(10, 6))
top_features.sort_values().plot(kind='barh', color='teal')
plt.title(f"생존/중단에 영향을 미치는 피처")
plt.xlabel('Feature Importance')
plt.tight_layout()
plt.show()

# 10. 중요도 순 피처 출력
sorted_features = importances.sort_values(ascending=False)
feature_list = sorted_features.index.tolist()
print("🎯 중요도 순 피처 리스트:")
for i, col in enumerate(feature_list, 1):
        print(f"{i:>2}. {col} ({sorted_features[col]:.4f})")


# In[13]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred, labels=['생존', '중단'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['생존', '중단'])
disp.plot(cmap='Blues')


# In[14]:


# 생존/중단 분류 모델의 성능 평가지표 해석
'''
- accuracy : 88.05%
전체 데이터중 88.05%를 정확히 분류했다는 의미

- precision(정밀도) 예측한 생존 중 94%가 실제 생존 (예측한 것 중 맞게 예측한 것)

- recall(재현율) 실제 생존 중 92% 정확히 맞춤. (실제값중 맞게 예측한 것)

- F1-score : 정밀도와 재현율의 조화평균

- mcro avg : 단순 평균

- weighted avg : 실제 데이터 비율 반영한 전체 성능 평균
'''




# In[22]:


variable = [
            'reviews_per_month', 
             'host_acceptance_rate',
             'review_active_days',
             #'host_response_rate', #삭제 (사유: 호스트응답률이 1시간 이내 응답여부와 충돌 되는 것 같아서 평균차이도 100점 만점에 5점차이)
            'response_time_within an hour',
            'host_is_superhost']
df.groupby('survival_label')[variable].mean()
# 더 자주 리뷰가 달리는 숙소일수록 생존 가능성 ↑
# 예약 수락률이 높을수록 생존 가능성 ↑
# 응답이 빠를수록 생존에 유리
# 리뷰가 지속적으로 달린 기간이 길수록 생존 가능성 ↑
# 어메니티 개수가 많을수록 생존에 유리
# 슈퍼 호스트일수록 생존에


# In[16]:


df.groupby('survival_label')[features2].mean()
# 더 자주 리뷰가 달리는 숙소일수록 생존 가능성 ↑
# 예약 수락률이 높을수록 생존 가능성 ↑
# 응답이 빠를수록 생존에 유리
# 리뷰가 지속적으로 달린 기간이 길수록 생존 가능성 ↑
# 어메니티 개수가 많을수록 생존에 유리
# 슈퍼 호스트일수록 생존에


# # 통계

# 1. groupby().mean() -> 평균 차이 먼저 눈으로 보기
# 2. t-test 수행 -> 실제로 유의미한 차이인지 검정
# 3. 시각화(boxplot, barplot 등) -> 인사이트 전달력 높이기
# 

# In[23]:


from scipy.stats import ttest_ind

#생존/중단 그룹 나누기
alive = df[df['survival_label'] == '생존']
dead = df[df['survival_label'] == '중단']

#연속형 변수
variable = [
            'reviews_per_month', 
             'host_acceptance_rate',
             'review_active_days',
             #'host_response_rate', #삭제 (사유: 호스트응답률이 1시간 이내 응답여부와 충돌 되는 것 같아서 평균차이도 100점 만점에 5점차이)
            'response_time_within an hour',
            'host_is_superhost']

for feature in variable:
    t_stat, p_val = ttest_ind(alive[feature], dead[feature], nan_policy='omit')
    print(f"{feature}")
    print(f"  생존 평균: {alive[feature].mean():.3f}")
    print(f"  중단 평균: {dead[feature].mean():.3f}")
    print(f"  T-Statistic: {t_stat:.4f}, P-Value: {p_val:.4f}")
    print("-" * 50)


# In[20]:


palette = {'생존': '#2ca02c',  # 초록색
           '중단': '#d62728'}  # 빨간색

import seaborn as sns
import matplotlib.pyplot as plt

# 생존/중단 색상 지정
palette = {'생존': '#2ca02c',  # 초록
           '중단': '#d62728'}  # 빨강

for feature in variable:
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x='survival_label', y=feature, data=df, ci=95, palette=palette)

    # 평균 값 표시
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + (height * 0.01),  # 위치 조정
                f'{height:.2f}', ha="center", va="bottom", fontsize=9)

    plt.title(f"{feature} 평균 비교 (with 95% CI)") #표현된 막대선의 오차선(세로선)은 평균의 "95%" 신뢰구간을 나타낸다. 즉 평균 값은 95%확률로 이 선 사이에 어딘가에 있을 것이다.
    plt.xlabel("생존 여부")
    plt.ylabel(feature)
    plt.tight_layout()
    plt.show()


# In[ ]:


# 규제 이후 숙소들이 생존 vs 중단으로 나뉘었을 때 
# 어떤 특성이 생존에 유리한지 파악하여
# 향후 호스트나 플랫폼 전략 수립에 반영하기
# 
# 1. 월 평균 리뷰수 (reviews_per_month)
# - 생존 평균 1.4
# - 중단 평균 0.3
# - 약 4.7개차이 p-value는 거의 0
# - 해석
#     리뷰가 활발한 숙소일수록 생존확률이 높음
# - 전략 제안
#     - 리뷰 작성 유도 기능 강화 (리뷰 작성 시 포인트 일부 제공)
#     - 리뷰 작성 요청 자동 메시지 전송 기능 생성
# 
# 2. 수략률 (host_acceptance_rate)
# - 생존 84%
# - 중단 68
# - 해석
#     예약 요청을 잘 수락하는 숙소가 생존에 유리
# - 전략제안
#     수락률이 80% 이상인 경우 숙소 우선 노출
# 
# 3. 한 시간 이내 응답률 
# - 생존 61%가 한시간 이내 응답
# - 중단은 21%
# - 해석
#     빠른 응답은 생존과 강하게 연관
# - 전략
#     응답 독려 메시지 "게스트가 응답을 애타게 기다린지 30분/1시간/3시간 되었어요."
# 
# 4. 슈퍼 호스트 비율
# - 생존 그룹 슈퍼 호스트 비율 높음
# - 해석
#     호스트의 신뢰성과 숙소 생존이 연결됨
# - 전략
#     신규 호스트 - 슈퍼호스트를 연결해 일대일 문의응답을 할 수 있는 관계망을 조성한다.
# 
# 5. 운영기간 
# - 오래 운영된 숙소일수록 생존 확률 높음
# - 해석
#     신규 호스트의 장기 운영을 도와주는 시스템 설계 (초창기-중반기-각종 어려움에 닥쳤을 때)
# 
# 
# 생존한 숙소의 공통점은 고객 응대가 빠르고 적극적이며,
# 운영을 오래 했고, 리뷰가 많이 쌓여 있으며, 신뢰도 높은 호스트였다는 점이다.
# 따라서 플랫폼은 리뷰 활성화, 응답 개선, 수락률 향상, 신뢰 구축을 핵심 전략으로 삼아야 한다.
# 

# In[1]:





# In[ ]:




