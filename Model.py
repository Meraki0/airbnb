import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# 한글 폰트 설정 (matplotlib 시각화용)
matplotlib.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
df = pd.read_excel(r'C:\Users\rlaxo\Desktop\2025_Airbnb_NYC_listings.xlsx')

############
df['region'] = df['neighbourhood_group_cleansed']
###########

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

df.reset_index(inplace=True)

# 이상치 제거된 데이터 추출
filtered = df[(df['log_price'] >= lower_bound) & (df['log_price'] <= upper_bound)]

filtered = filtered.set_index('id')

# 타겟 변수: log_price
target = 'log_price'

# 사용할 feature 목록 정의
features = [
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
    
    # 'host_response_time' 대신 실제 생성된 더미 컬럼들 추가
    'response_time_within an hour', 'response_time_within a few hours', 
    'response_time_within a day', 'response_time_a few days or more', 
    'response_time_missing',
    'has_all_popular_amenity', 'instant_bookable',

    # neighbourhood_group_cleansed dummy
    'neighbourhood_group_cleansed_Bronx','neighbourhood_group_cleansed_Brooklyn',
    'neighbourhood_group_cleansed_Manhattan','neighbourhood_group_cleansed_Queens','neighbourhood_group_cleansed_Staten Island',

    # room_type dummy
    'room_type_Entire home/apt', 'room_type_Private room', 'room_type_Shared room',
]

# 학습/평가용 데이터셋 분리
X = filtered[features]
y = filtered[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 기본 파라미터로 모델 정의
xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# 모델 학습
xgb.fit(X_train, y_train)

# 예측
y_pred_xgb = xgb.predict(X_test)
# 성능 지표 계산
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f'XGBoost RMSE(로그 스케일): {rmse_xgb:.4f}')
print(f'XGBoost MAE(로그 스케일): {mae_xgb:.4f}')
print(f'XGBoost R²(로그 스케일): {r2_xgb:.4f}')

# 로그 값 → 실제 달러 가격으로 역변환
actual_price = np.expm1(y_test)
predicted_price = np.expm1(y_pred_xgb)

actual_rmse = np.sqrt(np.mean((actual_price - predicted_price) ** 2))
actual_mae = np.mean(np.abs(actual_price - predicted_price))
actual_r2 = r2_score(actual_price, predicted_price)

print(f'실제 RMSE (달러): ${actual_rmse:.2f}')
print(f'실제 MAE (달러): ${actual_mae:.2f}')
print(f'실제 가격 기준 R²: {actual_r2:.4f}')

# 중요도 추출 및 정렬
importance = xgb.feature_importances_
feature_names = X.columns

# 전체 중요도 데이터프레임 정렬
fi_df_xgb = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
fi_df_xgb_sorted = fi_df_xgb.sort_values(by='Importance', ascending=False)
top_20 = fi_df_xgb_sorted.head(20)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=top_20)
plt.title('Top 20 Feature Importances (XGBoost)')
plt.tight_layout()
plt.show()

# Bottom 10 중요 변수 .tail(10)
bottom_20 = fi_df_xgb_sorted.tail(10).sort_values(by='Importance')

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=bottom_20)
plt.title('Bottom 10 Feature Importances (XGBoost)')
plt.tight_layout()
plt.show()

from sklearn.model_selection import RandomizedSearchCV
# 파라미터 그리드
param_dist = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
}

xgb = XGBRegressor(random_state=42)

random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=30,  # 시도할 조합 수 (더 늘리면 더 정밀)
    scoring='neg_root_mean_squared_error',
    cv=3,
    verbose=2,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print("Best Params:", random_search.best_params_)

# 베스트 모델로 예측
best_xgb = random_search.best_estimator_
y_pred_best = best_xgb.predict(X_test)

# 성능 지표 계산
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_best))
mae_xgb = mean_absolute_error(y_test, y_pred_best)
r2_xgb = r2_score(y_test, y_pred_best)

print(f'튜닝된 XGBoost RMSE(로그 스케일): {rmse_xgb:.4f}') 
print(f'튜닝된 XGBoost MAE(로그 스케일): {mae_xgb:.4f}') 
print(f'튜닝된 XGBoost R²(로그 스케일): {r2_xgb:.4f}') 
# XGBoost RMSE(로그 스케일): 0.3769 -> 0.3717
# XGBoost MAE(로그 스케일): 0.2782 -> 0.2728
# XGBoost R²(로그 스케일): 0.6703 ->  0.6795

# 역변환 후 실제 성능 측정
actual_price = np.expm1(y_test)
predicted_price = np.expm1(y_pred_best)

actual_rmse = np.sqrt(np.mean((actual_price - predicted_price) ** 2))
actual_mae = np.mean(np.abs(actual_price - predicted_price))
actual_r2 = r2_score(actual_price, predicted_price)

print(f'튜닝된 XGBoost 실제 RMSE: ${actual_rmse:.2f}')
print(f'튜닝된 XGBoost 실제 MAE: ${actual_mae:.2f}')
print(f'튜닝된 XGBoost 실제 R²: {actual_r2:.4f}')

# 실제 RMSE (달러): $84.83 -> $84.32
# 실제 MAE (달러): $47.92 ->  $47.23
# 실제 가격 기준 R²: 0.5284 -> 0.5340

# 중요도 추출 및 정렬 (튜닝된 best_xgb 모델 기준)
importance_best = best_xgb.feature_importances_
feature_names = X.columns

fi_df_best = pd.DataFrame({'Feature': feature_names, 'Importance': importance_best})
fi_df_best_sorted = fi_df_best.sort_values(by='Importance', ascending=False)

# Top 20 중요 변수 시각화
top_20_best = fi_df_best_sorted.head(20)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=top_20_best)
plt.title('Top 20 Feature Importances (Best XGBoost)')
plt.tight_layout()
plt.show()

# Bottom 10 중요 변수 시각화
bottom_10_best = fi_df_best_sorted.tail(10).sort_values(by='Importance')

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=bottom_10_best)
plt.title('Bottom 10 Feature Importances (Best XGBoost)')
plt.tight_layout()
plt.show()

import shap

# SHAP TreeExplainer 생성 (튜닝된 XGBoost 모델 사용)
explainer = shap.Explainer(best_xgb)

# SHAP 값 계산 (테스트 데이터 기준)
shap_values = explainer(X_test)

# Summary plot - 전체 feature 영향도 시각화 (기여도 + 분포)
shap.summary_plot(shap_values, X_test, max_display=20)

# # 테스트셋 전체 예측
# log_preds = best_xgb.predict(X_test)
# price_preds = np.expm1(log_preds)

# # 결과를 원본과 합쳐서 보기
# result_df = X_test.copy()
# result_df['actual_price'] = np.expm1(y_test)
# result_df['predicted_price'] = price_preds
# result_df['error'] = result_df['predicted_price'] - result_df['actual_price']

# # 상위 10개 확인
# result_df[['actual_price', 'predicted_price', 'error']]

# 전체 데이터셋 예측
log_preds_all = best_xgb.predict(X)
price_preds_all = np.expm1(log_preds_all)

result_all = X.copy()
result_all['actual_price'] = np.expm1(y)  # 전체 정답
result_all['predicted_price'] = price_preds_all
result_all['error'] = result_all['predicted_price'] - result_all['actual_price']


# 위경도 정보 id 기준 병합
location_info = df[['id', 'latitude', 'longitude', 'region']].drop_duplicates().set_index('id')
result_all = result_all.merge(location_info, left_index=True, right_index=True, how='left')

print(result_all.shape)
result_all[['actual_price', 'predicted_price', 'error']]

plt.figure(figsize=(8, 6))
sns.scatterplot(x=result_all['actual_price'], y=result_all['predicted_price'], alpha=0.4)
plt.plot([0, 1000], [0, 1000], color='red', linestyle='--', label='예측선 (y = x)')
plt.xlabel("실제 숙소 가격 ($)")
plt.ylabel("예측된 숙소 가격 ($)")
plt.title("전체 데이터: 실제 vs 예측 숙소 가격")
plt.xlim(0, 1000)
plt.ylim(0, 1000)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(result_all['error'], bins=50, kde=True)
plt.axvline(0, color='red', linestyle='--')
plt.title("전체 데이터: 예측 오차 분포 (예측값 - 실제값)")
plt.xlabel("예측 오차 ($)")
plt.ylabel("빈도")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(result_all['error'], bins=100, kde=True)
plt.axvline(0, color='red', linestyle='--', label='예측 정확선')
plt.xlim(-200, 200)
plt.title("전체 데이터: 예측 오차 분포 (예측값 - 실제값)")
plt.xlabel("예측 오차 ($)")
plt.ylabel("빈도")
plt.legend()
plt.tight_layout()
plt.show()

# 절대 오차 계산
abs_error = np.abs(result_all['error'])

# 오차가 ±50 이내인 숙소 수
count_in_50 = (abs_error <= 50).sum()

# 전체 숙소 수
total_count = len(result_all)

# 비율 계산
in_50_ratio = count_in_50 / total_count

print(f'±$50 이내 예측 숙소 수: {count_in_50}개 / 오차 범위 벗어난 숙소: {total_count-count_in_50}개')
print(f'±$50 이내 예측 비율: {in_50_ratio * 100:.2f}%')

def show_underpriced_on_map(result_df, X_features, margin=5, error_threshold=50, features_to_show=None):
    # 저평가 판단용 특성
    judgment_features = ['accommodates', 'bedrooms', 'bathrooms', 'beds', 'review_scores_rating']
    
    # room_type 및 지역 컬럼
    room_type_cols = [
        'room_type_Entire home/apt', 'room_type_Private room', 'room_type_Shared room'
    ]
    region_cols = [
        'neighbourhood_group_cleansed_Bronx','neighbourhood_group_cleansed_Brooklyn',
        'neighbourhood_group_cleansed_Manhattan','neighbourhood_group_cleansed_Queens',
        'neighbourhood_group_cleansed_Staten Island'
    ]

    # 저평가 숙소 필터링
    underpriced = result_df[
        (result_df['error'] > error_threshold) &
        (result_df['actual_price'] <= result_df['predicted_price'] - error_threshold)
    ].copy()

    if 'latitude' not in underpriced.columns or 'longitude' not in underpriced.columns:
        raise ValueError("위도(latitude)와 경도(longitude) 컬럼이 result_df에 필요합니다.")

    # 지도 초기화
    m = folium.Map(location=[40.75, -73.98], zoom_start=11)
    cluster = MarkerCluster().add_to(m)

    for idx, row in underpriced.iterrows():
        lat, lon = row['latitude'], row['longitude']
        pred_price = row['predicted_price']
        actual_price = row['actual_price']
        region = row['region']

        feat_values = X_features.loc[idx, judgment_features]

        # 예측가 ±margin 실제 숙소 그룹 평균
        similar_group = result_df[
            (result_df['actual_price'] >= pred_price - margin) &
            (result_df['actual_price'] <= pred_price + margin)
        ]
        group_avg = X_features.loc[similar_group.index, judgment_features].mean()

        # 저평가 판단
        diff_ratio = (feat_values - group_avg) / group_avg
        positive_ratio = (diff_ratio > 0).sum() / len(diff_ratio)
        underpriced_flag = "✅ 저평가 가능성 있음" if positive_ratio >= 0.7 else "⚠️ 저평가 근거 부족"

        # 실제 room_type 구하기
        for col in room_type_cols:
            if X_features.loc[idx, col] == 1:
                room_type = col.replace('room_type_', '')
                break
        else:
            room_type = 'Unknown'

        # 지역/room_type 비중 계산
        region_avg = X_features.loc[similar_group.index, region_cols].mean().round(2)
        room_avg = X_features.loc[similar_group.index, room_type_cols].mean().round(2)

        # 팝업 텍스트
        popup_text = (
            f"<b>지역:</b> {region}<br>"
            f"<b>실제 가격:</b> ${actual_price:.2f}<br>"
            f"<b>예측 가격:</b> ${pred_price:.2f}<br>"
            f"<b>오차:</b> ${pred_price - actual_price:.2f}<br>"
            f"<b>판단:</b> {underpriced_flag}<br>"
            f"<hr style='margin:4px 0'>"
            f"<b><u>이 숙소 특성</u></b><br>"
        )

        for f in judgment_features:
            popup_text += f"{f}: {feat_values[f]:.2f}<br>"
        popup_text += f"room_type: {room_type}<br>"

        popup_text += "<hr style='margin:4px 0'><b><u>숙소의 예측 가격을 실제 가격으로 가진 다른 숙소(±$5)</u></b><br>"
        for f in judgment_features:
            popup_text += f"{f}: {group_avg[f]:.2f}<br>"
        
        # room_type 비중
        popup_text += "<hr style='margin:4px 0'><b><u>room_type 비중</u></b><br>"
        for f in room_avg.index:
            name = f.replace('room_type_', '')
            popup_text += f"{name}: {int(room_avg[f]*100)}%<br>"

        # neighbourhood_group_cleansed 비중
        popup_text += "<br><b><u>지역 비중</u></b><br>"
        for f in region_avg.index:
            name = f.replace('neighbourhood_group_cleansed_', '')
            popup_text += f"{name} 비중: {int(region_avg[f]*100)}%<br>"

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_text, max_width=350),
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(cluster)

    return m
underpriced_map = show_underpriced_on_map(
    result_df=result_all,
    X_features=X
    )
underpriced_map