import pandas as pd
import matplotlib

# 한글 폰트 설정 (matplotlib 시각화용)
matplotlib.rc('font', family='Malgun Gothic')

# 데이터 불러오기
df = pd.read_excel(r'C:\Users\rlaxo\Desktop\2025_Airbnb_NYC_listings.xlsx')

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

# 확인할 주요 컬럼 리스트
columns_to_check = [
    'reviews_per_month',
    'review_scores_rating',
    'review_scores_cleanliness',
    'review_scores_communication',
    'first_review',
    'last_review',
    'review_active_days',
    'days_since_last_review'
]

# 각 컬럼별 결측치 개수, 비율 확인
missing_info = pd.DataFrame({
    'Missing Count': df[columns_to_check].isnull().sum(),
    'Total': len(df),
    'Missing Ratio (%)': df[columns_to_check].isnull().mean() * 100
})

print(missing_info)