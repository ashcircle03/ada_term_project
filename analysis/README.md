# 분석 (Analysis)

후르츠패밀리 셀러 시그니처 분석 코드. 크롤러로 수집된 SQLite DB를 입력으로 받아
가설 1·2·3을 검증한다.

## 디렉토리

```
analysis/
├── __init__.py
├── data_loader.py      # DB → DataFrame + parquet 캐시
├── features.py         # 가공 변수 (signature_text, consistency, matching)
├── utils.py            # 한글 폰트, 결과 저장, 콘솔 헬퍼
├── h1_clustering.py    # H1: 셀러 시그니처 군집 (TF-IDF + K-means)
├── h2_anova.py         # H2: 클러스터별 가격 차이 (Kruskal + 매칭 분석)
├── h3_prediction.py    # H3: 가격 예측 (XGBoost + paired t-test)
├── run_all.py          # 전체 파이프라인
└── results/            # 가설별 JSON 결과 + figures/
```

## 의존성

기존 크롤러 의존성에 분석용 추가:

```bash
pip install pandas numpy scipy scikit-learn matplotlib
pip install xgboost          # H3 (없으면 GradientBoosting으로 폴백)
pip install pyarrow          # parquet 캐시
pip install hdbscan          # H1 보조 (없으면 K-means만)
pip install scikit-posthocs  # H2 사후검정 (없으면 직접 구현)
```

## 사용

```bash
# 데이터 상태 확인
python -m analysis.data_loader

# 가공 변수 생성 결과 미리보기
python -m analysis.features

# 가설별 개별 실행
python -m analysis.h1_clustering   # 먼저
python -m analysis.h2_anova        # H1 결과 사용
python -m analysis.h3_prediction   # H1 결과 사용

# 전체 한 번에
python -m analysis.run_all
```

## 가설 → 코드 매핑

| 가설 | 모듈 | 입력 | 출력 |
|---|---|---|---|
| **H1** 셀러 시그니처 클러스터 존재 | `h1_clustering.py` | listing.title/desc/brand | `seller_clusters.parquet`, k, silhouette, 클러스터별 키워드/브랜드 |
| **H2** 시그니처별 가격 프리미엄 | `h2_anova.py` | listing + seller_clusters | Kruskal-Wallis p-value, 매칭 그룹 내 가격비, 일관성-판매율 상관 |
| **H3** 시그니처가 가격 예측 개선 | `h3_prediction.py` | listing + 모든 가공 변수 | Model A vs B RMSE, R², paired t-test |

## 데이터 흐름

```
data/fruitsfamily.db
  ↓ data_loader
DataFrame (listing, seller, review)
  ↓ features
가공 변수 (signature_text, consistency, matched_pairs, listing_features)
  ↓ h1_clustering
seller_clusters.parquet (셀러 → 클러스터 라벨)
  ↓
h2_anova ← 통계 검정
h3_prediction ← XGBoost 비교
  ↓
analysis/results/*.json + figures/*.png
```

## 최소 데이터 요건

| 분석 | 최소 권장 | 이유 |
|---|---|---|
| H1 | 셀러 50명 이상, 매물 3+ 보유 | 클러스터링 안정성 |
| H2 | 매물 500건 이상 | 매칭 그룹 충분히 확보 |
| H3 | 매물 1,000건 이상 | XGBoost 학습 + 5-fold CV |

현재 단계에서 분석 가능 여부는 `python -m analysis.data_loader` 로 확인.

## 향후 확장

- **리뷰 텍스트 추가** — `review` 테이블이 채워지면 셀러 시그니처 입력에 결합 (외부 시각의 평가)
- **시계열 분석** — `posted_relative` 추정값으로 판매 소요시간 종속변수 추가
- **NLP 고도화** — KoNLPy/Soynlp로 형태소 분석, FastText/SBERT 임베딩으로 시그니처 표현력 강화
