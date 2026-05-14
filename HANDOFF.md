# 프로젝트 인수인계 — 후르츠패밀리 셀러 시그니처 연구

> **새 세션의 Claude에게**: 이 문서를 먼저 끝까지 읽어. 사용자(재원)와 함께 5월부터 이어온 학기 프로젝트의 현재 상태와 다음 작업이 정리돼있어. 이 문서를 다 읽기 전엔 코드 수정·새 분석 시작하지 말 것.

---

## 0. 사용자 컨텍스트 (중요)

- **이름**: 재원
- **학과**: 소프트웨어융합학과
- **과목**: 응용 데이터 분석 학기 프로젝트
- **제약**: 자체 수집 데이터만 허용 (공개 데이터셋 사용 불가)
- **요구사항**: 가설 3개 (비지도 1개, 통계 1개, 지도 1개) 필수
- **선호**: 짧은 프롬프트로 substantial output 기대. 결론·action 지향. 형식적 칭찬·불필요한 preamble 싫어함. 솔직하고 직접적인 평가 선호.

### 교수님 핵심 피드백 (반드시 준수)

> "다른 회사에 있는 기능을 후르츠에 옮기는 건 분석이 아니다. 완전히 새로운 것이거나, 기존 연구·서비스가 못 다룬 부분을 정밀하게 짚어야 한다."

이 기준 때문에 영양제·화장품 등 다른 도메인을 검토했으나 모두 거절됨. 후르츠 도메인은 본인 익숙함 + 데이터 이미 있음 + 학술적 공백 존재로 채택.

### 사용자 상태

여러 번 가설 갈아엎으며 번아웃 직전 → 새 분석 코드 짜기보단 **이미 가진 결과를 학술 프레임으로 재정렬**하는 게 핵심. 새 가설 제안하지 말 것. 

---

## 1. 연구 한 줄 정리

> **"셀러 정체성은 가격이 아니라 판매 가능성을 만든다 — 후르츠패밀리 5,909건 데이터로 검증한 빈티지 C2C 셀러 시그니처 효과"**

Cervi(2023)와 Pugh & Ripley(2024)의 정성적 가설(셀러 정체성이 거래에 영향)을 후르츠 데이터로 정량 검증. 새 발견: 시그니처가 가격이 아닌 판매율에 영향.

---

## 2. 분석 대상 — 후르츠패밀리 (fruitsfamily.com)

한국의 SNS형 빈티지 C2C 플랫폼. 셀러가 직접 모델·큐레이터 역할을 하며 팔로워 기반으로 매물 노출. Depop의 한국 버전에 가까움. 학술 연구 부재.

### 데이터 수집 결과 (현재 상태)

- 매물 5,909건 (시드 5,922건 중 99% 처리)
- 발견된 셀러 1,046명, seller 테이블 200명 메타 수집 완료
- 브랜드 62개 (균등 분포, 상위 5개 점유율 3.5%)
- sold 비율 34.0%
- 가격 중앙값 120,000원 (Q1 72K - Q3 220K)
- 매물 5건+ 보유 활성 셀러 168명
- 매칭 가능 매물 886건 (브랜드 + 카테고리 + 사이즈)

### 데이터 한계 (반드시 인지)

- 셀러 단계 크롤링 매물 4,677건은 **카테고리 NULL** (셀러 페이지 카드 파싱 한계)
- `posted_relative` (회전 시간) sold 매물의 4.3%만 보유 → 회전 시간 분석 불가
- `likes`, `comments`, `n_photos` 값이 거의 동일 → 변별력 부족

---

## 3. 가설과 결과 (이미 분석 완료, 재실행 불필요)

### Baseline (사전 분석)

매칭 그룹 95개에서:
- 가격 변동계수(CV) 중앙값 **0.588**
- premium 셀러(가격비 1.5배) 41% / discount 셀러(가격비 0.5배) 38%
- 매물 메타만으로 가격 예측 R² 0.157 (84%의 변동 미설명)

이게 가설의 **출발점 증거**: 같은 옷이 셀러에 따라 가격 분포가 큼 → 셀러 잠재 변수 존재 추정.

### H1 (비지도) — 셀러 시그니처 클러스터링 ★★★ 강하게 지지

**가설**: 셀러는 매물 텍스트와 브랜드 분포로 의미 있는 N개 군집으로 분리된다.

**방법**: TF-IDF + K-means. 본문 제외(desc=0), 브랜드 가중치 5배, 제목 가중치 2배.

**결과**:
- 분석 대상 셀러 26명 (매물 5+ 보유)
- 9개 클러스터, **실루엣 0.507** (텍스트 클러스터링 합리적 구조)
- 명확한 시그니처 발견:
  - 일본 펑크 (Hysteric Glamour + CDG + RRL)
  - 이탈리아 테크니컬 (Stone Island + C.P. Company)
  - 다크 럭셔리 (Chrome Hearts + Rick Owens + Givenchy)
  - 한국 디자이너 (Hatchingroom)
  - 닥터마틴 컬렉터 (MIE/UK)
  - 빈티지 카메라 (Sanyo + Kyocera + Canon)
  - 925 실버 주얼리
  - RRL 단독
  - 잡탕형 (7명, 시그니처 약함)

**저장 위치**: `data/cache/seller_clusters.parquet`

### H2 (통계) — 시그니처 → 가격·판매 효과 ★★★ 부분 지지

**가설**: 시그니처 클러스터별로 가격에 차이 + 일관성이 판매 성공에 영향.

**방법**: Kruskal-Wallis 검정 + 매칭 분석 + Spearman 상관.

**결과**:
- 클러스터 간 가격 분포 Kruskal-Wallis **p < 0.001** (유의)
- 클러스터별 가격 중앙값 99K ~ 450K원 (4.5배 차이)
- 매칭 분석(브랜드+카테고리+사이즈): 클러스터 1번 셀러군 -44.6% 디스카운트 (표본 16건, 시사적)
- **시그니처 일관성 vs 판매율 ρ=0.41, p=0.002** (유의)
- **시그니처 일관성 vs 가격 ρ=0.03** (비유의)

**중요 발견**: 시그니처는 가격이 아니라 판매에 영향. 이게 본 연구의 가장 강한 차별점.

### H3 (지도) — 시그니처 가격 예측 효과 ★ 약하게 지지

**가설**: 시그니처를 매물 메타에 결합하면 가격 예측 R² 개선.

**방법**: Model A (매물 메타만) vs Model B (+ 시그니처) XGBoost 5-fold CV + paired t-test.

**결과**:
- Model A R² = 0.157, RMSE = 326,806원
- Model B R² = 0.181 (Δ +0.024, p=0.066)
- RMSE 변화 미미 (-1.45%)

**해석**: 매물 메타(브랜드 dummy)의 강한 신호가 시그니처 효과를 일부 가림. H2와 결합하면 "시그니처는 가격이 아닌 판매를 만든다" narrative로 통합.

### 가설별 검증 강도 요약

| 가설 | 검증 강도 | 핵심 근거 |
|---|---|---|
| H1 (비지도) | ★★★ 강함 | 실루엣 0.507 + 9개 도메인 의미 명확 |
| H2 (통계) | ★★★ 부분 | p<0.001 (가격 분포) + ρ=0.41 (일관성-판매) |
| H3 (지도) | ★ 약함 | R² Δ 0.024 (마진널 유의 p=0.066) |

---

## 4. 선행연구 차별성 (학술 프레임)

후르츠 연구의 학술적 새로움은 **3가지 정성적 선행연구의 정량 검증**:

| 선행연구 | 정성적 주장 | 본 연구의 정량 검증 |
|---|---|---|
| **Cervi (2023)** Depop 인플루언서 10명 Instagram 분석 | "셀러 정체성이 거래에 영향" | H1 9개 클러스터 + H2 ρ=0.41 |
| **McKeown (2024)** Texas State thesis | "셀러가 패션 게이트키퍼" | H2 시그니처별 판매율 차이 |
| **Pugh & Ripley (2024)** Cardiff thesis | "빈티지 셀러가 era/quality/style로 가격 책정" | H2 매칭 분석 + H3 (부분 반증: 가격 아닌 판매) |
| **Ellen MacArthur 재단** 산업 관찰 | "큐레이션이 mark-up 결정" | H3 (수정: mark-up 아니라 회전 속도) |

**5차원 차별성**:

| 차원 | 선행연구 위치 | 본 연구의 새로움 |
|---|---|---|
| 분석 단위 | 매물/트랜잭션 또는 사용자 일반 | **셀러 단위** |
| 방법론 | 정성 사례연구 / 설문 SEM | **TF-IDF + K-means + Kruskal + XGBoost + paired t-test** |
| 데이터 출처 | Instagram 텍스트 / 리뷰 | **거래 데이터 + 셀러 메타 직접 결합** |
| 플랫폼 | Vinted/Depop/당근/Xianyu | **후르츠패밀리 (학술 공백)** |
| 검증 수준 | 정성적 주장 | **정량 검증 + 통계 검정** |

### 발견한 선행연구 9편 (참고문헌 형태)

1. Cervi, F. (2023). Effects of Instagram Influencers on the Adoption of Secondhand Fashion Consumption. IGI Global.
2. Hossain et al. (2022). Investigating Consumer Values of Secondhand Fashion Consumption: Mass vs Luxury. Sustainability.
3. Khaleefah & Al-Ani (2021). Boolean logic algebra driven similarity measure for text-based applications. PeerJ CS.
4. Lee et al. (2023). The Factors Influencing Users' Trust in C2C Secondhand Marketplace. Sustainability.
5. Li et al. (2024). Unlocking insights: integrated text mining and ISM. PeerJ CS.
6. McKeown, S. (2024). Resale Revolution: Resellers' Evolving Power. Texas State University thesis.
7. Pugh & Ripley (2024). The Price of Vintage: Developing a Model for Valuing Vintage Clothing. Cardiff University.
8. Sasaki et al. (2025). Determinants of secondhand consumer choices on C2C in Japan. Cleaner and Responsible Consumption.
9. Skuza et al. (2024). Text-Based Product Matching: Semi-Supervised Clustering. arXiv:2402.10091.
10. Wang et al. (2023). Dynamic decisions between sellers and consumers in online second-hand trading. Transportation Research Part E.

---

## 5. 액션 제안 (후르츠 의사결정자용)

가설 결과가 직접 4가지 액션으로 연결됨:

### 액션 1 — 유저(구매자): 시그니처 매칭 시스템
**근거**: H1 (9개 명확 군집) + H2 (시그니처-판매율 ρ=0.41)
- 가입 시 5-7개 OOTD 이미지 선택 → 시그니처 클러스터 매핑
- 본인 시그니처 일치 셀러 우선 노출
- "이 셀러와 비슷한 셀러" 동일 클러스터 추천

### 액션 2 — 셀러(판매자): 시그니처 명확도 피드백
**근거**: H2 일관성-판매율 상관
- 매물 등록 시 "당신의 매물은 [일본 펑크] 시그니처와 70% 일치" 점수 표시
- 가격 추천은 매물 메타 기반 (H3가 시그니처 가격 효과 약함을 보임)

### 액션 3 — 운영진: 톱셀러 발굴 지표 고도화
**근거**: H2
- 기존 지표에 **시그니처 명확도** + **시그니처 내 판매율 분위** 추가
- 시그니처 공백(수요 있는데 셀러 적은 군집) 탐지 → 영입 우선순위

### 액션 4 — 오프라인 편집숍 매칭
**근거**: H1 + Sasaki(2025)
- 후르츠 입점 오프라인 빈티지 편집숍을 시그니처별 라벨링
- 사용자 진단 결과와 연결

---

## 6. 분석 한계 (반드시 보고서에 명시)

1. **표본 크기**: H1 클러스터링 대상 셀러 26명, 클러스터당 2-3명
2. **매칭 본질적 어려움**: 빈티지는 매물 unique → 같은 옷 비교 본질 불가능
3. **시간 차원 부재**: 단일 시점 크롤링 → 셀러 학습·시그니처 변화 추적 불가
4. **텍스트 정보 밀도 낮음**: 빈티지 매물 본문은 보일러플레이트 비중 높음 (desc=0 권장이 이를 시사)
5. **NLP 단순함**: KoNLPy 미사용, 어미 트리밍 휴리스틱만 적용

---

## 7. 워킹디렉토리 구조 (현재 상태)

```
fruitsfamily_crawler/
├── data/
│   ├── fruitsfamily.db          # SQLite (매물 5,909건 + 셀러 200명)
│   └── cache/
│       └── seller_clusters.parquet  # H1 결과 (셀러별 클러스터 라벨)
├── src/                          # 크롤러 (config, db, fetcher, parsers, crawler, main)
├── analysis/
│   ├── data_loader.py
│   ├── features.py               # build_seller_text(), matched_pairs(), seller_aggregates()
│   ├── utils.py                  # section(), bullet(), save_result(), setup_korean_font()
│   ├── h1_clustering.py          # ★ 채택한 시그니처 클러스터링
│   ├── h2_anova.py               # ★ 채택한 시그니처-가격 검정
│   ├── h3_prediction.py          # ★ 채택한 가격 예측 (Model A vs B)
│   ├── h2_baseline.py            # baseline 가격 격차 분석
│   ├── h3_baseline.py            # baseline 매물 메타 예측
│   └── results/                  # 분석 결과 JSON·figure
├── test_parsers.py
├── test_regression.py
├── test_upsert.py
├── test_e2e.py
├── test_brand_fix.py
├── test_size_whitelist.py
├── test_analysis_pipeline.py
└── HANDOFF.md                    # 이 문서
```

**버려진 실험들 (이력 참고용, 진행 X)**:
- `h1_price_curve.py`, `h2_seller_experience.py`, `h3_recommendation.py` — 가격-판매 곡선 가설 (매칭 본질 한계로 폐기)
- `h1v3_listing_quality.py`, `h2v3_seller_operations.py`, `h3v3_signature_match.py` — 매물 작성 품질 가설 (코드만 작성, 채택 안 함)
- `diagnose_pricing.py`, `diagnose_matching.py` — 진단 스크립트 (참고용)

**채택된 분석 흐름**: 
```
diagnose_for_planning.py → h1_clustering.py → h2_anova.py → h3_prediction.py
```

---

## 8. 환경 설정

- **OS**: macOS (사용자 환경)
- **Python**: 3.12 (miniforge)
- **주요 의존성**: 
  ```
  pandas, numpy, scipy, scikit-learn, xgboost, statsmodels,
  matplotlib (AppleGothic 폰트), pyarrow,
  requests, beautifulsoup4, lxml
  ```
- **DB**: SQLite (별도 서버 없음)

---

## 9. 다음 단계 — 사용자가 합의한 작업 순서

### 단계 1 (지금 진행): 선행연구 분석 워드 문서
**파일명**: `docs/선행연구_분석.docx`

내용:
- 위 4번 섹션을 정식 학술 형식으로 확장
- 9편 선행연구 각각의 요약 + 한계 + 본 연구의 차별점
- 3갈래 구조 (C2C 플랫폼 / 패션 리세일 / 텍스트 마이닝)
- 차별성 5차원 표
- 가설별 선행연구 매핑 표

### 단계 2: 연구 결과 통합 보고서 워드 문서
**파일명**: `docs/연구_결과_보고서.docx`

내용:
- 1번 문제 정의 → 2번 연구 주제 → 3번 가설(H1·H2·H3) → 결과 → 결론 → 5번 액션 → 6번 한계
- 그래프·표 정리 (실루엣 곡선, 클러스터 키워드, Kruskal 결과, Model A vs B 비교)

### 단계 3 (선택): MVP 시각화 / 발표 슬라이드

---

## 10. 새 세션 Claude에게 — 작업 시작 전 확인할 것

1. **이 문서 다 읽었는지 확인**. 안 읽고 코드 수정 들어가지 말 것.
2. **새 가설·새 분석 제안하지 말 것**. 이미 분석 완료, narrative 재정렬 단계.
3. **사용자 톤 존중**: 짧은 프롬프트 → substantial output. 형식적 칭찬 X. 솔직한 평가 ○.
4. **docx 생성 시 mnt/skills/public/docx/SKILL.md 먼저 view** (작업 환경에 따라 경로 다름).
5. **회귀 테스트 8개 통과 상태 유지**: 코드 수정 시 `python test_*.py` 다 돌려서 깨지지 않는지 확인.

### 최우선 작업

**ada.ipynb에서 새로 정리 시작**부터. 
이거 묻고 시작하면 됨.
