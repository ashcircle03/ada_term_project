"""후르츠패밀리 셀러 시그니처 분석 패키지.

모듈 구성:
  data_loader.py   — DB → DataFrame 로더 + 캐시
  features.py      — 가공 변수 생성 (signature_text, consistency 등)
  h1_clustering.py — 가설 1: 셀러 시그니처 군집 (비지도)
  h2_anova.py      — 가설 2: 시그니처별 가격 차이 (통계)
  h3_prediction.py — 가설 3: 가격 예측 (지도학습)
  utils.py         — 시각화, 한글 폰트, 공통 유틸

실행 방식:
  python -m analysis.h1_clustering
  python -m analysis.h2_anova
  python -m analysis.h3_prediction
"""
