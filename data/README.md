# Data Directory

이 경로는 원자료와 분석 캐시가 생성되는 위치입니다. 제출용 GitHub ZIP에는 경로 설명만 포함하고, 실제 데이터 파일은 포함하지 않습니다.

- `fruitsfamily.db`: 크롤러가 생성하는 SQLite 원천 DB
- `raw_html/`: 크롤링 중 저장되는 원본 HTML
- `cache/`: `analysis.build_features`와 노트북이 생성하는 parquet 캐시

원자료부터 재실행하려면 크롤러로 DB를 다시 구축한 뒤 다음 순서로 피처와 노트북 산출물을 재생성합니다.

```bash
python -m analysis.build_features
python -m analysis.nbmake notebooks/00_report_eda.py
python -m analysis.nbmake notebooks/01_report_h1_general_guides.py
python -m analysis.nbmake notebooks/02_report_h2_structural_signals.py
python -m analysis.nbmake notebooks/03_report_h3_wishlist_onboarding.py
```
