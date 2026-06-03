# FruitsFamily Seller Analysis

빈티지 C2C 플랫폼 FruitsFamily의 공개 상품·셀러 데이터를 바탕으로 신규 셀러 초기 안내 가설을 검토한 데이터 분석 프로젝트입니다. 최종 보고서는 별도로 제출하며, 보고서에 쓰인 산출물은 `results/`에 보관합니다.

## Repository Layout

- `src/`: 크롤러, 파서, SQLite 저장소, CLI
- `analysis/`: 공용 피처 생성과 노트북 변환 도구
- `notebooks/`: 보고서 절과 연결된 실행 스크립트·노트북 쌍
  - `00_report_eda.*`: 데이터 개요와 판매 전환 문제
  - `01_report_h1_general_guides.*`: 사진 수·설명 길이 등 수량형 등록 안내 검토
  - `02_report_h2_structural_signals.*`: 구조 변수와 표현 변수의 예측 신호 비교
  - `03_report_h3_wishlist_onboarding.*`: 위시리스트 기반 취향 정렬과 초기 안내 가설
- `results/`: 보고서에 인용한 JSON 결과와 그림

`archive/`, `data/`, `AGENTS.md`, `REPORT.md`, dot-prefixed 로컬 경로는 제출용 GitHub ZIP에서 제외합니다. `REPORT.md`는 로컬 보고서 초안이며, 최종 보고서는 별도 파일로 제출합니다.

## Environment

Python 3.11 conda 환경을 사용합니다. base 환경에는 패키지를 설치하지 않습니다.

```bash
conda create -p .conda python=3.11 -y
conda activate ./.conda
pip install -r requirements.txt
python -m ipykernel install --user --name ada --display-name "ada (Python 3.11)"
```

macOS에서 XGBoost 실행 시 OpenMP가 필요하면 `brew install libomp`를 먼저 실행합니다. matplotlib 캐시 경고를 피하려면 다음 환경 변수를 사용합니다.

```bash
export MPLCONFIGDIR=.cache/matplotlib
```

## Reproducing Report Outputs

최종 보고서 수치와 그림은 Git 추적 대상인 `results/*.json`과 `results/figures/*.png`에 들어 있습니다. 원자료부터 다시 실행하려면 SQLite 원천 데이터가 필요합니다.

```bash
python -m analysis.build_features
python -m analysis.nbmake notebooks/00_report_eda.py
python -m analysis.nbmake notebooks/01_report_h1_general_guides.py
python -m analysis.nbmake notebooks/02_report_h2_structural_signals.py
python -m analysis.nbmake notebooks/03_report_h3_wishlist_onboarding.py
```

`data/fruitsfamily.db`, `data/cache/`, `data/raw_html/`은 원자료·캐시 파일이라 GitHub ZIP에는 들어가지 않습니다. 따라서 ZIP만으로는 보고서 산출물을 확인할 수 있지만, 원자료 수집부터 완전 재실행하려면 크롤러로 DB를 다시 구축하거나 별도 데이터 파일을 제공해야 합니다.

## Crawler Commands

작은 단위로 검증한 뒤 크롤링 범위를 늘립니다.

```bash
python -m src.main init
python -m src.main listings --limit 100
python -m src.main sellers --limit 50
python -m src.main stats
python -m src.reparse --limit 5000
```
