# 위시리스트 크롤 — 클라우드 실행 런북 (전체 DB scp 방식)

로컬 5시간 부담을 피해 위시 크롤을 클라우드 VM에서 돌린다. DB는 git에서 제외했으므로
**전체 DB를 scp로 양방향** 옮긴다. 위시 크롤은 resumable이라 어디서 멈춰도 안전.

전제: 클라우드 접속을 `CLOUD=user@host` 로 둔다. 로컬 DB는 이미 WAL 체크포인트로
단일 파일 정리됨(`data/fruitsfamily.db`, 191MB, 현재 done≈1,372 / 남은≈9,724).

## 0) (로컬) 보내기 전 확인
```bash
# 크롤러가 안 돌고 있어야 함
pgrep -f "src.main wishlists" && echo "아직 돌고있음 → 종료필요" || echo "clean"
# 단일 .db 인지 (-wal/-shm 없어야 함)
ls data/fruitsfamily.db*
```

## 1) (로컬→클라우드) 코드 + DB 전송
```bash
CLOUD=user@host        # 예: opc@<oracle-vm-ip>
ssh $CLOUD 'git clone https://github.com/ashcircle03/ada_term_project.git ada || (cd ada && git pull)'
ssh $CLOUD 'mkdir -p ada/data'
scp data/fruitsfamily.db $CLOUD:ada/data/fruitsfamily.db     # ~191MB, 1회
```

## 2) (클라우드) 경량 환경 + 크롤 실행
크롤러는 `requests / beautifulsoup4 / lxml` 만 필요(분석 라이브러리 불필요).
```bash
ssh $CLOUD
cd ada
python3 -m venv .venv && source .venv/bin/activate
pip install requests beautifulsoup4 lxml
python -m src.main init          # 스키마/마이그레이션 정합성 보정(안전, 데이터 보존)

# 남은 셀러 위시 크롤 (resumable; done 표시된 ~1,372명은 자동 skip)
nohup python -m src.main wishlists --limit 11000 > wishlist.log 2>&1 &
tail -f wishlist.log             # 진행 로그

# 진행 수 확인
python -c "import sqlite3;print('done=',sqlite3.connect('data/fruitsfamily.db').execute(\"SELECT COUNT(*) FROM crawl_state WHERE key LIKE 'wishlist:%:done'\").fetchone()[0])"
```
- rate limit 1–2초/요청(`src/config.py`) → 약 4.5~5시간. `config.py`의 USER_AGENT는
  학술용 식별자 그대로 둘 것(사이트가 허용/차단을 선택할 수 있게).
- ⚠️ 클라우드에서 **git commit 하지 말 것**(DB가 main에선 아직 추적됨). 크롤만 돌린다.

## 3) (클라우드→로컬) 결과 회수
```bash
# 클라우드: 다시 단일 .db로 정리 후
ssh $CLOUD 'cd ada && .venv/bin/python -c "import sqlite3;c=sqlite3.connect(\"data/fruitsfamily.db\");c.execute(\"PRAGMA wal_checkpoint(TRUNCATE)\");c.close()"'
# 로컬로 회수 (클라우드 DB가 최신 = canonical)
scp $CLOUD:ada/data/fruitsfamily.db data/fruitsfamily.db
```

## 4) (로컬) 검증 + 위시 분석 재실행
```bash
python -c "import sqlite3;c=sqlite3.connect('data/fruitsfamily.db');\
print('owners=',c.execute('SELECT COUNT(DISTINCT owner_seller_id) FROM wishlist').fetchone()[0],\
'done=',c.execute(\"SELECT COUNT(*) FROM crawl_state WHERE key LIKE 'wishlist:%:done'\").fetchone()[0])"
# 기대: done≈11,096, owners 대폭 증가(크롤된 셀러의 ~91%)

python -m analysis.build_features
jupyter nbconvert --to notebook --execute --inplace notebooks/03_h3_archetypes.ipynb   # H3/P4 위시 분석 갱신
```

## 메모
- 끊겨도 안전: 셀러마다 즉시 commit. 재실행하면 done 건너뜀.
- 더 빠르게: 위시 크롤은 셀러 독립적이라 `--limit`을 나눠 여러 VM에 분산 가능
  (단 각 VM이 같은 not-done 셀러를 가져오므로, 분산하려면 DB를 나눠 seed하거나
  seller_id 해시로 샤딩하는 보완 필요 — 단일 VM이면 신경 안 써도 됨).
