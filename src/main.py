"""크롤러 CLI 진입점.

사용법:
  python -m src.main init          # DB 초기화
  python -m src.main seed          # 카테고리에서 매물 시드 수집
  python -m src.main listings      # 매물 상세 크롤링
  python -m src.main sellers       # 셀러 메타 크롤링
  python -m src.main reviews       # 리뷰 크롤링
  python -m src.main wishlists     # 셀러 공개 위시리스트 크롤링
  python -m src.main fill          # condition/like_count/view_count 채우기 (Apollo 재크롤)
  python -m src.main stats         # 진행 상황 조회
  python -m src.main full          # seed → listings → sellers 일괄 실행

권장 워크플로:
  1) init 으로 DB 만들고
  2) seed 로 시드만 먼저 받아서 규모 확인
  3) listings 를 small batch (limit=100)로 돌려 파서 정상 작동 검증
  4) 정상이면 listings → sellers → reviews 순으로 본격 수집
"""
import argparse
import logging
import sys
from . import config, db
from .crawler import Crawler


def setup_logging():
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = config.LOG_DIR / "crawler.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def cmd_init(args):
    db.init_db()
    print(f"DB 초기화 완료: {config.DB_PATH}")


def cmd_seed(args):
    crawler = Crawler()
    with db.get_conn() as conn:
        crawler.seed_categories(conn)


def cmd_listings(args):
    crawler = Crawler()
    with db.get_conn() as conn:
        crawler.crawl_listings(conn, limit=args.limit)


def cmd_sellers(args):
    crawler = Crawler()
    with db.get_conn() as conn:
        crawler.crawl_sellers(conn, limit=args.limit)


def cmd_reviews(args):
    crawler = Crawler()
    with db.get_conn() as conn:
        crawler.crawl_reviews(conn, min_sales=args.min_sales, limit=args.limit)


def cmd_wishlists(args):
    crawler = Crawler()
    with db.get_conn() as conn:
        crawler.crawl_wishlists(conn, limit=args.limit)


def cmd_fill(args):
    """condition/like_count/view_count 없는 매물 상세 페이지 재크롤링."""
    crawler = Crawler()
    with db.get_conn() as conn:
        crawler.fill_listing_details(conn, limit=args.limit)


def cmd_stats(args):
    db.init_db()  # 없으면 만들어주고 0으로 보여주기
    with db.get_conn() as conn:
        s = db.stats(conn)
    print("=" * 50)
    print("크롤링 진행 상황")
    print("=" * 50)
    for k, v in s.items():
        print(f"  {k:15} : {v:,}")


def cmd_full(args):
    """end-to-end. 권장하지 않음 — 단계별 검증 후 사용."""
    crawler = Crawler()
    with db.get_conn() as conn:
        crawler.seed_categories(conn)
        crawler.crawl_listings(conn, limit=args.limit)
        crawler.crawl_sellers(conn, limit=args.limit)


def main():
    setup_logging()
    db.init_db()

    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init")
    sub.add_parser("seed")
    sub.add_parser("stats")

    sp = sub.add_parser("listings")
    sp.add_argument("--limit", type=int, default=100)

    sp = sub.add_parser("sellers")
    sp.add_argument("--limit", type=int, default=50)

    sp = sub.add_parser("reviews")
    sp.add_argument("--limit", type=int, default=50)
    sp.add_argument("--min-sales", type=int, default=5, dest="min_sales")

    sp = sub.add_parser("wishlists")
    sp.add_argument("--limit", type=int, default=1100)

    sp = sub.add_parser("fill")
    sp.add_argument("--limit", type=int, default=30000)

    sp = sub.add_parser("full")
    sp.add_argument("--limit", type=int, default=100)

    args = p.parse_args()

    handlers = {
        "init": cmd_init, "seed": cmd_seed, "listings": cmd_listings,
        "sellers": cmd_sellers, "reviews": cmd_reviews, "wishlists": cmd_wishlists,
        "fill": cmd_fill, "stats": cmd_stats, "full": cmd_full,
    }
    handlers[args.cmd](args)


if __name__ == "__main__":
    main()
