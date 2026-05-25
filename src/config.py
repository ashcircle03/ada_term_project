"""크롤러 전역 설정.

모든 매직 넘버와 URL 패턴을 한 곳에 모은다.
실험 중에는 여기만 바꾸면 전체 동작이 바뀌도록 한다.
"""
from pathlib import Path

# ============================================================
# 경로
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_HTML_DIR = DATA_DIR / "raw_html"
LOG_DIR = PROJECT_ROOT / "logs"
DB_PATH = DATA_DIR / "fruitsfamily.db"

# ============================================================
# URL 패턴
# ============================================================
BASE_URL = "https://fruitsfamily.com"

URL_PATTERNS = {
    "product": "https://fruitsfamily.com/product/{product_id}/{slug}",
    "seller": "https://fruitsfamily.com/seller/{seller_id}/{username}",
    "seller_review": "https://fruitsfamily.com/seller/{seller_id}/{username}/review",
    "brand": "https://fruitsfamily.com/brand/{brand}",
    "search_category": "https://fruitsfamily.com/search?gender={gender}&subcategoryIds={sub_id}&sort={sort}",
    "vintage_shops": "https://fruitsfamily.com/vintage-shops",
}

# 시드 정렬 정책
# 'POPULAR'는 인기 브랜드(Ignota 등) 매물이 모든 카테고리에 도배되어
# 브랜드 다양성이 0에 수렴 — 셀러 시그니처 클러스터링이 brand bias로 무너짐
# 'RECENT'(최신순)로 변경하여 다양한 브랜드·셀러가 균등하게 노출되게 함
SEARCH_SORT = "RECENT"

# 카테고리 시드 — 매물·셀러를 발견하기 위한 시작점
# 의류 주요 서브카테고리 (subcategoryIds는 실측 검색 페이지에서 추출)
SEED_CATEGORIES = [
    # (gender, subcategory_id, label)
    ("MEN", 1, "men_short_tshirt"),
    ("MEN", 6, "men_hoodie"),
    ("MEN", 26, "men_jacket"),
    ("MEN", 132, "men_leather_jacket"),
    ("MEN", 11, "men_denim_pants"),
    ("MEN", 35, "men_sneakers"),
    ("WOMEN", 1, "women_short_tshirt"),
    ("WOMEN", 5, "women_knit"),
    ("WOMEN", 26, "women_jacket"),
    ("WOMEN", 124, "women_mini_skirt"),
    ("WOMEN", 128, "women_mini_dress"),
]

# 브랜드 시드 — 카테고리 시드와 병행하여 셀러·매물 다양성 확보
# 후르츠 인기 브랜드 페이지 상단 30개를 추출 (실측, 2026-04 기준)
# 각 브랜드 페이지의 매물 첫 페이지(약 30~40건)를 시드로 흡수
SEED_BRANDS = [
    "Ignota", "A.Presse", "Chrome Hearts", "Hysteric Glamour", "SANSAN GEAR",
    "RRL", "Levi's", "Supreme", "KAPITAL", "Adidas",
    "C.P. Company", "Balenciaga", "Hatchingroom", "Prada", "Maison Margiela",
    "Polo Ralph Lauren", "Comme des Garcons", "Bape", "Lemaire", "Our Legacy",
    "XLIM", "Rick Owens", "Stussy", "Montbell", "Stone Island",
    "PLASTICPRODUCT", "ARCTERYX", "Vivienne Westwood", "Dr. Martens", "Carhartt",
]

# ============================================================
# 크롤링 정책 (서버 부담 최소화)
# ============================================================
# 약관 정신 존중 + 차단 회피를 위한 보수적 값
REQUEST_DELAY_MIN = 1.0  # 최소 요청 간격 (초)
REQUEST_DELAY_MAX = 2.0  # 최대 요청 간격 (랜덤 지터)
TIMEOUT = 15  # HTTP 타임아웃
MAX_RETRIES = 3  # 실패 시 재시도 횟수
BACKOFF_BASE = 5  # 재시도 간 exponential backoff 기본 초

# User-Agent: 일반 브라우저로 위장하지 말고 명시적 식별자 사용
# (학술 연구 명시 — 운영자가 차단할지 허용할지 명확히 판단하게)
USER_AGENT = (
    "Mozilla/5.0 (compatible; AcademicResearchBot/1.0; "
    "kyunghee Univ. data analysis term project; "
    "contact: ashcircle03@gmail.com)"
)

# ============================================================
# 수집 규모 한도 (실수로 무한 루프 방지)
# ============================================================
MAX_PAGES_PER_CATEGORY = 5  # 카테고리당 페이지네이션 최대 깊이
MAX_LISTINGS_PER_SELLER = 200  # 셀러당 매물 수집 한도
MAX_TOTAL_LISTINGS = 30_000  # 전체 매물 수집 한도
MAX_TOTAL_SELLERS = 1_500  # 전체 셀러 수집 한도

# ============================================================
# 익명화
# ============================================================
# seller_id를 SHA256으로 해싱할 때 사용할 솔트 (분석 단계에서만 본인이 보관)
ANONYMIZATION_SALT = "39cnejch"  # 실행 전 반드시 교체
