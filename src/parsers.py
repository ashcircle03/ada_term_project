"""HTML → 구조화 데이터.

후르츠패밀리는 React SPA + SSR 구조. HTML에는 두 가지 데이터 소스가 있다:
  1. __APOLLO_STATE__ (script#__APOLLO_STATE__ JSON) — 구조화된 GraphQL 캐시.
     가장 신뢰할 수 있는 소스. condition, like_count, view_count, createdAt 등 포함.
  2. 텍스트 휴리스틱 — Apollo state가 없거나 필드가 누락됐을 때 폴백.

파서 우선순위: Apollo state → 텍스트 휴리스틱
누락 필드는 None 반환, 절대 예외 던지지 않는다.
"""
import re
import json
import logging
from urllib.parse import unquote
from bs4 import BeautifulSoup


logger = logging.getLogger(__name__)


# ============================================================
# Apollo __APOLLO_STATE__ 파서 (1순위)
# ============================================================

def _extract_apollo(soup) -> dict:
    """script#__APOLLO_STATE__ JSON 파싱. 없으면 빈 dict."""
    tag = soup.find("script", {"id": "__APOLLO_STATE__"})
    if not tag or not tag.string:
        return {}
    try:
        return json.loads(tag.string)
    except Exception:
        return {}


def _apollo_product(apollo: dict, product_id: str) -> dict:
    """Apollo state에서 현재 페이지의 메인 product 항목 반환.

    fruitsfamily는 shortcode URL (/product/5a27w/x)이지만 Apollo state는
    numeric ID (ProductNotMine:8867516)를 키로 씀. ROOT_QUERY의
    seeProductResponse 레퍼런스로 실제 numeric key를 찾는다.
    """
    # 1순위: ROOT_QUERY.seeProductResponse.seeProduct.__ref → numeric key
    rq = apollo.get("ROOT_QUERY", {})
    for k, v in rq.items():
        if k.startswith("seeProductResponse(") and isinstance(v, dict):
            # response wrapper: {seeProduct: {__ref: "ProductNotMine:xxxx"}}
            inner = v.get("seeProduct") or v
            ref = inner.get("__ref", "") if isinstance(inner, dict) else ""
            if ref and ref in apollo:
                return apollo[ref]

    # 2순위: shortcode가 곧 numeric ID인 경우 (드문 케이스)
    for key in (f"ProductNotMine:{product_id}", f"ProductMine:{product_id}"):
        if key in apollo:
            return apollo[key]

    # 3순위: id 필드 매칭
    for v in apollo.values():
        if isinstance(v, dict) and v.get("id") == product_id and "price" in v:
            return v
    return {}


def _apollo_seller_from_product(apollo: dict) -> dict:
    """상품 페이지 Apollo state에서 셀러 정보 추출."""
    for k, v in apollo.items():
        if isinstance(v, dict) and k.startswith("SellerNotMe:") and "rating" in v:
            return v
    return {}


def _apollo_seller(apollo: dict, seller_id: str) -> dict:
    """셀러 페이지 Apollo state에서 셀러 정보 추출."""
    for key in (f"SellerNotMe:{seller_id}", f"SellerMe:{seller_id}"):
        if key in apollo:
            return apollo[key]
    return {}


# ============================================================
# 헬퍼
# ============================================================

# 가격 정규식: 반드시 숫자로 시작해야 하고, 콤마와 숫자가 이어진 뒤 '원'으로 끝남
# 잘못된 케이스를 방지하기 위해 \d로 강제 시작하고, [\d,]*로 이어붙임
KOREAN_PRICE_RE = re.compile(r"(\d[\d,]*)\s*원")
PERCENT_RE = re.compile(r"(\d+)\s*%")
PRODUCT_URL_RE = re.compile(r"/product/([^/?]+)")
SELLER_URL_RE = re.compile(r"/seller/([^/]+)/([^/?#]+)")
BRAND_URL_RE = re.compile(r"/brand/([^/?#]+)")


def _won(text: str) -> int | None:
    """'225,000원' → 225000. 매치 없거나 변환 실패 시 None.

    방어적 변환:
      - 빈 입력 → None
      - 정규식 매치 실패 → None
      - 숫자 부분이 비어있음 → None (regex가 강화됐어도 안전망)
      - int() 변환 실패 → None
    """
    if not text:
        return None
    m = KOREAN_PRICE_RE.search(text)
    if not m:
        return None
    digits = m.group(1).replace(",", "")
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def _percent(text: str) -> int:
    if not text:
        return 0
    m = PERCENT_RE.search(text)
    return int(m.group(1)) if m else 0


def _int_safe(text: str) -> int:
    if not text:
        return 0
    digits = re.sub(r"[^\d]", "", text)
    return int(digits) if digits else 0


# ============================================================
# 매물 페이지 파서
# ============================================================

def parse_product_page(html: str, url: str) -> dict | None:
    """매물 상세 페이지 → 매물 dict.

    우선순위: __APOLLO_STATE__ JSON → 텍스트 휴리스틱 폴백
    """
    if not html:
        return None

    soup = BeautifulSoup(html, "lxml")
    data: dict = {}

    # 1. ID
    m = PRODUCT_URL_RE.search(url)
    data["product_id"] = m.group(1) if m else None
    if not data["product_id"]:
        logger.warning(f"product_id 추출 실패: {url}")
        return None

    # ── Apollo state 파싱 ─────────────────────────────────────
    apollo = _extract_apollo(soup)
    ap = _apollo_product(apollo, data["product_id"])
    # ROOT_QUERY 경유 ap는 view_count가 없는 partial 객체일 수 있음.
    # ap의 numeric id로 ProductNotMine/ProductMine 전체 객체를 재조회.
    _numeric_id = ap.get("id")
    if _numeric_id:
        _full_ap = (
            apollo.get(f"ProductNotMine:{_numeric_id}")
            or apollo.get(f"ProductMine:{_numeric_id}")
        )
        if _full_ap and _full_ap.get("view_count") is not None:
            ap = _full_ap

    # 2. 제목
    data["title"] = ap.get("title") or None
    if not data["title"]:
        h1 = soup.find("h1")
        data["title"] = h1.get_text(strip=True) if h1 else None

    # 3. 본문
    data["description"] = ap.get("description") or None
    h1 = soup.find("h1") if not data["description"] else None
    if not data["description"] and h1:
        for sib in h1.find_all_next(string=True):
            text = sib.strip()
            if len(text) > 30 and "원" not in text and "%" not in text:
                data["description"] = text
                break

    # 4. 브랜드
    brand = None
    brand_link = None
    # Apollo: brand 필드가 {"__ref": "Brand:xxx"} 형태 → apollo에서 실제 name 추출
    ap_brand = ap.get("brand")
    if isinstance(ap_brand, dict):
        brand_ref = ap_brand.get("__ref", "")
        brand_obj = apollo.get(brand_ref, {})
        brand = brand_obj.get("name") or brand_obj.get("nameKo") or None
        if not brand and brand_ref:
            # __ref 키 자체에서 슬래시 뒤 추출 (e.g. "Brand:Chrome Hearts")
            parts = brand_ref.split(":", 1)
            if len(parts) == 2 and parts[1]:
                brand = parts[1]

    # 텍스트 휴리스틱 폴백
    if not brand:
        h1_tag = soup.find("h1")
        if h1_tag:
            for prev in h1_tag.find_all_previous("a", href=BRAND_URL_RE):
                href = prev.get("href", "")
                bm = BRAND_URL_RE.search(href)
                if bm:
                    brand = unquote(bm.group(1))
                    brand_link = prev
                    break
        if not brand:
            for a in soup.find_all("a", href=BRAND_URL_RE):
                if a.find_parent(["nav", "header"]):
                    continue
                href = a.get("href", "")
                bm = BRAND_URL_RE.search(href)
                if bm:
                    brand = unquote(bm.group(1))
                    brand_link = a
                    break
    data["brand"] = brand

    # 5. 카테고리
    ap_cat = ap.get("category")
    if isinstance(ap_cat, dict):
        cat_ref = ap_cat.get("__ref", "")
        cat_obj = apollo.get(cat_ref, {})
        data["category_l1"] = cat_obj.get("gender") or cat_obj.get("topCategory") or None
        data["category_l2"] = cat_obj.get("name") or cat_obj.get("nameKo") or None

    if not data.get("category_l1"):
        page_text_cat = soup.get_text(" ", strip=True)
        cat_match = re.search(r"(남자|여자|라이프|굿즈)\s*>\s*([^\s|·]+)", page_text_cat)
        if cat_match:
            data["category_l1"] = cat_match.group(1)
            data["category_l2"] = cat_match.group(2)

    # 6. 가격
    ap_price = ap.get("price")
    ap_original = ap.get("originalPrice")
    ap_discount = ap.get("discountRate") or 0

    if ap_price is not None:
        try:
            data["price_final"] = int(ap_price)
        except (TypeError, ValueError):
            pass
    if ap_original is not None:
        try:
            data["price_original"] = int(ap_original)
        except (TypeError, ValueError):
            pass
    try:
        data["discount_pct"] = int(ap_discount)
    except (TypeError, ValueError):
        data["discount_pct"] = 0

    # 가격 폴백: 텍스트 휴리스틱
    if "price_final" not in data:
        page_text = soup.get_text(" ", strip=True)
        prices_in_page = []
        for digits in KOREAN_PRICE_RE.findall(page_text):
            cleaned = digits.replace(",", "")
            if not cleaned:
                continue
            try:
                n = int(cleaned)
            except ValueError:
                continue
            if 1000 <= n <= 100_000_000:
                prices_in_page.append(n)

        if not data.get("discount_pct"):
            data["discount_pct"] = _percent(page_text)

        if data["discount_pct"] > 0 and len(prices_in_page) >= 2:
            sorted_prices = sorted(prices_in_page[:4])
            data["price_final"] = sorted_prices[0]
            data.setdefault("price_original", sorted_prices[-1])
        elif prices_in_page:
            data["price_final"] = prices_in_page[0]
            data.setdefault("price_original", None)

    # 7. 사이즈
    SIZE_LETTER_RE = re.compile(r"^(XXS|XS|S|M|L|XL|XXL|XXXL|OS|FREE)$")
    SIZE_BRACKET_RE = re.compile(r"\[\s*([A-Z0-9]{1,4})\s*\]")

    def _is_valid_size(s: str) -> bool:
        if not s:
            return False
        if SIZE_LETTER_RE.match(s):
            return True
        if s.isdigit():
            n = int(s)
            return 220 <= n <= 320 or 24 <= n <= 56
        return False

    size = None
    # Apollo size 필드 시도
    ap_size = ap.get("size") or ap.get("sizeInfo")
    if ap_size and isinstance(ap_size, str):
        candidate = ap_size.strip().upper()
        if _is_valid_size(candidate):
            size = candidate

    if not size:
        if brand_link:
            next_node = brand_link.find_next()
            if next_node:
                cand = next_node.get_text(strip=True) if hasattr(next_node, "get_text") else str(next_node).strip()
                if _is_valid_size(cand):
                    size = cand
        if not size and data.get("title"):
            for bm in SIZE_BRACKET_RE.finditer(data["title"]):
                cand = bm.group(1)
                if _is_valid_size(cand):
                    size = cand
                    break
        if not size and data.get("description"):
            dm = re.search(r"(?:사이즈|size|Size)\s*[:=]\s*([0-9A-Z]{1,5})", data["description"])
            if dm and _is_valid_size(dm.group(1)):
                size = dm.group(1)
    data["size"] = size

    # 8. SOLD 여부
    ap_sold = ap.get("isSold")
    if ap_sold is not None:
        data["is_sold"] = bool(ap_sold)
    else:
        pt = soup.get_text(" ", strip=True)
        data["is_sold"] = ("Sold" in pt) or ("판매완료" in pt)

    # 9. 등록 시점 (상대시간 폴백 / Apollo createdAt)
    ap_created = ap.get("createdAt") or ap.get("created_at")
    if ap_created:
        data["created_at"] = str(ap_created)
    else:
        pt = soup.get_text(" ", strip=True)
        time_match = re.search(r"(\d+)\s+(day|hour|minute|month|year)s?\s+ago", pt)
        if time_match:
            data["posted_relative"] = time_match.group(0)

    # 10. 셀러 정보 — seller_id는 항상 HTML /seller/ 링크의 shortcode 사용.
    # Apollo의 SellerNotMe:XXXXXXX 는 numeric ID라 DB 스키마와 맞지 않음.
    seller_link = soup.find("a", href=SELLER_URL_RE)
    if seller_link:
        sm = SELLER_URL_RE.search(seller_link.get("href", ""))
        if sm:
            data["seller_id"] = sm.group(1)
            data["seller_username"] = sm.group(2)

    # 11. 사진 수 — Apollo resizedSmallImages 배열 길이가 가장 정확
    ap_images = ap.get("resizedSmallImages") or ap.get("images") or []
    if isinstance(ap_images, list) and ap_images:
        data["n_photos"] = len(ap_images)
    else:
        photo_imgs = soup.find_all("img", src=re.compile(r"product/resized"))
        data["n_photos"] = max(1, min(len(photo_imgs), 10))

    # 12. 새 필드 — Apollo에서만 얻을 수 있는 구조화 정보
    # Apollo JSON은 snake_case 사용 (like_count, view_count)
    data["like_count"] = ap.get("like_count") or ap.get("likeCount") or 0
    data["view_count"] = ap.get("view_count") or ap.get("viewCount") or 0
    data["condition"] = ap.get("condition") or None  # NEW/GOOD_CONDITION/LIGHTLY_WORN/WORN
    ap_gender = ap.get("gender")
    data["gender"] = str(ap_gender) if ap_gender else None

    # likes/comments (하위호환)
    data["likes"] = data["like_count"]
    data["comments"] = 0

    return data


# ============================================================
# 셀러 페이지 파서 (메타 + 매물 카드 리스트)
# ============================================================

def parse_seller_page(html: str, url: str) -> dict | None:
    """셀러 페이지 → {seller 메타, listings: [...]}"""
    if not html:
        return None
    soup = BeautifulSoup(html, "lxml")

    sm = SELLER_URL_RE.search(url)
    if not sm:
        logger.warning(f"seller_id 추출 실패: {url}")
        return None

    seller = {
        "seller_id": sm.group(1),
        "username": sm.group(2),
    }

    page_text = soup.get_text(" ", strip=True)

    # 팔로워, 판매수, 평점 패턴 추출
    fol = re.search(r"팔로워\s*(\d+)", page_text)
    sales = re.search(r"판매수\s*(\d+)", page_text)
    rating = re.search(r"평점\s*([\d.]+)", page_text)

    seller["followers"] = int(fol.group(1)) if fol else None
    seller["total_sales"] = int(sales.group(1)) if sales else None
    seller["rating"] = float(rating.group(1)) if rating else None

    # 매물 카드 추출: 모든 /product/ 링크가 카드의 진입점
    listings = []
    seen = set()
    for a in soup.find_all("a", href=PRODUCT_URL_RE):
        href = a.get("href", "")
        pm = PRODUCT_URL_RE.search(href)
        if not pm:
            continue
        product_id = pm.group(1)
        if product_id in seen:
            continue
        seen.add(product_id)

        # 카드 컨테이너 — 이 a 태그의 가장 가까운 li/div 조상
        card = a.find_parent(["li", "div"])
        card_text = card.get_text(" ", strip=True) if card else ""

        # 카드 안의 브랜드, 사이즈, 가격, sold 등 추출
        brand_a = card.find("a", href=BRAND_URL_RE) if card else None
        brand = None
        if brand_a:
            bm = BRAND_URL_RE.search(brand_a.get("href", ""))
            if bm:
                brand = unquote(bm.group(1))
            else:
                brand = brand_a.get_text(strip=True)

        prices = []
        for digits in KOREAN_PRICE_RE.findall(card_text):
            cleaned = digits.replace(",", "")
            if not cleaned:
                continue
            try:
                n = int(cleaned)
            except ValueError:
                continue
            if 1000 <= n <= 100_000_000:
                prices.append(n)

        discount = _percent(card_text)
        is_sold = ("Sold" in card_text) or ("판매완료" in card_text)

        # 사이즈/제목 추출은 카드 텍스트에서 휴리스틱하게
        # 정밀한 값이 필요하면 매물 상세 페이지를 추가로 요청
        listings.append({
            "product_id": product_id,
            "seller_id": seller["seller_id"],
            "brand": brand,
            "price_final": prices[0] if prices else None,
            "price_original": max(prices) if discount > 0 and len(prices) > 1 else None,
            "discount_pct": discount,
            "is_sold": is_sold,
            "card_text": card_text[:200],  # 디버깅용
        })

    seller["listings"] = listings
    return seller


# ============================================================
# 검색/카테고리 페이지 파서 — 매물 ID 시드 발견용
# ============================================================

def parse_search_page(html: str) -> list[str]:
    """카테고리 검색 페이지 → product_id 리스트."""
    if not html:
        return []
    soup = BeautifulSoup(html, "lxml")
    seen = set()
    ids = []
    for a in soup.find_all("a", href=PRODUCT_URL_RE):
        m = PRODUCT_URL_RE.search(a.get("href", ""))
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            ids.append(m.group(1))
    return ids


# ============================================================
# 위시리스트 페이지 파서 (셀러가 공개 찜한 매물 목록)
# ============================================================

def parse_wishlist_page(html: str, owner_seller_id: str) -> list[dict]:
    """위시리스트 페이지 → [{product_id, rank}].

    매핑: ROOT_QUERY.seeUserLikes 리스트 ↔ HTML /product/ 카드 순서 1:1.
    길이 불일치면 안전을 위해 빈 리스트 반환.
    """
    if not html:
        return []
    soup = BeautifulSoup(html, "lxml")
    apollo = _extract_apollo(soup)

    # 1. seeUserLikes (Apollo 순서)
    refs: list[str] = []
    for k, v in apollo.get("ROOT_QUERY", {}).items():
        if k.startswith("seeUserLikes") and isinstance(v, list):
            refs = [item.get("__ref") for item in v if isinstance(item, dict) and item.get("__ref")]
            break

    # 2. HTML 카드 shortcode 순서 (dedup)
    shortcodes: list[str] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=PRODUCT_URL_RE):
        m = PRODUCT_URL_RE.search(a.get("href", ""))
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            shortcodes.append(m.group(1))

    # 3. Apollo 리스트가 비어있으면 위시리스트가 빈 것 (정상) → [] 반환
    if not refs:
        return []

    # 4. 길이 불일치 → 매핑 신뢰 불가 → 빈 리스트 (실패 표시는 호출자가)
    if len(refs) != len(shortcodes):
        logger.warning(
            f"위시 매핑 길이 불일치 owner={owner_seller_id}: refs={len(refs)} cards={len(shortcodes)}"
        )
        return []

    return [
        {"owner_seller_id": owner_seller_id, "product_id": sc, "rank": i}
        for i, sc in enumerate(shortcodes)
    ]


# ============================================================
# 리뷰 페이지 파서
# ============================================================

def parse_review_page(html: str, seller_id: str) -> list[dict]:
    """셀러 리뷰 페이지 → [review dict]."""
    if not html:
        return []
    soup = BeautifulSoup(html, "lxml")
    reviews = []

    # 리뷰는 대체로 별점(★ 또는 숫자) + 텍스트 블록 형태
    # 정확한 구조는 실제 페이지를 받아본 뒤 fine-tune 필요
    # 일단 보수적으로: 별점 블록을 찾고 그 인접 텍스트를 review_text로 잡음
    for star_block in soup.find_all(string=re.compile(r"^[1-5](\.\d)?$")):
        try:
            rating = float(star_block.strip())
            if not 1 <= rating <= 5:
                continue
        except ValueError:
            continue
        # 별점 옆 텍스트 추출 — 부모 또는 형제에서
        parent = star_block.parent
        text = parent.find_next("p") or parent.find_next("div")
        review_text = text.get_text(strip=True)[:1000] if text else ""
        if review_text:
            reviews.append({
                "seller_id": seller_id,
                "review_rating": int(rating),
                "review_text": review_text,
            })

    return reviews
