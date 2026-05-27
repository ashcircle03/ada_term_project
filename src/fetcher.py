"""HTTP 요청 책임 모듈.

핵심 기능:
- 랜덤 지터를 곁들인 rate limit
- 실패 시 exponential backoff 재시도
- 모든 raw HTML을 디스크에 보관 (파싱 로직 변경 시 재파싱 가능)
- 4xx는 즉시 포기, 5xx는 재시도
"""
import logging
import random
import time
import hashlib
import requests
from pathlib import Path
from . import config


logger = logging.getLogger(__name__)


class Fetcher:
    """세션 재사용 + rate limit + 재시도 래퍼."""

    def __init__(self, save_html: bool = True):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": config.USER_AGENT,
            "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.7",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        })
        self.save_html = save_html
        if save_html and not config.RAW_HTML_DIR.is_dir():
            config.RAW_HTML_DIR.resolve().mkdir(parents=True, exist_ok=True)

    def _delay(self):
        """매 요청 사이 랜덤 지연. 일정한 패턴이면 봇 탐지 쉬움."""
        wait = random.uniform(config.REQUEST_DELAY_MIN, config.REQUEST_DELAY_MAX)
        time.sleep(wait)

    def _save_raw(self, url: str, html: str) -> Path:
        """URL의 SHA256을 파일명으로 raw HTML 보존."""
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        path = config.RAW_HTML_DIR / f"{url_hash}.html"
        path.write_text(html, encoding="utf-8")
        return path

    def get(self, url: str) -> str | None:
        """
        URL을 가져와 HTML 텍스트 반환. 실패 시 None.

        재시도 정책:
        - 5xx, 네트워크 오류: backoff 후 재시도 (최대 MAX_RETRIES회)
        - 4xx: 즉시 포기 (404, 403 등은 재시도해도 의미 없음)
        - 200: HTML 반환 + raw 보존
        """
        for attempt in range(config.MAX_RETRIES):
            self._delay()
            try:
                resp = self.session.get(url, timeout=config.TIMEOUT)
            except requests.RequestException as e:
                logger.warning(f"네트워크 오류 (시도 {attempt+1}/{config.MAX_RETRIES}) {url}: {e}")
                self._backoff(attempt)
                continue

            if resp.status_code == 200:
                if self.save_html:
                    self._save_raw(url, resp.text)
                return resp.text

            if 400 <= resp.status_code < 500:
                logger.error(f"클라이언트 오류 {resp.status_code} (재시도 안 함) {url}")
                return None

            # 5xx
            logger.warning(f"서버 오류 {resp.status_code} (시도 {attempt+1}/{config.MAX_RETRIES}) {url}")
            self._backoff(attempt)

        logger.error(f"최대 재시도 초과: {url}")
        return None

    @staticmethod
    def _backoff(attempt: int):
        """지수 백오프 + 지터."""
        wait = config.BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 2)
        logger.info(f"  → {wait:.1f}초 대기 후 재시도")
        time.sleep(wait)
