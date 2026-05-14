"""분석 공통 유틸 — 한글 폰트, 결과 저장 경로, 시각화 헬퍼."""
import json
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def ensure_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def setup_korean_font():
    """matplotlib에서 한글 깨짐 방지.

    macOS는 'AppleGothic', Linux는 'NanumGothic' 또는 'Malgun Gothic'.
    실패해도 분석 자체는 진행되도록 try/except.
    """
    try:
        import matplotlib
        import matplotlib.font_manager as fm
        import platform

        candidates = []
        sys = platform.system()
        if sys == "Darwin":
            candidates = ["AppleGothic", "Apple SD Gothic Neo"]
        elif sys == "Windows":
            candidates = ["Malgun Gothic", "맑은 고딕"]
        else:
            candidates = ["NanumGothic", "Noto Sans CJK KR", "DejaVu Sans"]

        available = {f.name for f in fm.fontManager.ttflist}
        for name in candidates:
            if name in available:
                matplotlib.rcParams["font.family"] = name
                matplotlib.rcParams["axes.unicode_minus"] = False
                return name
        # 폴백: 한글 안 깨질 수도 있지만 적어도 음수 부호는 정상
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass
    return None


# ============================================================
# 결과 저장 — JSON 으로 가설별 결과를 한 곳에 모음
# ============================================================

def save_result(name: str, payload: dict):
    """가설 검정 결과를 JSON 으로 저장.

    Args:
        name: 'h1_clustering', 'h2_anova', 'h3_xgb' 등
        payload: 결과 dict — 통계량, p-value, 클러스터 라벨 등
    """
    ensure_dirs()
    payload = {
        **payload,
        "_saved_at": datetime.utcnow().isoformat(),
    }
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return path


def load_result(name: str) -> dict:
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_figure(fig, name: str):
    """matplotlib figure 저장."""
    ensure_dirs()
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


# ============================================================
# 콘솔 헬퍼
# ============================================================

def section(title: str):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def bullet(label: str, value, indent: int = 2):
    print(" " * indent + f"• {label:30} {value}")
