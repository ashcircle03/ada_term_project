"""percent-format .py → .ipynb 변환기 (jupytext 없이, nbformat만 사용).

규칙:
  '# %%'            → 코드 셀 시작
  '# %% [markdown]' → 마크다운 셀 시작 (이후 줄의 선행 '# ' 제거)

사용:  python -m analysis.nbmake notebooks/00_eda.py
"""
import sys
from pathlib import Path

import nbformat as nbf


def convert(py_path: str) -> Path:
    src = Path(py_path).read_text(encoding="utf-8")
    nb = nbf.v4.new_notebook()
    cells, cur, is_md = [], [], False

    def flush():
        if not cur:
            return
        text = "\n".join(cur).strip("\n")
        if not text.strip():
            return
        if is_md:
            lines = [ln[2:] if ln.startswith("# ") else ln.lstrip("#")
                     for ln in text.split("\n")]
            cells.append(nbf.v4.new_markdown_cell("\n".join(lines)))
        else:
            cells.append(nbf.v4.new_code_cell(text))

    for line in src.split("\n"):
        if line.startswith("# %%"):
            flush()
            cur, is_md = [], ("[markdown]" in line)
        else:
            cur.append(line)
    flush()

    nb.cells = cells
    out = Path(py_path).with_suffix(".ipynb")
    nbf.write(nb, str(out))
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print("wrote", convert(p))
