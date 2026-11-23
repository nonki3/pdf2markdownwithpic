#!/usr/bin/env python3
"""PDF を画像付き Markdown に変換するツール.

出力:
    <出力先>/<名前>.md
    <出力先>/<名前>_images/figure-1.jpeg, figure-2.png, ...

主な機能:
    - フォントサイズ（または PDF のしおり）から見出しレベル (#, ##, ###) を推定
    - 埋め込み画像を元の形式のまま抽出し、本文中の位置に ![図N](...) として挿入
    - 改行で分断された日本語の段落を結合
    - PDF 内リンク（目次など）を Markdown の見出しアンカーリンクに変換
    - 繰り返し現れるヘッダー/フッター・ページ番号の除去
    - (任意) ベクター描画の図をラスタ画像として切り出し (--vector-figures)

使い方:
    python pdf2md.py input.pdf
    python pdf2md.py input.pdf -o out/book.md --toc
"""
from __future__ import annotations

import argparse
import collections
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

BULLET_CHARS = "●○◎◆◇■□▲△▼▽•※★☆➢►▶"
# 行頭にあっても折り返し（例: 「ロビン」+「・フッド」）の可能性がある記号
WEAK_BULLET_CHARS = "・-－*＊"
# 和文文字（全角記号・かな・漢字・全角英数）
CJK = "　-ヿ㐀-䶿一-鿿豈-﫿＀-￯"
SPACE_BEFORE_CJK = re.compile(rf"(?<=[!-~]) +(?=[{CJK}])")
LIST_NUMBER = re.compile(r"^(\d+|[０-９]+)[.．)）](?![0-9０-９])")
SENTENCE_END = re.compile(r"[。．！？!?][」』）)】]*$")
LATIN_WORD = re.compile(r"[A-Za-z0-9.,:;/%\-_~?=&#@()+'\"]+")
LINK_ONLY = re.compile(r"^\[[^\]]*\]\([^)]*\)$")
LINK_PLACEHOLDER = re.compile(r"\x00LINK(\d+):(-?\d+(?:\.\d+)?)\x00")


# --------------------------------------------------------------------------- #
# データ構造
# --------------------------------------------------------------------------- #
@dataclass
class Line:
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str  # Markdown 装飾済み（リンク・上付き）
    plain: str  # プレーンテキスト
    size: float
    raw_indent: bool = False  # 行頭が全角/半角スペースで始まる
    local_right: float = 0.0  # 周辺行から推定した本文の右端

    @property
    def is_blank(self) -> bool:
        return not self.plain.strip()


@dataclass
class Figure:
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    path: str  # Markdown からの相対パス
    alt: str
    vector: bool = False  # ベクター描画を切り出した図


@dataclass
class Heading:
    page: int
    y0: float
    level: int
    text: str
    slug: str = ""


@dataclass
class Stats:
    body_size: float
    right_edge: float
    left_edge: float
    page_right: dict[int, float] = field(default_factory=dict)
    page_left: dict[int, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# ユーティリティ
# --------------------------------------------------------------------------- #
def slugify(text: str) -> str:
    """GitHub / VS Code と同じ規則の見出しアンカーを作る."""
    s = re.sub(r"<[^>]+>", "", text)
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
    s = s.strip().lower()
    s = re.sub(r"[^\w\- ]", "", s)
    return s.replace(" ", "-")


def join_text(a: str, b: str) -> str:
    """2 行を連結する。欧文同士の場合のみ空白を入れる."""
    if not a:
        return b
    if not b:
        return a
    if a.endswith("-") and re.match(r"[a-z]", b) and re.search(r"[A-Za-z]-$", a):
        return a[:-1] + b  # ハイフネーション解除
    if re.search(r"[A-Za-z0-9,.;:!?)\]'\"]$", a) and re.match(r"[A-Za-z0-9(\['\"]", b):
        return a + " " + b
    return a + b


def normalize_spaces(text: str) -> str:
    """英数字の直後に PDF が挿入した半角スペース（例: "2020 年"）を取り除く."""
    return SPACE_BEFORE_CJK.sub("", text)


def escape_md_line(text: str) -> str:
    """本文行頭が Markdown 記法として解釈されないようにする."""
    if re.match(r"#{1,6}(\s|$)", text):
        return "\\" + text
    if re.match(r"[>]", text):
        return "\\" + text
    return text


def md_path(p: str) -> str:
    return f"<{p}>" if re.search(r"[\s()]", p) else p


def rect_intersect_ratio(a: tuple, b: tuple) -> float:
    """a の面積に対する a∩b の割合."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    area = max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)
    return (x1 - x0) * (y1 - y0) / area


# --------------------------------------------------------------------------- #
# 変換器
# --------------------------------------------------------------------------- #
class PdfToMarkdown:
    def __init__(
        self,
        pdf_path: Path,
        md_path_: Path,
        *,
        max_heading_level: int = 3,
        min_image_size: float = 24.0,
        vector_figures: bool = False,
        dpi: int = 150,
        toc: bool = False,
        cover_alt: str = "表紙",
        figure_alt: str = "図{n}",
        use_outline: str = "auto",
        keep_spaces: bool = False,
        verbose: bool = True,
    ) -> None:
        self.pdf_path = pdf_path
        self.md_file = md_path_
        self.images_dirname = f"{md_path_.stem}_images"
        self.images_dir = md_path_.parent / self.images_dirname
        self.max_heading_level = max_heading_level
        self.min_image_size = min_image_size
        self.vector_figures = vector_figures
        self.dpi = dpi
        self.toc = toc
        self.cover_alt = cover_alt
        self.figure_alt = figure_alt
        self.use_outline = use_outline
        self.keep_spaces = keep_spaces
        self.verbose = verbose

        self.doc = pymupdf.open(pdf_path)
        self.figure_count = 0
        self.saved_xrefs: dict[int, str] = {}
        self.headings: list[Heading] = []

    # ------------------------------------------------------------------ log
    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg, file=sys.stderr)

    # ------------------------------------------------------------ 行の抽出
    def extract_lines(self, page: pymupdf.Page, pno: int) -> list[Line]:
        links = []
        for lk in page.get_links():
            r = lk.get("from")
            if r is None:
                continue
            if lk.get("kind") == pymupdf.LINK_GOTO and lk.get("page", -1) >= 0:
                to = lk.get("to")
                y = to.y if to is not None else 0
                links.append((tuple(r), f"\x00LINK{lk['page']}:{y:.1f}\x00"))
            elif lk.get("kind") == pymupdf.LINK_URI and lk.get("uri"):
                links.append((tuple(r), lk["uri"]))

        def link_for(bbox: tuple) -> str | None:
            for r, target in links:
                if rect_intersect_ratio(bbox, r) > 0.5:
                    return target
            return None

        lines: list[Line] = []
        d = page.get_text("dict", sort=True, flags=pymupdf.TEXTFLAGS_TEXT)
        for block in d["blocks"]:
            if block.get("type") != 0:
                continue
            for ln in block["lines"]:
                spans = ln["spans"]
                if not spans:
                    continue
                plain = "".join(s["text"] for s in spans)
                # Markdown 用テキスト（上付き・リンク）
                pieces: list[tuple[str | None, str]] = []
                for s in spans:
                    t = s["text"]
                    if s["flags"] & 1 and t.strip():  # superscript
                        t = f"<sup>{t.strip()}</sup>"
                    target = link_for(tuple(s["bbox"])) if links else None
                    if pieces and pieces[-1][0] == target:
                        pieces[-1] = (target, pieces[-1][1] + t)
                    else:
                        pieces.append((target, t))
                parts = []
                for target, t in pieces:
                    if target and t.strip():
                        lead = t[: len(t) - len(t.lstrip())]
                        trail = t[len(t.rstrip()):]
                        parts.append(f"{lead}[{t.strip()}]({target}){trail}")
                    else:
                        parts.append(t)
                text = "".join(parts)
                if not self.keep_spaces:
                    text = normalize_spaces(text)
                    plain = normalize_spaces(plain)

                sizes = [s["size"] for s in spans if s["text"].strip() and not s["flags"] & 1]
                if not sizes:
                    sizes = [s["size"] for s in spans]
                x0, y0, x1, y1 = ln["bbox"]
                lines.append(
                    Line(
                        page=pno,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        text=text.strip(),
                        plain=plain.strip(),
                        size=round(max(sizes), 1),
                        raw_indent=bool(re.match(r"^[ 　]", plain)) and bool(plain.strip()),
                    )
                )
        return lines

    # ------------------------------------------------------------ 統計
    def compute_stats(self, pages: list[list[Line]]) -> Stats:
        size_chars: collections.Counter = collections.Counter()
        for lines in pages:
            for ln in lines:
                if not ln.is_blank:
                    size_chars[ln.size] += len(ln.plain)
        body = size_chars.most_common(1)[0][0] if size_chars else 12.0

        rights: collections.Counter = collections.Counter()
        lefts: collections.Counter = collections.Counter()
        page_right: dict[int, float] = {}
        page_left: dict[int, float] = {}
        for pno, lines in enumerate(pages):
            body_lines = [ln for ln in lines if abs(ln.size - body) < 0.6 and len(ln.plain) >= 8]
            for ln in body_lines:
                rights[round(ln.x1)] += 1
                lefts[round(ln.x0)] += 1
            if len(body_lines) >= 3:
                # はみ出した行に引きずられないよう、2 行以上が揃う最も右の位置を右端とする
                c = collections.Counter(round(ln.x1) for ln in body_lines)
                common = [x for x, k in c.items() if k >= 2]
                page_right[pno] = max(common) if common else max(ln.x1 for ln in body_lines)
                page_left[pno] = min(ln.x0 for ln in body_lines)
        right = rights.most_common(1)[0][0] if rights else 0
        left = lefts.most_common(1)[0][0] if lefts else 0
        # 短い行しかないページ（箇条書きだけ等）では右端を推定できないので全体の値を使う
        page_right = {p: r for p, r in page_right.items() if abs(r - right) <= body * 3}
        return Stats(body, right, left, page_right, page_left)

    @staticmethod
    def assign_local_right(pages: list[list[Line]], stats: Stats, window: int = 3) -> None:
        """前後の行から本文の右端を推定する（同じページ内でも組幅が変わる PDF 対策）."""
        for pno, lines in enumerate(pages):
            page_right = stats.page_right.get(pno, stats.right_edge)
            for i, ln in enumerate(lines):
                near = [
                    o.x1
                    for o in lines[max(0, i - window): i + window + 1]
                    if o is not ln
                    and not o.is_blank
                    and abs(o.size - ln.size) < 0.6
                    and o.x1 > page_right - o.size * 3
                ]
                if near:
                    near.sort()
                    ln.local_right = near[len(near) // 2]  # 中央値
                else:
                    ln.local_right = page_right

    def heading_levels(self, pages: list[list[Line]], body: float) -> dict[float, int]:
        """本文より大きいフォントサイズを見出しレベルに割り当てる."""
        counter: collections.Counter = collections.Counter()
        for lines in pages:
            for ln in lines:
                if not ln.is_blank and ln.size >= body * 1.1 and len(ln.plain) <= 120:
                    counter[ln.size] += 1
        sizes = sorted(counter, reverse=True)
        # 近いサイズは同じレベルにまとめる
        groups: list[list[float]] = []
        for s in sizes:
            if groups and groups[-1][-1] - s < 0.6:
                groups[-1].append(s)
            else:
                groups.append([s])
        levels: dict[float, int] = {}
        for i, g in enumerate(groups[: self.max_heading_level]):
            for s in g:
                levels[s] = i + 1
        return levels

    def detect_headers_footers(self, pages: list[list[Line]]) -> set[tuple[int, int]]:
        """複数ページの上下余白に同じ文字列が出るものをヘッダー/フッターとみなす."""
        n = len(pages)
        removed: set[tuple[int, int]] = set()
        if n < 3:
            # ページ番号だけは除去
            pass
        keys: collections.defaultdict = collections.defaultdict(list)
        for pno, lines in enumerate(pages):
            h = self.doc[pno].rect.height
            for idx, ln in enumerate(lines):
                if ln.is_blank:
                    continue
                in_margin = ln.y1 < h * 0.1 or ln.y0 > h * 0.9
                if not in_margin:
                    continue
                if re.fullmatch(r"[-–—\s]*\d{1,4}[-–—\s]*|[ivxlcdm]{1,6}", ln.plain, re.I):
                    removed.add((pno, idx))  # ページ番号
                    continue
                key = (re.sub(r"\d+", "#", ln.plain), round(ln.y0 / 5), ln.y0 < h / 2)
                keys[key].append((pno, idx))
        threshold = max(3, int(n * 0.3))
        for key, occ in keys.items():
            if len({p for p, _ in occ}) >= threshold:
                removed.update(occ)
        return removed

    # ------------------------------------------------------------ 画像
    def _next_figure(self, page_no: int) -> tuple[int, str]:
        self.figure_count += 1
        n = self.figure_count
        alt = self.cover_alt if (n == 1 and page_no == 0 and self.cover_alt) else self.figure_alt.format(n=n)
        return n, alt

    def _save_xref(self, xref: int, n: int) -> str | None:
        info = self.doc.extract_image(xref)
        if not info:
            return None
        ext = info.get("ext", "png")
        smask = info.get("smask", 0)
        cs = info.get("colorspace", 3)
        name_base = f"figure-{n}"
        if ext in ("jpeg", "jpg", "png") and not smask and cs in (1, 3):
            fname = f"{name_base}.{'jpeg' if ext == 'jpg' else ext}"
            (self.images_dir / fname).write_bytes(info["image"])
            return fname
        # 透過マスク・CMYK・JPX 等は PNG に変換
        pix = pymupdf.Pixmap(self.doc, xref)
        if smask:
            try:
                pix = pymupdf.Pixmap(pix, pymupdf.Pixmap(self.doc, smask))
            except Exception:
                pass
        if pix.colorspace and pix.colorspace.n not in (1, 3):
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        fname = f"{name_base}.png"
        pix.save(self.images_dir / fname)
        return fname

    def _save_clip(self, page: pymupdf.Page, rect: pymupdf.Rect, n: int) -> str:
        pix = page.get_pixmap(clip=rect, dpi=self.dpi)
        fname = f"figure-{n}.png"
        pix.save(self.images_dir / fname)
        return fname

    def extract_figures(self, page: pymupdf.Page, pno: int, xref_pages: dict[int, int]) -> list[Figure]:
        # 候補を集めてから読み順 (上→下, 左→右) に並べ、その順で通し番号を振る
        candidates: list[tuple[pymupdf.Rect, int, bool]] = []  # (bbox, xref, is_vector)
        seen_on_page: set[int] = set()
        for info in page.get_image_info(xrefs=True):
            bbox = pymupdf.Rect(info["bbox"]) & page.rect
            if bbox.is_empty or bbox.width < self.min_image_size or bbox.height < self.min_image_size:
                continue
            xref = info.get("xref", 0)
            # 多数のページに出る画像はロゴ・装飾とみなしスキップ
            if xref and self.doc.page_count >= 5 and xref_pages.get(xref, 0) > max(3, self.doc.page_count * 0.3):
                continue
            if xref and xref in seen_on_page:
                continue
            seen_on_page.add(xref)
            candidates.append((bbox, xref, False))
        if self.vector_figures:
            raster = [tuple(b) for b, _, _ in candidates]
            candidates.extend((r, 0, True) for r in self.find_vector_figures(page, raster))
        candidates.sort(key=lambda c: (round(c[0].y0), c[0].x0))

        figures: list[Figure] = []
        for bbox, xref, is_vector in candidates:
            if xref and xref in self.saved_xrefs:  # 既出の画像は同じファイルを参照
                fname = self.saved_xrefs[xref]
                alt = self.figure_alt.format(n=Path(fname).stem.split("-")[-1])
            else:
                n, alt = self._next_figure(pno)
                fname = None
                if xref:
                    try:
                        fname = self._save_xref(xref, n)
                    except Exception as e:  # 壊れた画像などはページから切り出す
                        self.log(f"  [warn] p.{pno + 1} xref={xref}: {e}; 領域をレンダリングします")
                if fname is None:
                    clip = bbox + (-2, -2, 2, 2) if is_vector else bbox
                    fname = self._save_clip(page, clip & page.rect, n)
                if xref:
                    self.saved_xrefs[xref] = fname
            figures.append(
                Figure(pno, bbox.x0, bbox.y0, bbox.x1, bbox.y1, f"{self.images_dirname}/{fname}", alt, vector=is_vector)
            )
        return figures

    def find_vector_figures(self, page: pymupdf.Page, raster: list[tuple]) -> list[pymupdf.Rect]:
        """ベクター描画（グラフ・図形）の領域を返す. 埋め込み画像と重なるものは除く."""
        try:
            clusters = page.cluster_drawings()
        except Exception:
            return []
        min_side = max(self.min_image_size, 40)
        out = []
        for r in clusters:
            r = pymupdf.Rect(r) & page.rect
            if r.width < min_side or r.height < min_side:
                continue
            if r.width > page.rect.width * 0.95 and r.height > page.rect.height * 0.95:
                continue  # ページ全体の枠・背景
            if any(rect_intersect_ratio(tuple(r), b) > 0.5 for b in raster):
                continue
            out.append(r)
        return out

    # ------------------------------------------------------------ しおり
    def outline_headings(self) -> dict[int, list[tuple[int, str]]]:
        by_page: dict[int, list[tuple[int, str]]] = collections.defaultdict(list)
        for level, title, page in self.doc.get_toc(simple=True):
            if page >= 1 and level <= self.max_heading_level:
                by_page[page - 1].append((level, re.sub(r"\s+", "", title)))
        return by_page

    # ------------------------------------------------------------ 本体
    def convert(self) -> Path:
        self.images_dir.mkdir(parents=True, exist_ok=True)
        n = self.doc.page_count
        self.log(f"{self.pdf_path.name}: {n} ページ")

        pages = [self.extract_lines(self.doc[p], p) for p in range(n)]
        stats = self.compute_stats(pages)
        self.assign_local_right(pages, stats)
        removed = self.detect_headers_footers(pages)

        outline = self.outline_headings() if self.use_outline != "never" else {}
        levels = self.heading_levels(pages, stats.body_size) if self.use_outline != "always" else {}
        # auto: フォントサイズで見出しが見つからない場合だけしおりを使う
        use_outline = bool(outline) and (self.use_outline == "always" or (self.use_outline == "auto" and not levels))
        self.log(
            f"本文サイズ {stats.body_size}pt / 見出し: "
            + ("PDFのしおり" if use_outline else ", ".join(f"{s}pt→{'#' * lv}" for s, lv in sorted(levels.items(), key=lambda x: x[1])) or "なし")
        )

        # xref ごとの出現ページ数
        xref_pages: collections.Counter = collections.Counter()
        for p in range(n):
            for x in {img[0] for img in self.doc[p].get_images(full=True)}:
                xref_pages[x] += 1

        blocks: list[str] = []  # Markdown ブロック
        para: list[Line] = []
        last_heading: tuple[Line, int] | None = None  # (line, blocks index)

        def heading_level(ln: Line) -> int:
            if ln.is_blank or len(ln.plain) > 120:
                return 0
            if use_outline:
                key = re.sub(r"\s+", "", ln.plain)
                for lv, title in outline.get(ln.page, []):
                    if key and (key == title or (len(key) >= 4 and title.startswith(key))):
                        return lv
                return 0
            return levels.get(ln.size, 0)

        def is_list_item(ln: Line) -> bool:
            return ln.plain[:1] in BULLET_CHARS or bool(LIST_NUMBER.match(ln.plain))

        def wraps_into(prev: Line, cur: Line) -> bool:
            """prev の右側の余白に cur の先頭語が収まらない = 折り返しによる改行."""
            gap = prev.local_right - prev.x1
            if SENTENCE_END.search(prev.plain) and gap >= prev.size * 0.9:
                return False  # 文末で行が終わり、右端まで達していない
            m = LATIN_WORD.match(cur.plain)
            token_w = len(m.group()) * cur.size * 0.6 if m else cur.size
            return gap < token_w + cur.size * 0.5

        def starts_new(first: Line, prev: Line, cur: Line) -> bool:
            if abs(prev.size - cur.size) >= 0.6:
                return True
            if is_list_item(cur) or cur.raw_indent:
                return True
            if cur.plain[:1] in WEAK_BULLET_CHARS:
                # 「・」で始まる段落の途中に「・」行が来たら箇条書きの次の項目
                if first.plain[:1] == cur.plain[:1] or not wraps_into(prev, cur):
                    return True
            if LINK_ONLY.match(prev.text) or LINK_ONLY.match(cur.text):
                return True  # 目次などのリンク行は 1 行ずつ
            left = stats.page_left.get(cur.page, stats.left_edge)
            if prev.x0 > left + prev.size * 2.5:
                return True  # 右寄せ・中央寄せの行
            if cur.x0 > left + cur.size * 0.8:
                # 箇条書きのぶら下げインデントは継続とみなす
                if not (is_list_item(first) and cur.x0 <= first.x0 + cur.size * 3):
                    return True
            if prev.page == cur.page and cur.y0 - prev.y1 > cur.size * 0.9:
                return True
            return not wraps_into(prev, cur)

        def flush() -> None:
            nonlocal para
            if para:
                text = ""
                for ln in para:
                    text = join_text(text, ln.text)
                blocks.append(escape_md_line(text))
            para = []

        for pno in range(n):
            if self.verbose and (pno + 1) % 50 == 0:
                self.log(f"  {pno + 1}/{n} ページ処理済み")
            page = self.doc[pno]
            figures = self.extract_figures(page, pno, xref_pages)
            fig_rects = [(f.x0, f.y0, f.x1, f.y1) for f in figures if f.vector]

            items: list[tuple[float, float, int, object]] = []
            for idx, ln in enumerate(pages[pno]):
                if (pno, idx) in removed:
                    continue
                # ベクター図として切り出した領域内の文字は図に含まれるので除外
                if fig_rects and not ln.is_blank and any(
                    rect_intersect_ratio((ln.x0, ln.y0, ln.x1, ln.y1), r) > 0.8 for r in fig_rects
                ):
                    continue
                items.append((ln.y0, ln.x0, 1, ln))
            for f in figures:
                items.append((f.y0, f.x0, 0, f))
            items.sort(key=lambda t: (round(t[0], 0), t[1], t[2]))

            for _, _, _, item in items:
                if isinstance(item, Figure):
                    flush()
                    last_heading = None
                    blocks.append(f"![{item.alt}]({md_path(item.path)})")
                    continue
                ln: Line = item
                if ln.is_blank:
                    flush()
                    continue
                lv = heading_level(ln)
                if lv:
                    flush()
                    # 複数行にまたがる見出しを結合
                    if (
                        last_heading
                        and last_heading[1] == len(blocks) - 1
                        and last_heading[0].page == ln.page
                        and abs(last_heading[0].size - ln.size) < 0.6
                        and 0 <= ln.y0 - last_heading[0].y1 < ln.size * 0.8
                        and last_heading[0].x1 >= last_heading[0].local_right - ln.size * 2  # 折り返した見出し
                        and not use_outline
                    ):
                        h = self.headings[-1]
                        h.text = join_text(h.text, ln.plain)
                        blocks[-1] = f"{'#' * h.level} {h.text}"
                        last_heading = (ln, len(blocks) - 1)
                        continue
                    text = re.sub(r"\s+", " ", ln.plain).strip()
                    self.headings.append(Heading(ln.page, ln.y0, lv, text))
                    blocks.append(f"{'#' * lv} {text}")
                    last_heading = (ln, len(blocks) - 1)
                    continue
                last_heading = None
                if para and starts_new(para[0], para[-1], ln):
                    flush()
                para.append(ln)
            # ページ末の段落はページをまたいで継続しうるので flush しない
        flush()

        # アンカー生成
        counts: collections.Counter = collections.Counter()
        for h in self.headings:
            base = slugify(h.text)
            h.slug = base if counts[base] == 0 else f"{base}-{counts[base]}"
            counts[base] += 1

        md = "\n\n".join(blocks)
        md = LINK_PLACEHOLDER.sub(lambda m: "#" + self.resolve_anchor(int(m.group(1)), float(m.group(2))), md)
        md = re.sub(r"\[([^\]]*)\]\(#\)", r"\1", md)  # 解決できなかったリンクは文字だけ残す

        if self.toc and self.headings:
            toc_lines = ["## 目次", ""]
            base_lv = min(h.level for h in self.headings)
            for h in self.headings:
                toc_lines.append(f"{'  ' * (h.level - base_lv)}- [{h.text}](#{h.slug})")
            md = "\n".join(toc_lines) + "\n\n" + md

        self.md_file.parent.mkdir(parents=True, exist_ok=True)
        self.md_file.write_text(md.rstrip() + "\n", encoding="utf-8")
        if self.figure_count == 0:
            try:
                self.images_dir.rmdir()
            except OSError:
                pass
        self.log(f"完了: {self.md_file} (見出し {len(self.headings)} 件, 画像 {self.figure_count} 枚)")
        return self.md_file

    def resolve_anchor(self, page: int, y: float) -> str:
        """リンク先 (ページ, y座標) に最も近い見出しのアンカーを返す."""
        on_page = [h for h in self.headings if h.page == page]
        after = [h for h in on_page if h.y0 >= y - 10]
        if after:
            return after[0].slug
        before = [h for h in self.headings if h.page < page or (h.page == page and h.y0 < y)]
        if on_page:
            return on_page[-1].slug
        if before:
            return before[-1].slug
        return ""


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PDF を画像付き Markdown に変換します。")
    ap.add_argument("pdf", type=Path, help="入力 PDF ファイル")
    ap.add_argument("-o", "--output", type=Path, help="出力 .md ファイル (既定: PDF と同じ場所・同じ名前)")
    ap.add_argument("--max-heading-level", type=int, default=3, choices=range(1, 7), help="見出しとして扱う最大レベル (既定: 3)")
    ap.add_argument("--min-image-size", type=float, default=24.0, help="これより小さい画像 (pt) は無視 (既定: 24)")
    ap.add_argument("--vector-figures", action="store_true", help="ベクター描画の図も画像として切り出す")
    ap.add_argument("--dpi", type=int, default=150, help="図をレンダリングするときの解像度 (既定: 150)")
    ap.add_argument("--toc", action="store_true", help="先頭に見出しから生成した目次を追加")
    ap.add_argument("--outline", choices=["auto", "always", "never"], default="auto",
                    help="見出し判定に PDF のしおりを使うか (auto: フォントサイズで判定できない時だけ / always: しおりのみ / never: 使わない)")
    ap.add_argument("--cover-alt", default="表紙", help="1ページ目の最初の画像の代替テキスト (空文字で無効)")
    ap.add_argument("--figure-alt", default="図{n}", help="画像の代替テキスト書式 ({n} は通し番号)")
    ap.add_argument("--keep-spaces", action="store_true", help="和文と英数字の間の半角スペースを残す")
    ap.add_argument("-q", "--quiet", action="store_true", help="進捗を表示しない")
    args = ap.parse_args(argv)

    if not args.pdf.is_file():
        ap.error(f"ファイルが見つかりません: {args.pdf}")
    out = args.output or args.pdf.with_suffix(".md")
    if out.suffix.lower() != ".md":
        out = out / f"{args.pdf.stem}.md" if out.is_dir() or not out.suffix else out.with_suffix(".md")

    PdfToMarkdown(
        args.pdf,
        out,
        max_heading_level=args.max_heading_level,
        min_image_size=args.min_image_size,
        vector_figures=args.vector_figures,
        dpi=args.dpi,
        toc=args.toc,
        cover_alt=args.cover_alt,
        figure_alt=args.figure_alt,
        use_outline=args.outline,
        keep_spaces=args.keep_spaces,
        verbose=not args.quiet,
    ).convert()
    return 0


if __name__ == "__main__":
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    sys.exit(main())
