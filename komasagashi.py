# -*- coding: utf-8 -*-
"""
komasagashi.py  (KomaSagashi / コマ探し)

フォルダ内の画像を OCR してテキストをインデックス化し、
キーワード検索でヒットした画像へ即ジャンプできる検索システム。

OCR 結果をファイル名に埋め込む方式と違い、全文を SQLite に保持するため
  ・255 文字のファイル名制限を受けない
  ・元のファイル名を保ったまま
  ・何度でも高速に再検索できる
  ・検索ヒットからその場で元画像を開ける
  ・ヒットした吹き出しを含むコマを自動で囲み、枠を微調整（補正で一回り大きく・小さくも可）して
    クリップボードへコピー・画像で保存できる（出力倍率を指定可、プレビューは拡大縮小可）
  ・一度インデックスを作ったフォルダを記憶し、次回は一覧から選べる
  ・（任意）横書きの文字しかないページ（目次・扉など）を PaddleOCR で読み直す
  ・検索で表記ゆれ（カタカナ/ひらがな・濁点・全角/半角など）を無視できる
  ・フォルダ内の書庫（zip / cbz / rar / cbr / 7z / cb7）の中の画像も、取り出さずに検索できる
OCR エンジンは mokuro / manga-ocr-base を使用する。
コマの検出は OpenCV による画像処理（コマ間の白い隙間を手がかりにする）。

【セットアップ】
    Windows なら setup.bat をダブルクリック → run.bat で起動。
    手動で入れる場合の手順は README.md を参照。
"""
from __future__ import annotations

# 修正・機能追加のたびに 0.01 ずつ上げる。変更内容は CHANGELOG.md に書く。
__version__ = "1.08"

import hashlib
import importlib.util
import io
import json
import os
import queue
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
import unicodedata
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Callable, Iterator

from PIL import Image

try:
    from PIL import ImageTk
    _IMAGETK_AVAILABLE = True
except ImportError:
    _IMAGETK_AVAILABLE = False

try:
    # どちらも mokuro の依存として入る。無ければコマ検出だけ無効にする。
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

# 画像上の矩形 (x1, y1, x2, y2)。座標は元画像の px。
Box = tuple[int, int, int, int]

# ── 画像拡張子 ────────────────────────────────────────────────────────────
_IMAGE_EXTS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}
)
# 書庫（拡張子 → 形式）。中の画像を取り出さずに OCR・検索・プレビューする。
_ARCHIVE_EXTS: dict[str, str] = {
    ".zip": "zip", ".cbz": "zip",
    ".rar": "rar", ".cbr": "rar",
    ".7z": "7z", ".cb7": "7z",
}
# 書庫内の画像は「書庫の相対パス::書庫内のパス」で記録する（Windows のパスに : は使えないので区切れる）
_ARCHIVE_SEP = "::"

# ── デフォルト設定値 ──────────────────────────────────────────────────────
_DEFAULT_MAX_IMAGE_SIZE = 1500   # 長辺の上限px（0=原寸）
# 索引・設定はすべてプログラムのフォルダ内 data/ に置く（画像フォルダには何も作らない。
# アンインストールはフォルダを消すだけで済む）。KOMASAGASHI_DATA_DIR で場所を変えられる（テスト用）。
APP_DIR   = Path(__file__).resolve().parent
DATA_DIR  = Path(os.environ.get("KOMASAGASHI_DATA_DIR") or APP_DIR / "data")
INDEX_DIR = DATA_DIR / "indexes"
TMP_DIR   = DATA_DIR / "tmp"   # 書庫から一時的に取り出した画像（読み終えたら・終了時に消す）
MODELS_DIR = DATA_DIR / "models"   # OCR のモデル（ユーザーフォルダの共用キャッシュを使わない）
# ライブラリがモデルを置く場所。transformers などを読み込む前に決める必要がある。
os.environ["HF_HOME"] = str(MODELS_DIR / "huggingface")            # manga-ocr（transformers）
os.environ["PADDLE_PDX_CACHE_HOME"] = str(MODELS_DIR / "paddlex")  # PaddleOCR
os.environ["TORCH_HOME"] = str(MODELS_DIR / "torch")
_LEGACY_DB_FILENAME = "_画像テキスト検索.db"   # 旧版は対象フォルダ直下に置いていた
_SCHEMA_VERSION         = "4"     # 2: 吹き出しの位置 / 3: OCR エンジン・横書き率・正規化文字列 / 4: 書庫内画像の sig
_NORM_VERSION           = "1"     # fold_text() を変えたら上げる（norm 列を作り直す）
_SNIPPET_PAD            = 30      # スニペットでヒット語の前後に残す文字数

# ── OCR エンジン ──────────────────────────────────────────────────────────
_ENGINE_MOKURO     = "mokuro"
_PADDLE_MODELS     = ("small", "medium")   # PP-OCRv6 のモデルの大きさ
_PADDLE_MIN_HRATIO = 0.8    # 文字の面積のうち横書きがこの割合以上のページを PaddleOCR で読み直す
# mokuro が読めた文字（記号・空白を除く）がこれ未満のページは読み直さない。
# 英字ロゴだけのページなどが「横書きのページ」に入り、PaddleOCR が 1 枚 15 秒以上かけて
# 意味のない文字を読んでいたため（目次は 200 文字以上、ロゴのページは 10 文字前後）。
_PADDLE_MIN_CHARS  = 20

# ── コマ検出 ──────────────────────────────────────────────────────────────
_PANEL_WORK_SIZE   = 1000   # 検出はこの長辺まで縮小して行う（速度のため）
_PANEL_MIN_AREA    = 0.015  # ページ面積に対するコマの最小面積
_PANEL_MIN_SIDE    = 0.05   # ページの幅・高さに対するコマの最小の辺
_PANEL_GUTTER_FILL = 0.15   # 行（列）のうちコマが占める割合がこれ未満なら隙間とみなす
_PANEL_CLEAN_GUTTER = 0.02  # 隙間の行（列）がすべてこれ未満なら、完全な空白とみなす
_PANEL_MIN_DENSITY = 0.4    # 分割後のコマの中身がこれ未満なら、枠線の有無で判断する
_PANEL_MIN_COVER   = 0.5    # コマ間が黒いページとしての検出結果は、ページのこの割合以上を覆う場合だけ使う
_PANEL_FRAME_COVER = 0.8    # 辺のこの割合以上に線があれば枠線とみなす
_HIT_PAD           = 0.05   # コマが見つからないとき、吹き出しの周りに足す余白（長辺比）


# ════════════════════════════════════════════════════════════════════════
# ユーティリティ
# ════════════════════════════════════════════════════════════════════════

def _format_elapsed(seconds: float) -> str:
    seconds = int(seconds)
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def _format_seconds(seconds: float) -> str:
    """1 枚ごとの時間の表示。1 分未満は小数 1 桁（切り捨ての「00:02」だと 2.9 秒でも 2 秒に見えるため）。"""
    return f"{seconds:.1f}秒" if seconds < 60 else _format_elapsed(seconds)


def _meaningful_chars(text: str) -> int:
    """記号・空白を除いた文字数（OCR が読めた文字の量の目安）。"""
    return sum(1 for c in text if c.isalnum())


def _natural_sort_key(name: str) -> list:
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", name)]


def _resize_image(image: Image.Image, max_size: int) -> tuple[Image.Image, bool]:
    """長辺が max_size を超える場合のみアスペクト比を保ってリサイズする。0=原寸。"""
    if max_size <= 0:
        return image, False
    w, h = image.size
    if max(w, h) <= max_size:
        return image, False
    scale = max_size / max(w, h)
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS), True


def _collect_files(folder: Path, recursive: bool) -> list[Path]:
    """フォルダ内の画像と書庫を自然順で収集する。recursive=True でサブフォルダも走査。"""
    it: Iterator[Path] = folder.rglob("*") if recursive else folder.iterdir()
    exts = _IMAGE_EXTS | _ARCHIVE_EXTS.keys()
    files = [p for p in it if p.is_file() and p.suffix.lower() in exts]
    return sorted(files, key=lambda p: _natural_sort_key(str(p.relative_to(folder))))


def split_relpath(relpath: str) -> tuple[str, str | None]:
    """'巻1.zip::p/001.jpg' → ('巻1.zip', 'p/001.jpg')。書庫内でなければ (relpath, None)。"""
    outer, sep, member = relpath.partition(_ARCHIVE_SEP)
    return (outer, member) if sep else (relpath, None)


def display_name(relpath: str) -> str:
    """
    一覧に出す名前。書庫内の画像は「画像名（書庫名）」。
    書庫名が長いと欄に収まらず、どのページか見えなくなるので画像名を先にする。
    """
    outer, member = split_relpath(relpath)
    if member is None:
        return Path(outer).name
    return f"{PurePosixPath(member).name}（{Path(outer).name}）"


def display_path(relpath: str) -> str:
    """ログ・画面に出す相対パス。書庫内の画像は「書庫の相対パス › 書庫内のパス」。"""
    outer, member = split_relpath(relpath)
    return outer if member is None else f"{outer} › {member}"


def _area(box: Box) -> int:
    return (box[2] - box[0]) * (box[3] - box[1])


def _folder_key(folder: Path) -> str:
    """フォルダの同一性を判定するキー（Windows では大文字小文字・区切り文字の違いを無視）。"""
    return os.path.normcase(os.path.abspath(str(folder)))


def index_path_for(folder: Path) -> Path:
    """
    対象フォルダの索引 DB の場所（data/indexes/フォルダ名_ハッシュ.db）。
    フォルダ名は見分けやすさのため、ハッシュはフルパスの違いを区別するため。
    """
    digest = hashlib.sha1(_folder_key(folder).encode("utf-8")).hexdigest()[:10]
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", Path(folder).name or "root")[:40].strip(" .")
    return INDEX_DIR / f"{name or 'folder'}_{digest}.db"


def migrate_legacy_index(folder: Path) -> bool:
    """
    旧版が対象フォルダ直下に作った索引を data/indexes/ へ移す。移したら True。
    新しい場所に索引が既にある場合は何もしない（旧い方は消さずに残す）。
    """
    legacy = Path(folder) / _LEGACY_DB_FILENAME
    target = index_path_for(folder)
    if not legacy.is_file() or target.exists():
        return False
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    shutil.move(str(legacy), str(target))
    for suffix in ("-journal", "-wal", "-shm"):
        extra = legacy.with_name(legacy.name + suffix)
        if extra.exists():
            shutil.move(str(extra), str(target.with_name(target.name + suffix)))
    return True


# ════════════════════════════════════════════════════════════════════════
# 表記ゆれの吸収（検索用の文字の正規化）
# ════════════════════════════════════════════════════════════════════════

_SMALL_KANA = str.maketrans("ぁぃぅぇぉっゃゅょゎゕゖ", "あいうえおつやゆよわかけ")
_FOLD_SYMBOLS = str.maketrans({
    "·": "・", "•": "・", "∙": "・",
    "－": "ー", "—": "ー", "―": "ー", "‐": "ー", "-": "ー", "─": "ー", "〜": "ー", "~": "ー",
})
_DAKUTEN_MARKS = {"゙", "゚", "゛", "゜"}


def _fold_char(ch: str) -> str:
    """1 文字を、OCR が取り違えやすい違いを無視した形にする（0〜数文字を返す）。"""
    out = []
    for c in unicodedata.normalize("NFKC", ch):          # 全角英数→半角、半角カナ→全角
        code = ord(c)
        if 0x30A1 <= code <= 0x30F6:                        # カタカナ → ひらがな
            c = chr(code - 0x60)
        for d in unicodedata.normalize("NFD", c):           # 濁点・半濁点を外す（ぺ→へ）
            if d not in _DAKUTEN_MARKS:
                out.append(d)
    return "".join(out).translate(_SMALL_KANA).translate(_FOLD_SYMBOLS).lower()


def fold_text(text: str) -> str:
    """
    検索用に表記ゆれを吸収した文字列にする。
    カタカナ/ひらがな・濁点/半濁点・小さい仮名・全角/半角・大文字/小文字・中点や長音の字形の違いを無視する。
    """
    return "".join(_fold_char(c) for c in text)


def fold_with_map(text: str) -> tuple[str, list[int]]:
    """fold_text と同じ変換をし、変換後の各文字が元の何文字目から来たかも返す（強調表示用）。"""
    chars: list[str] = []
    origin: list[int] = []
    for i, c in enumerate(text):
        folded = _fold_char(c)
        chars.append(folded)
        origin.extend([i] * len(folded))
    return "".join(chars), origin


def find_terms(text: str, terms: list[str], fuzzy: bool) -> list[tuple[int, int]]:
    """text 中に現れる各検索語の位置 (開始, 終了) をすべて返す（元の文字位置）。"""
    if fuzzy:
        hay, origin = fold_with_map(text)
        needles = [fold_text(t) for t in terms]
    else:
        hay, origin = text.lower(), list(range(len(text)))
        needles = [t.lower() for t in terms]
    spans = []
    for needle in needles:
        if not needle:
            continue
        start = 0
        while (i := hay.find(needle, start)) != -1:
            spans.append((origin[i], origin[i + len(needle) - 1] + 1))
            start = i + len(needle)
    return sorted(spans)


def contains_any(text: str, terms: list[str], fuzzy: bool) -> bool:
    if fuzzy:
        folded = fold_text(text)
        return any(fold_text(t) and fold_text(t) in folded for t in terms)
    low = text.lower()
    return any(t.lower() in low for t in terms)


# ════════════════════════════════════════════════════════════════════════
# コマ検出（OpenCV）
# ════════════════════════════════════════════════════════════════════════

def detect_panels(image: Image.Image) -> list[Box]:
    """
    漫画ページからコマの矩形を検出する。
      1. 紙の白より暗い部分（枠線・絵）をつながりごとに取り出す。
         枠線で閉じたコマは中身ごと 1 つの塊になる。
      2. 吹き出しなどがコマ間の隙間をまたいで複数のコマがつながった塊は、
         ほぼ空白の行・列（＝隙間）で再帰的に切り分ける。
      3. ページ全体が 1 コマになったときは、コマ間が黒いページとみなして
         明暗を反転してやり直す。
    枠線のないコマや、絵がページ全体に広がるページはうまく取れないことがある。
    """
    if not _CV2_AVAILABLE:
        return []
    gray = np.asarray(image.convert("L"))
    full_h, full_w = gray.shape
    scale = min(1.0, _PANEL_WORK_SIZE / max(full_h, full_w))
    if scale < 1.0:
        gray = cv2.resize(gray, (round(full_w * scale), round(full_h * scale)),
                          interpolation=cv2.INTER_AREA)
    h, w = gray.shape

    # 紙の白さ・黒さはスキャンによって違うので、明るい側・暗い側の画素から推定する
    paper = float(np.percentile(gray, 95))
    boxes = _find_panels(gray < min(235.0, paper - 20))
    if len(boxes) <= 1 and (not boxes or _area(boxes[0]) >= 0.8 * w * h):
        black = float(np.percentile(gray, 5))
        dark_boxes = _find_panels(gray > max(40.0, black + 30))
        # 表紙の文字などをコマと誤認しないよう、ページの大部分を覆うときだけ採用する
        if len(dark_boxes) >= 2 and sum(map(_area, dark_boxes)) >= _PANEL_MIN_COVER * w * h:
            boxes = dark_boxes

    inv = 1.0 / scale
    result = [(round(b[0] * inv), round(b[1] * inv),
               min(full_w, round(b[2] * inv)), min(full_h, round(b[3] * inv)))
              for b in boxes]
    # 漫画の読み順（上から、右から）
    return sorted(result, key=lambda b: (b[1] // max(1, full_h // 20), -b[2]))


def _find_panels(ink_mask: "np.ndarray") -> list[Box]:
    """ink_mask（True = 隙間以外）からコマの矩形を取り出す。座標は ink_mask の px。"""
    h, w = ink_mask.shape
    ink = cv2.morphologyEx(ink_mask.astype(np.uint8), cv2.MORPH_CLOSE,
                           np.ones((3, 3), np.uint8))
    min_w, min_h = _PANEL_MIN_SIDE * w, _PANEL_MIN_SIDE * h
    contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[Box] = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh < _PANEL_MIN_AREA * w * h or bw < min_w or bh < min_h:
            continue
        filled = np.zeros((h, w), np.uint8)
        cv2.drawContours(filled, [c], -1, 1, thickness=cv2.FILLED)
        boxes += _split_panel(filled, (x, y, x + bw, y + bh), min_w, min_h, 0)

    # 他のコマにほぼ収まる小さな矩形（はみ出した吹き出しなど）は捨てる
    boxes.sort(key=_area, reverse=True)
    kept: list[Box] = []
    for b in boxes:
        if not any(_area(_intersect(b, k)) >= 0.8 * _area(b) for k in kept):
            kept.append(b)
    return kept


def _intersect(a: Box, b: Box) -> Box:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return (x1, y1, max(x1, x2), max(y1, y2))


def _tighten(mask: "np.ndarray", box: Box) -> Box | None:
    """box の中で mask が立っている範囲に縮める。空なら None。"""
    x1, y1, x2, y2 = box
    sub = mask[y1:y2, x1:x2]
    ys = np.flatnonzero(sub.any(axis=1))
    xs = np.flatnonzero(sub.any(axis=0))
    if len(ys) == 0:
        return None
    return (x1 + int(xs[0]), y1 + int(ys[0]), x1 + int(xs[-1]) + 1, y1 + int(ys[-1]) + 1)


def _split_panel(mask: "np.ndarray", box: Box, min_w: float, min_h: float,
                 depth: int) -> list[Box]:
    """塗りつぶしたコマの塊を、隙間の行・列で再帰的に切り分ける（XY-cut）。"""
    tight = _tighten(mask, box)
    if tight is None:
        return []
    x1, y1, x2, y2 = tight
    if depth >= 8:
        return [tight]
    sub = mask[y1:y2, x1:x2]
    for axis in (1, 0):   # 1: 横線で上下に切る, 0: 縦線で左右に切る
        profile = sub.mean(axis=axis)
        n = len(profile)
        min_len = min_h if axis == 1 else min_w
        # 隙間の候補 = 連続した空白の行・列のうち、両側にコマがあるもの。
        # 端まで続く空白（斜めのコマの先端など）は隙間ではない。
        gaps: list[tuple[int, int, bool]] = []   # (隙間の太さ, 切る位置, 完全な空白か)
        i = 0
        while i < n:
            if profile[i] >= _PANEL_GUTTER_FILL:
                i += 1
                continue
            j = i
            while j < n and profile[j] < _PANEL_GUTTER_FILL:
                j += 1
            cut = (i + j) // 2
            if i > 0 and j < n and min_len <= cut <= n - min_len:
                gaps.append((j - i, cut, float(profile[i:j].max()) < _PANEL_CLEAN_GUTTER))
            i = j
        # 太い隙間から順に試し、両側がコマとして成り立つところで切る
        for _width, cut, clean in sorted(gaps, reverse=True):
            if axis == 1:
                parts = [(x1, y1, x2, y1 + cut), (x1, y1 + cut, x2, y2)]
            else:
                parts = [(x1, y1, x1 + cut, y2), (x1 + cut, y1, x2, y2)]
            if _can_split(mask, parts, min_w, min_h, clean):
                return [b for p in parts
                        for b in _split_panel(mask, p, min_w, min_h, depth + 1)]
    return [tight]


def _can_split(mask: "np.ndarray", parts: list[Box], min_w: float, min_h: float,
               clean_gap: bool) -> bool:
    """parts の境目で切ってよいか。両側とも十分な大きさがあることが前提。"""
    tights = [_tighten(mask, p) for p in parts]
    if any(t is None or t[2] - t[0] < min_w or t[3] - t[1] < min_h for t in tights):
        return False
    solid = sum(_is_solid_part(mask, t) for t in tights)
    # 両側がコマらしければ切る。背景が白く、ページ端で枠線が切れたコマは中身が空に
    # 見えるので、隙間が完全な空白なら片側だけコマらしければよい（集中線などの誤分割を防ぐ）。
    return solid == 2 or (solid == 1 and clean_gap)


def _is_solid_part(mask: "np.ndarray", box: Box) -> bool:
    """box（_tighten 済み）がコマらしいか（中身が詰まっている、または枠線が 3 辺以上ある）。"""
    x1, y1, x2, y2 = box
    sub = mask[y1:y2, x1:x2]
    if float(sub.mean()) >= _PANEL_MIN_DENSITY:
        return True
    band = 3
    sides = (sub[:band].any(axis=0), sub[-band:].any(axis=0),
             sub[:, :band].any(axis=1), sub[:, -band:].any(axis=1))
    return sum(float(s.mean()) >= _PANEL_FRAME_COVER for s in sides) >= 3


def pick_panel(panels: list[Box], hit: Box, image_size: tuple[int, int]) -> Box:
    """ヒットした吹き出しを含むコマを選ぶ。無ければ吹き出しの周りを少し広げた枠。"""
    cx, cy = (hit[0] + hit[2]) / 2, (hit[1] + hit[3]) / 2
    containing = [p for p in panels if p[0] <= cx <= p[2] and p[1] <= cy <= p[3]]
    if containing:
        return min(containing, key=_area)
    w, h = image_size
    pad = round(max(w, h) * _HIT_PAD)
    return (max(0, hit[0] - pad), max(0, hit[1] - pad),
            min(w, hit[2] + pad), min(h, hit[3] + pad))


FRAME_ADJUST_MIN, FRAME_ADJUST_MAX = -50, 100   # 枠の補正（%）の範囲


def scale_box(box, factor: float) -> list[float]:
    """枠を中心から幅・高さとも factor 倍にする（画像の外にはみ出してもよい）。"""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = (x2 - x1) * factor / 2, (y2 - y1) * factor / 2
    return [cx - hw, cy - hh, cx + hw, cy + hh]


def adjust_box(box, percent: float, image_size: tuple[int, int]) -> list[float]:
    """枠の補正: 幅・高さとも percent % 大きく（マイナスなら小さく）し、画像の外は切り詰める。"""
    x1, y1, x2, y2 = scale_box(box, 1 + percent / 100)
    w, h = image_size
    return [max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)]


def format_adjust(percent: int) -> str:
    return f"{percent:+d}" if percent else "0"


# ════════════════════════════════════════════════════════════════════════
# クリップボード（Windows）
# ════════════════════════════════════════════════════════════════════════

def copy_image_to_clipboard(image: Image.Image) -> None:
    """画像を Windows のクリップボードへ CF_DIB 形式で置く（ペイント・Word・チャット等に貼れる）。"""
    if os.name != "nt":
        raise RuntimeError("画像のコピーは Windows のみ対応です。")
    import ctypes
    from ctypes import wintypes

    buf = io.BytesIO()
    image.convert("RGB").save(buf, "BMP")
    data = buf.getvalue()[14:]   # BMP ファイルヘッダー（14 バイト）を除くと DIB になる

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
    user32.SetClipboardData.restype = wintypes.HANDLE

    GMEM_MOVEABLE, CF_DIB = 0x0002, 8
    handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    ptr = kernel32.GlobalLock(handle)
    ctypes.memmove(ptr, data, len(data))
    kernel32.GlobalUnlock(handle)

    # 他のアプリが一瞬クリップボードを開いていることがあるので少し待って再試行
    for _ in range(10):
        if user32.OpenClipboard(None):
            break
        time.sleep(0.05)
    else:
        kernel32.GlobalFree(handle)
        raise RuntimeError("クリップボードを開けませんでした。もう一度試してください。")
    try:
        user32.EmptyClipboard()
        if not user32.SetClipboardData(CF_DIB, handle):
            kernel32.GlobalFree(handle)   # 成功時はクリップボードが所有するので解放しない
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        user32.CloseClipboard()


# ════════════════════════════════════════════════════════════════════════
# インフラ: SQLite インデックス
# ════════════════════════════════════════════════════════════════════════

@dataclass
class ArchiveMember:
    name: str   # 書庫内のパス（/ 区切り）
    sig: str    # 変更検出用（CRC32:サイズ）。書庫が更新されても中身が同じ画像は OCR し直さない


class ArchiveError(Exception):
    """
    書庫を開けない・読めない（壊れている・パスワード付き・展開する道具が無いなど）。
    retry=True は、道具やライブラリを入れれば読めるようになる失敗（次回も開き直す）。
    """

    def __init__(self, message: str, retry: bool = False) -> None:
        super().__init__(message)
        self.retry = retry


def _safe_member_name(name: str) -> bool:
    """一時フォルダの外に書き出されるような名前（絶対パス・..）は扱わない。"""
    p = PurePosixPath(name.replace("\\", "/"))
    return not p.is_absolute() and ".." not in p.parts and ":" not in name


def _is_image_member(name: str) -> bool:
    return PurePosixPath(name).suffix.lower() in _IMAGE_EXTS and _safe_member_name(name)


class ArchiveReader:
    """書庫の中の画像を列挙し、読み出す。with 文で使う。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.skipped_encrypted = 0   # パスワード付きで読めなかった画像の数
        self._tmp: Path | None = None

    def __enter__(self) -> "ArchiveReader":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def members(self) -> list[ArchiveMember]:
        raise NotImplementedError

    def read(self, name: str) -> bytes:
        raise NotImplementedError

    def prefetch(self, names: list[str]) -> None:
        """まとめて読む画像を先に展開しておく（rar・7z の固め圧縮を 1 回の展開で済ませる）。"""

    def _tmp_dir(self) -> Path:
        if self._tmp is None:
            TMP_DIR.mkdir(parents=True, exist_ok=True)
            self._tmp = Path(tempfile.mkdtemp(prefix="arc_", dir=TMP_DIR))
        return self._tmp

    def _read_extracted(self, name: str) -> bytes | None:
        """prefetch で展開済みなら読む（一時ファイルは close で消す）。"""
        if self._tmp is None:
            return None
        p = self._tmp.joinpath(*PurePosixPath(name).parts)
        return p.read_bytes() if p.is_file() else None

    def close(self) -> None:
        if self._tmp is not None:
            shutil.rmtree(self._tmp, ignore_errors=True)
            self._tmp = None


def _zip_member_name(info: zipfile.ZipInfo) -> str:
    """
    zip のファイル名を正しく読む。UTF-8 の印が無い名前は、Python が cp437 として読んでしまう。
    日本語の Windows で作った zip は Shift_JIS（cp932）が多いので、UTF-8 → cp932 の順に読み直す。
    """
    name = info.filename
    if not info.flag_bits & 0x800:
        try:
            raw = name.encode("cp437")
        except UnicodeEncodeError:
            return name
        for encoding in ("utf-8", "cp932"):
            try:
                name = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
    return name.replace("\\", "/")


class ZipReader(ArchiveReader):
    """zip / cbz（Python 標準の zipfile で直接読む）。"""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        try:
            self._zf = zipfile.ZipFile(path)
        except (zipfile.BadZipFile, OSError) as exc:
            raise ArchiveError(f"zip として開けません（{exc}）") from exc
        self._infos: dict[str, zipfile.ZipInfo] = {}
        for info in self._zf.infolist():
            name = _zip_member_name(info)
            if info.is_dir() or not _is_image_member(name):
                continue
            if info.flag_bits & 0x1:
                self.skipped_encrypted += 1
                continue
            self._infos[name] = info

    def members(self) -> list[ArchiveMember]:
        return [ArchiveMember(n, f"{i.CRC:08x}:{i.file_size}") for n, i in self._infos.items()]

    def read(self, name: str) -> bytes:
        try:
            return self._zf.read(self._infos[name])
        except KeyError as exc:
            raise ArchiveError(f"書庫内に見つかりません: {name}") from exc

    def close(self) -> None:
        self._zf.close()
        super().close()


class SevenZipReader(ArchiveReader):
    """7z / cb7（py7zr）。固め圧縮が多いので、まとめて展開してから読む。"""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        try:
            import py7zr
        except ImportError as exc:
            raise ArchiveError("7z を読むには py7zr が必要です"
                               "（setup.bat を実行し直すと入ります）", retry=True) from exc
        self._py7zr = py7zr
        try:
            with py7zr.SevenZipFile(path) as z:
                if z.needs_password():
                    raise ArchiveError("パスワード付きの 7z は読めません")
                infos = z.list()
        except ArchiveError:
            raise
        except Exception as exc:
            raise ArchiveError(f"7z として開けません（{type(exc).__name__}: {exc}）") from exc
        self._members = {
            fi.filename.replace("\\", "/"): fi for fi in infos
            if not fi.is_directory and _is_image_member(fi.filename)}

    def members(self) -> list[ArchiveMember]:
        return [ArchiveMember(n, f"{(fi.crc32 or 0):08x}:{fi.uncompressed}")
                for n, fi in self._members.items()]

    def prefetch(self, names: list[str]) -> None:
        targets = [n for n in names if n in self._members]
        if not targets:
            return
        try:
            with self._py7zr.SevenZipFile(self.path) as z:
                z.extract(path=self._tmp_dir(), targets=targets)
        except Exception as exc:
            raise ArchiveError(f"7z を展開できません（{type(exc).__name__}: {exc}）") from exc

    def read(self, name: str) -> bytes:
        if name not in self._members:
            raise ArchiveError(f"書庫内に見つかりません: {name}")
        data = self._read_extracted(name)
        if data is None:
            self.prefetch([name])
            data = self._read_extracted(name)
        if data is None:
            raise ArchiveError(f"展開できませんでした: {name}")
        return data


_RAR_TOOL_READY: bool | None = None
_RAR_EXTRACTOR: tuple[str, str] | None = None   # (種類, 実行ファイル) まとめて展開するときに使う


def _setup_rar_tool(rarfile) -> None:
    """
    rar の展開に使う道具を探す（rarfile は一覧は自前で読めるが、展開には外部の道具が要る）。
    WinRAR の UnRAR → 7-Zip → Windows 標準の tar（libarchive）の順。
    Windows 10 / 11 には tar が入っているので、多くの PC では追加のインストールが要らない。
    """
    global _RAR_TOOL_READY, _RAR_EXTRACTOR
    if _RAR_TOOL_READY is not None:
        if not _RAR_TOOL_READY:
            raise ArchiveError(_RAR_TOOL_MISSING, retry=True)
        return
    program_files = [os.environ.get(k) for k in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432")]
    def find(names: list[str], rel_paths: list[str]) -> str | None:
        for n in names:
            if found := shutil.which(n):
                return found
        for base in filter(None, program_files):
            for rel in rel_paths:
                if (p := Path(base) / rel).is_file():
                    return str(p)
        return None
    if unrar := find(["unrar"], [r"WinRAR\UnRAR.exe"]):
        rarfile.UNRAR_TOOL = unrar
    if sevenzip := find(["7z"], [r"7-Zip\7z.exe"]):
        rarfile.SEVENZIP_TOOL = sevenzip
    tar = shutil.which("bsdtar") or str(Path(os.environ.get("SystemRoot", r"C:\Windows"))
                                        / "System32" / "tar.exe")
    rarfile.BSDTAR_TOOL = tar
    try:
        rarfile.tool_setup(unrar=True, unar=False, sevenzip=True, sevenzip2=False,
                           bsdtar=True, force=True)
        _RAR_TOOL_READY = True
    except rarfile.RarCannotExec as exc:
        _RAR_TOOL_READY = False
        raise ArchiveError(_RAR_TOOL_MISSING, retry=True) from exc
    for kind, exe in (("unrar", unrar), ("7z", sevenzip), ("tar", tar)):
        if exe and Path(exe).is_file():
            _RAR_EXTRACTOR = (kind, exe)
            break


def _extract_rar_all(archive: Path, dest: Path, names: list[str]) -> None:
    """rar を dest に丸ごと展開する（固め圧縮でも 1 回の展開で済む）。"""
    if _RAR_EXTRACTOR is None:
        raise ArchiveError(_RAR_TOOL_MISSING, retry=True)
    kind, exe = _RAR_EXTRACTOR
    if kind == "unrar":
        cmd = [exe, "x", "-y", "-o+", "-idq", "-p-", str(archive), str(dest) + os.sep]
    elif kind == "7z":
        cmd = [exe, "x", "-y", "-bso0", "-bsp0", "-p", f"-o{dest}", str(archive)]
    else:
        cmd = [exe, "-x", "-f", str(archive), "-C", str(dest)]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    result = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True,
                            creationflags=flags)
    missing = [n for n in names if not dest.joinpath(*PurePosixPath(n).parts).is_file()]
    if not missing:
        return
    if kind == "tar":
        # Windows 標準の tar（libarchive）は古い形式（RAR4）の固め圧縮などを展開できない
        raise ArchiveError(_RAR_TAR_UNSUPPORTED, retry=True)
    err = (result.stderr or result.stdout).decode("utf-8", "replace").strip()[:200]
    raise ArchiveError(f"rar を展開できません（{kind}: {err or result.returncode}）")


_RAR_TOOL_MISSING = ("rar を展開できる道具が見つかりません"
                     "（WinRAR か 7-Zip を入れると読めるようになります）")
_RAR_TAR_UNSUPPORTED = ("この rar は Windows 標準の tar では展開できません（古い形式の固め圧縮など）。"
                        "WinRAR か 7-Zip を入れると読めるようになります")


class RarReader(ArchiveReader):
    """rar / cbr（rarfile。展開は _setup_rar_tool で見つけた道具）。"""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        try:
            import rarfile
        except ImportError as exc:
            raise ArchiveError("rar を読むには rarfile が必要です"
                               "（setup.bat を実行し直すと入ります）", retry=True) from exc
        self._rarfile = rarfile
        _setup_rar_tool(rarfile)
        try:
            self._rf = rarfile.RarFile(str(path))
            if self._rf.needs_password():
                raise ArchiveError("パスワード付きの rar は読めません")
            infos = self._rf.infolist()
        except ArchiveError:
            raise
        except Exception as exc:
            raise ArchiveError(f"rar として開けません（{type(exc).__name__}: {exc}）") from exc
        self._infos = {}
        for info in infos:
            name = info.filename.replace("\\", "/")
            if info.is_dir() or not _is_image_member(name):
                continue
            if info.needs_password():
                self.skipped_encrypted += 1
                continue
            self._infos[name] = info

    def members(self) -> list[ArchiveMember]:
        return [ArchiveMember(n, f"{(i.CRC or 0):08x}:{i.file_size}")
                for n, i in self._infos.items()]

    def prefetch(self, names: list[str]) -> None:
        # rarfile の extractall は 1 枚ごとに道具を起動し、固め圧縮では毎回先頭から展開し直すので、
        # 道具で丸ごと 1 回展開する。展開できなかった画像は read() が 1 枚ずつ読む。
        targets = [n for n in names if n in self._infos]
        if targets:
            _extract_rar_all(self.path, self._tmp_dir(), targets)

    def read(self, name: str) -> bytes:
        if name not in self._infos:
            raise ArchiveError(f"書庫内に見つかりません: {name}")
        data = self._read_extracted(name)
        if data is not None:
            return data
        try:
            return self._rf.read(self._infos[name])
        except Exception:
            # 1 枚だけ読む方法が使えない道具（Windows の tar など）は、丸ごと展開して読む
            self.prefetch([name])
            data = self._read_extracted(name)
            if data is None:
                raise ArchiveError(f"rar から読めません: {name}")
            return data

    def close(self) -> None:
        self._rf.close()
        super().close()


def open_archive(path: Path) -> ArchiveReader:
    kind = _ARCHIVE_EXTS.get(path.suffix.lower())
    if kind == "zip":
        return ZipReader(path)
    if kind == "7z":
        return SevenZipReader(path)
    if kind == "rar":
        return RarReader(path)
    raise ArchiveError(f"対応していない書庫です: {path.suffix}")


def load_image(root: Path, relpath: str, reader: ArchiveReader | None = None) -> Image.Image:
    """索引の relpath から画像を読む。書庫内の画像は reader（無ければその場で開く）から読む。"""
    outer, member = split_relpath(relpath)
    if member is None:
        return Image.open(root / outer).convert("RGB")
    if reader is not None:
        data = reader.read(member)
    else:
        with open_archive(root / outer) as r:
            data = r.read(member)
    return Image.open(io.BytesIO(data)).convert("RGB")


def clean_tmp_dir() -> None:
    """前回の終了時に消し損ねた一時ファイルを消す。"""
    shutil.rmtree(TMP_DIR, ignore_errors=True)


@dataclass
class IndexedRow:
    """既存インデックスの 1 行（更新が必要かの判定に使う）。"""
    mtime: float
    size: int
    has_blocks: bool       # 吹き出しの位置を持っているか（schema 1 の行は False）
    engine: str            # 読み取った OCR エンジン（"mokuro" / "paddle-small" など）
    hratio: float | None   # 文字の面積のうち横書きの割合（mokuro の判定）。None は判定前の古い行
    sig: str | None = None # 書庫内の画像の CRC32:サイズ（mtime / size は書庫ファイルのもの）
    n_chars: int = 0       # 登録済みの文字数（記号・空白を除く）


class TextIndex:
    """
    OCR 結果を保持する SQLite インデックス（対象フォルダ 1 つにつき 1 ファイル）。
    DB は data/indexes/ に置き（index_path_for）、各画像は対象フォルダ（root）からの
    相対パスで記録する。
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._db_path = index_path_for(self._root)
        self._ready = False

    @property
    def root(self) -> Path:
        return self._root

    @property
    def db_path(self) -> Path:
        return self._db_path

    def exists(self) -> bool:
        return self._db_path.is_file()

    def _connect(self) -> sqlite3.Connection:
        if not self._ready:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        if self._ready:
            return conn
        # blocks: 吹き出しごとの [[x1, y1, x2, y2, 文字], ...]（JSON・元画像の px）
        # norm:   表記ゆれを吸収した検索用の文字列（fold_text）
        conn.execute(
            "CREATE TABLE IF NOT EXISTS images ("
            "  relpath TEXT PRIMARY KEY,"
            "  text    TEXT NOT NULL,"
            "  mtime   REAL NOT NULL,"
            "  size    INTEGER NOT NULL,"
            "  blocks  TEXT,"
            "  engine  TEXT,"
            "  hratio  REAL,"
            "  norm    TEXT,"
            "  sig     TEXT"
            ")"
        )
        conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        # 書庫ごとの処理状況。書庫ファイルの日時・サイズが同じなら、次回は state に応じて:
        #   done    … 中の画像を最後まで処理した → 開かずに飛ばす
        #   pending … 途中で止めた・失敗した画像がある → 開き直して残りを処理する
        #   empty / error … 読める画像が無い・壊れている → 開かずに飛ばす
        conn.execute(
            "CREATE TABLE IF NOT EXISTS archives ("
            "  relpath TEXT PRIMARY KEY, mtime REAL NOT NULL, size INTEGER NOT NULL,"
            "  state TEXT NOT NULL, note TEXT)"
        )
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "empty_archives" in tables:   # v1.01 の記録を引き継ぐ
            conn.execute("INSERT OR IGNORE INTO archives "
                         "SELECT relpath, mtime, size, 'empty', note FROM empty_archives")
            conn.execute("DROP TABLE empty_archives")
        cols = {row[1] for row in conn.execute("PRAGMA table_info(images)")}
        # 古い索引に列を足す。blocks が NULL の行（schema 1）は次回の更新で OCR し直される。
        for col, sql_type in (("blocks", "TEXT"), ("engine", "TEXT"),
                              ("hratio", "REAL"), ("norm", "TEXT"), ("sig", "TEXT")):
            if col not in cols:
                conn.execute(f"ALTER TABLE images ADD COLUMN {col} {sql_type}")
        row = conn.execute("SELECT value FROM meta WHERE key = 'norm_version'").fetchone()
        if row is None or row[0] != _NORM_VERSION:
            conn.executemany("UPDATE images SET norm = ? WHERE relpath = ?",
                             [(fold_text(t), r) for r, t in
                              conn.execute("SELECT relpath, text FROM images").fetchall()])
        # root は参照用（どのフォルダの索引かを DB ファイルだけで分かるようにする）
        conn.executemany("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                         [("schema", _SCHEMA_VERSION), ("norm_version", _NORM_VERSION),
                          ("root", str(self._root))])
        conn.commit()
        self._ready = True
        return conn

    def load_existing(self) -> dict[str, IndexedRow]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT relpath, mtime, size, blocks IS NOT NULL, engine, hratio, sig, text"
                " FROM images").fetchall()
        return {r[0]: IndexedRow(r[1], r[2], bool(r[3]), r[4] or _ENGINE_MOKURO, r[5], r[6],
                                 _meaningful_chars(r[7]))
                for r in rows}

    def upsert(self, relpath: str, page: "PageText", mtime: float, size: int,
               sig: str | None = None) -> None:
        blocks_json = json.dumps(page.blocks, ensure_ascii=False)
        values = (page.text, mtime, size, blocks_json, page.engine, page.hratio,
                  fold_text(page.text), sig)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO images (relpath, text, mtime, size, blocks, engine, hratio, norm, sig) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(relpath) DO UPDATE SET text=?, mtime=?, size=?, blocks=?, "
                "engine=?, hratio=?, norm=?, sig=?",
                (relpath, *values, *values),
            )

    def load_archives(self) -> dict[str, tuple[float, int, str]]:
        """{書庫の relpath: (mtime, size, state)}"""
        with self._connect() as conn:
            rows = conn.execute("SELECT relpath, mtime, size, state FROM archives").fetchall()
        return {r[0]: (r[1], r[2], r[3]) for r in rows}

    def set_archive(self, relpath: str, mtime: float, size: int, state: str,
                    note: str = "") -> None:
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO archives VALUES (?, ?, ?, ?, ?)",
                         (relpath, mtime, size, state, note))

    def forget_archives(self, relpaths: list[str]) -> None:
        if not relpaths:
            return
        with self._connect() as conn:
            conn.executemany("DELETE FROM archives WHERE relpath = ?",
                             [(r,) for r in relpaths])

    def update_stats(self, rows: list[tuple[float, int, str]]) -> None:
        """[(mtime, size, relpath), ...] の日時・サイズだけを更新する（中身が同じ書庫内の画像）。"""
        if not rows:
            return
        with self._connect() as conn:
            conn.executemany("UPDATE images SET mtime = ?, size = ? WHERE relpath = ?", rows)

    def export_rows(self, relpaths: list[str] | None = None) -> list[dict]:
        """
        書き出し用に登録内容を返す。relpaths を渡せばその順で、None なら全件を自然順で。
        各要素: {"relpath", "text", "engine", "blocks": [((x1, y1, x2, y2), 文字), ...]}
        """
        with self._connect() as conn:
            rows = conn.execute("SELECT relpath, text, engine, blocks FROM images").fetchall()
        by_rel = {r[0]: {"relpath": r[0], "text": r[1], "engine": r[2] or _ENGINE_MOKURO,
                         "blocks": [((b[0], b[1], b[2], b[3]), b[4])
                                    for b in (json.loads(r[3]) if r[3] else [])]}
                  for r in rows}
        if relpaths is None:
            return [by_rel[r] for r in sorted(by_rel, key=_natural_sort_key)]
        return [by_rel[r] for r in relpaths if r in by_rel]

    def get_blocks(self, relpath: str) -> list[tuple[Box, str]] | None:
        """吹き出しごとの (矩形, 文字) を返す。位置情報のない古い索引なら None。"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT blocks FROM images WHERE relpath = ?", (relpath,)).fetchone()
        if row is None or row[0] is None:
            return None
        return [((b[0], b[1], b[2], b[3]), b[4]) for b in json.loads(row[0])]

    def delete_many(self, relpaths: list[str]) -> None:
        if not relpaths:
            return
        with self._connect() as conn:
            conn.executemany("DELETE FROM images WHERE relpath = ?",
                             [(r,) for r in relpaths])

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]

    @staticmethod
    def _like_param(term: str) -> str:
        """LIKE 用にワイルドカード文字をエスケープして %term% にする。"""
        esc = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        return f"%{esc}%"

    def search(self, terms: list[str], fuzzy: bool = False) -> list[tuple[str, str]]:
        """
        全 term を含む（AND・部分一致・大文字小文字無視）画像を
        [(relpath, text), ...] で返す。terms が空なら空リスト。
        fuzzy=True なら表記ゆれを吸収して比べる（fold_text）。
        """
        if fuzzy:
            terms = [fold_text(t) for t in terms]
        terms = [t for t in terms if t]
        if not terms:
            return []
        column = "norm" if fuzzy else "text"
        where = " AND ".join([f"{column} LIKE ? ESCAPE '\\'"] * len(terms))
        params = [self._like_param(t) for t in terms]
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT relpath, text FROM images WHERE {where}", params
            ).fetchall()
        return sorted(rows, key=lambda r: _natural_sort_key(r[0]))


def make_snippet(text: str, terms: list[str], fuzzy: bool = False) -> str:
    """ヒット語の周辺を抜き出した1行スニペットを作る。"""
    flat = re.sub(r"\s+", " ", text).strip()
    spans = find_terms(flat, terms, fuzzy)
    if not spans:
        return flat[:80] + ("…" if len(flat) > 80 else "")
    pos, end_hit = spans[0]
    start = max(0, pos - _SNIPPET_PAD)
    end = min(len(flat), end_hit + _SNIPPET_PAD)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(flat) else ""
    return f"{prefix}{flat[start:end]}{suffix}"


# ════════════════════════════════════════════════════════════════════════
# インフラ: OCR エンジン（mokuro / PaddleOCR）
# ════════════════════════════════════════════════════════════════════════

@dataclass
class PageText:
    """1 ページの OCR 結果。"""
    text: str                                          # 全文（吹き出し・行ごとに改行）
    blocks: list[list] = field(default_factory=list)   # [[x1, y1, x2, y2, 文字], ...]（元画像の px）
    engine: str = _ENGINE_MOKURO
    hratio: float = 0.0                                # 文字の面積のうち横書きの割合（文字なしは 0）


def horizontal_ratio(blocks: list[tuple[list, bool]]) -> float | None:
    """[(矩形, 横書きか), ...] から、文字の面積のうち横書きの割合を返す。文字が無ければ None。"""
    total = sum(max(0, b[2] - b[0]) * max(0, b[3] - b[1]) for b, _h in blocks)
    if total <= 0:
        return None
    horiz = sum(max(0, b[2] - b[0]) * max(0, b[3] - b[1]) for b, h in blocks if h)
    return horiz / total


def _patch_numpy_compat() -> None:
    """
    comic-text-detector（mokuro が使用）は numpy 1.24 / 2.0 で削除された
    np.int・np.bool8 などを使っている。PaddleOCR は numpy 1.24 以上を要求するので、
    削除された別名を補って両方を動かす。
    """
    try:
        import numpy
    except ImportError:
        return
    for name, value in (("int", int), ("float", float), ("bool", bool),
                        ("object", object), ("bool8", numpy.bool_)):
        if name not in numpy.__dict__:
            setattr(numpy, name, value)


def _use_local_mokuro_cache() -> None:
    """
    mokuro は文字位置検出のモデルを ~/.cache/manga-ocr に置く（読み込んだ時点でフォルダも作る）。
    data/models/manga-ocr を使うように切り替え、読み込みで作られた空のフォルダは消す。
    """
    try:
        import mokuro.cache as mokuro_cache
    except ImportError:
        return
    default = Path(mokuro_cache.cache.root)
    local = MODELS_DIR / "manga-ocr"
    local.mkdir(parents=True, exist_ok=True)
    mokuro_cache.cache.root = local
    try:
        if default != local and default.is_dir() and not any(default.iterdir()):
            default.rmdir()   # ほかのツールのモデルが入っていれば消さない
    except OSError:
        pass


class MokuroEngine:
    """mokuro の MangaPageOcr（comic-text-detector + manga-ocr-base）を使う OCR。"""

    def __init__(self) -> None:
        self._mpocr = None

    def load(self, progress_cb: "Callable[[str], None] | None" = None) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch が必要です: pip install torch") from exc
        _patch_numpy_compat()
        try:
            from mokuro.manga_page_ocr import MangaPageOcr
        except ImportError as exc:
            missing = getattr(exc, "name", "") or ""
            if missing == "pkg_resources":
                hint = ('pkg_resources がありません。setuptools 81+ で削除されたためです。'
                        '解決: pip install "setuptools<81"')
            elif missing == "mokuro":
                hint = "mokuro が必要です: pip install mokuro"
            else:
                hint = f"mokuro の読み込みに失敗しました（{type(exc).__name__}: {exc}）"
            raise RuntimeError(hint) from exc
        _use_local_mokuro_cache()

        force_cpu = not torch.cuda.is_available()
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            if progress_cb:
                progress_cb(f"GPU 検出: {name} (VRAM {vram:.1f} GB) → CUDA で推論します")
        else:
            if progress_cb:
                progress_cb("⚠ GPU が見つかりません → CPU で推論します（遅くなります）")

        if progress_cb:
            progress_cb("テキスト検出モデル・manga-ocr-base をロード中...")

        t0 = time.time()
        self._mpocr = MangaPageOcr(
            pretrained_model_name_or_path="kha-white/manga-ocr-base",
            force_cpu=force_cpu,
        )
        if progress_cb:
            progress_cb(f"ロード完了 ({time.time() - t0:.1f}秒)")

    @property
    def is_loaded(self) -> bool:
        return self._mpocr is not None

    def recognize(self, image: Image.Image, max_image_size: int = _DEFAULT_MAX_IMAGE_SIZE
                  ) -> PageText:
        """画像内の文字を吹き出しごとに読む。座標はリサイズ前の元画像の px。"""
        if not self.is_loaded:
            raise RuntimeError("モデルがロードされていません。")

        orig_w = image.width
        image, _ = _resize_image(image, max_image_size)
        scale = orig_w / image.width

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".png")
        try:
            os.close(tmp_fd)
            image.save(tmp_path)
            result = self._mpocr(tmp_path)
        finally:
            os.unlink(tmp_path)

        texts: list[str] = []
        blocks: list[list] = []
        layout: list[tuple[list, bool]] = []
        for block in result.get("blocks", []):
            lines = block.get("lines", [])
            if not lines:
                continue
            text = "".join(lines)
            texts.append(text)
            box = [int(round(float(v) * scale)) for v in block["box"]]
            blocks.append([*box, text])
            layout.append((box, not block.get("vertical", True)))
        return PageText("\n".join(texts), blocks, _ENGINE_MOKURO,
                        horizontal_ratio(layout) or 0.0)


class PaddleEngine:
    """
    PaddleOCR（PP-OCRv6）で横書きの文字を読む。目次・扉・説明文などの印刷文字に強い。
    任意機能: paddlepaddle と paddleocr が入っていなければ使えない（is_installed）。
    CPU で動かす（GPU 版は torch と別の CUDA 環境が要り、導入が重いため）。

    PaddleOCR は同じプロセスで動かすと Python の処理を長く握り、画面が数秒ずつ固まる。
    そのため別プロセス（このファイルを --paddle-worker で起動）で動かし、
    1 行の JSON で画像のパスと結果をやり取りする。優先度を下げて画面の操作を妨げないようにする。
    """
    _LOG_NAME = "paddle.log"

    def __init__(self, model: str) -> None:
        self.model = model
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return f"paddle-{self.model}"

    @staticmethod
    def is_installed() -> bool:
        return importlib.util.find_spec("paddleocr") is not None

    @property
    def is_loaded(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def log_path(self) -> Path:
        return DATA_DIR / "logs" / self._LOG_NAME

    def load(self, progress_cb: "Callable[[str], None] | None" = None) -> None:
        if self.is_loaded:
            return
        if progress_cb:
            progress_cb(f"PaddleOCR（{self.model}）を起動中..."
                        "（初回はモデルを自動でダウンロードするため、数分かかることがあります）")
        t0 = time.time()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, "w", encoding="utf-8")
        flags = (getattr(subprocess, "CREATE_NO_WINDOW", 0)
                 | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0))
        try:
            self._proc = subprocess.Popen(
                [sys.executable, "-W", "ignore", str(Path(__file__).resolve()),
                 "--paddle-worker", self.model],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                cwd=str(APP_DIR), creationflags=flags,
                # paddle は設定に関係なくホームフォルダに .cache\paddle などを作るので、
                # 子プロセスのホームフォルダを data\models の中に向ける
                env={**os.environ, "PYTHONIOENCODING": "utf-8",
                     "USERPROFILE": str(MODELS_DIR / "paddle_home"),
                     "HOME": str(MODELS_DIR / "paddle_home")})
        finally:
            log.close()   # 子プロセスが自分の分を持っている
        reply = self._receive()
        if "error" in reply:
            self.close()
            raise RuntimeError(f"PaddleOCR を読み込めませんでした（{reply['error']}）")
        if progress_cb:
            progress_cb(f"PaddleOCR 起動完了 ({time.time() - t0:.1f}秒)")

    def _receive(self) -> dict:
        line = self._proc.stdout.readline() if self._proc else b""
        if not line:
            raise RuntimeError(f"PaddleOCR が止まりました（詳しくは {self.log_path}）")
        return json.loads(line.decode("utf-8"))

    def recognize(self, image: Image.Image) -> PageText:
        """行ごとに読み、同じ高さに並ぶ項目（「第14話」「タイトル…003」など）は 1 行にまとめる。"""
        with self._lock:
            if not self.is_loaded:
                self.load()   # 途中で止まっていたら起動し直す
            TMP_DIR.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix="paddle_", suffix=".bmp", dir=TMP_DIR)
            os.close(fd)
            try:
                image.convert("RGB").save(tmp)   # 圧縮しない BMP で素早く渡す
                self._proc.stdin.write((json.dumps({"path": tmp}) + "\n").encode("utf-8"))
                self._proc.stdin.flush()
                reply = self._receive()
            finally:
                Path(tmp).unlink(missing_ok=True)
        if "error" in reply:
            raise RuntimeError(f"PaddleOCR で読めませんでした（{reply['error']}）")
        items = [(text, [int(v) for v in box]) for text, box in reply["items"] if text.strip()]

        rows: list[list[tuple[str, list[int]]]] = []
        for text, box in sorted(items, key=lambda it: (it[1][1] + it[1][3]) / 2):
            cy, h = (box[1] + box[3]) / 2, box[3] - box[1]
            if rows:
                last = rows[-1][-1][1]
                if abs(cy - (last[1] + last[3]) / 2) <= max(h, last[3] - last[1]) / 2:
                    rows[-1].append((text, box))
                    continue
            rows.append([(text, box)])
        lines = [" ".join(t for t, _b in sorted(row, key=lambda it: it[1][0])) for row in rows]
        blocks = [[*box, text] for text, box in items]
        return PageText("\n".join(lines), blocks, self.name, 1.0)

    def close(self) -> None:
        """子プロセスを終える（アプリの終了時など）。"""
        proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.stdin.close()   # 子プロセスは入力の終わりで終了する
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


def _paddle_worker_main(model: str) -> int:
    """
    PaddleOCR の子プロセス（PaddleEngine が起動する）。
    標準入力から {"path": 画像} を 1 行ずつ受け取り、{"items": [[文字, [x1, y1, x2, y2]], ...]} を返す。
    """
    # ライブラリが標準出力に書く文字がやり取りに混ざらないよう、やり取りは複製した出力で行い、
    # 標準出力はログ（標準エラー）へ回す
    reply = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def send(obj: dict) -> None:
        reply.write(json.dumps(obj, ensure_ascii=False) + "\n")
        reply.flush()

    # 共有環境では setuptools と標準の distutils が食い違って paddle の import が失敗することがある
    if sys.version_info < (3, 12):
        os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "stdlib")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    _patch_numpy_compat()
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(
            text_detection_model_name=f"PP-OCRv6_{model}_det",
            text_recognition_model_name=f"PP-OCRv6_{model}_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,   # CPU 高速化（oneDNN）は一部のモデルで NotImplementedError になる
        )
    except Exception as exc:
        send({"error": f"{type(exc).__name__}: {exc}"})
        return 1
    send({"ready": True})
    for line in sys.stdin:
        try:
            path = json.loads(line)["path"]
            # cv2.imread は日本語を含むパスを読めないので、バイト列から読む
            bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            result = ocr.predict(bgr)[0]
            send({"items": [[t, [int(v) for v in b]]
                            for t, b in zip(result["rec_texts"], result["rec_boxes"])]})
        except Exception as exc:
            send({"error": f"{type(exc).__name__}: {exc}"})
    return 0


# ════════════════════════════════════════════════════════════════════════
# アプリケーション: インデックス作成ワーカー
# ════════════════════════════════════════════════════════════════════════

@dataclass
class _WorkerMessage:
    kind: str    # "log" | "progress" | "done" | "error"
    payload: object = None


class IndexControl:
    """
    インデックス作成の一時停止・停止。画面のボタンから操作し、作業スレッドが 1 枚ごとに確かめる。
    停止しても登録済みの分は索引に残り、次回は続き（まだ登録していない画像）から処理される。
    """

    def __init__(self) -> None:
        self._running = threading.Event()
        self._running.set()
        self._stop = threading.Event()
        self._paused_at: float | None = None
        self._paused_total = 0.0

    def pause(self) -> None:
        if self._paused_at is None:
            self._paused_at = time.time()
        self._running.clear()

    def resume(self) -> None:
        if self._paused_at is not None:
            self._paused_total += time.time() - self._paused_at
            self._paused_at = None
        self._running.set()

    def stop(self) -> None:
        self._stop.set()
        self.resume()   # 一時停止中でもすぐ止められるように待ちを解く

    @property
    def paused_seconds(self) -> float:
        """一時停止していた時間の合計（処理時間から除くため）。"""
        current = time.time() - self._paused_at if self._paused_at is not None else 0.0
        return self._paused_total + current

    @property
    def paused(self) -> bool:
        return not self._running.is_set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def checkpoint(self) -> bool:
        """一時停止中なら再開まで待つ。続けてよければ True、停止なら False。"""
        self._running.wait()
        return not self._stop.is_set()


@dataclass
class _Job:
    """OCR する画像 1 枚（通常の画像、または書庫内の画像）。"""
    relpath: str                # 索引に記録する名前
    path: Path                  # 画像ファイル、または書庫ファイル
    member: str | None          # 書庫内のパス（通常の画像は None）
    prev: IndexedRow | None     # 前回の登録内容（読み直しで使う）
    mode: str                   # "full"（mokuro から） / "reread"（PaddleOCR で読み直すだけ）
    stat: os.stat_result        # 画像ファイル（書庫内なら書庫ファイル）の日時・サイズ
    sig: str | None             # 書庫内の画像の CRC32:サイズ


class IndexWorker(threading.Thread):
    """
    フォルダ内画像を走査し、新規・更新分だけ OCR してインデックスへ反映する。
    paddle を渡すと、横書きの文字しか見つからなかったページ（目次・扉・説明文など）を
    PaddleOCR で読み直す。漫画の本編（縦書きの吹き出し）は mokuro の結果のまま。
    """

    def __init__(
        self,
        engine: MokuroEngine,
        index: TextIndex,
        files: list[Path],
        msg_queue: "queue.Queue[_WorkerMessage]",
        *,
        max_image_size: int,
        paddle: PaddleEngine | None = None,
        control: IndexControl | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self._engine         = engine
        self._index          = index
        self._files          = files
        self._q              = msg_queue
        self._max_image_size = max_image_size
        self._paddle         = paddle
        self._control        = control or IndexControl()

    def run(self) -> None:
        try:
            self._process_all()
        except Exception as exc:
            self._q.put(_WorkerMessage("error", str(exc)))

    def _log(self, msg: str) -> None:
        self._q.put(_WorkerMessage("log", msg))

    def _wants_paddle(self, hratio: float | None, n_chars: int) -> bool:
        """横書きの文字がほとんどで、ある程度の量の文字があるページだけ PaddleOCR で読み直す。"""
        return (self._paddle is not None and hratio is not None and hratio >= _PADDLE_MIN_HRATIO
                and n_chars >= _PADDLE_MIN_CHARS)

    def _ensure_paddle(self) -> bool:
        """PaddleOCR を必要になった時点で読み込む。失敗したら以降は使わない。"""
        if self._paddle is None:
            return False
        if not self._paddle.is_loaded:
            try:
                self._paddle.load(progress_cb=self._log)
            except Exception as exc:
                self._log(f"⚠ {exc}\n   PaddleOCR を使わずに続けます（mokuro の結果を登録します）。")
                self._paddle = None
                return False
        return True

    def _classify(self, prev: IndexedRow | None, unchanged: bool) -> str:
        """
        画像 1 枚をどう扱うか: "full"（mokuro から読む） / "reread"（PaddleOCR で読み直すだけ）
        / "skip"（今の登録のまま）。
        """
        if prev is None or not unchanged:
            return "full"
        if not prev.has_blocks:
            self._upgraded += 1   # 画像は同じだが、吹き出しの位置が未登録（古い索引）
            return "full"
        if self._paddle is not None and prev.hratio is None:
            self._unjudged += 1   # 縦書き・横書きの判定が未登録（古い索引）なので mokuro から
            return "full"
        if self._wants_paddle(prev.hratio, prev.n_chars) and prev.engine != self._paddle.name:
            return "reread"
        return "skip"

    def _plan_archive(self, path: Path, rel: str, st: os.stat_result,
                      existing: dict[str, IndexedRow], keep: set[str],
                      jobs: list[_Job], restat: list[tuple[float, int, str]]) -> int:
        """書庫の中の画像を洗い出す。スキップした枚数を返す。"""
        prefix = rel + _ARCHIVE_SEP
        prev_rows = {r: row for r, row in existing.items() if r.startswith(prefix)}
        record = self._archives.get(rel)
        state = (record[2] if record and abs(record[0] - st.st_mtime) < 1e-6
                 and record[1] == st.st_size else None)   # 書庫が変わっていれば None
        if state in ("empty", "error") and not prev_rows:
            self._archive_keep.add(rel)   # 前回、読める画像が無いと分かった書庫（変わっていない）
            return 0
        skipped = 0
        if state == "done":
            # 前回最後まで処理した書庫で、変わっていない → 開かずに済ませる。
            # 途中で止めた書庫（pending）は開き直して、まだ登録していない画像を処理する。
            self._archive_keep.add(rel)
            for r, row in prev_rows.items():
                keep.add(r)
                mode = self._classify(row, True)
                if mode == "skip":
                    skipped += 1
                else:
                    jobs.append(_Job(r, path, split_relpath(r)[1], row, mode, st, row.sig))
            return skipped
        try:
            with open_archive(path) as archive:
                members = archive.members()
                encrypted = archive.skipped_encrypted
        except Exception as exc:
            kept = "（前回の登録内容は残します）" if prev_rows else ""
            self._log(f"⚠ 書庫を開けません: {rel}\n    {exc}{kept}")
            keep.update(prev_rows)
            if not prev_rows and not getattr(exc, "retry", False):
                # 壊れているなど。書庫が変わるまでは開き直さない（道具が無いだけなら次回も試す）
                self._index.set_archive(rel, st.st_mtime, st.st_size, "error", str(exc))
                self._archive_keep.add(rel)
            return len(prev_rows)
        self._n_archives += 1
        self._archive_keep.add(rel)
        if encrypted:
            self._log(f"⚠ {rel}: パスワード付きの画像 {encrypted} 枚は読めないため飛ばします")
        if not members:
            if not encrypted:
                self._log(f"ℹ {rel}: 画像が入っていません")
            self._index.set_archive(rel, st.st_mtime, st.st_size, "empty",
                                    "パスワード付き" if encrypted else "画像なし")
            return 0
        n_jobs = 0
        for member in members:
            r = prefix + member.name
            keep.add(r)
            prev = existing.get(r)
            mode = self._classify(prev, prev is not None and prev.sig == member.sig)
            if mode == "skip":
                # 中身は同じ。書庫の日時・サイズだけ更新して、次回は書庫を開かずに済むようにする
                restat.append((st.st_mtime, st.st_size, r))
                skipped += 1
            else:
                jobs.append(_Job(r, path, member.name, prev, mode, st, member.sig))
                n_jobs += 1
        if n_jobs:
            # 処理し終えるまでは pending（途中で止めても、次回は開き直して残りを処理する）
            self._index.set_archive(rel, st.st_mtime, st.st_size, "pending")
            self._archive_left[rel] = n_jobs
        else:
            self._index.set_archive(rel, st.st_mtime, st.st_size, "done")
        return skipped

    def _archive_job_finished(self, job: _Job, ok: bool) -> None:
        """書庫内の画像を 1 枚処理した。その書庫の分が全部成功したら done にする。"""
        rel = split_relpath(job.relpath)[0]
        if rel not in self._archive_left:
            return   # done の書庫の読み直し（書庫の状態は変わらない）
        if not ok:
            self._archive_failed.add(rel)
        self._archive_left[rel] -= 1
        if self._archive_left[rel] == 0 and rel not in self._archive_failed:
            self._index.set_archive(rel, job.stat.st_mtime, job.stat.st_size, "done")

    def _process_all(self) -> None:
        begin = time.time()   # 書庫の確認なども含めた処理時間を出すため
        root = self._index.root
        existing = self._index.load_existing()

        # 新規・更新が必要な画像を洗い出す（書庫は中の画像まで）
        jobs: list[_Job] = []
        restat: list[tuple[float, int, str]] = []
        keep: set[str] = set()
        skipped = 0
        self._upgraded = self._unjudged = self._n_archives = 0
        self._archives = self._index.load_archives()
        self._archive_keep: set[str] = set()        # 今回も存在する書庫
        self._archive_left: dict[str, int] = {}     # 書庫 → まだ処理していない画像の数
        self._archive_failed: set[str] = set()      # 失敗した画像がある書庫（次回も開き直す）
        for p in self._files:
            if not self._control.checkpoint():
                # 洗い出しの途中で止めた。どの画像が消えたかは分からないので、削除もしない
                self._log("■ 停止しました（索引は変更していません）")
                self._q.put(_WorkerMessage("stopped", str(self._index.count())))
                return
            rel = str(p.relative_to(root))
            try:
                st = p.stat()
            except OSError:
                continue
            if p.suffix.lower() in _ARCHIVE_EXTS:
                skipped += self._plan_archive(p, rel, st, existing, keep, jobs, restat)
                continue
            keep.add(rel)
            prev = existing.get(rel)
            unchanged = (prev is not None and abs(prev.mtime - st.st_mtime) < 1e-6
                         and prev.size == st.st_size)
            mode = self._classify(prev, unchanged)
            if mode == "skip":
                skipped += 1
            else:
                jobs.append(_Job(rel, p, None, prev if mode == "reread" else None, mode, st, None))

        # 消えた画像（書庫ごと消えた・書庫から消えたものも）をインデックスから削除
        removed = [r for r in existing if r not in keep]
        if removed:
            self._index.delete_many(removed)
            self._log(f"🗑 消えた画像 {len(removed)} 件をインデックスから削除しました")
        self._index.update_stats(restat)
        # 消えた書庫の記録を外す
        self._index.forget_archives([r for r in self._archives if r not in self._archive_keep])

        # 書庫ごとにまとめて処理する（書庫は 1 回だけ開いて展開する）
        jobs.sort(key=lambda j: (_natural_sort_key(str(j.path.relative_to(root))),
                                 _natural_sort_key(j.member or "")))
        n_reread = sum(1 for j in jobs if j.mode == "reread")
        total = len(jobs)
        size_label = f"上限 {self._max_image_size}px" if self._max_image_size > 0 else "原寸"
        if self._upgraded:
            self._log(f"ℹ 吹き出しの位置が未登録の画像 {self._upgraded} 件を OCR し直します"
                      "（コマの切り抜きに使います。初回のみ）")
        if self._unjudged:
            self._log(f"ℹ 縦書き・横書きの判定が未登録の画像 {self._unjudged} 件を OCR し直します"
                      "（横書きのページを見分けるため。初回のみ）")
        if n_reread:
            self._log(f"ℹ 横書きのページ {n_reread} 件を PaddleOCR（{self._paddle.model}）"
                      "で読み直します")
        paddle_label = f"PaddleOCR {self._paddle.model}" if self._paddle else "使わない"
        archive_label = f"（書庫 {self._n_archives} 個を開きました）" if self._n_archives else ""
        self._log(
            f"対象画像: {total + skipped} 枚{archive_label}  "
            f"（OCR 実行: {total} 枚 / スキップ: {skipped} 枚）\n"
            f"[画像サイズ: {size_label} / 横書きページの読み直し: {paddle_label}]\n"
        )
        self._q.put(_WorkerMessage("progress", (0, max(total, 1))))

        if total == 0:
            self._log("更新はありませんでした（インデックスは最新です）。")
            self._q.put(_WorkerMessage("done", str(self._index.count())))
            return

        start = time.time()
        n_ok = n_err = n_paddle = 0
        stopped = False
        reader: ArchiveReader | None = None
        reader_error: str | None = None
        try:
            for idx, job in enumerate(jobs, start=1):
                if not self._control.checkpoint():   # 一時停止中はここで待つ
                    stopped = True
                    break
                self._log(f"[{idx}/{total}] {display_path(job.relpath)}")
                t0 = time.time()
                try:
                    if job.member is not None and (reader is None or reader.path != job.path):
                        # 次の書庫: 前の書庫を閉じ（一時ファイルも消える）、この書庫の分をまとめて展開
                        if reader is not None:
                            reader.close()
                        reader, reader_error = None, None
                        try:
                            reader = open_archive(job.path)
                            reader.prefetch([j.member for j in jobs if j.path == job.path])
                        except ArchiveError as exc:
                            reader_error = str(exc)
                    if job.member is not None and reader is None:
                        raise ArchiveError(reader_error or "書庫を開けません")
                    image = load_image(root, job.relpath, reader)
                    page, n_used_paddle = self._recognize(job, image)
                    if page is None:
                        skipped += 1   # PaddleOCR が使えなくなった読み直し分は今のまま
                        self._log("    ― PaddleOCR が使えないため、今の登録内容のままにします")
                    else:
                        self._index.upsert(job.relpath, page, job.stat.st_mtime,
                                           job.stat.st_size, job.sig)
                        n_ok += 1
                        n_paddle += n_used_paddle
                        self._log(f"    ✅ 完了 ({_format_seconds(time.time() - t0)})")
                    ok = True
                except Exception as exc:
                    n_err += 1
                    ok = False
                    self._log(f"    ⚠ エラー ({_format_seconds(time.time() - t0)}): {exc}")
                if job.member is not None:
                    self._archive_job_finished(job, ok)
                self._q.put(_WorkerMessage("progress", (idx, total)))
        finally:
            if reader is not None:
                reader.close()

        # 一時停止していた時間は数えない
        paused = self._control.paused_seconds
        ocr_time = max(0.0, time.time() - start - paused)
        whole = max(0.0, time.time() - begin - paused)
        done = n_ok + n_err
        self._log(
            f"\n{'=' * 50}\n"
            + ("■ 停止しました（残りは次回の「インデックス作成/更新」で続きから処理します）\n"
               if stopped else "✅ インデックス更新完了\n")
            + f"OCR 成功: {n_ok} 件（うち PaddleOCR: {n_paddle} 件） / 失敗: {n_err} 件 / "
            f"スキップ: {skipped} 件" + (f" / 未処理: {total - done} 件" if stopped else "") + "\n"
            f"インデックス総数: {self._index.count()} 件\n"
            f"処理時間: {_format_elapsed(whole)}（うち OCR {_format_elapsed(ocr_time)}"
            + (f"・1 枚あたり平均 {ocr_time / done:.1f}秒" if done else "") + "）"
        )
        self._q.put(_WorkerMessage("stopped" if stopped else "done", str(self._index.count())))

    def _recognize(self, job: "_Job", image: Image.Image) -> tuple[PageText | None, int]:
        """mokuro で読み、横書きのページなら PaddleOCR で読み直す。(結果, PaddleOCR を使ったか)"""
        if job.mode == "full":
            t1 = time.time()
            page = self._engine.recognize(image, max_image_size=self._max_image_size)
            hratio, n_chars = page.hratio, _meaningful_chars(page.text)
            mokuro_time = time.time() - t1
        else:
            page, hratio, n_chars = None, job.prev.hratio, job.prev.n_chars
        if self._wants_paddle(hratio, n_chars) and self._ensure_paddle():
            t1 = time.time()
            paddle_page = self._paddle.recognize(image)
            paddle_page.hratio = hratio
            if paddle_page.text.strip() or page is None:
                where = f"mokuro {_format_seconds(mokuro_time)} → " if page is not None else ""
                self._log(f"    ↳ 横書きのページ → PaddleOCR で読み直し "
                          f"({where}PaddleOCR {_format_seconds(time.time() - t1)})")
                return paddle_page, 1
        return page, 0


# ════════════════════════════════════════════════════════════════════════
# 設定の保存（最近インデックスを作ったフォルダなど）
# ════════════════════════════════════════════════════════════════════════

class Settings:
    """
    設定を data/settings.json に保存する。
    読み書きに失敗しても動作は続ける（設定が引き継がれないだけ）。
    """
    _MAX_RECENT = 20

    def __init__(self) -> None:
        self._path = DATA_DIR / "settings.json"
        self._migrate_from_appdata()
        self._data: dict = {}
        try:
            # メモ帳などで保存すると先頭に BOM が付くことがあるので utf-8-sig で読む
            self._data = json.loads(self._path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            pass

    def _migrate_from_appdata(self) -> None:
        """開発途中の版が %APPDATA%\\KomaSagashi に置いていた設定を data/ へ移す。"""
        appdata = os.environ.get("APPDATA")
        if not appdata or self._path.exists():
            return
        old = Path(appdata) / "KomaSagashi" / "settings.json"
        try:
            if old.is_file():
                self._path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(old), str(self._path))
                old.parent.rmdir()   # 空になったフォルダも消す（他のファイルがあれば残る）
        except OSError:
            pass

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            tmp.replace(self._path)
        except OSError:
            pass

    def recent_folders(self) -> list[str]:
        """インデックスのあるフォルダを新しい順に返す（索引が消えたもの・フォルダが無いものは除く）。"""
        return [f for f in self._data.get("recent_folders", [])
                if isinstance(f, str) and Path(f).is_dir()
                and (index_path_for(Path(f)).is_file()
                     or (Path(f) / _LEGACY_DB_FILENAME).is_file())]

    def add_recent_folder(self, folder: Path) -> None:
        path = str(folder)
        key = os.path.normcase(os.path.abspath(path))
        rest = [f for f in self._data.get("recent_folders", [])
                if isinstance(f, str) and os.path.normcase(os.path.abspath(f)) != key]
        self._data["recent_folders"] = [path] + rest[: self._MAX_RECENT - 1]
        self._save()

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def set(self, key: str, value) -> None:
        if self._data.get(key) != value:
            self._data[key] = value
            self._save()


def parse_percent(text: str) -> float | None:
    """'150%' や '150' を 1.5 にする。数値でなければ None。"""
    try:
        value = float(text.strip().rstrip("%").strip())
    except ValueError:
        return None
    return value / 100 if value > 0 else None


def format_percent(scale: float) -> str:
    pct = scale * 100
    return f"{pct:.0f}%" if pct >= 10 else f"{pct:.1f}%"


# ════════════════════════════════════════════════════════════════════════
# 書き出し（CSV / JSON / TXT）
# ════════════════════════════════════════════════════════════════════════

EXPORT_FORMATS = {".csv": "csv", ".json": "json", ".txt": "txt"}


def export_results(path: Path, rows: list[dict], *, folder: Path,
                   terms: list[str] | None = None, fuzzy: bool = True) -> int:
    """
    登録内容（TextIndex.export_rows）を path の拡張子の形式で書き出し、件数を返す。
    terms を渡すと検索結果として書き出す（ヒット箇所の列・検索条件が付く）。
      CSV  … Excel 向け。1 ページ 1 行。UTF-8（BOM 付き。Excel で文字化けしない）
      JSON … すべての情報（吹き出しの位置・OCR エンジンなど）。別のプログラムで使う向け
      TXT  … 読む・文章を貼り付ける向け。ページ名と全文を順に並べる
    """
    fmt = EXPORT_FORMATS.get(path.suffix.lower())
    if fmt is None:
        raise ValueError(f"対応していない形式です: {path.suffix}（.csv / .json / .txt）")
    items = []
    for row in rows:
        outer, member = split_relpath(row["relpath"])
        items.append({
            "name": PurePosixPath(member).name if member else Path(outer).name,
            "archive": Path(outer).name if member else "",
            "location": display_path(row["relpath"]),
            "snippet": make_snippet(row["text"], terms, fuzzy) if terms else "",
            **row,
        })
    exported_at = time.strftime("%Y-%m-%d %H:%M:%S")
    if fmt == "csv":
        import csv
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            header = ["画像名", "書庫名", "場所"] + (["ヒット箇所"] if terms else []) + ["全文", "OCR エンジン"]
            w.writerow(header)
            for it in items:
                w.writerow([it["name"], it["archive"], it["location"]]
                           + ([it["snippet"]] if terms else []) + [it["text"], it["engine"]])
    elif fmt == "json":
        data = {
            "app": "KomaSagashi",
            "version": __version__,
            "exported_at": exported_at,
            "folder": str(folder),
            "search": {"terms": terms, "ignore_variants": fuzzy} if terms else None,
            "count": len(items),
            "items": [{
                "name": it["name"],
                "archive": it["archive"] or None,
                "relpath": it["relpath"],
                **({"snippet": it["snippet"]} if terms else {}),
                "text": it["text"],
                "lines": it["text"].splitlines(),
                "engine": it["engine"],
                # 吹き出し（PaddleOCR なら行）ごとの位置。座標は元画像の px
                "blocks": [{"box": list(box), "text": text} for box, text in it["blocks"]],
            } for it in items],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        title = (f"検索結果: {' '.join(terms)}（{len(items)} 件）" if terms
                 else f"索引全体（{len(items)} 件）")
        out = [f"KomaSagashi v{__version__} 書き出し", f"対象フォルダ: {folder}", title,
               f"書き出し日時: {exported_at}", "=" * 60, ""]
        for it in items:
            out.append(f"■ {it['name']}" + (f"（{it['archive']}）" if it["archive"] else ""))
            out.append(it["text"] if it["text"].strip() else "（文字なし）")
            out.append("")
        path.write_text("\n".join(out), encoding="utf-8")
    return len(items)


def export_file_name(stem: str) -> str:
    """書き出すファイルの名前に使えない文字を置き換える。"""
    return re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", stem).strip(" .")[:80] or "komasagashi"


# ════════════════════════════════════════════════════════════════════════
# UI: 切り抜き枠を調整できる画像キャンバス
# ════════════════════════════════════════════════════════════════════════

class CropCanvas(tk.Canvas):
    """
    画像を表示し、切り抜き枠（赤）をマウスで調整できるキャンバス。
      ・枠の辺や角をドラッグ       → 大きさを変える
      ・枠の内側をドラッグ         → 枠を移動する
      ・点線のコマをクリック       → そのコマを枠にする
      ・Shift + ドラッグ           → 新しく枠を描く
      ・枠の外をドラッグ / ホイールボタンでドラッグ / 手モードでドラッグ
                                   → 表示位置を動かす（制限なし。「全体表示」で中央に戻る）
      ・ホイール                   → マウスの位置を中心に拡大・縮小
    座標は元画像の px で持ち、表示するときだけ拡大・縮小する。
    枠は「補正する前の枠」（_base）と補正（%）から求める。手で直した枠は、
    補正を差し引いて _base に戻しておくので、補正を変えても直した形を基準に広がる・狭まる。
    表示は見えている範囲だけを切り出して描くので、高倍率でも重くならない。
    """
    ZOOM_LEVELS = (0.1, 0.15, 0.2, 0.25, 0.33, 0.5, 0.67, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0,
                   6.0, 8.0)
    MIN_ZOOM, MAX_ZOOM = 0.05, 8.0
    _FIT_MAX    = 2.0    # 「全体表示」で小さい画像を拡大する上限
    _WHEEL_STEP = 1.25   # ホイール 1 目盛りあたりの倍率
    _EDGE_TOL   = 6      # 辺をつかめる距離（表示 px）
    _HANDLE     = 4      # 角・辺の中点に描く四角の半径（表示 px）
    _DRAG_START = 4      # これ以上動いたらドラッグとみなす（表示 px）
    _MIN_SIZE   = 8      # 枠の最小サイズ（元画像 px）
    _CURSORS = {
        "nw": "top_left_corner", "ne": "top_right_corner",
        "sw": "bottom_left_corner", "se": "bottom_right_corner",
        "n": "top_side", "s": "bottom_side", "w": "left_side", "e": "right_side",
        "move": "fleur",
    }

    def __init__(self, master: tk.Misc, on_change: Callable[[Box | None], None],
                 on_zoom: Callable[[float], None], **kwargs) -> None:
        super().__init__(master, highlightthickness=0, background="#d9d9d9",
                         cursor="arrow", **kwargs)
        self._on_change = on_change
        self._on_zoom = on_zoom
        self._image: Image.Image | None = None
        self._imgtk = None                   # GC 防止のため参照を保持
        self._panels: list[Box] = []
        self._hits: list[Box] = []
        self._sel: list[float] | None = None  # [x1, y1, x2, y2]（元画像の px、補正後）
        self._base: list[float] | None = None # 補正する前の枠（画像の外にはみ出してもよい）
        self._adjust = 0                      # 枠の補正（%）
        self._message = ""
        self._fit = True                      # True: 画面に収まる倍率に自動で合わせる
        self._scale, self._ox, self._oy = 1.0, 0.0, 0.0   # 倍率と、画像左上の表示位置
        self._hand = False                    # 手モード: 左ドラッグは常に表示位置の移動
        self._drag: dict | None = None
        self._resize_job: str | None = None
        self._render_job: str | None = None

        self.bind("<Configure>", self._on_configure)
        self.bind("<Destroy>", self._on_destroy)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<ButtonPress-2>", self._on_pan_start)
        self.bind("<B2-Motion>", self._on_drag)
        self.bind("<ButtonRelease-2>", self._on_release)
        self.bind("<MouseWheel>", self._on_wheel)
        self.bind("<Motion>", self._on_hover)

    # ── 外部 API ──────────────────────────────────────────────────────

    def show(self, image: Image.Image, panels: list[Box], hits: list[Box],
             selection: Box | None) -> None:
        self._image, self._panels, self._hits = image, panels, hits
        self._set_base(selection)
        self._drag = None
        self._fit = True
        self._render()
        self._on_change(self.selection())

    def clear(self, message: str) -> None:
        self._image, self._imgtk = None, None
        self._panels, self._hits, self._sel, self._base = [], [], None, None
        self._message = message
        self._render()
        self._on_change(None)

    @property
    def has_image(self) -> bool:
        return self._image is not None

    @property
    def scale(self) -> float:
        """現在の表示倍率。"""
        return self._scale

    def selection(self) -> Box | None:
        if self._image is None or self._sel is None:
            return None
        x1, y1, x2, y2 = (int(round(v)) for v in self._sel)
        return (x1, y1, x2, y2)

    def crop(self) -> Image.Image | None:
        box = self.selection()
        return self._image.crop(box) if self._image is not None and box else None

    def set_adjust(self, percent: int) -> None:
        """枠の補正（%）を変え、今の枠をその場で広げる・狭める。"""
        self._adjust = percent
        if self._image is None or self._base is None:
            return
        self._sel = adjust_box(self._base, percent, self._image.size)
        self._draw_overlay()
        self._on_change(self.selection())

    def _set_base(self, box) -> None:
        """補正する前の枠を決め、補正した枠を表示用に求める。"""
        self._base = list(box) if box else None
        self._sel = (adjust_box(self._base, self._adjust, self._image.size)
                     if self._base and self._image is not None else None)

    def _sync_base(self) -> None:
        """手で動かした枠から補正を差し引いて、補正する前の枠に戻す。"""
        self._base = (scale_box(self._sel, 1 / (1 + self._adjust / 100))
                      if self._sel else None)

    def set_hand(self, on: bool) -> None:
        self._hand = on
        self.config(cursor="hand2" if on else "arrow")

    def zoom_fit(self) -> None:
        self._fit = True
        self._request_render()

    def zoom_to(self, scale: float, anchor: tuple[float, float] | None = None) -> None:
        """倍率を変える。anchor（表示座標）の下にある画像の点が動かないようにする。"""
        if self._image is None:
            return
        scale = min(max(scale, self.MIN_ZOOM), self.MAX_ZOOM)
        if anchor is None:
            anchor = (self.winfo_width() / 2, self.winfo_height() / 2)
        ax, ay = anchor
        ix, iy = (ax - self._ox) / self._scale, (ay - self._oy) / self._scale
        self._fit = False
        self._scale = scale
        self._ox, self._oy = ax - ix * scale, ay - iy * scale
        self._request_render()

    def zoom_step(self, direction: int) -> None:
        """ZOOM_LEVELS の次の段階へ（direction: +1 拡大 / -1 縮小）。"""
        cur = self._scale
        if direction > 0:
            nxt = next((z for z in self.ZOOM_LEVELS if z > cur * 1.01), self.MAX_ZOOM)
        else:
            nxt = next((z for z in reversed(self.ZOOM_LEVELS) if z < cur / 1.01), self.MIN_ZOOM)
        self.zoom_to(nxt)

    # ── 描画 ──────────────────────────────────────────────────────────

    def _on_configure(self, _event=None) -> None:
        # ウィンドウのリサイズ中に何度も描き直さないよう、少し待ってから描く
        if self._resize_job:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(60, self._render)

    def _request_render(self) -> None:
        """ドラッグ中の連続した描き直しを 1 回にまとめる。"""
        if self._render_job is None:
            self._render_job = self.after_idle(self._render)

    def _on_destroy(self, event) -> None:
        # 閉じた後に予約済みの描画が走らないようにする
        if event.widget is not self:
            return
        for job in (self._resize_job, self._render_job):
            if job:
                self.after_cancel(job)
        self._resize_job = self._render_job = None

    def _fit_scale(self) -> float:
        iw, ih = self._image.size
        cw, ch = max(1, self.winfo_width()), max(1, self.winfo_height())
        return min(cw / iw, ch / ih, self._FIT_MAX)

    def _center_view(self) -> None:
        """画像を画面の中央に置く（全体表示のとき）。"""
        iw, ih = self._image.size
        cw, ch = self.winfo_width(), self.winfo_height()
        self._ox = (cw - iw * self._scale) / 2
        self._oy = (ch - ih * self._scale) / 2

    def _render(self) -> None:
        for job in (self._resize_job, self._render_job):
            if job:
                self.after_cancel(job)
        self._resize_job = self._render_job = None
        self.delete("all")
        cw, ch = max(1, self.winfo_width()), max(1, self.winfo_height())
        if self._image is None:
            self.create_text(cw / 2, ch / 2, text=self._message, fill="#555555",
                             width=max(100, cw - 40), justify="center")
            return
        # 全体表示のときだけ中央に置く。それ以外は表示位置を制限しない
        # （画像の端を画面の中央まで持ってくるなど、自由に動かせる）
        if self._fit:
            self._scale = self._fit_scale()
            self._center_view()

        # 見えている範囲だけを切り出して拡大・縮小する
        s = self._scale
        iw, ih = self._image.size
        x0 = max(0, int(-self._ox / s))
        y0 = max(0, int(-self._oy / s))
        x1 = min(iw, int((cw - self._ox) / s) + 1)
        y1 = min(ih, int((ch - self._oy) / s) + 1)
        if x1 > x0 and y1 > y0:
            region = self._image.crop((x0, y0, x1, y1))
            size = (max(1, round((x1 - x0) * s)), max(1, round((y1 - y0) * s)))
            if s < 1:
                region = region.resize(size, Image.LANCZOS, reducing_gap=3.0)
            else:
                # 高倍率では画素の境目が分かるよう補間しない
                region = region.resize(size, Image.NEAREST if s >= 3 else Image.BICUBIC)
            self._imgtk = ImageTk.PhotoImage(region)
            self.create_image(x0 * s + self._ox, y0 * s + self._oy,
                              image=self._imgtk, anchor="nw")
        self._draw_overlay()
        self._on_zoom(s)

    def _draw_overlay(self) -> None:
        self.delete("overlay")
        for b in self._panels:
            self.create_rectangle(*self._to_canvas(b), outline="#1e6fd9",
                                  dash=(4, 3), width=1, tags="overlay")
        for b in self._hits:
            self.create_rectangle(*self._to_canvas(b), outline="#ff9900",
                                  width=2, tags="overlay")
        if self._sel is None:
            return
        x1, y1, x2, y2 = self._to_canvas(self._sel)
        self.create_rectangle(x1, y1, x2, y2, outline="#e00000", width=2, tags="overlay")
        r = self._HANDLE
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        for hx, hy in ((x1, y1), (mx, y1), (x2, y1), (x2, my),
                       (x2, y2), (mx, y2), (x1, y2), (x1, my)):
            self.create_rectangle(hx - r, hy - r, hx + r, hy + r, fill="white",
                                  outline="#e00000", tags="overlay")

    # ── 座標変換 ──────────────────────────────────────────────────────

    def _to_canvas(self, box) -> tuple[float, float, float, float]:
        s = self._scale
        return (box[0] * s + self._ox, box[1] * s + self._oy,
                box[2] * s + self._ox, box[3] * s + self._oy)

    def _to_image(self, cx: float, cy: float) -> tuple[float, float]:
        """表示座標 → 元画像の座標（画像の外は端に寄せる）。"""
        iw, ih = self._image.size
        x = (cx - self._ox) / self._scale
        y = (cy - self._oy) / self._scale
        return min(max(x, 0.0), iw), min(max(y, 0.0), ih)

    # ── マウス操作 ────────────────────────────────────────────────────

    def _hit_test(self, cx: float, cy: float) -> str | None:
        """カーソル位置が枠のどこか（'n' 'se' などの辺・角 / 'move' / None）。"""
        if self._sel is None:
            return None
        x1, y1, x2, y2 = self._to_canvas(self._sel)
        t = self._EDGE_TOL
        if not (x1 - t <= cx <= x2 + t and y1 - t <= cy <= y2 + t):
            return None
        edge = ""
        if abs(cy - y1) <= t:
            edge += "n"
        elif abs(cy - y2) <= t:
            edge += "s"
        if abs(cx - x1) <= t:
            edge += "w"
        elif abs(cx - x2) <= t:
            edge += "e"
        if edge:
            return edge
        return "move" if x1 < cx < x2 and y1 < cy < y2 else None

    def _on_hover(self, event) -> None:
        if self._image is None or self._drag is not None:
            return
        if self._hand:
            self.config(cursor="hand2")
            return
        part = self._hit_test(event.x, event.y)
        self.config(cursor=self._CURSORS.get(part, "arrow"))

    def _on_wheel(self, event) -> None:
        if self._image is None:
            return
        steps = event.delta / 120 if event.delta else 0
        if steps:
            # Tk のバージョンによって x, y が画面座標で来ることがあるので、画面座標から求める
            x = event.x_root - self.winfo_rootx()
            y = event.y_root - self.winfo_rooty()
            if not (0 <= x <= self.winfo_width() and 0 <= y <= self.winfo_height()):
                x, y = event.x, event.y
            self.zoom_to(self._scale * self._WHEEL_STEP ** steps, anchor=(x, y))

    def _start_pan(self, event) -> None:
        self._drag = {"mode": "pan", "press": (event.x, event.y),
                      "view": (self._ox, self._oy)}
        self.config(cursor="fleur")

    def _on_pan_start(self, event) -> None:
        if self._image is not None:
            self._start_pan(event)

    def _on_press(self, event) -> None:
        self.focus_set()   # Ctrl+C をキャンバスで受けるため
        if self._image is None:
            return
        if self._hand:
            self._start_pan(event)
            return
        shift = bool(event.state & 0x0001)
        part = None if shift else self._hit_test(event.x, event.y)
        self._drag = {
            # new: 新しい枠 / pending: クリック（コマ選択）か表示位置の移動かまだ不明
            "mode": "new" if shift else (part or "pending"),
            "press": (event.x, event.y),
            "view": (self._ox, self._oy),
            "start": self._to_image(event.x, event.y),
            "sel": list(self._sel) if self._sel else None,
            "base": list(self._base) if self._base else None,
        }

    def _on_drag(self, event) -> None:
        d = self._drag
        if d is None:
            return
        dx, dy = event.x - d["press"][0], event.y - d["press"][1]
        mode = d["mode"]
        if mode == "pending":
            if max(abs(dx), abs(dy)) < self._DRAG_START:
                return
            mode = d["mode"] = "pan"
            self.config(cursor="fleur")
        if mode == "pan":
            self._fit = False   # 動かした位置を保つ（全体表示のままだと中央に戻ってしまう）
            self._ox, self._oy = d["view"][0] + dx, d["view"][1] + dy
            self._request_render()
            return

        px, py = self._to_image(event.x, event.y)
        iw, ih = self._image.size
        if mode == "new":
            if max(abs(dx), abs(dy)) < self._DRAG_START:
                return
            sx, sy = d["start"]
            self._sel = [min(sx, px), min(sy, py), max(sx, px), max(sy, py)]
        elif mode == "move":
            x1, y1, x2, y2 = d["sel"]
            mx = min(max(px - d["start"][0], -x1), iw - x2)
            my = min(max(py - d["start"][1], -y1), ih - y2)
            self._sel = [x1 + mx, y1 + my, x2 + mx, y2 + my]
        else:
            x1, y1, x2, y2 = d["sel"]
            m = self._MIN_SIZE
            if "w" in mode:
                x1 = min(px, x2 - m)
            if "e" in mode:
                x2 = max(px, x1 + m)
            if "n" in mode:
                y1 = min(py, y2 - m)
            if "s" in mode:
                y2 = max(py, y1 + m)
            self._sel = [max(0.0, x1), max(0.0, y1), min(iw, x2), min(ih, y2)]
        self._sync_base()
        self._draw_overlay()
        self._on_change(self.selection())

    def _on_release(self, event) -> None:
        d, self._drag = self._drag, None
        if d is None or self._image is None:
            return
        if d["mode"] == "pending":
            # クリックだけ → その位置のコマ（重なっていれば小さい方）を枠にする
            px, py = self._to_image(event.x, event.y)
            under = [p for p in self._panels if p[0] <= px <= p[2] and p[1] <= py <= p[3]]
            if under:
                self._set_base(min(under, key=_area))
        elif d["mode"] == "new" and self._sel is not None:
            x1, y1, x2, y2 = self._sel
            if x2 - x1 < self._MIN_SIZE or y2 - y1 < self._MIN_SIZE:
                self._sel, self._base = d["sel"], d["base"]   # 小さすぎる枠は取り消す
        self._draw_overlay()
        self._on_change(self.selection())
        self._on_hover(event)


# ════════════════════════════════════════════════════════════════════════
# UI: メインアプリケーション
# ════════════════════════════════════════════════════════════════════════

class ImageSearchApp:
    _POLL_MS = 100
    _PREVIEW_PLACEHOLDER = "（結果を選択すると画像を表示）"
    _CROP_HINT = ("赤枠が切り抜く範囲（オレンジはヒットした吹き出し）。辺・角のドラッグで調整、"
                  "内側のドラッグで枠を移動、点線のコマのクリックで切り替え、Shift+ドラッグで新しい枠。"
                  "「枠の補正」で枠を一回り大きく（＋）・小さく（－）。"
                  "ホイールで拡大縮小、枠の外・ホイールボタン・✋ のドラッグで表示位置を移動。"
                  "右クリックでコピー・保存。")
    _CROP_HINT_NO_BLOCKS = ("この画像は吹き出しの位置が未登録です（古い索引）。"
                            "① でインデックスを更新すると、ヒットしたコマが自動で選ばれます。"
                            "今は点線のコマをクリックするか、Shift+ドラッグで枠を描いてください。")
    _ZOOM_FIT_LABEL = "全体表示"
    _OUTPUT_SCALES = ("25%", "50%", "75%", "100%", "150%", "200%", "300%", "400%")
    _OUTPUT_MIN, _OUTPUT_MAX = 0.05, 8.0

    def __init__(self, root: tk.Tk) -> None:
        self._root = root
        self._root.title(f"KomaSagashi v{__version__} - 画像テキスト検索")
        self._root.geometry("1100x820")

        self._engine        = MokuroEngine()
        self._paddle_engines: dict[str, PaddleEngine] = {}   # モデル名 → エンジン
        self._settings      = Settings()
        self._msg_queue: "queue.Queue[_WorkerMessage]" = queue.Queue()
        self._is_processing = False
        self._indexing_folder: Path | None = None
        self._control = IndexControl()           # インデックス作成の一時停止・停止
        self._paused_at: float | None = None     # 一時停止した時刻（経過時間から除くため）
        self._migrated_note = ""
        self._timer_id: str | None = None
        self._process_start = 0.0

        # 検索結果 iid → (絶対パス, 全文, 相対パス) の対応
        self._result_map: dict[str, tuple[Path, str, str]] = {}   # iid → (対象フォルダ, 全文, relpath)
        self._preview_reader: ArchiveReader | None = None           # プレビュー中の書庫
        clean_tmp_dir()   # 前回の一時ファイルが残っていれば消す
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._last_terms: list[str] = []
        self._last_fuzzy = True
        save_dir = self._settings.get("last_save_dir")
        self._last_save_dir: Path | None = Path(save_dir) if save_dir else None
        scale = self._settings.get("output_scale", 1.0)
        self._output_scale = (float(scale) if isinstance(scale, (int, float))
                              and self._OUTPUT_MIN <= scale <= self._OUTPUT_MAX else 1.0)
        adjust = self._settings.get("frame_adjust", 0)
        self._frame_adjust = (adjust if isinstance(adjust, int) and not isinstance(adjust, bool)
                              and FRAME_ADJUST_MIN <= adjust <= FRAME_ADJUST_MAX else 0)

        self._build_ui()
        self._show_start_message()

    # ── UI 構築 ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        pad = {"padx": 8, "pady": 4}

        # 共有: 対象フォルダ
        f = ttk.LabelFrame(self._root, text="対象フォルダ")
        f.pack(fill="x", **pad)
        self._folder_var = tk.StringVar()
        # ▼ で、これまでにインデックスを作ったフォルダから選べる
        self._folder_box = ttk.Combobox(f, textvariable=self._folder_var,
                                        postcommand=self._fill_recent_folders)
        self._folder_box.pack(side="left", fill="x", expand=True, padx=4, pady=4)
        self._folder_box.bind("<<ComboboxSelected>>", lambda _e: self._on_folder_changed())
        self._folder_box.bind("<Return>", lambda _e: self._on_folder_changed())
        self._folder_box.bind("<FocusOut>", lambda _e: self._refresh_index_status())
        ttk.Button(f, text="フォルダ選択", command=self._select_folder).pack(side="left", padx=2)
        self._index_status_var = tk.StringVar(value="インデックス: 未確認")
        ttk.Label(self._root, textvariable=self._index_status_var,
                  foreground="blue", anchor="w").pack(fill="x", padx=12)

        # タブ
        self._nb = ttk.Notebook(self._root)
        self._nb.pack(fill="both", expand=True, **pad)
        self._build_index_tab()
        self._build_search_tab()

    def _build_index_tab(self) -> None:
        tab = ttk.Frame(self._nb)
        self._nb.add(tab, text="① インデックス作成")

        f = ttk.LabelFrame(tab, text="設定")
        f.pack(fill="x", padx=8, pady=4)
        self._recursive_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="サブフォルダも含める",
                        variable=self._recursive_var).grid(
            row=0, column=0, columnspan=3, padx=4, pady=4, sticky="w")
        # 1 行ずつ横に並べる（grid の列にすると、幅の広い下の行に合わせて隙間が空く）
        size_row = ttk.Frame(f)
        size_row.grid(row=1, column=0, columnspan=3, sticky="w")
        ttk.Label(size_row, text="画像リサイズ上限 (px):").pack(side="left", padx=4, pady=4)
        self._max_image_size_var = tk.StringVar(value=str(_DEFAULT_MAX_IMAGE_SIZE))
        ttk.Entry(size_row, textvariable=self._max_image_size_var, width=8).pack(
            side="left", padx=4, pady=4)
        ttk.Label(size_row, text="0 = 原寸で処理（おすすめは 1500。小さい文字が多いときは 0）",
                  foreground="gray").pack(side="left", padx=4)

        # 任意: 横書きのページ（目次・扉・説明文など）を PaddleOCR で読み直す
        paddle_row = ttk.Frame(f)
        paddle_row.grid(row=2, column=0, columnspan=3, padx=0, pady=(0, 4), sticky="w")
        installed = PaddleEngine.is_installed()
        model = self._settings.get("paddle_model", _PADDLE_MODELS[0])
        self._paddle_var = tk.BooleanVar(
            value=installed and bool(self._settings.get("paddle_enabled", True)))
        self._paddle_model_var = tk.StringVar(
            value=model if model in _PADDLE_MODELS else _PADDLE_MODELS[0])
        paddle_check = ttk.Checkbutton(
            paddle_row, text="横書きのページ（目次・扉・説明文など）は PaddleOCR で読み直す",
            variable=self._paddle_var, command=self._save_paddle_settings)
        paddle_check.pack(side="left", padx=4)
        ttk.Label(paddle_row, text="モデル:").pack(side="left", padx=(8, 2))
        model_box = ttk.Combobox(paddle_row, textvariable=self._paddle_model_var,
                                 values=_PADDLE_MODELS, width=8, state="readonly")
        model_box.pack(side="left")
        model_box.bind("<<ComboboxSelected>>", lambda _e: self._save_paddle_settings())
        ttk.Label(paddle_row,
                  text=("small: 速い（約4秒/枚）  medium: 高精度（約15秒/枚）" if installed else
                        "PaddleOCR が入っていません（setup.bat を実行し直すと追加できます）"),
                  foreground="gray").pack(side="left", padx=8)
        if not installed:
            paddle_check.state(["disabled"])
            model_box.state(["disabled"])

        f = ttk.Frame(tab)
        f.pack(fill="x", padx=8, pady=4)
        self._progress = ttk.Progressbar(f, mode="determinate")
        self._progress.pack(side="left", fill="x", expand=True)
        self._elapsed_var = tk.StringVar(value="00:00")
        ttk.Label(f, textvariable=self._elapsed_var, width=8, anchor="center").pack(
            side="left", padx=(6, 0))
        self._run_btn = ttk.Button(f, text="▶ インデックス作成/更新", command=self._run_index)
        self._run_btn.pack(side="left", padx=6)
        # 作成中だけ押せる。停止しても登録済みの分は残り、次回は続きから処理する。
        self._pause_btn = ttk.Button(f, text="⏸ 一時停止", command=self._toggle_pause,
                                     state="disabled")
        self._pause_btn.pack(side="left", padx=2)
        self._stop_btn = ttk.Button(f, text="■ 停止", command=self._stop_index,
                                    state="disabled")
        self._stop_btn.pack(side="left", padx=2)
        self._export_all_btn = ttk.Button(f, text="索引全体を書き出す…",
                                          command=self._export_index)
        self._export_all_btn.pack(side="left", padx=(8, 0))

        f = ttk.LabelFrame(tab, text="ログ")
        f.pack(fill="both", expand=True, padx=8, pady=4)
        self._log_text = scrolledtext.ScrolledText(
            f, height=16, state="disabled", wrap="word")
        self._log_text.pack(fill="both", expand=True, padx=4, pady=4)

    def _build_search_tab(self) -> None:
        tab = ttk.Frame(self._nb)
        self._nb.add(tab, text="② 検索")

        f = ttk.Frame(tab)
        f.pack(fill="x", padx=8, pady=6)
        ttk.Label(f, text="検索（スペース区切りで AND）:").pack(side="left")
        self._query_var = tk.StringVar()
        self._query_entry = ttk.Entry(f, textvariable=self._query_var)
        self._query_entry.pack(side="left", fill="x", expand=True, padx=6)
        self._query_entry.bind("<Return>", lambda _e: self._run_search())
        ttk.Button(f, text="検索", command=self._run_search).pack(side="left")
        # OCR の読み違い（ベ/ペ、ぺ/ペ など）でも見つかるよう、既定で表記ゆれを無視する
        self._fuzzy_var = tk.BooleanVar(value=bool(self._settings.get("fuzzy_search", True)))
        ttk.Checkbutton(f, text="表記ゆれを無視", variable=self._fuzzy_var,
                        command=self._on_fuzzy_changed).pack(side="left", padx=(8, 0))
        self._hit_var = tk.StringVar(value="")
        ttk.Label(f, textvariable=self._hit_var, width=14, anchor="e").pack(side="left", padx=4)
        self._export_hits_btn = ttk.Button(f, text="検索結果を書き出す…",
                                           command=self._export_hits, state="disabled")
        self._export_hits_btn.pack(side="left")

        paned = ttk.Panedwindow(tab, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=8, pady=4)

        # 左: 結果一覧
        left = ttk.Frame(paned)
        paned.add(left, weight=1)
        cols = ("file", "snippet")
        self._tree = ttk.Treeview(left, columns=cols, show="headings", selectmode="browse")
        self._tree.heading("file", text="ファイル")
        self._tree.heading("snippet", text="ヒット箇所")
        self._tree.column("file", width=180, anchor="w")
        self._tree.column("snippet", width=240, anchor="w")
        vsb = ttk.Scrollbar(left, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self._tree.bind("<<TreeviewSelect>>", self._on_result_select)
        self._tree.bind("<Double-1>", lambda _e: self._open_default())
        self._tree.bind("<Control-c>", self._copy_crop)

        # 右: プレビュー（切り抜き枠つき）+ 詳細
        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        # 上: 表示倍率の操作
        toolbar = ttk.Frame(right)
        toolbar.pack(side="top", fill="x", padx=4, pady=(4, 0))
        ttk.Button(toolbar, text="－", width=3,
                   command=lambda: self._canvas.zoom_step(-1)).pack(side="left")
        ttk.Button(toolbar, text="＋", width=3,
                   command=lambda: self._canvas.zoom_step(+1)).pack(side="left", padx=(2, 0))
        self._zoom_var = tk.StringVar(value="")
        zoom_box = ttk.Combobox(
            toolbar, textvariable=self._zoom_var, width=9,
            values=[self._ZOOM_FIT_LABEL] + [format_percent(z) for z in CropCanvas.ZOOM_LEVELS])
        zoom_box.pack(side="left", padx=4)
        zoom_box.bind("<<ComboboxSelected>>", lambda _e: self._apply_zoom_text())
        zoom_box.bind("<Return>", lambda _e: self._apply_zoom_text())
        ttk.Button(toolbar, text=self._ZOOM_FIT_LABEL,
                   command=lambda: self._canvas.zoom_fit()).pack(side="left")
        # 押し込むと手モード（枠の内側でもドラッグで表示位置を移動）。もう一度押すと解除。
        self._hand_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(toolbar, text="✋ 手", style="Toolbutton", variable=self._hand_var,
                        command=lambda: self._canvas.set_hand(self._hand_var.get())
                        ).pack(side="left", padx=(8, 0))

        # 下の操作部を先に配置し、残りの高さをすべて画像に使う
        details = ttk.Frame(right)
        details.pack(side="bottom", fill="x")
        self._canvas = CropCanvas(right, on_change=self._on_crop_change,
                                  on_zoom=self._on_zoom_change)
        self._canvas.pack(side="top", fill="both", expand=True, padx=4, pady=4)
        self._canvas.bind("<Control-c>", self._copy_crop)
        self._canvas.bind("<Button-3>", self._show_crop_menu)

        # 右クリックメニュー
        self._crop_menu = tk.Menu(self._root, tearoff=0)
        self._crop_menu.add_command(label="枠内をコピー", accelerator="Ctrl+C",
                                    command=self._copy_crop)
        self._crop_menu.add_command(label="枠内を画像で保存…", command=self._save_crop)

        crop = ttk.Frame(details)
        crop.pack(fill="x", padx=4)
        self._btn_copy = ttk.Button(crop, text="枠内をコピー (Ctrl+C)",
                                    command=self._copy_crop, state="disabled")
        self._btn_copy.pack(side="left", padx=2)
        self._btn_save = ttk.Button(crop, text="枠内を画像で保存…",
                                    command=self._save_crop, state="disabled")
        self._btn_save.pack(side="left", padx=2)
        # コピー・保存するときの拡大率（元画像の解像度に対する倍率）
        ttk.Label(crop, text="出力倍率:").pack(side="left", padx=(8, 2))
        self._out_scale_var = tk.StringVar(value=format_percent(self._output_scale))
        out_box = ttk.Combobox(crop, textvariable=self._out_scale_var, width=6,
                               values=self._OUTPUT_SCALES)
        out_box.pack(side="left")
        out_box.bind("<<ComboboxSelected>>", lambda _e: self._apply_output_scale())
        out_box.bind("<Return>", lambda _e: self._apply_output_scale())
        out_box.bind("<FocusOut>", lambda _e: self._apply_output_scale())
        # 枠の補正: 枠を中心から幅・高さとも ±% 広げる・狭める
        ttk.Label(crop, text="枠の補正:").pack(side="left", padx=(8, 2))
        self._frame_adjust_var = tk.StringVar(value=format_adjust(self._frame_adjust))
        adj_box = ttk.Spinbox(crop, textvariable=self._frame_adjust_var, width=5,
                              from_=FRAME_ADJUST_MIN, to=FRAME_ADJUST_MAX, increment=5,
                              command=self._apply_frame_adjust)
        adj_box.pack(side="left")
        adj_box.bind("<Return>", lambda _e: self._apply_frame_adjust())
        adj_box.bind("<FocusOut>", lambda _e: self._apply_frame_adjust())
        ttk.Label(crop, text="%").pack(side="left", padx=(2, 0))
        self._canvas.set_adjust(self._frame_adjust)
        self._crop_var = tk.StringVar(value="")
        # 枠の大きさは行が溢れないよう次の行に出す（出力倍率があると長くなる）
        ttk.Label(details, textvariable=self._crop_var, anchor="w").pack(
            fill="x", padx=6, pady=(2, 0))
        self._crop_hint_var = tk.StringVar(value="")
        ttk.Label(details, textvariable=self._crop_hint_var, foreground="gray",
                  anchor="w", wraplength=520).pack(fill="x", padx=4, pady=(2, 0))
        self._canvas.clear(self._PREVIEW_PLACEHOLDER)

        self._path_var = tk.StringVar(value="")
        ttk.Label(details, textvariable=self._path_var, foreground="gray",
                  anchor="w", wraplength=520).pack(fill="x", padx=4, pady=(4, 0))

        btns = ttk.Frame(details)
        btns.pack(fill="x", padx=4, pady=4)
        self._btn_explorer = ttk.Button(
            btns, text="エクスプローラーで表示", command=self._open_in_explorer,
            state="disabled")
        self._btn_explorer.pack(side="left", padx=2)
        self._btn_open = ttk.Button(
            btns, text="既定のアプリで開く", command=self._open_default, state="disabled")
        self._btn_open.pack(side="left", padx=2)

        ttk.Label(details, text="全文（ヒット語を強調）:").pack(anchor="w", padx=4)
        self._fulltext = scrolledtext.ScrolledText(details, height=6, wrap="word",
                                                    state="disabled")
        self._fulltext.pack(fill="x", padx=4, pady=(0, 4))
        self._fulltext.tag_configure("hit", background="yellow")

    # ── フォルダ / DB ─────────────────────────────────────────────────

    def _folder_or_none(self) -> Path | None:
        folder = self._folder_var.get().strip()
        return Path(folder) if folder else None

    def _index_or_none(self) -> TextIndex | None:
        """対象フォルダの索引。旧版が対象フォルダ内に作った索引があれば、先に data/ へ移す。"""
        folder = self._folder_or_none()
        if folder is None:
            return None
        try:
            if migrate_legacy_index(folder):
                self._migrated_note = "（旧版が画像フォルダ内に作った索引を、プログラムのフォルダへ移しました）"
        except OSError as exc:
            messagebox.showwarning(
                "索引の移動に失敗",
                f"画像フォルダ内の旧い索引を移せませんでした:\n{exc}\n\n"
                "このフォルダは新しく索引を作ることになります。")
        return TextIndex(folder)

    def _select_folder(self) -> None:
        current = self._folder_var.get().strip()
        path = filedialog.askdirectory(
            title="フォルダを選択",
            initialdir=current if current and Path(current).is_dir() else None)
        if not path:
            return
        self._folder_var.set(str(Path(path)))
        self._on_folder_changed()

    def _fill_recent_folders(self) -> None:
        """▼ を開いたときに、インデックスを作ったことのあるフォルダを並べる。"""
        self._folder_box["values"] = self._settings.recent_folders()

    def _on_folder_changed(self) -> None:
        self._refresh_index_status()
        index = self._index_or_none()
        if index is not None and index.exists():
            self._settings.add_recent_folder(index.root)
            # 索引があればすぐ検索できるようにする
            self._nb.select(1)
            self._query_entry.focus_set()

    def _show_start_message(self) -> None:
        """起動時は対象フォルダを空欄にし、▼ から前に使ったフォルダを選べることを案内する。"""
        if self._settings.recent_folders():
            self._index_status_var.set(
                "対象フォルダを選んでください（▼ から前にインデックスを作ったフォルダを選ぶと、すぐ検索できます）")
        else:
            self._index_status_var.set("対象フォルダを選んでください")

    def _refresh_index_status(self) -> None:
        self._migrated_note = ""
        if self._folder_or_none() is None:
            self._show_start_message()
            return
        index = self._index_or_none()
        if index is not None and index.exists():
            try:
                n = index.count()
                self._index_status_var.set(f"インデックス: あり（{n} 件）{self._migrated_note}")
                return
            except sqlite3.Error as exc:
                self._index_status_var.set(f"インデックス: 読み込みエラー（{exc}）")
                return
        self._index_status_var.set("インデックス: なし（①で作成してください）")

    # ── インデックス作成 ───────────────────────────────────────────────

    def _run_index(self) -> None:
        if self._is_processing:
            return
        folder_str = self._folder_var.get().strip()
        if not folder_str:
            messagebox.showerror("エラー", "対象フォルダを指定してください。"); return
        folder = Path(folder_str)
        if not folder.is_dir():
            messagebox.showerror("エラー", f"フォルダが存在しません:\n{folder}"); return

        try:
            max_image_size = int(self._max_image_size_var.get().strip())
        except ValueError:
            max_image_size = _DEFAULT_MAX_IMAGE_SIZE

        self._refresh_index_status()   # 旧版の索引が画像フォルダ内にあれば、ここで data/ へ移す
        self._is_processing = True
        self._indexing_folder = folder
        self._control = IndexControl()
        self._paused_at: float | None = None
        self._set_running(True)
        self._progress["value"] = 0
        self._elapsed_var.set("00:00")
        self._log_text.config(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.config(state="disabled")

        self._process_start = time.time()
        self._tick_elapsed()

        threading.Thread(
            target=self._load_and_index,
            args=(folder, self._recursive_var.get(), max_image_size, self._selected_paddle(),
                  self._control),
            daemon=True,
        ).start()
        self._root.after(self._POLL_MS, self._poll_queue)

    def _set_running(self, running: bool) -> None:
        self._run_btn.state(["disabled"] if running else ["!disabled"])
        self._export_all_btn.state(["disabled"] if running else ["!disabled"])
        for btn in (self._pause_btn, self._stop_btn):
            btn.state(["!disabled"] if running else ["disabled"])
        self._pause_btn.config(text="⏸ 一時停止")

    def _toggle_pause(self) -> None:
        if not self._is_processing or self._control.stopped:
            return
        if self._control.paused:
            self._control.resume()
            if self._paused_at is not None:   # 止めていた時間は経過時間に数えない
                self._process_start += time.time() - self._paused_at
                self._paused_at = None
            self._pause_btn.config(text="⏸ 一時停止")
            self._append_log("▶ 再開します")
        else:
            self._control.pause()
            self._paused_at = time.time()
            self._pause_btn.config(text="▶ 再開")
            self._append_log("⏸ 一時停止します（処理中の 1 枚が終わったところで止まります）")

    def _stop_index(self) -> None:
        if not self._is_processing or self._control.stopped:
            return
        self._control.stop()
        if self._paused_at is not None:
            self._process_start += time.time() - self._paused_at
            self._paused_at = None
        self._pause_btn.state(["disabled"])
        self._stop_btn.state(["disabled"])
        self._append_log("■ 停止します（処理中の 1 枚が終わったところで止まります）")

    def _tick_elapsed(self) -> None:
        if not self._is_processing:
            return
        if self._paused_at is None:
            self._elapsed_var.set(_format_elapsed(time.time() - self._process_start))
        self._timer_id = self._root.after(1000, self._tick_elapsed)

    def _load_and_index(self, folder: Path, recursive: bool, max_image_size: int,
                        paddle: PaddleEngine | None, control: "IndexControl") -> None:
        try:
            if not self._engine.is_loaded:
                self._engine.load(
                    progress_cb=lambda m: self._msg_queue.put(_WorkerMessage("log", m)))
            files = _collect_files(folder, recursive)
            index = TextIndex(folder)   # 旧い索引の移動は _run_index で済ませてある
        except Exception as exc:
            self._msg_queue.put(_WorkerMessage("error", str(exc)))
            return
        IndexWorker(self._engine, index, files, self._msg_queue,
                    max_image_size=max_image_size, paddle=paddle, control=control).start()

    def _selected_paddle(self) -> PaddleEngine | None:
        """① の設定で PaddleOCR を使うなら、そのモデルのエンジン（読み込み済みなら使い回す）。"""
        if not (self._paddle_var.get() and PaddleEngine.is_installed()):
            return None
        model = self._paddle_model_var.get()
        for other, engine in self._paddle_engines.items():
            if other != model:
                engine.close()   # 使わなくなったモデルの子プロセスは終える（メモリを空ける）
        if model not in self._paddle_engines:
            self._paddle_engines[model] = PaddleEngine(model)
        return self._paddle_engines[model]

    def _save_paddle_settings(self) -> None:
        self._settings.set("paddle_enabled", bool(self._paddle_var.get()))
        self._settings.set("paddle_model", self._paddle_model_var.get())

    def _poll_queue(self) -> None:
        still_working = True
        while not self._msg_queue.empty():
            msg: _WorkerMessage = self._msg_queue.get_nowait()
            if   msg.kind == "log":      self._append_log(str(msg.payload))
            elif msg.kind == "progress":
                cur, tot = msg.payload
                self._progress["maximum"] = tot
                self._progress["value"]   = cur
            elif msg.kind == "done":
                still_working = False; self._on_index_done(str(msg.payload))
            elif msg.kind == "stopped":
                still_working = False; self._on_index_stopped(str(msg.payload))
            elif msg.kind == "error":
                still_working = False; self._on_error(str(msg.payload))
        if still_working:
            self._root.after(self._POLL_MS, self._poll_queue)

    def _append_log(self, text: str) -> None:
        self._log_text.config(state="normal")
        self._log_text.insert("end", text + "\n")
        self._log_text.see("end")
        self._log_text.config(state="disabled")

    def _stop_timer(self) -> None:
        if self._timer_id:
            self._root.after_cancel(self._timer_id)
            self._timer_id = None

    def _on_index_stopped(self, count: str) -> None:
        self._is_processing = False
        self._stop_timer()
        self._set_running(False)
        self._elapsed_var.set(_format_elapsed(time.time() - self._process_start))
        self._refresh_index_status()
        if self._indexing_folder is not None and TextIndex(self._indexing_folder).exists():
            self._settings.add_recent_folder(self._indexing_folder)
        messagebox.showinfo(
            "停止しました",
            f"インデックス作成を停止しました。\n"
            f"登録済み: {count} 件（ここまでの分は検索できます）\n\n"
            "もう一度「インデックス作成/更新」を押すと、続きから処理します。")

    def _on_index_done(self, count: str) -> None:
        self._is_processing = False
        self._stop_timer()
        self._set_running(False)
        self._elapsed_var.set(_format_elapsed(time.time() - self._process_start))
        self._refresh_index_status()
        if self._indexing_folder is not None:
            # 次回からフォルダ欄の ▼ で選べるように記憶する
            self._settings.add_recent_folder(self._indexing_folder)
        messagebox.showinfo(
            "完了",
            f"インデックスを更新しました。\n"
            f"総登録数: {count} 件\n\n"
            "「② 検索」タブで検索できます。")

    def _on_error(self, error_msg: str) -> None:
        self._is_processing = False
        self._stop_timer()
        self._set_running(False)
        self._append_log(f"❌ エラー: {error_msg}")
        messagebox.showerror("エラー", error_msg)

    # ── 検索 ──────────────────────────────────────────────────────────

    def _run_search(self) -> None:
        index = self._index_or_none()
        if index is None:
            messagebox.showerror("エラー", "対象フォルダを指定してください。"); return
        if not index.exists():
            messagebox.showwarning(
                "インデックスなし",
                "このフォルダにはインデックスがありません。\n"
                "先に「① インデックス作成」を実行してください。")
            return

        terms = self._query_var.get().split()
        fuzzy = bool(self._fuzzy_var.get())
        self._last_terms, self._last_fuzzy = terms, fuzzy
        self._tree.delete(*self._tree.get_children())
        self._result_map.clear()
        self._clear_preview()
        self._export_hits_btn.state(["disabled"])

        if not terms:
            self._hit_var.set("")
            return

        try:
            rows = index.search(terms, fuzzy=fuzzy)
        except sqlite3.Error as exc:
            messagebox.showerror("エラー", f"検索に失敗しました:\n{exc}"); return

        for relpath, text in rows:
            iid = self._tree.insert(
                "", "end",
                values=(display_name(relpath), make_snippet(text, terms, fuzzy)))
            self._result_map[iid] = (index.root, text, relpath)

        self._hit_var.set(f"ヒット: {len(rows)} 件")
        if not rows:
            self._hit_var.set("ヒット: 0 件")
        else:
            self._export_hits_btn.state(["!disabled"])

    # ── 書き出し ──────────────────────────────────────────────────────

    def _ask_export_path(self, initial_name: str) -> Path | None:
        """保存先を尋ねる。拡張子を書かなかったときは、選んだ「ファイルの種類」の拡張子を付ける。"""
        initial_dir = self._settings.get("last_export_dir")
        if not initial_dir or not Path(initial_dir).is_dir():
            docs = Path.home() / "Documents"
            initial_dir = str(docs if docs.is_dir() else Path.home())
        kinds = [("CSV（Excel で開く）", ".csv"), ("テキスト（読む・貼り付ける）", ".txt"),
                 ("JSON（別のプログラムで使う）", ".json")]
        type_var = tk.StringVar(value=kinds[0][0])
        filename = filedialog.asksaveasfilename(
            title="書き出し", initialdir=initial_dir, initialfile=initial_name + ".csv",
            filetypes=[(label, "*" + ext) for label, ext in kinds], typevariable=type_var)
        if not filename:
            return None
        path = Path(filename)
        if path.suffix.lower() not in EXPORT_FORMATS:
            ext = next((e for label, e in kinds if label == type_var.get()), ".csv")
            path = path.with_name(path.name + ext)
        self._settings.set("last_export_dir", str(path.parent))
        return path

    def _do_export(self, path: Path, rows: list[dict], index: TextIndex,
                   terms: list[str] | None) -> None:
        try:
            n = export_results(path, rows, folder=index.root, terms=terms,
                               fuzzy=self._last_fuzzy)
        except Exception as exc:
            messagebox.showerror("エラー", f"書き出せませんでした:\n{exc}")
            return
        if messagebox.askyesno("書き出しました",
                               f"{n} 件を書き出しました。\n{path}\n\n保存したフォルダを開きますか？"):
            if os.name == "nt":
                subprocess.run(["explorer", "/select,", str(path)])

    def _export_hits(self) -> None:
        """② 検索タブ: いまの検索結果（一覧の順）を書き出す。"""
        index = self._index_or_none()
        relpaths = [self._result_map[iid][2] for iid in self._tree.get_children()
                    if iid in self._result_map]
        if index is None or not relpaths:
            return
        path = self._ask_export_path(
            export_file_name(f"検索結果_{' '.join(self._last_terms)}"))
        if path:
            self._do_export(path, index.export_rows(relpaths), index, self._last_terms)

    def _export_index(self) -> None:
        """① タブ: 対象フォルダの索引全体を書き出す。"""
        index = self._index_or_none()
        if index is None:
            messagebox.showerror("エラー", "対象フォルダを指定してください。"); return
        if not index.exists():
            messagebox.showwarning("インデックスなし",
                                   "このフォルダにはインデックスがありません。\n"
                                   "先に「インデックス作成/更新」を実行してください。")
            return
        path = self._ask_export_path(export_file_name(f"索引_{index.root.name}"))
        if path:
            self._do_export(path, index.export_rows(), index, None)

    def _on_fuzzy_changed(self) -> None:
        self._settings.set("fuzzy_search", bool(self._fuzzy_var.get()))
        index = self._index_or_none()
        if self._query_var.get().strip() and index is not None and index.exists():
            self._run_search()

    def _current_entry(self) -> tuple[Path, str] | None:
        """選択中の結果の (対象フォルダ, relpath)。"""
        sel = self._tree.selection()
        if not sel:
            return None
        entry = self._result_map.get(sel[0])
        return (entry[0], entry[2]) if entry else None

    def _current_file(self) -> Path | None:
        """選択中の結果のファイル（書庫内の画像なら書庫ファイル）。"""
        entry = self._current_entry()
        return entry[0] / split_relpath(entry[1])[0] if entry else None

    def _on_result_select(self, _event=None) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        entry = self._result_map.get(sel[0])
        if not entry:
            return
        root, text, relpath = entry
        outer, member = split_relpath(relpath)
        self._path_var.set(str(root / outer) + (f" › {member}" if member else ""))
        self._btn_explorer.state(["!disabled"])
        self._btn_open.state(["!disabled"])
        self._show_preview(root, relpath)
        self._show_fulltext(text)

    def _preview_archive(self, path: Path) -> ArchiveReader:
        """プレビュー用に書庫を開く。同じ書庫のページを続けて見るときは開いたまま使い回す。"""
        if self._preview_reader is not None and self._preview_reader.path == path:
            return self._preview_reader
        self._close_preview_archive()
        self._preview_reader = open_archive(path)
        return self._preview_reader

    def _close_preview_archive(self) -> None:
        if self._preview_reader is not None:
            self._preview_reader.close()
            self._preview_reader = None

    def _load_result_image(self, root: Path, relpath: str) -> Image.Image:
        outer, member = split_relpath(relpath)
        if member is None:
            self._close_preview_archive()
            return load_image(root, relpath)
        return load_image(root, relpath, self._preview_archive(root / outer))

    def _show_preview(self, root: Path, relpath: str) -> None:
        if not _IMAGETK_AVAILABLE:
            self._canvas.clear("（プレビューには Pillow の ImageTk が必要）")
            return
        try:
            img = self._load_result_image(root, relpath)
        except Exception as exc:
            self._canvas.clear(f"（画像を表示できません: {exc}）")
            self._crop_hint_var.set("")
            return

        panels = detect_panels(img)
        index = self._index_or_none()
        try:
            blocks = index.get_blocks(relpath) if index else None
        except (sqlite3.Error, ValueError):
            blocks = None

        hits: list[Box] = []
        if blocks is not None:
            hits = [box for box, btext in blocks
                    if contains_any(btext, self._last_terms, self._last_fuzzy)]
        selection = pick_panel(panels, hits[0], img.size) if hits else None
        self._canvas.show(img, panels, hits, selection)
        self._crop_hint_var.set(
            self._CROP_HINT if blocks is not None else self._CROP_HINT_NO_BLOCKS)

    def _show_fulltext(self, text: str) -> None:
        self._fulltext.config(state="normal")
        self._fulltext.delete("1.0", "end")
        self._fulltext.insert("1.0", text)
        # ヒット語を全てハイライト（表記ゆれを無視した検索なら、ゆれた表記も強調する）
        for start, end in find_terms(text, self._last_terms, self._last_fuzzy):
            self._fulltext.tag_add("hit", f"1.0+{start}c", f"1.0+{end}c")
        self._fulltext.config(state="disabled")

    def _clear_preview(self) -> None:
        self._canvas.clear(self._PREVIEW_PLACEHOLDER)
        self._crop_hint_var.set("")
        self._path_var.set("")
        self._btn_explorer.state(["disabled"])
        self._btn_open.state(["disabled"])
        self._fulltext.config(state="normal")
        self._fulltext.delete("1.0", "end")
        self._fulltext.config(state="disabled")

    # ── 切り抜き ──────────────────────────────────────────────────────

    def _on_crop_change(self, box: Box | None) -> None:
        state = ["!disabled"] if box else ["disabled"]
        self._btn_copy.state(state)
        self._btn_save.state(state)
        if not box:
            self._crop_var.set("")
            return
        w, h = box[2] - box[0], box[3] - box[1]
        text = f"枠: {w} × {h} px"
        if abs(self._output_scale - 1.0) > 1e-9:
            ow, oh = self._output_size(w, h)
            text += f" → 出力 {ow} × {oh} px"
        self._crop_var.set(text)

    def _output_size(self, w: int, h: int) -> tuple[int, int]:
        return max(1, round(w * self._output_scale)), max(1, round(h * self._output_scale))

    def _output_image(self) -> Image.Image | None:
        """枠内を切り抜き、出力倍率に合わせて拡大・縮小した画像。"""
        crop = self._canvas.crop()
        if crop is None or abs(self._output_scale - 1.0) <= 1e-9:
            return crop
        return crop.resize(self._output_size(crop.width, crop.height), Image.LANCZOS)

    def _apply_output_scale(self) -> None:
        scale = parse_percent(self._out_scale_var.get())
        if scale is not None and self._OUTPUT_MIN <= scale <= self._OUTPUT_MAX:
            self._output_scale = scale
            self._settings.set("output_scale", scale)
        self._out_scale_var.set(format_percent(self._output_scale))
        self._on_crop_change(self._canvas.selection())

    def _apply_frame_adjust(self) -> None:
        try:
            value = round(float(self._frame_adjust_var.get().strip().rstrip("%")))
        except ValueError:
            value = self._frame_adjust
        value = min(max(value, FRAME_ADJUST_MIN), FRAME_ADJUST_MAX)
        if value != self._frame_adjust:
            self._frame_adjust = value
            self._settings.set("frame_adjust", value)
            self._canvas.set_adjust(value)
        self._frame_adjust_var.set(format_adjust(value))

    def _on_zoom_change(self, scale: float) -> None:
        self._zoom_var.set(format_percent(scale))

    def _apply_zoom_text(self) -> None:
        text = self._zoom_var.get().strip()
        if text == self._ZOOM_FIT_LABEL:
            self._canvas.zoom_fit()
            return
        scale = parse_percent(text)
        if scale is not None:
            self._canvas.zoom_to(scale)
        else:
            self._zoom_var.set(format_percent(self._canvas.scale))

    def _show_crop_menu(self, event) -> None:
        if not self._canvas.has_image:
            return
        state = "normal" if self._canvas.selection() else "disabled"
        for i in range(self._crop_menu.index("end") + 1):
            self._crop_menu.entryconfigure(i, state=state)
        try:
            self._crop_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._crop_menu.grab_release()

    def _copy_crop(self, _event=None) -> str:
        crop = self._output_image()
        if crop is not None:
            try:
                copy_image_to_clipboard(crop)
            except Exception as exc:
                messagebox.showerror("エラー", f"コピーできませんでした:\n{exc}")
            else:
                self._crop_var.set(f"コピーしました（{crop.width} × {crop.height} px）")
        return "break"   # Treeview 既定の Ctrl+C 動作を止める

    def _save_crop(self) -> None:
        crop = self._output_image()
        entry = self._current_entry()
        if crop is None or entry is None:
            return
        outer, member = split_relpath(entry[1])
        stem = PurePosixPath(member).stem if member else Path(outer).stem
        # 対象フォルダに保存すると次回のインデックス作成で OCR されるので、既定は「ピクチャ」
        initial_dir = self._last_save_dir
        if initial_dir is None or not initial_dir.is_dir():
            pictures = Path.home() / "Pictures"
            initial_dir = pictures if pictures.is_dir() else Path.home()
        filename = filedialog.asksaveasfilename(
            title="枠内を画像で保存",
            initialdir=str(initial_dir),
            initialfile=f"{stem}_koma.png",
            defaultextension=".png",
            filetypes=[("PNG 画像", "*.png"), ("JPEG 画像", "*.jpg *.jpeg"),
                       ("WebP 画像", "*.webp")],
        )
        if not filename:
            return
        out = Path(filename)
        try:
            if out.suffix.lower() in (".jpg", ".jpeg"):
                crop.save(out, quality=95)
            else:
                crop.save(out)
        except Exception as exc:
            messagebox.showerror("エラー", f"保存できませんでした:\n{exc}")
            return
        self._last_save_dir = out.parent
        self._settings.set("last_save_dir", str(out.parent))
        self._crop_var.set(f"保存しました: {out.name}（{crop.width} × {crop.height} px）")

    # ── 元画像を開く ──────────────────────────────────────────────────

    def _open_in_explorer(self) -> None:
        path = self._current_file()   # 書庫内の画像なら書庫ファイルを選択した状態で開く
        if not path:
            return
        if not path.exists():
            messagebox.showwarning("見つかりません",
                                   f"ファイルが見つかりません:\n{path}"); return
        if os.name == "nt":
            # /select でファイルを選択状態にしてエクスプローラーを開く
            subprocess.run(["explorer", "/select,", str(path)])
        else:
            subprocess.run(["xdg-open", str(path.parent)])

    def _open_default(self) -> None:
        entry = self._current_entry()
        path = self._current_file()
        if not entry or not path:
            return
        if not path.exists():
            messagebox.showwarning("見つかりません",
                                   f"ファイルが見つかりません:\n{path}"); return
        try:
            member = split_relpath(entry[1])[1]
            if member is not None:
                # 書庫内の画像は、そのページだけ一時ファイルに取り出して開く（終了時に消す）
                path = self._extract_for_viewer(entry[0], entry[1])
            if os.name == "nt":
                os.startfile(str(path))   # type: ignore[attr-defined]
            else:
                subprocess.run(["xdg-open", str(path)])
        except Exception as exc:
            messagebox.showerror("エラー", f"開けませんでした:\n{exc}")

    def _extract_for_viewer(self, root: Path, relpath: str) -> Path:
        outer, member = split_relpath(relpath)
        data = self._preview_archive(root / outer).read(member)
        dest_dir = TMP_DIR / "open"
        dest_dir.mkdir(parents=True, exist_ok=True)
        name = re.sub(r'[\\/:*?"<>|]', "_", PurePosixPath(member).name)
        dest = Path(tempfile.mkdtemp(dir=dest_dir)) / name
        dest.write_bytes(data)
        return dest

    def _on_close(self) -> None:
        if self._is_processing:
            if not messagebox.askyesno(
                    "インデックス作成中",
                    "インデックス作成中です。停止して終了しますか？\n"
                    "（登録済みの分は残り、次回は続きから処理します）"):
                return
            self._control.stop()
        for engine in self._paddle_engines.values():
            engine.close()   # PaddleOCR の子プロセスを終える
        self._close_preview_archive()
        clean_tmp_dir()   # 取り出した一時ファイル（開いているアプリがあれば消せずに次回起動時に消す）
        self._root.destroy()


# ════════════════════════════════════════════════════════════════════════
# エントリーポイント
# ════════════════════════════════════════════════════════════════════════

def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "--paddle-worker":
        sys.exit(_paddle_worker_main(sys.argv[2]))   # PaddleEngine が起動する子プロセス
    root = tk.Tk()
    ImageSearchApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
