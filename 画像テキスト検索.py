# -*- coding: utf-8 -*-
"""
画像テキスト検索.py

フォルダ内の画像を OCR してテキストをインデックス化し、
キーワード検索でヒットした画像へ即ジャンプできる検索システム。

「画像テキストをファイル名に.py」がファイル名にテキストを埋め込むのに対し、
こちらは全文を SQLite に保持するため
  ・255 文字のファイル名制限を受けない
  ・元のファイル名を保ったまま
  ・何度でも高速に再検索できる
  ・検索ヒットからその場で元画像を開ける
という違いがある。OCR エンジンは同じ mokuro / manga-ocr-base を使用する。

【必要なライブラリ】
    pip install mokuro Pillow
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    # setuptools 81+ では pkg_resources が無いため: pip install "setuptools<81"
"""
from __future__ import annotations

import os
import queue
import re
import sqlite3
import subprocess
import tempfile
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Callable, Iterator

from PIL import Image

try:
    from PIL import ImageTk
    _IMAGETK_AVAILABLE = True
except ImportError:
    _IMAGETK_AVAILABLE = False

# ── 画像拡張子 ────────────────────────────────────────────────────────────
_IMAGE_EXTS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}
)

# ── デフォルト設定値 ──────────────────────────────────────────────────────
_DEFAULT_MAX_IMAGE_SIZE = 1500   # 長辺の上限px（0=原寸）
_DB_FILENAME            = "_画像テキスト検索.db"   # インデックスDB（対象フォルダ直下に置く）
_SCHEMA_VERSION         = "1"
_PREVIEW_BOX            = 460     # プレビュー画像の最大表示辺(px)
_SNIPPET_PAD            = 30      # スニペットでヒット語の前後に残す文字数


# ════════════════════════════════════════════════════════════════════════
# ユーティリティ
# ════════════════════════════════════════════════════════════════════════

def _format_elapsed(seconds: float) -> str:
    seconds = int(seconds)
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


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


def _collect_images(folder: Path, recursive: bool) -> list[Path]:
    """フォルダ内の画像を自然順で収集する。recursive=True でサブフォルダも走査。"""
    it: Iterator[Path] = folder.rglob("*") if recursive else folder.iterdir()
    files = [p for p in it if p.is_file() and p.suffix.lower() in _IMAGE_EXTS]
    return sorted(files, key=lambda p: _natural_sort_key(str(p.relative_to(folder))))


# ════════════════════════════════════════════════════════════════════════
# インフラ: SQLite インデックス
# ════════════════════════════════════════════════════════════════════════

class TextIndex:
    """
    OCR 結果を保持する SQLite インデックス。
    DB ファイルは対象フォルダの直下に置き、DB の場所を基準（root）として
    各画像を相対パスで記録する。フォルダごと移動しても検索・ジャンプできる。
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._root = db_path.parent

    @property
    def root(self) -> Path:
        return self._root

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS images ("
            "  relpath TEXT PRIMARY KEY,"
            "  text    TEXT NOT NULL,"
            "  mtime   REAL NOT NULL,"
            "  size    INTEGER NOT NULL"
            ")"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
        )
        conn.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema', ?)",
            (_SCHEMA_VERSION,),
        )
        return conn

    def load_existing(self) -> dict[str, tuple[float, int]]:
        """既存インデックスを {relpath: (mtime, size)} で返す。"""
        with self._connect() as conn:
            rows = conn.execute("SELECT relpath, mtime, size FROM images").fetchall()
        return {r[0]: (r[1], r[2]) for r in rows}

    def upsert(self, relpath: str, text: str, mtime: float, size: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO images (relpath, text, mtime, size) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(relpath) DO UPDATE SET text=?, mtime=?, size=?",
                (relpath, text, mtime, size, text, mtime, size),
            )

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

    def search(self, terms: list[str]) -> list[tuple[str, str]]:
        """
        全 term を含む（AND・部分一致・大文字小文字無視）画像を
        [(relpath, text), ...] で返す。terms が空なら空リスト。
        """
        if not terms:
            return []
        where = " AND ".join(["text LIKE ? ESCAPE '\\'"] * len(terms))
        params = [self._like_param(t) for t in terms]
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT relpath, text FROM images WHERE {where}", params
            ).fetchall()
        return sorted(rows, key=lambda r: _natural_sort_key(r[0]))


def make_snippet(text: str, terms: list[str]) -> str:
    """ヒット語の周辺を抜き出した1行スニペットを作る。"""
    flat = re.sub(r"\s+", " ", text).strip()
    low = flat.lower()
    # 最初にヒットした語の位置を探す
    pos, hit = -1, ""
    for t in terms:
        i = low.find(t.lower())
        if i != -1 and (pos == -1 or i < pos):
            pos, hit = i, t
    if pos == -1:
        return flat[:80] + ("…" if len(flat) > 80 else "")
    start = max(0, pos - _SNIPPET_PAD)
    end = min(len(flat), pos + len(hit) + _SNIPPET_PAD)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(flat) else ""
    return f"{prefix}{flat[start:end]}{suffix}"


# ════════════════════════════════════════════════════════════════════════
# インフラ: mokuro OCR エンジン（画像テキストをファイル名に.py と同一）
# ════════════════════════════════════════════════════════════════════════

class MokuroEngine:
    """mokuro の MangaPageOcr（comic-text-detector + manga-ocr-base）を使う OCR。"""

    def __init__(self) -> None:
        self._mpocr = None

    def load(self, progress_cb: "Callable[[str], None] | None" = None) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch が必要です: pip install torch") from exc
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

    def recognize(self, image: Image.Image, max_image_size: int = _DEFAULT_MAX_IMAGE_SIZE) -> str:
        if not self.is_loaded:
            raise RuntimeError("モデルがロードされていません。")

        image, _ = _resize_image(image, max_image_size)

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".png")
        try:
            os.close(tmp_fd)
            image.save(tmp_path)
            result = self._mpocr(tmp_path)
        finally:
            os.unlink(tmp_path)

        blocks = result.get("blocks", [])
        texts: list[str] = []
        for block in blocks:
            lines = block.get("lines", [])
            if lines:
                texts.append("".join(lines))
        return "\n".join(texts)


# ════════════════════════════════════════════════════════════════════════
# アプリケーション: インデックス作成ワーカー
# ════════════════════════════════════════════════════════════════════════

@dataclass
class _WorkerMessage:
    kind: str    # "log" | "progress" | "done" | "error"
    payload: object = None


class IndexWorker(threading.Thread):
    """フォルダ内画像を走査し、新規・更新分だけ OCR してインデックスへ反映する。"""

    def __init__(
        self,
        engine: MokuroEngine,
        index: TextIndex,
        files: list[Path],
        msg_queue: "queue.Queue[_WorkerMessage]",
        *,
        max_image_size: int,
    ) -> None:
        super().__init__(daemon=True)
        self._engine         = engine
        self._index          = index
        self._files          = files
        self._q              = msg_queue
        self._max_image_size = max_image_size

    def run(self) -> None:
        try:
            self._process_all()
        except Exception as exc:
            self._q.put(_WorkerMessage("error", str(exc)))

    def _log(self, msg: str) -> None:
        self._q.put(_WorkerMessage("log", msg))

    def _process_all(self) -> None:
        root = self._index.root
        existing = self._index.load_existing()
        current_rel = {str(p.relative_to(root)) for p in self._files}

        # 消えた画像をインデックスから削除
        removed = [r for r in existing if r not in current_rel]
        if removed:
            self._index.delete_many(removed)
            self._log(f"🗑 消えた画像 {len(removed)} 件をインデックスから削除しました")

        # 新規・更新が必要な画像を洗い出す
        todo: list[Path] = []
        skipped = 0
        for p in self._files:
            rel = str(p.relative_to(root))
            try:
                st = p.stat()
            except OSError:
                todo.append(p)
                continue
            prev = existing.get(rel)
            if prev and abs(prev[0] - st.st_mtime) < 1e-6 and prev[1] == st.st_size:
                skipped += 1
            else:
                todo.append(p)

        total = len(todo)
        size_label = f"上限 {self._max_image_size}px" if self._max_image_size > 0 else "原寸"
        self._log(
            f"対象画像: {len(self._files)} 枚  "
            f"（OCR 実行: {total} 枚 / スキップ: {skipped} 枚）\n"
            f"[画像サイズ: {size_label}]\n"
        )
        self._q.put(_WorkerMessage("progress", (0, max(total, 1))))

        if total == 0:
            self._log("更新はありませんでした（インデックスは最新です）。")
            self._q.put(_WorkerMessage("done", str(self._index.count())))
            return

        start = time.time()
        n_ok = n_err = 0
        for idx, p in enumerate(todo, start=1):
            rel = str(p.relative_to(root))
            self._log(f"[{idx}/{total}] {rel}")
            t0 = time.time()
            try:
                image = Image.open(p).convert("RGB")
                text = self._engine.recognize(image, max_image_size=self._max_image_size)
                st = p.stat()
                self._index.upsert(rel, text, st.st_mtime, st.st_size)
                n_ok += 1
                self._log(f"    ✅ 完了 ({_format_elapsed(time.time() - t0)})")
            except Exception as exc:
                n_err += 1
                self._log(f"    ⚠ エラー ({_format_elapsed(time.time() - t0)}): {exc}")
            self._q.put(_WorkerMessage("progress", (idx, total)))

        elapsed = time.time() - start
        self._log(
            f"\n{'=' * 50}\n"
            f"✅ インデックス更新完了\n"
            f"OCR 成功: {n_ok} 件 / 失敗: {n_err} 件 / スキップ: {skipped} 件\n"
            f"インデックス総数: {self._index.count()} 件\n"
            f"処理時間: {_format_elapsed(elapsed)}"
            + (f" (平均 {elapsed / total:.1f}秒/枚)" if total else "")
        )
        self._q.put(_WorkerMessage("done", str(self._index.count())))


# ════════════════════════════════════════════════════════════════════════
# UI: メインアプリケーション
# ════════════════════════════════════════════════════════════════════════

class ImageSearchApp:
    _POLL_MS = 100

    def __init__(self, root: tk.Tk) -> None:
        self._root = root
        self._root.title("KomaSagashi - 画像テキスト検索 (mokuro / manga-ocr-base)")
        self._root.geometry("980x680")

        self._engine        = MokuroEngine()
        self._msg_queue: "queue.Queue[_WorkerMessage]" = queue.Queue()
        self._is_processing = False
        self._timer_id: str | None = None
        self._process_start = 0.0

        # 検索結果 iid → (絶対パス, 全文) の対応
        self._result_map: dict[str, tuple[Path, str]] = {}
        self._preview_imgtk = None   # GC 防止のため参照を保持
        self._last_terms: list[str] = []

        self._build_ui()

    # ── UI 構築 ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        pad = {"padx": 8, "pady": 4}

        # 共有: 対象フォルダ
        f = ttk.LabelFrame(self._root, text="対象フォルダ")
        f.pack(fill="x", **pad)
        self._folder_var = tk.StringVar()
        ttk.Entry(f, textvariable=self._folder_var).pack(
            side="left", fill="x", expand=True, padx=4, pady=4)
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
        ttk.Label(f, text="画像リサイズ上限 (px):").grid(
            row=1, column=0, padx=4, pady=4, sticky="w")
        self._max_image_size_var = tk.StringVar(value=str(_DEFAULT_MAX_IMAGE_SIZE))
        ttk.Entry(f, textvariable=self._max_image_size_var, width=8).grid(
            row=1, column=1, padx=4, pady=4, sticky="w")
        ttk.Label(f, text="0 = 原寸で処理", foreground="gray").grid(
            row=1, column=2, padx=4, sticky="w")

        f = ttk.Frame(tab)
        f.pack(fill="x", padx=8, pady=4)
        self._progress = ttk.Progressbar(f, mode="determinate")
        self._progress.pack(side="left", fill="x", expand=True)
        self._elapsed_var = tk.StringVar(value="00:00")
        ttk.Label(f, textvariable=self._elapsed_var, width=8, anchor="center").pack(
            side="left", padx=(6, 0))
        self._run_btn = ttk.Button(f, text="▶ インデックス作成/更新", command=self._run_index)
        self._run_btn.pack(side="left", padx=6)

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
        entry = ttk.Entry(f, textvariable=self._query_var)
        entry.pack(side="left", fill="x", expand=True, padx=6)
        entry.bind("<Return>", lambda _e: self._run_search())
        ttk.Button(f, text="検索", command=self._run_search).pack(side="left")
        self._hit_var = tk.StringVar(value="")
        ttk.Label(f, textvariable=self._hit_var, width=14, anchor="e").pack(side="left", padx=4)

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

        # 右: プレビュー + 詳細
        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        self._preview_label = ttk.Label(
            right, text="（結果を選択すると画像を表示）",
            anchor="center", relief="groove")
        self._preview_label.pack(fill="both", expand=True, padx=4, pady=4)

        self._path_var = tk.StringVar(value="")
        ttk.Label(right, textvariable=self._path_var, foreground="gray",
                  anchor="w", wraplength=440).pack(fill="x", padx=4)

        btns = ttk.Frame(right)
        btns.pack(fill="x", padx=4, pady=4)
        self._btn_explorer = ttk.Button(
            btns, text="エクスプローラーで表示", command=self._open_in_explorer,
            state="disabled")
        self._btn_explorer.pack(side="left", padx=2)
        self._btn_open = ttk.Button(
            btns, text="既定のアプリで開く", command=self._open_default, state="disabled")
        self._btn_open.pack(side="left", padx=2)

        ttk.Label(right, text="全文（ヒット語を強調）:").pack(anchor="w", padx=4)
        self._fulltext = scrolledtext.ScrolledText(right, height=8, wrap="word",
                                                    state="disabled")
        self._fulltext.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        self._fulltext.tag_configure("hit", background="yellow")

    # ── フォルダ / DB ─────────────────────────────────────────────────

    def _db_path(self) -> Path | None:
        folder = self._folder_var.get().strip()
        if not folder:
            return None
        return Path(folder) / _DB_FILENAME

    def _index_or_none(self) -> TextIndex | None:
        db = self._db_path()
        return TextIndex(db) if db else None

    def _select_folder(self) -> None:
        path = filedialog.askdirectory(title="フォルダを選択")
        if not path:
            return
        self._folder_var.set(path)
        self._refresh_index_status()

    def _refresh_index_status(self) -> None:
        db = self._db_path()
        if db and db.exists():
            try:
                n = TextIndex(db).count()
                self._index_status_var.set(f"インデックス: あり（{n} 件）")
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

        self._is_processing = True
        self._run_btn.state(["disabled"])
        self._progress["value"] = 0
        self._elapsed_var.set("00:00")
        self._log_text.config(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.config(state="disabled")

        self._process_start = time.time()
        self._tick_elapsed()

        threading.Thread(
            target=self._load_and_index,
            args=(folder, self._recursive_var.get(), max_image_size),
            daemon=True,
        ).start()
        self._root.after(self._POLL_MS, self._poll_queue)

    def _tick_elapsed(self) -> None:
        if not self._is_processing:
            return
        self._elapsed_var.set(_format_elapsed(time.time() - self._process_start))
        self._timer_id = self._root.after(1000, self._tick_elapsed)

    def _load_and_index(self, folder: Path, recursive: bool, max_image_size: int) -> None:
        try:
            if not self._engine.is_loaded:
                self._engine.load(
                    progress_cb=lambda m: self._msg_queue.put(_WorkerMessage("log", m)))
            files = _collect_images(folder, recursive)
            index = TextIndex(folder / _DB_FILENAME)
        except Exception as exc:
            self._msg_queue.put(_WorkerMessage("error", str(exc)))
            return
        IndexWorker(self._engine, index, files, self._msg_queue,
                    max_image_size=max_image_size).start()

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

    def _on_index_done(self, count: str) -> None:
        self._is_processing = False
        self._stop_timer()
        self._run_btn.state(["!disabled"])
        self._elapsed_var.set(_format_elapsed(time.time() - self._process_start))
        self._refresh_index_status()
        messagebox.showinfo(
            "完了",
            f"インデックスを更新しました。\n"
            f"総登録数: {count} 件\n\n"
            "「② 検索」タブで検索できます。")

    def _on_error(self, error_msg: str) -> None:
        self._is_processing = False
        self._stop_timer()
        self._run_btn.state(["!disabled"])
        self._append_log(f"❌ エラー: {error_msg}")
        messagebox.showerror("エラー", error_msg)

    # ── 検索 ──────────────────────────────────────────────────────────

    def _run_search(self) -> None:
        index = self._index_or_none()
        db = self._db_path()
        if index is None:
            messagebox.showerror("エラー", "対象フォルダを指定してください。"); return
        if db is None or not db.exists():
            messagebox.showwarning(
                "インデックスなし",
                "このフォルダにはインデックスがありません。\n"
                "先に「① インデックス作成」を実行してください。")
            return

        terms = self._query_var.get().split()
        self._last_terms = terms
        self._tree.delete(*self._tree.get_children())
        self._result_map.clear()
        self._clear_preview()

        if not terms:
            self._hit_var.set("")
            return

        try:
            rows = index.search(terms)
        except sqlite3.Error as exc:
            messagebox.showerror("エラー", f"検索に失敗しました:\n{exc}"); return

        for relpath, text in rows:
            iid = self._tree.insert(
                "", "end",
                values=(Path(relpath).name, make_snippet(text, terms)))
            self._result_map[iid] = (index.root / relpath, text)

        self._hit_var.set(f"ヒット: {len(rows)} 件")
        if not rows:
            self._hit_var.set("ヒット: 0 件")

    def _current_path(self) -> Path | None:
        sel = self._tree.selection()
        if not sel:
            return None
        entry = self._result_map.get(sel[0])
        return entry[0] if entry else None

    def _on_result_select(self, _event=None) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        entry = self._result_map.get(sel[0])
        if not entry:
            return
        path, text = entry
        self._path_var.set(str(path))
        self._btn_explorer.state(["!disabled"])
        self._btn_open.state(["!disabled"])
        self._show_preview(path)
        self._show_fulltext(text)

    def _show_preview(self, path: Path) -> None:
        if not _IMAGETK_AVAILABLE:
            self._preview_label.config(image="", text="（プレビューには Pillow の ImageTk が必要）")
            return
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((_PREVIEW_BOX, _PREVIEW_BOX), Image.LANCZOS)
            self._preview_imgtk = ImageTk.PhotoImage(img)
            self._preview_label.config(image=self._preview_imgtk, text="")
        except Exception as exc:
            self._preview_imgtk = None
            self._preview_label.config(image="", text=f"（画像を表示できません: {exc}）")

    def _show_fulltext(self, text: str) -> None:
        self._fulltext.config(state="normal")
        self._fulltext.delete("1.0", "end")
        self._fulltext.insert("1.0", text)
        # ヒット語を全てハイライト
        low = text.lower()
        for term in self._last_terms:
            t = term.lower()
            if not t:
                continue
            start = 0
            while True:
                i = low.find(t, start)
                if i == -1:
                    break
                self._fulltext.tag_add(
                    "hit", f"1.0+{i}c", f"1.0+{i + len(term)}c")
                start = i + len(term)
        self._fulltext.config(state="disabled")

    def _clear_preview(self) -> None:
        self._preview_imgtk = None
        self._preview_label.config(image="", text="（結果を選択すると画像を表示）")
        self._path_var.set("")
        self._btn_explorer.state(["disabled"])
        self._btn_open.state(["disabled"])
        self._fulltext.config(state="normal")
        self._fulltext.delete("1.0", "end")
        self._fulltext.config(state="disabled")

    def _open_in_explorer(self) -> None:
        path = self._current_path()
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
        path = self._current_path()
        if not path:
            return
        if not path.exists():
            messagebox.showwarning("見つかりません",
                                   f"ファイルが見つかりません:\n{path}"); return
        try:
            if os.name == "nt":
                os.startfile(str(path))   # type: ignore[attr-defined]
            else:
                subprocess.run(["xdg-open", str(path)])
        except Exception as exc:
            messagebox.showerror("エラー", f"開けませんでした:\n{exc}")


# ════════════════════════════════════════════════════════════════════════
# エントリーポイント
# ════════════════════════════════════════════════════════════════════════

def main() -> None:
    root = tk.Tk()
    ImageSearchApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
