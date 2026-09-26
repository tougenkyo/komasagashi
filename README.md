# KomaSagashi（コマ探し）

**画像の中の文字で、画像を探す。**

フォルダ内の画像（漫画のページ・スクリーンショットなど）を OCR して索引を作り、
キーワード検索でヒットした画像をその場でプレビュー・エクスプローラーで開けるデスクトップツールです。
日本語の縦書き・吹き出しに強い [manga-ocr](https://github.com/kha-white/manga-ocr) / [mokuro](https://github.com/kha-white/mokuro) を OCR エンジンに使っています。

<!-- TODO: 検索 → プレビュー → エクスプローラーで表示 の流れが分かる GIF を置く -->

## こんなときに

- 自炊した漫画から「あのセリフのページ」を探したい
- 大量のスクリーンショットから、特定の文字が写っている1枚を見つけたい
- OCR 結果を 1 つのテキストファイルに書き出しても、元の画像を探すのが面倒
- ファイル名にテキストを入れると 255 文字の制限にかかる

## 特徴

- **索引化して何度でも高速検索** — OCR は重いので一度だけ。結果は SQLite に保存
- **増分更新** — 再実行時は新規・更新された画像だけ OCR。消えた画像は索引から自動で削除
- **スペース区切りの AND 検索** — `猫 名前` で両方を含む画像を絞り込み（部分一致・英字の大文字小文字は区別しない）
- **ヒットから元画像へすぐ飛べる** — 結果一覧で選ぶと画像プレビューと全文（ヒット語を強調）を表示。
  「エクスプローラーで表示」でファイルを選択した状態でフォルダが開く
- **元のファイル名はそのまま** — 画像ファイルには一切手を加えません
- **フォルダごと持ち運べる** — 索引は対象フォルダ直下に置かれ、画像は相対パスで記録

## 動作環境

- Windows（Python 3.10 で動作確認）
- NVIDIA GPU があれば CUDA で高速に動作。なくても CPU で動きますが遅くなります

## インストール

```bash
git clone https://github.com/tougenkyo/komasagashi.git
cd komasagashi
```

GPU を使う場合は、**先に** CUDA 版 torch を入れます（PyPI の Windows 版 torch は CPU 専用のため）。
コマンドは環境に合わせて [PyTorch 公式](https://pytorch.org/get-started/locally/) で確認してください。例（CUDA 12.8）:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

続いて残りの依存ライブラリを入れます。

```bash
pip install -r requirements.txt
```

初回の OCR 実行時に、テキスト検出モデルと manga-ocr-base のモデル（合わせて数百 MB）が自動でダウンロードされます。

## 使い方

```bash
python 画像テキスト検索.py
```

1. 上部の **対象フォルダ** で画像のあるフォルダを選ぶ
2. **① インデックス作成** タブで「▶ インデックス作成/更新」
   - 「サブフォルダも含める」（既定 ON）でサブフォルダの画像も対象になります
   - 2 回目以降は新しい・変更された画像だけを OCR します
3. **② 検索** タブでキーワードを入力して Enter
   - 左の一覧でヒットした画像を選ぶと、右に画像と全文が表示されます
   - 「エクスプローラーで表示」「既定のアプリで開く」、または一覧のダブルクリックで元画像を開けます

対応形式: `.jpg` `.jpeg` `.png` `.webp` `.bmp` `.gif` `.tiff`

### 索引ファイルについて

索引は対象フォルダ直下に `_画像テキスト検索.db` として作られます。
削除すれば次回は全画像を OCR し直します。
画像フォルダを Git 管理している場合は `.gitignore` に追加してください。

## トラブルシューティング

**`pkg_resources がありません` と表示される**

setuptools 81 以降で `pkg_resources` が削除されたためです（mokuro が依存する comic-text-detector が使用）。

```bash
pip install "setuptools<81"
```

**`GPU が見つかりません → CPU で推論します` と表示される**

CPU 版 torch が入っています。上の「インストール」の手順で CUDA 版を入れ直してください。

**文字がうまく読み取れない**

manga-ocr は日本語（特に漫画）向けのモデルです。英語主体の画像や写真の中の文字は苦手です。
また、① タブの「画像リサイズ上限」を大きくする（`0` で原寸）と、小さい文字の精度が上がる場合があります（処理は遅くなります）。

## 関連プロジェクト

- [mokuro](https://github.com/kha-white/mokuro) — 本ツールの OCR エンジン。漫画をテキスト選択可能な形で読むリーダー
- [manga-ocr](https://github.com/kha-white/manga-ocr) — 日本語の漫画向け OCR モデル
- [comic-text-detector](https://github.com/dmMaze/comic-text-detector) — 漫画の吹き出し・テキスト領域検出
- [Poricom](https://github.com/blueaxis/Poricom) — manga-ocr を使った画面 OCR の GUI ツール

本ツールは「文字を抜き出す」ことよりも「**索引化して検索し、元の画像にたどり着く**」ことを目的にしています。

## ライセンス

[GPL-3.0](LICENSE)

OCR エンジンとして使用している mokuro が GPL-3.0 のため、本ツールも GPL-3.0 で公開しています。
