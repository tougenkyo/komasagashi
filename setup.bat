@echo off
rem ---- Restart in a new cmd with the UTF-8 code page. Keep this part ASCII only. ----
rem cmd misreads lines that contain Japanese when the code page is changed (chcp 65001)
rem in the middle of a running batch file, so switch first and run this file again.
if "%KOMA_UTF8%"=="1" goto :main
for /f "tokens=2 delims=:." %%a in ('chcp') do for %%b in (%%a) do set "KOMA_OLDCP=%%b"
chcp 65001 >nul
set "KOMA_UTF8=1"
cmd /c ""%~f0" %*"
set "KOMA_RC=%errorlevel%"
if defined KOMA_OLDCP chcp %KOMA_OLDCP% >nul
exit /b %KOMA_RC%

:main
setlocal
cd /d "%~dp0"
title KomaSagashi セットアップ

rem ====================================================================
rem  KomaSagashi セットアップ（初回に 1 回だけダブルクリック）
rem    1. Python を探す（動作確認済みの 3.10 を優先）
rem    2. このフォルダに仮想環境 .venv を作る
rem    3. NVIDIA GPU があれば CUDA 版、なければ CPU 版の torch を入れる
rem    4. 残りのライブラリを入れ、読み込めるか確認する
rem    5. 任意で PaddleOCR（横書きのページ用）を入れる
rem  何度実行しても大丈夫（入っているものはスキップされる）。
rem ====================================================================

echo ================================================================
echo   KomaSagashi セットアップ
echo   GPU 版は約 3 GB のダウンロードがあり、10〜30 分ほどかかります
echo ================================================================
echo.

rem ---- 1. Python を探す ----------------------------------------------
rem  py ランチャーで python.org 版を探す。<nul は、バージョンが無いときに
rem  インストールの確認を求められても止まらないようにするため。
echo [1/5] Python を探しています...
set "PY="
for %%V in (3.10 3.11 3.12 3.13) do (
    if not defined PY (
        py -%%V -c "import sys" <nul >nul 2>nul && set "PY=py -%%V"
    )
)
if defined PY goto :python_found
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" <nul >nul 2>nul && set "PY=python"
if defined PY goto :python_found

echo.
echo   Python 3.10 以上が見つかりませんでした。
echo   https://www.python.org/downloads/ から Python をインストールしてから、
echo   もう一度 setup.bat をダブルクリックしてください。
echo   （動作確認は Python 3.10 です。3.11〜3.13 でも動く見込みです）
goto :fail

:python_found
for /f "delims=" %%i in ('%PY% --version') do echo       %%i を使います
echo.

rem ---- 2. 仮想環境を作る ---------------------------------------------
rem  PC 全体の Python を汚さないよう、ライブラリはこのフォルダの .venv に入れる。
echo [2/5] 仮想環境 .venv を準備しています...
if exist ".venv\Scripts\python.exe" goto :venv_ready
%PY% -m venv .venv
if errorlevel 1 goto :fail
:venv_ready
set "VPY=.venv\Scripts\python.exe"
"%VPY%" -m pip install --quiet --upgrade pip
if errorlevel 1 goto :fail
echo       OK
echo.

rem ---- 3. torch / torchvision -----------------------------------------
rem  torch と torchvision は必ず同じ配布元からまとめて入れる。
rem  torchvision だけ後から PyPI で入ると、torch が CPU 版に置き換えられることがある。
echo [3/5] PyTorch をインストールしています...
where nvidia-smi >nul 2>nul
if errorlevel 1 goto :torch_cpu
echo       NVIDIA GPU を検出しました → GPU 版（CUDA 12.8）を入れます
set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
goto :torch_install
:torch_cpu
echo       NVIDIA GPU が見つかりません → CPU 版を入れます（OCR は遅くなります）
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
:torch_install
"%VPY%" -m pip install torch torchvision --index-url %TORCH_INDEX%
if errorlevel 1 goto :fail
echo.

rem ---- 4. 残りのライブラリ -------------------------------------------
echo [4/5] 残りのライブラリをインストールしています...
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 goto :fail
echo.

echo インストールを確認しています...
"%VPY%" -W ignore -c "import torch, mokuro.manga_page_ocr, py7zr, rarfile; print('      torch', torch.__version__, '/ GPU:', '使えます' if torch.cuda.is_available() else '使いません（CPU で動作）')"
if errorlevel 1 goto :fail

rem ---- 5. PaddleOCR（任意） -------------------------------------------
rem  目次・扉など横書きのページを読み直すための任意機能。大きいので入れるか尋ねる。
rem  すでに入っていれば尋ねずに更新だけ確認する。
echo.
echo [5/5] PaddleOCR（任意機能）
"%VPY%" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('paddleocr') else 1)" <nul >nul 2>nul
if not errorlevel 1 goto :paddle_install
echo       目次・扉・説明文など、横書きのページの読み取り精度が上がります。
echo       ダウンロードは約 550 MB です。あとで setup.bat を実行し直して追加することもできます。
choice /c YN /n /m "      PaddleOCR を追加しますか？ [Y/N]: "
if errorlevel 2 goto :paddle_done
:paddle_install
"%VPY%" -m pip install -r requirements-paddle.txt
if errorlevel 1 goto :fail
"%VPY%" -W ignore -c "import os, sys; sys.version_info < (3, 12) and os.environ.setdefault('SETUPTOOLS_USE_DISTUTILS', 'stdlib'); import paddleocr, importlib.metadata as m; print('      PaddleOCR', m.version('paddleocr'), 'OK')"
if errorlevel 1 goto :fail
:paddle_done

echo.
echo ================================================================
echo   セットアップ完了！
echo   run.bat をダブルクリックすると KomaSagashi が起動します。
echo   初回の OCR 実行時に、モデル（数百 MB）が自動でダウンロードされます。
echo   （PaddleOCR を入れた場合は、そのモデルも初回に自動でダウンロードされます）
echo ================================================================
echo.
pause
exit /b 0

:fail
echo.
echo ****************************************************************
echo   セットアップに失敗しました。上に表示されたメッセージを確認してください。
echo   よくある原因:
echo     - インターネットに接続されていない
echo     - Python がインストールされていない
echo     - Python が新しすぎる（3.14 以降は未確認。3.10 をおすすめします）
echo   原因を直してから、もう一度 setup.bat をダブルクリックしてください。
echo ****************************************************************
echo.
pause
exit /b 1
