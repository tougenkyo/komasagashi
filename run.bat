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
title KomaSagashi

rem KomaSagashi を起動する（先に setup.bat を 1 回実行しておくこと）。
rem この黒い画面はアプリの動作ログ用。閉じるとアプリも終了する。

if exist ".venv\Scripts\python.exe" goto :run
echo.
echo   セットアップがまだです。
echo   先に setup.bat をダブルクリックしてください。
echo.
pause
exit /b 1

:run
echo KomaSagashi を起動しています...（この画面は閉じないでください）
".venv\Scripts\python.exe" -W ignore komasagashi.py
if errorlevel 1 (
    echo.
    echo   エラーで終了しました。上に表示されたメッセージを確認してください。
    echo.
    pause
)
