@echo off
chcp 65001 >nul
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
