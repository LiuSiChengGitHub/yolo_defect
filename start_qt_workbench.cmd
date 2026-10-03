@echo off
rem Double-click launcher for the Qt workbench in a source checkout.
rem Builds the client once if it is missing, then starts it through qt.cmd run
rem with the default FP32 config, sample image and results\qt output folder.
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
title Qt defect inspection workbench

set "QT_TOOL=%~dp0cpp_infer\tools\qt.cmd"
set "QT_EXE=%~dp0cpp_infer\build\qt-msvc-release\bin\yolo_defect_qt.exe"

if exist "%QT_EXE%" goto :run
echo [Qt] Client not built yet. Building once; this can take a few minutes...
call "%QT_TOOL%" build
if not "%ERRORLEVEL%"=="0" goto :failed

:run
echo [Qt] Starting the workbench. This window closes when the workbench exits.
call "%QT_TOOL%" run
if not "%ERRORLEVEL%"=="0" goto :failed
endlocal & exit /b 0

:failed
echo.
echo [Qt] Launch failed. Check cpp_infer\.qt.local.psd1 and cpp_infer\.stage1.local.psd1,
echo      then see cpp_infer\apps\qt\README.md for setup details.
pause
endlocal & exit /b 1
