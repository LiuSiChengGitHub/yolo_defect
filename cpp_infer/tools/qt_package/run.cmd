@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
set "PATH=%~dp0;%SystemRoot%\System32;%SystemRoot%"
set "QT_PLUGIN_PATH=%~dp0"
set "QT_QPA_PLATFORM_PLUGIN_PATH=%~dp0platforms"
set "QT_QPA_PLATFORM=windows"
start "" /wait "%~dp0yolo_defect_qt.exe" --config "configs\default_config.txt" --image "samples\crazing_241.jpg" --output-dir "outputs" %*
set "DEMO_EXIT=%ERRORLEVEL%"
if not "%DEMO_EXIT%"=="0" (
  echo Workbench exited with code %DEMO_EXIT%. See demo\index.html and run verify.ps1 for diagnostics.
  pause
)
endlocal & exit /b %DEMO_EXIT%
