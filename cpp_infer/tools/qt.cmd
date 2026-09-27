@echo off
setlocal EnableExtensions DisableDelayedExpansion

set "QT_SCRIPT_DIR=%~dp0"
if "%~1"=="" goto :show_help
if /I "%~1"=="help" goto :run_without_vs
if /I "%~1"=="run" goto :run_without_vs
if /I "%~1"=="configure" goto :prepare_vs
if /I "%~1"=="build" goto :prepare_vs
if /I "%~1"=="test" goto :prepare_vs
goto :run_without_vs

:prepare_vs
set "QT_VSDEVCMD=%YOLO_DEFECT_VSDEVCMD%"
set "QT_VSWHERE=%SystemDrive%\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
if defined QT_VSDEVCMD goto :validate_vs
if not exist "%QT_VSWHERE%" (
  echo Qt workflow: vswhere.exe not found. Install the Visual Studio C++ workload or set YOLO_DEFECT_VSDEVCMD to VsDevCmd.bat. 1>&2
  exit /b 1
)
for /f "usebackq delims=" %%I in (`"%QT_VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "QT_VSINSTALL=%%I"
if defined QT_VSINSTALL set "QT_VSDEVCMD=%QT_VSINSTALL%\Common7\Tools\VsDevCmd.bat"

:validate_vs
if not defined QT_VSDEVCMD (
  echo Qt workflow: MSVC was not found. Install the Visual Studio C++ workload or set YOLO_DEFECT_VSDEVCMD. 1>&2
  exit /b 1
)
if not exist "%QT_VSDEVCMD%" (
  echo Qt workflow: VsDevCmd.bat not found at "%QT_VSDEVCMD%". Correct YOLO_DEFECT_VSDEVCMD. 1>&2
  exit /b 1
)
call "%QT_VSDEVCMD%" -arch=amd64 -host_arch=amd64 >nul
if errorlevel 1 exit /b %ERRORLEVEL%

:run_without_vs
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%QT_SCRIPT_DIR%qt.ps1" %*
set "QT_WORKFLOW_EXIT=%ERRORLEVEL%"
endlocal & exit /b %QT_WORKFLOW_EXIT%

:show_help
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%QT_SCRIPT_DIR%qt.ps1" help
set "QT_WORKFLOW_EXIT=%ERRORLEVEL%"
endlocal & exit /b %QT_WORKFLOW_EXIT%
