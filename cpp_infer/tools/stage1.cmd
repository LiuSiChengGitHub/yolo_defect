@echo off
setlocal EnableExtensions DisableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "VSDEVCMD=%YOLO_DEFECT_VSDEVCMD%"
set "VSWHERE=%SystemDrive%\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"

if "%~1"=="" goto :show_help
if /I "%~1"=="help" goto :run_without_vs

if defined VSDEVCMD goto :validate_vsdevcmd

if not exist "%VSWHERE%" (
  echo Stage-1 environment preflight failed: object=vswhere.exe; expected=the Visual Studio Installer discovery tool; actual=not found at "%VSWHERE%"; action=set YOLO_DEFECT_VSDEVCMD to the full VsDevCmd.bat path. 1>&2
  exit /b 1
)

for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%I"
if defined VSINSTALL set "VSDEVCMD=%VSINSTALL%\Common7\Tools\VsDevCmd.bat"

:validate_vsdevcmd
if not defined VSDEVCMD (
  echo Stage-1 environment preflight failed: object=VsDevCmd.bat; expected=an x64 MSVC developer environment entry; actual=Visual Studio was not discovered; action=install the C++ workload or set YOLO_DEFECT_VSDEVCMD. 1>&2
  exit /b 1
)

if not exist "%VSDEVCMD%" (
  echo Stage-1 environment preflight failed: object=VsDevCmd.bat; expected=an existing batch file; actual="%VSDEVCMD%"; action=correct YOLO_DEFECT_VSDEVCMD or repair Visual Studio. 1>&2
  exit /b 1
)

call "%VSDEVCMD%" -arch=amd64 -host_arch=amd64 >nul
if errorlevel 1 (
  echo Stage-1 environment preflight failed: object=VsDevCmd.bat; expected=successful x64 initialization; actual=nonzero exit; action=repair the MSVC x64 workload and rerun. 1>&2
  exit /b 1
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%stage1.ps1" %*
set "STAGE1_EXIT=%ERRORLEVEL%"
endlocal & exit /b %STAGE1_EXIT%

:run_without_vs
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%stage1.ps1" %*
set "STAGE1_EXIT=%ERRORLEVEL%"
endlocal & exit /b %STAGE1_EXIT%

:show_help
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%stage1.ps1" help
set "STAGE1_EXIT=%ERRORLEVEL%"
endlocal & exit /b %STAGE1_EXIT%
