[CmdletBinding()]
param(
  [Parameter(Position = 0)]
  [ValidateSet('help', 'configure', 'build', 'test', 'run', 'package', 'media')]
  [string]$Action = 'help',
  [string]$QtRoot = '',
  [string]$BuildDir = '',
  [string]$OrtRoot = '',
  [string]$OpenCvDir = '',
  [string]$OpenCvBin = '',
  [string]$PythonExe = '',
  [string]$GTestSource = '',
  [string]$Config = '',
  [string]$Image = '',
  [string]$OutputDir = '',
  [string]$PackageDir = '',
  [switch]$Screenshots,
  [string]$ScreenshotDir = ''
)

Set-StrictMode -Version 3.0
$ErrorActionPreference = 'Stop'

if ($Action -eq 'help') {
  Write-Host @'
YOLO Defect Qt desktop workflow (Windows, MSVC x64, Release)

Usage from PowerShell or CMD:
  cpp_infer\tools\qt.cmd configure
  cpp_infer\tools\qt.cmd build
  cpp_infer\tools\qt.cmd test [-Screenshots | -ScreenshotDir <directory>]
  cpp_infer\tools\qt.cmd run [-Config <file>] [-Image <file>] [-OutputDir <dir>]
  cpp_infer\tools\qt.cmd package [-PackageDir <new-or-empty-directory>]
  cpp_infer\tools\qt.cmd media

configure/build disable BUILD_TESTING; test enables it, builds the Qt tests
and CLI, then runs only yolo_defect_qt_client. Test requires a local GoogleTest
source tree and the project's Python validation environment. No dependencies
are downloaded or installed by this script. run uses an existing executable.
help/run do not initialize or require the MSVC compiler. package builds Release
Qt/CLI executables and creates a portable Windows demo in dist/yolo-defect-qt.
It refuses a nonempty destination. media captures real UI interactions and
rebuilds the README/HTML media; it uses the same test dependencies as test.

Path options:
  -QtRoot -BuildDir -OrtRoot -OpenCvDir -OpenCvBin -PythonExe -GTestSource
  -Config -Image -OutputDir -PackageDir

Settings precedence:
  explicit parameter > cpp_infer/.qt.local.psd1 > .stage1.local.psd1
  > environment > portable defaults
Shared stage1 settings: OrtRoot, OpenCvDir, OpenCvBin, PythonExe, GTestSource.
Copy tools/qt.local.example.psd1 to cpp_infer/.qt.local.psd1 to configure Qt.
Relative settings-file paths use that file's directory; command-line and
environment paths use the caller's working directory.

Environment equivalents:
  YOLO_DEFECT_QT_ROOT, YOLO_DEFECT_QT_BUILD_DIR, ONNXRUNTIME_ROOT, OpenCV_DIR,
  YOLO_DEFECT_OPENCV_BIN, YOLO_DEFECT_PYTHON, YOLO_DEFECT_GTEST_SOURCE,
  YOLO_DEFECT_QT_CONFIG, YOLO_DEFECT_QT_IMAGE, YOLO_DEFECT_QT_OUTPUT_DIR,
  YOLO_DEFECT_QT_PACKAGE_DIR
  YOLO_DEFECT_VSDEVCMD (optional custom VsDevCmd.bat for the CMD wrapper)

Defaults: cpp_infer/build/qt-msvc-release; configs/default_config.txt;
data/images/val/crazing_241.jpg; results/qt. All are anchored to this checkout.
Screenshots are opt-in and default to <build>/screenshots. QtTest loads the
Windows Chinese font when available. Test logs remain in the build directory.
'@
  exit 0
}

$sourceRoot = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent $sourceRoot
$invocationDirectory = (Get-Location).ProviderPath
$explicitSettings = $PSBoundParameters
$exitCode = 1
$sharedSettings = @{}
$qtSettings = @{}

function Read-LocalSettings {
  param([string]$Path)
  $tokens = $null
  $parseErrors = $null
  $ast = [Management.Automation.Language.Parser]::ParseFile(
    $Path, [ref]$tokens, [ref]$parseErrors)
  if ($parseErrors.Count -ne 0 -or $ast.EndBlock.Statements.Count -ne 1 -or
      $ast.EndBlock.Statements[0].PipelineElements.Count -ne 1) {
    throw "Expected one literal settings hashtable in '$Path'."
  }
  $expression = $ast.EndBlock.Statements[0].PipelineElements[0].Expression
  if ($expression -isnot [Management.Automation.Language.HashtableAst]) {
    throw "Expected one literal settings hashtable in '$Path'."
  }
  return $expression.SafeGetValue()
}

function Resolve-SettingPath {
  param([string]$Key, [string]$EnvironmentName, [string]$Default = '')
  $baseDirectory = $invocationDirectory
  if ($explicitSettings.ContainsKey($Key)) {
    $value = $explicitSettings[$Key]
  } elseif ($qtSettings.ContainsKey($Key)) {
    $value = $qtSettings[$Key]
    $baseDirectory = $sourceRoot
  } elseif ($sharedSettings.ContainsKey($Key)) {
    $value = $sharedSettings[$Key]
    $baseDirectory = $sourceRoot
  } elseif (-not [string]::IsNullOrWhiteSpace(
      [Environment]::GetEnvironmentVariable($EnvironmentName))) {
    $value = [Environment]::GetEnvironmentVariable($EnvironmentName)
  } else {
    $value = $Default
  }
  if ($value -isnot [string]) {
    throw "$Key must be a path string in the local settings file."
  }
  if ([string]::IsNullOrWhiteSpace($value)) { return '' }
  if (-not [IO.Path]::IsPathRooted($value)) {
    $value = Join-Path $baseDirectory $value
  }
  return [IO.Path]::GetFullPath($value)
}

function Require-Path {
  param([string]$Path, [string]$Name, [string]$Kind = 'Leaf')
  if ([string]::IsNullOrWhiteSpace($Path) -or
      -not (Test-Path -LiteralPath $Path -PathType $Kind)) {
    throw "$Name is missing: '$Path'. Set the corresponding path option or local setting."
  }
}

function Invoke-Checked {
  param([string]$Command, [string[]]$Arguments)
  & $Command @Arguments
  if ($LASTEXITCODE -ne 0) {
    $script:exitCode = $LASTEXITCODE
    throw "$Command failed with exit code $LASTEXITCODE."
  }
}

$savedEnvironment = @{}
foreach ($name in @('PATH', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM',
    'YOLO_DEFECT_QT_SCREENSHOT_DIR', 'YOLO_DEFECT_QT_TEST_FONT',
    'YOLO_DEFECT_QT_TEST_HIDDEN')) {
  $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name)
}

try {
  if (($Screenshots -or $ScreenshotDir) -and $Action -ne 'test') {
    throw '-Screenshots and -ScreenshotDir are options for test only.'
  }
  if ($PackageDir -and $Action -ne 'package') {
    throw '-PackageDir is an option for package only.'
  }
  $sharedFile = Join-Path $sourceRoot '.stage1.local.psd1'
  if (Test-Path -LiteralPath $sharedFile -PathType Leaf) {
    $allSharedSettings = Read-LocalSettings $sharedFile
    foreach ($name in @('OrtRoot', 'OpenCvDir', 'OpenCvBin', 'PythonExe', 'GTestSource')) {
      if ($allSharedSettings.ContainsKey($name)) {
        $sharedSettings[$name] = $allSharedSettings[$name]
      }
    }
  }
  $qtFile = Join-Path $sourceRoot '.qt.local.psd1'
  if (Test-Path -LiteralPath $qtFile -PathType Leaf) {
    $qtSettings = Read-LocalSettings $qtFile
  }

  $resolvedQt = Resolve-SettingPath QtRoot YOLO_DEFECT_QT_ROOT
  if (-not $resolvedQt) {
    throw 'QtRoot is unset. Configure .qt.local.psd1, -QtRoot, or YOLO_DEFECT_QT_ROOT with your Qt 6 MSVC x64 kit.'
  }
  $resolvedBuild = Resolve-SettingPath BuildDir YOLO_DEFECT_QT_BUILD_DIR `
    (Join-Path $sourceRoot 'build\qt-msvc-release')
  if (-not $resolvedBuild) { throw 'BuildDir must not be empty.' }
  if ($Action -eq 'package') {
    $resolvedPackage = Resolve-SettingPath PackageDir YOLO_DEFECT_QT_PACKAGE_DIR `
      (Join-Path $repoRoot 'dist\yolo-defect-qt')
    if (-not $resolvedPackage) { throw 'PackageDir must not be empty.' }
    if (Test-Path -LiteralPath $resolvedPackage) {
      Require-Path $resolvedPackage 'PackageDir' Container
      if (@(Get-ChildItem -LiteralPath $resolvedPackage -Force).Count -ne 0) {
        throw "PackageDir is not empty: '$resolvedPackage'. Choose a new or empty directory."
      }
    }
  }
  $resolvedOpenCvDir = Resolve-SettingPath OpenCvDir OpenCV_DIR
  $defaultOpenCvBin = ''
  if ($resolvedOpenCvDir) {
    $defaultOpenCvBin = Join-Path (Split-Path -Parent $resolvedOpenCvDir) 'bin'
  }
  $resolvedOpenCvBin = Resolve-SettingPath OpenCvBin YOLO_DEFECT_OPENCV_BIN $defaultOpenCvBin
  Require-Path (Join-Path $resolvedQt 'bin\Qt6Widgets.dll') 'Qt 6 Widgets runtime'
  Require-Path (Join-Path $resolvedQt 'plugins\platforms\qwindows.dll') 'Qt Windows plugin'
  Require-Path $resolvedOpenCvBin 'OpenCV runtime directory (OpenCvBin)' Container
  $env:PATH = "$(Join-Path $resolvedQt 'bin');$resolvedOpenCvBin;$env:PATH"
  $env:QT_PLUGIN_PATH = Join-Path $resolvedQt 'plugins'

  Write-Host "[Qt] SDK: $resolvedQt"
  Write-Host "[Qt] Build: $resolvedBuild"
  if ($Action -eq 'run') {
    $executable = Join-Path $resolvedBuild 'bin\yolo_defect_qt.exe'
    Require-Path $executable 'Qt executable; run qt.cmd build first'
    $resolvedConfig = Resolve-SettingPath Config YOLO_DEFECT_QT_CONFIG `
      (Join-Path $sourceRoot 'configs\default_config.txt')
    $resolvedImage = Resolve-SettingPath Image YOLO_DEFECT_QT_IMAGE `
      (Join-Path $repoRoot 'data\images\val\crazing_241.jpg')
    $resolvedOutput = Resolve-SettingPath OutputDir YOLO_DEFECT_QT_OUTPUT_DIR `
      (Join-Path $repoRoot 'results\qt')
    Require-Path $resolvedConfig 'Runtime configuration'
    Require-Path $resolvedImage 'Input image'
    if (-not $resolvedOutput) { throw 'OutputDir must not be empty.' }
    $env:QT_QPA_PLATFORM = 'windows'
    # Start-Process joins ArgumentList into a Windows command line. Quote each
    # path and double trailing backslashes so the closing quote stays literal.
    $runArguments = @('--config', $resolvedConfig, '--image', $resolvedImage,
      '--output-dir', $resolvedOutput) | ForEach-Object {
      '"' + ($_ -replace '(\\+)$', '$1$1') + '"'
    }
    $runProcess = Start-Process -FilePath $executable -ArgumentList $runArguments `
      -Wait -PassThru
    if ($runProcess.ExitCode -ne 0) {
      $script:exitCode = $runProcess.ExitCode
      throw "Qt client failed with exit code $($runProcess.ExitCode)."
    }
  } else {
    if ($env:VSCMD_ARG_TGT_ARCH -ne 'x64') {
      throw 'Use qt.cmd to initialize the x64 MSVC toolchain, or run qt.ps1 in an x64 Developer PowerShell.'
    }
    Get-Command cmake -ErrorAction Stop | Out-Null
    Get-Command nmake -ErrorAction Stop | Out-Null
    Require-Path (Join-Path $resolvedQt 'lib\cmake\Qt6\Qt6Config.cmake') 'Qt development kit'
    Require-Path $resolvedOpenCvDir 'OpenCvDir' Container
    Require-Path (Join-Path $resolvedOpenCvDir 'OpenCVConfig.cmake') 'OpenCvDir'
    $resolvedOrt = Resolve-SettingPath OrtRoot ONNXRUNTIME_ROOT
    Require-Path $resolvedOrt 'ONNX Runtime SDK (OrtRoot)' Container
    Require-Path (Join-Path $resolvedOrt 'include\onnxruntime_cxx_api.h') 'ONNX Runtime SDK (OrtRoot)'
    $testing = if ($Action -in @('test', 'media')) { 'ON' } else { 'OFF' }
    $configureArgs = @('-S', $sourceRoot, '-B', $resolvedBuild, '-G', 'NMake Makefiles',
      '-DCMAKE_BUILD_TYPE=Release', '-DYOLO_DEFECT_BUILD_QT=ON', '-DYOLO_DEFECT_CORE_ONLY=OFF',
      "-DBUILD_TESTING=$testing", '-UQt6*_DIR', "-DCMAKE_PREFIX_PATH=$resolvedQt",
      "-DONNXRUNTIME_ROOT=$resolvedOrt", "-DOpenCV_DIR=$resolvedOpenCvDir")
    if ($Action -in @('test', 'media')) {
      Get-Command ctest -ErrorAction Stop | Out-Null
      Require-Path (Join-Path $resolvedQt 'lib\cmake\Qt6Test\Qt6TestConfig.cmake') 'Qt Test development component'
      $resolvedGTest = Resolve-SettingPath GTestSource YOLO_DEFECT_GTEST_SOURCE `
        (Join-Path $sourceRoot 'build\googletest')
      $resolvedPython = Resolve-SettingPath PythonExe YOLO_DEFECT_PYTHON
      Require-Path $resolvedGTest 'Local GoogleTest source (GTestSource)' Container
      Require-Path (Join-Path $resolvedGTest 'CMakeLists.txt') 'Local GoogleTest source (GTestSource)'
      Require-Path $resolvedPython 'Python validation interpreter (PythonExe)'
      $configureArgs += @("-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=$resolvedGTest",
        '-DFETCHCONTENT_FULLY_DISCONNECTED=ON', "-DPython3_EXECUTABLE=$resolvedPython")
    }
    Invoke-Checked cmake $configureArgs
    if ($Action -ne 'configure') {
      $targets = @('yolo_defect_qt')
      if ($Action -eq 'test') { $targets += 'yolo_defect_qt_tests' }
      if ($Action -eq 'package') { $targets += 'yolo_defect_cpp' }
      if ($Action -eq 'media') { $targets += 'yolo_defect_qt_capture' }
      Invoke-Checked cmake (@('--build', $resolvedBuild, '--target') + $targets)
    }
    if ($Action -eq 'package') {
      & (Join-Path $PSScriptRoot 'qt_package.ps1') -RepoRoot $repoRoot `
        -BuildDir $resolvedBuild -PackageDir $resolvedPackage -QtRoot $resolvedQt `
        -OrtRoot $resolvedOrt -OpenCvBin $resolvedOpenCvBin
    }
    if ($Action -eq 'media') {
      $capture = Join-Path $resolvedBuild 'bin\yolo_defect_qt_capture.exe'
      Require-Path $capture 'Qt media capture executable'
      $frames = Join-Path $resolvedBuild 'demo-frames'
      $env:QT_QPA_PLATFORM = 'windows'
      $env:YOLO_DEFECT_QT_TEST_HIDDEN = '1'
      Invoke-Checked $capture @('--repo-root', $repoRoot, '--frames', $frames)
      Invoke-Checked $resolvedPython @((Join-Path $PSScriptRoot 'qt_demo_media.py'),
        '--frames', $frames)
      Invoke-Checked $resolvedPython @((Join-Path $PSScriptRoot 'render_demo.py'))
    }
    if ($Action -eq 'test') {
      $env:YOLO_DEFECT_QT_SCREENSHOT_DIR = $null
      if ($Screenshots -or $ScreenshotDir) {
        $resolvedScreenshots = if ($ScreenshotDir) {
          Resolve-SettingPath ScreenshotDir YOLO_DEFECT_QT_SCREENSHOT_DIR
        } else { Join-Path $resolvedBuild 'screenshots' }
        $env:YOLO_DEFECT_QT_SCREENSHOT_DIR = $resolvedScreenshots
        Write-Host "[Qt] Screenshots: $resolvedScreenshots"
      }
      $chineseFont = Join-Path $env:WINDIR 'Fonts\msyh.ttc'
      if (Test-Path -LiteralPath $chineseFont -PathType Leaf) {
        $env:YOLO_DEFECT_QT_TEST_FONT = $chineseFont
      }
      # Reject an empty gate even if the installed CTest treats it as success.
      $testList = & ctest --test-dir $resolvedBuild -N -R '^yolo_defect_qt_client$'
      if ($LASTEXITCODE -ne 0) {
        $script:exitCode = $LASTEXITCODE
        throw 'CTest could not enumerate the Qt test.'
      }
      if (($testList -join "`n") -notmatch 'Total Tests: 1\s*$') {
        throw 'Expected exactly one yolo_defect_qt_client CTest entry; the Qt gate was not registered.'
      }
      Invoke-Checked ctest @('--test-dir', $resolvedBuild, '-R', '^yolo_defect_qt_client$', '--output-on-failure')
      $qtTestLog = Join-Path $resolvedBuild 'qt_client_test.txt'
      Require-Path $qtTestLog 'QtTest acceptance log'
      if ((Get-Content -LiteralPath $qtTestLog -Raw) -notmatch
          'Totals:\s+\d+ passed, 0 failed, 0 skipped,') {
        throw "Qt acceptance is incomplete (failure or skipped case). Inspect '$qtTestLog'; ensure the FP32/U8S8 models and test images are available."
      }
    }
  }
  $exitCode = 0
} catch {
  [Console]::Error.WriteLine("Qt workflow: $($_.Exception.Message)")
} finally {
  foreach ($name in $savedEnvironment.Keys) {
    [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], 'Process')
  }
}
exit $exitCode
