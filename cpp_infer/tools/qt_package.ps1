# Called by qt.ps1 after a Release build in the x64 MSVC environment.
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)][string]$RepoRoot,
  [Parameter(Mandatory = $true)][string]$BuildDir,
  [Parameter(Mandatory = $true)][string]$PackageDir,
  [Parameter(Mandatory = $true)][string]$QtRoot,
  [Parameter(Mandatory = $true)][string]$OrtRoot,
  [Parameter(Mandatory = $true)][string]$OpenCvBin
)

Set-StrictMode -Version 3.0
$ErrorActionPreference = 'Stop'
$utf8 = New-Object Text.UTF8Encoding($false)
$PackageDir = [IO.Path]::GetFullPath($PackageDir)
if (Test-Path -LiteralPath $PackageDir) {
  if (-not (Test-Path -LiteralPath $PackageDir -PathType Container) -or
      @(Get-ChildItem -LiteralPath $PackageDir -Force).Count -ne 0) {
    throw "Package destination must be new or empty: '$PackageDir'."
  }
}

function Copy-PackageFile([string]$Source, [string]$RelativeTarget) {
  if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
    throw "Package input is missing: '$Source'."
  }
  $target = Join-Path $PackageDir $RelativeTarget
  New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
  Copy-Item -LiteralPath $Source -Destination $target
}

function Write-PackageText([string]$RelativeTarget, [string]$Text) {
  $target = Join-Path $PackageDir $RelativeTarget
  New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
  [IO.File]::WriteAllText($target, $Text, $utf8)
}

function Find-MsvcCrt {
  # VsDevCmd selects the redistributables for the active toolset. Its patch
  # number can differ from VCToolsVersion; the major/minor toolset must match.
  if ($env:VCToolsVersion -notmatch '^\d+\.\d+\.\d+') {
    throw 'VCToolsVersion is unset. Use qt.cmd package in the x64 MSVC environment.'
  }
  $toolset = [version]$env:VCToolsVersion.Trim()
  $redistRoots = @()
  if ($env:VCToolsRedistDir) { $redistRoots += $env:VCToolsRedistDir }
  if ($env:VCINSTALLDIR) {
    $redistParent = Join-Path $env:VCINSTALLDIR 'Redist/MSVC'
    if (Test-Path -LiteralPath $redistParent -PathType Container) {
      $redistRoots += @(Get-ChildItem -LiteralPath $redistParent -Directory |
        Where-Object { $_.Name -match '^\d+\.\d+\.\d+$' } |
        Sort-Object { [version]$_.Name } -Descending |
        ForEach-Object { $_.FullName })
    }
  }
  foreach ($redistRoot in ($redistRoots | Select-Object -Unique)) {
    $x64Root = Join-Path $redistRoot 'x64'
    if (-not (Test-Path -LiteralPath $x64Root -PathType Container)) { continue }
    foreach ($crt in (Get-ChildItem -LiteralPath $x64Root -Directory -Filter 'Microsoft.VC*.CRT')) {
      $runtime = Join-Path $crt.FullName 'msvcp140.dll'
      if (-not (Test-Path -LiteralPath $runtime -PathType Leaf)) { continue }
      $version = (Get-Item -LiteralPath $runtime).VersionInfo
      if ($version.FileMajorPart -eq $toolset.Major -and
          $version.FileMinorPart -eq $toolset.Minor) {
        return [pscustomobject]@{
          Directory = $crt.FullName
          ToolsVersion = $toolset.ToString()
          RuntimeVersion = $version.FileVersion
        }
      }
    }
  }
  throw "No x64 MSVC CRT matching toolset $($toolset.Major).$($toolset.Minor) was found in VCToolsRedistDir or VCINSTALLDIR/Redist/MSVC."
}

# Validate the maintained presentation before creating a partial package.
$requiredMedia = @('workbench.png', 'walkthrough.gif', '01-ready.webp',
  '02-running.webp', '03-results.webp', '04-inspect.webp', '05-browse.webp',
  '06-overview.webp')
$requiredCharts = @('quantization.zh.svg', 'quantization.en.svg',
  'batch-throughput.zh.svg', 'batch-throughput.en.svg')
$requiredInputs = @('docs/demo/index.html', 'docs/demo/index.en.html', 'models/best.onnx') +
  @($requiredMedia | ForEach-Object { "docs/assets/qt/$_" }) +
  @($requiredCharts | ForEach-Object { "docs/assets/engineering/$_" })
foreach ($file in $requiredInputs) {
  if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot $file) -PathType Leaf)) {
    throw "Missing package input '$file'. Restore or regenerate the presentation inputs; see docs/demo/README.md."
  }
}
$deploy = Join-Path $QtRoot 'bin\windeployqt.exe'
if (-not (Test-Path -LiteralPath $deploy -PathType Leaf)) {
  throw "Qt deployment tool is missing: '$deploy'."
}
Get-Command dumpbin -ErrorAction Stop | Out-Null
$msvcCrt = Find-MsvcCrt

foreach ($name in @('yolo_defect_qt.exe', 'yolo_defect_cpp.exe')) {
  Copy-PackageFile (Join-Path $BuildDir "bin\$name") $name
}
& $deploy --release --no-compiler-runtime --no-translations --no-opengl-sw `
  --dir $PackageDir (Join-Path $PackageDir 'yolo_defect_qt.exe')
if ($LASTEXITCODE -ne 0) { throw "windeployqt failed with exit code $LASTEXITCODE." }
# windeployqt can supply only vc_redist.x64.exe on current Visual Studio
# installations. Deploy the selected x64 CRT app-locally; never run an installer.
foreach ($runtime in (Get-ChildItem -LiteralPath $msvcCrt.Directory -File -Filter '*.dll')) {
  Copy-PackageFile $runtime.FullName $runtime.Name
}
Write-Host "[Qt] MSVC tools $($msvcCrt.ToolsVersion); app-local CRT $($msvcCrt.RuntimeVersion)"

# Qt deploys its own graph. Copy the application's actual OpenCV/ORT imports,
# recursively, rather than shipping an entire SDK or hard-coding OpenCV 4.x.
$pending = New-Object 'Collections.Generic.Queue[string]'
$seen = New-Object 'Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
$pending.Enqueue((Join-Path $PackageDir 'yolo_defect_qt.exe'))
$pending.Enqueue((Join-Path $PackageDir 'yolo_defect_cpp.exe'))
while ($pending.Count -gt 0) {
  $binary = $pending.Dequeue()
  if (-not $seen.Add($binary)) { continue }
  $imports = & dumpbin /nologo /dependents $binary
  if ($LASTEXITCODE -ne 0) { throw "Cannot inspect dependencies of '$binary'." }
  foreach ($line in $imports) {
    if ($line -notmatch '^\s+([A-Za-z0-9_.-]+\.dll)\s*$') { continue }
    $dll = $Matches[1]
    if (Test-Path -LiteralPath (Join-Path $PackageDir $dll) -PathType Leaf) { continue }
    foreach ($searchDirectory in @($OpenCvBin, (Join-Path $OrtRoot 'lib'))) {
      $dependency = Join-Path $searchDirectory $dll
      if (Test-Path -LiteralPath $dependency -PathType Leaf) {
        Copy-PackageFile $dependency $dll
        $pending.Enqueue((Join-Path $PackageDir $dll))
        break
      }
    }
  }
}
foreach ($required in @('Qt6Core.dll', 'Qt6Gui.dll', 'Qt6Widgets.dll',
    'platforms/qwindows.dll', 'onnxruntime.dll', 'msvcp140.dll', 'vcruntime140.dll',
    'vcruntime140_1.dll')) {
  if (-not (Test-Path -LiteralPath (Join-Path $PackageDir $required) -PathType Leaf)) {
    throw "Deployment is incomplete: '$required' was not collected. Check the Qt/MSVC installation."
  }
}

$variants = @(@{ Config = 'default_config.txt'; Artifact = 'yolov8_neu_det.artifact.txt'; Model = 'best.onnx' })
if (Test-Path -LiteralPath (Join-Path $RepoRoot 'models/best.int8.qdq.u8s8.onnx') -PathType Leaf) {
  $variants += @{ Config = 'int8_u8s8_config.txt'; Artifact = 'yolov8_neu_det_int8_qdq_u8s8.artifact.txt'; Model = 'best.int8.qdq.u8s8.onnx' }
}
foreach ($variant in $variants) {
  Copy-PackageFile (Join-Path $RepoRoot "cpp_infer/configs/$($variant.Config)") "configs/$($variant.Config)"
  $artifact = Get-Content -LiteralPath (Join-Path $RepoRoot "cpp_infer/artifacts/$($variant.Artifact)") -Raw
  $artifact = $artifact -replace '(?m)^model_path\s*=.*$', "model_path = ../models/$($variant.Model)"
  Write-PackageText "artifacts/$($variant.Artifact)" $artifact
  Copy-PackageFile (Join-Path $RepoRoot "models/$($variant.Model)") "models/$($variant.Model)"
}
$samples = @('crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches') |
  ForEach-Object { "${_}_241.jpg" }
foreach ($sample in $samples) {
  Copy-PackageFile (Join-Path $RepoRoot "data/images/val/$sample") "samples/$sample"
}
Write-PackageText 'samples/manifest.txt' (($samples -join "`n") + "`n")
New-Item -ItemType Directory -Path (Join-Path $PackageDir 'outputs') -Force | Out-Null
Copy-PackageFile (Join-Path $PSScriptRoot 'qt_package/run.cmd') 'run.cmd'
Copy-PackageFile (Join-Path $PSScriptRoot 'qt_package/verify.ps1') 'verify.ps1'
Copy-PackageFile (Join-Path $RepoRoot 'docs/demo/index.html') 'demo/index.html'
Copy-PackageFile (Join-Path $RepoRoot 'docs/demo/index.en.html') 'demo/index.en.html'
foreach ($file in (Get-ChildItem -LiteralPath (Join-Path $RepoRoot 'docs/assets/qt') -File |
    Where-Object { $_.Extension -in @('.png', '.gif', '.webp', '.mp4') } | Sort-Object Name)) {
  Copy-PackageFile $file.FullName "assets/qt/$($file.Name)"
}
foreach ($chart in $requiredCharts) {
  Copy-PackageFile (Join-Path $RepoRoot "docs/assets/engineering/$chart") "assets/engineering/$chart"
}
Copy-PackageFile (Join-Path $RepoRoot 'LICENSE') 'LICENSE'

# Preserve notices that are actually present in the supplied SDKs. Model
# provenance and license declarations stay with their artifact specs.
$notices = @(
  @{ Source = (Join-Path $OrtRoot 'LICENSE'); Target = 'licenses/onnxruntime-LICENSE'; Origin = 'ONNX Runtime distribution: LICENSE' },
  @{ Source = (Join-Path $OrtRoot 'ThirdPartyNotices.txt'); Target = 'licenses/onnxruntime-ThirdPartyNotices.txt'; Origin = 'ONNX Runtime distribution: ThirdPartyNotices.txt' }
)
$opencvRoot = [IO.Path]::GetFullPath((Join-Path $OpenCvBin '../../../..'))
$qtInstallation = Split-Path -Parent (Split-Path -Parent $QtRoot)
$notices += @(
  @{ Source = (Join-Path $opencvRoot 'LICENSE.txt'); Target = 'licenses/opencv-LICENSE.txt'; Origin = 'OpenCV distribution: LICENSE.txt' },
  @{ Source = (Join-Path $opencvRoot 'sources/doc/LICENSE_CHANGE_NOTICE.txt'); Target = 'licenses/opencv-LICENSE_CHANGE_NOTICE.txt'; Origin = 'OpenCV distribution: sources/doc/LICENSE_CHANGE_NOTICE.txt' },
  @{ Source = (Join-Path $opencvRoot 'sources/doc/LICENSE_BSD.txt'); Target = 'licenses/opencv-LICENSE_BSD.txt'; Origin = 'OpenCV distribution: sources/doc/LICENSE_BSD.txt' },
  @{ Source = (Join-Path $qtInstallation 'Licenses/LICENSE'); Target = 'licenses/qt-LICENSE'; Origin = 'Qt distribution: Licenses/LICENSE' }
)
$copiedNotices = @()
foreach ($notice in $notices) {
  if (Test-Path -LiteralPath $notice.Source -PathType Leaf) {
    Copy-PackageFile $notice.Source $notice.Target
    $copiedNotices += "$($notice.Target) <- $($notice.Origin)"
  }
}
Write-PackageText 'THIRD_PARTY_NOTICES.txt' @"
YOLO Defect Windows demonstration package

Project source: https://github.com/LiuSiChengGitHub/yolo_defect
Project code license: LICENSE
Qt runtime: https://www.qt.io/ (notices copied from the installed Qt SDK)
OpenCV runtime: https://opencv.org/ (notices copied from the installed OpenCV SDK)
ONNX Runtime: https://onnxruntime.ai/ (license and third-party notices from the SDK)
Microsoft C++ runtime: app-local DLLs from the active Visual Studio x64 CRT redistributable directory.
MSVC toolset: $($msvcCrt.ToolsVersion); CRT file version: $($msvcCrt.RuntimeVersion).
Model source/provenance/license: artifacts/*.artifact.txt, kept from the repository.
Samples: six repository-tracked NEU-DET validation images; dataset terms are separate.
Copied SDK notice files:
$($copiedNotices -join "`n")

These notices describe the included inputs; they do not replace their terms.
See demo/index.html (Chinese) or demo/index.en.html (English) for the showcase and usage.
"@
$manifest = [ordered]@{
  format_version = 1
  created_utc = [DateTime]::UtcNow.ToString('o')
  platform = 'Windows x64'
  configuration = 'Release'
  qt_version = (Get-Item -LiteralPath (Join-Path $PackageDir 'Qt6Core.dll')).VersionInfo.ProductVersion
  msvc_tools_version = $msvcCrt.ToolsVersion
  msvc_crt_version = $msvcCrt.RuntimeVersion
  models = @($variants | ForEach-Object { "models/$($_.Model)" })
  samples = @($samples | ForEach-Object { "samples/$_" })
  entrypoint = 'run.cmd'
  guide = 'demo/index.html'
  guides = [ordered]@{ zh = 'demo/index.html'; en = 'demo/index.en.html' }
  verification = 'powershell -NoProfile -ExecutionPolicy Bypass -File .\verify.ps1'
}
Write-PackageText 'package-info.json' (($manifest | ConvertTo-Json -Depth 4) + "`n")
Write-Host "[Qt] Demo package: $PackageDir"
Write-Host '[Qt] Launch run.cmd; open demo/index.html for the guide.'
Write-Host '[Qt] Verify without a development PATH: powershell -NoProfile -ExecutionPolicy Bypass -File <package>\verify.ps1'
