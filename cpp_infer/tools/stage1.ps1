[CmdletBinding()]
param(
  [Parameter(Position = 0)]
  [string]$Action = 'help',

  [Parameter(Position = 1)]
  [string]$Image = '',

  [Parameter(Position = 2)]
  [string]$OutputDir = '',

  [string]$BuildDir = '',
  [string]$WorkflowFile = '',
  [string]$EnvironmentFile = '',
  [string]$Config = '',
  [string]$OrtRoot = '',
  [string]$OpenCvDir = '',
  [string]$OpenCvBin = '',
  [string]$PythonExe = '',
  [string]$GTestSource = '',

  [int]$Warmup = -1,

  [int]$Repeat = -1,

  [string]$ProfilePrefix = '',

  [int]$ProfileRuns = -1,

  [int]$Workers = -1,

  [int]$QueueCapacity = -1,

  [switch]$Overwrite,
  [switch]$OutputImages,
  [switch]$AllowGTestDownload
)

Set-StrictMode -Version 3.0
$ErrorActionPreference = 'Stop'

function Throw-ActionableError {
  param(
    [string]$Object,
    [string]$Expected,
    [string]$Actual,
    [string]$ActionText
  )

  throw "Stage-1 acceptance failed: object=$Object; expected=$Expected; actual=$Actual; action=$ActionText"
}

function Write-Stage {
  param([string]$Message)
  Write-Host ''
  Write-Host "[Stage-1] $Message" -ForegroundColor Cyan
}

function Show-Usage {
  Write-Host @'
YOLO Defect Stage-1 engineering workflow

Usage:
  stage1.cmd help
  stage1.cmd doctor
  stage1.cmd build
  stage1.cmd clean-build
  stage1.cmd test
  stage1.cmd detect <image> [output-directory] [-Config <path>] [-Overwrite]
  stage1.cmd batch <input-dir-or-manifest> [output-directory]
                   [-Config <path>] [-Workers <1..64>]
                   [-QueueCapacity <1..4096>] [-OutputImages] [-Overwrite]
  stage1.cmd batch-compare
  stage1.cmd demo
  stage1.cmd consistency
  stage1.cmd benchmark [-Warmup <n>] [-Repeat <n>]
  stage1.cmd profile [image] [-Config <path>] [-ProfilePrefix <path>]
                     [-ProfileRuns <n>]
  stage1.cmd all

Actions:
  help         Show this help without requiring Visual Studio or SDKs.
  doctor       Validate x64 MSVC, CMake/CTest, ORT SDK, OpenCV, Python,
               GoogleTest policy, workflow config, and resolved defaults.
  build        Incrementally build, or configure first when no build exists.
  clean-build  Recreate the guarded TEMP NMake Release build from scratch.
  test         Build current sources and run the complete CTest gate.
  detect       Run one arbitrary image through preprocess -> ORT ->
               postprocess -> JSON/PNG. The output directory is optional.
  batch        Run a directory or UTF-8 path-list manifest with bounded
               workers; JSON is always written per successful image.
  batch-compare Run workers=1 and workers=4 as separate Release processes,
               queue=8, FP32 CPU and JSON-only over data\images\val. Require
               identical detections and describe throughput/PWS deltas.
  demo         Build and validate the fixed three-detection Demo.
  consistency  Build and run the frozen 30-image Python/C++ comparison.
  benchmark    Build, rerun consistency, then run the configured benchmark.
  profile      Build, then run a separate profiling-enabled ORT session.
               Defaults: FP32 Runtime config, fixed crazing_241 sample,
               10 runs, and a fresh config-named trace prefix below TEMP.
  all          Clean build -> CTest -> Demo -> consistency -> benchmark ->
               frozen 30-image batch acceptance.

Configuration:
  tools\stage1.defaults.psd1    tracked machine-independent workflow defaults
  ..\.stage1.local.psd1         ignored machine paths and optional detect defaults
  configs\default_config.txt    Runtime thresholds/provider/artifact selection
  artifacts\*.artifact.txt      model identity and tensor contract

Detect examples:
  stage1.cmd detect "D:\images\sample.jpg"
  stage1.cmd detect "D:\images\sample.jpg" "D:\outputs"
  stage1.cmd detect "D:\images\sample.jpg" "D:\outputs" -Overwrite

Batch examples:
  stage1.cmd batch "D:\images" "D:\batch-output" -Workers 4 -QueueCapacity 8
  stage1.cmd batch cpp_infer\tests\fixtures\s2_03_consistency_manifest.txt -Workers 2
  stage1.cmd batch-compare

Profile examples:
  stage1.cmd profile
  stage1.cmd profile -Config cpp_infer\configs\int8_config.txt
  stage1.cmd profile -Config cpp_infer\configs\int8_config.txt -ProfilePrefix cpp_infer\results\s2_01\profile\int8_ort -ProfileRuns 10

Profile environment overrides (explicit parameters take precedence):
  YOLO_DEFECT_PROFILE_CONFIG, YOLO_DEFECT_PROFILE_IMAGE,
  YOLO_DEFECT_PROFILE_PREFIX, YOLO_DEFECT_PROFILE_RUNS

Profiling is diagnostic and includes instrumentation overhead. It is always
separate from the formal benchmark action and is not performance evidence.

Run stage1.cmd from an ordinary PowerShell or CMD. No manual VsDevCmd or
PowerShell switching is required.
'@
}

function Import-SafeSettingsFile {
  param(
    [string]$Path,
    [string]$ObjectName,
    [string]$RepairText
  )

  $tokens = $null
  $parseErrors = $null
  $ast = [System.Management.Automation.Language.Parser]::ParseFile(
      $Path, [ref]$tokens, [ref]$parseErrors)
  if ($parseErrors.Count -ne 0 -or
      $ast.EndBlock.Statements.Count -ne 1 -or
      $ast.EndBlock.Statements[0].PipelineElements.Count -ne 1) {
    Throw-ActionableError -Object $ObjectName `
      -Expected 'one literal PowerShell hashtable with no executable expressions' `
      -Actual $Path `
      -ActionText $RepairText
  }

  $expression = $ast.EndBlock.Statements[0].PipelineElements[0].Expression
  if (-not ($expression -is
      [System.Management.Automation.Language.HashtableAst])) {
    Throw-ActionableError -Object $ObjectName `
      -Expected 'one literal PowerShell hashtable' -Actual $Path `
      -ActionText $RepairText
  }

  try {
    return $expression.SafeGetValue()
  } catch {
    Throw-ActionableError -Object $ObjectName `
      -Expected 'literal keys and values without executable expressions' `
      -Actual $Path `
      -ActionText $RepairText
  }
}

function Get-ValueTypeName {
  param($Value)
  if ($null -eq $Value) {
    return '<null>'
  }
  return $Value.GetType().FullName
}

function Assert-SettingsObject {
  param(
    $Value,
    [string]$ObjectName,
    [string[]]$RequiredKeys,
    [string[]]$OptionalKeys = @()
  )

  if (-not ($Value -is [System.Collections.IDictionary])) {
    Throw-ActionableError -Object $ObjectName `
      -Expected 'a literal hashtable' -Actual (Get-ValueTypeName $Value) `
      -ActionText 'restore the documented key = value structure'
  }

  $allowedKeys = @($RequiredKeys) + @($OptionalKeys)
  $actualKeys = @($Value.Keys | ForEach-Object { [string]$_ })
  $missingKeys = @($RequiredKeys | Where-Object {
      -not $Value.Contains($_)
    })
  $unknownKeys = @($actualKeys | Where-Object { $_ -notin $allowedKeys })
  if ($missingKeys.Count -ne 0 -or $unknownKeys.Count -ne 0) {
    Throw-ActionableError -Object $ObjectName `
      -Expected ("required keys [{0}] and no unknown keys" -f
          ($RequiredKeys -join ', ')) `
      -Actual ("missing=[{0}], unknown=[{1}]" -f
          ($missingKeys -join ', '), ($unknownKeys -join ', ')) `
      -ActionText 'compare the file with tools\stage1.defaults.psd1 or tools\stage1.local.example.psd1'
  }
}

function Assert-NonEmptyStringSetting {
  param($Value, [string]$ObjectName)
  if (-not ($Value -is [string]) -or
      [string]::IsNullOrWhiteSpace([string]$Value)) {
    Throw-ActionableError -Object $ObjectName `
      -Expected 'a non-empty string' -Actual ([string]$Value) `
      -ActionText 'set the documented string value in the declaring settings file'
  }
}

function Assert-BooleanSetting {
  param($Value, [string]$ObjectName)
  if (-not ($Value -is [bool])) {
    Throw-ActionableError -Object $ObjectName `
      -Expected '$true or $false' -Actual ("{0} ({1})" -f
          ([string]$Value), (Get-ValueTypeName $Value)) `
      -ActionText 'use an unquoted PowerShell boolean literal'
  }
}

function Assert-WorkflowSettings {
  param($Settings)

  Assert-SettingsObject -Value $Settings -ObjectName 'workflow config root' `
    -RequiredKeys @('SchemaVersion', 'Build', 'Detect', 'Demo',
                    'Consistency', 'Benchmark')
  if (-not ($Settings.SchemaVersion -is [int]) -or
      $Settings.SchemaVersion -ne 1) {
    Throw-ActionableError -Object 'workflow.SchemaVersion' `
      -Expected 'integer 1' `
      -Actual ("{0} ({1})" -f ([string]$Settings.SchemaVersion),
          (Get-ValueTypeName $Settings.SchemaVersion)) `
      -ActionText 'use the current tools\stage1.defaults.psd1 schema'
  }

  Assert-SettingsObject -Value $Settings.Build -ObjectName 'workflow.Build' `
    -RequiredKeys @('Generator', 'Type', 'Testing',
                    'TemporaryDirectoryName')
  Assert-NonEmptyStringSetting $Settings.Build.Generator `
    'workflow.Build.Generator'
  Assert-NonEmptyStringSetting $Settings.Build.Type 'workflow.Build.Type'
  Assert-BooleanSetting $Settings.Build.Testing 'workflow.Build.Testing'
  Assert-NonEmptyStringSetting $Settings.Build.TemporaryDirectoryName `
    'workflow.Build.TemporaryDirectoryName'
  if ($Settings.Build.Generator -ne 'NMake Makefiles' -or
      $Settings.Build.Type -ne 'Release' -or
      -not $Settings.Build.Testing) {
    Throw-ActionableError -Object 'workflow.Build protocol' `
      -Expected 'Generator=NMake Makefiles, Type=Release, Testing=$true' `
      -Actual ("Generator={0}, Type={1}, Testing={2}" -f
          $Settings.Build.Generator, $Settings.Build.Type,
          $Settings.Build.Testing) `
      -ActionText 'keep the frozen Stage-1 evidence protocol; extend the script explicitly before adding another build profile'
  }
  if ($Settings.Build.TemporaryDirectoryName -notmatch
      '^yolo_defect_stage1_[A-Za-z0-9_.-]+$') {
    Throw-ActionableError -Object 'workflow.Build.TemporaryDirectoryName' `
      -Expected 'one safe leaf beginning with yolo_defect_stage1_' `
      -Actual ([string]$Settings.Build.TemporaryDirectoryName) `
      -ActionText 'remove directory separators and restore the protected TEMP naming prefix'
  }

  Assert-SettingsObject -Value $Settings.Detect -ObjectName 'workflow.Detect' `
    -RequiredKeys @('RuntimeConfig', 'OutputRoot', 'WriteJson', 'WriteImage')
  Assert-NonEmptyStringSetting $Settings.Detect.RuntimeConfig `
    'workflow.Detect.RuntimeConfig'
  Assert-NonEmptyStringSetting $Settings.Detect.OutputRoot `
    'workflow.Detect.OutputRoot'
  Assert-BooleanSetting $Settings.Detect.WriteJson `
    'workflow.Detect.WriteJson'
  Assert-BooleanSetting $Settings.Detect.WriteImage `
    'workflow.Detect.WriteImage'
  if (-not $Settings.Detect.WriteJson -and
      -not $Settings.Detect.WriteImage) {
    Throw-ActionableError -Object 'workflow.Detect outputs' `
      -Expected 'WriteJson or WriteImage to be $true' `
      -Actual 'both are $false' `
      -ActionText 'enable at least one durable single-image output'
  }

  Assert-SettingsObject -Value $Settings.Demo -ObjectName 'workflow.Demo' `
    -RequiredKeys @('Image')
  Assert-NonEmptyStringSetting $Settings.Demo.Image 'workflow.Demo.Image'
  Assert-SettingsObject -Value $Settings.Consistency `
    -ObjectName 'workflow.Consistency' -RequiredKeys @('Manifest')
  Assert-NonEmptyStringSetting $Settings.Consistency.Manifest `
    'workflow.Consistency.Manifest'
  Assert-SettingsObject -Value $Settings.Benchmark `
    -ObjectName 'workflow.Benchmark' -RequiredKeys @('Warmup', 'Repeat')
  if (-not ($Settings.Benchmark.Warmup -is [int]) -or
      $Settings.Benchmark.Warmup -lt 0 -or
      $Settings.Benchmark.Warmup -gt 100000) {
    Throw-ActionableError -Object 'workflow.Benchmark.Warmup' `
      -Expected 'an integer in [0,100000]' `
      -Actual ([string]$Settings.Benchmark.Warmup) `
      -ActionText 'restore a bounded warmup count'
  }
  if (-not ($Settings.Benchmark.Repeat -is [int]) -or
      $Settings.Benchmark.Repeat -lt 1 -or
      $Settings.Benchmark.Repeat -gt 100000) {
    Throw-ActionableError -Object 'workflow.Benchmark.Repeat' `
      -Expected 'an integer in [1,100000]' `
      -Actual ([string]$Settings.Benchmark.Repeat) `
      -ActionText 'restore a positive bounded repeat count'
  }
}

function Assert-LocalSettings {
  param($Settings)

  Assert-SettingsObject -Value $Settings -ObjectName 'local settings' `
    -RequiredKeys @() `
    -OptionalKeys @('OrtRoot', 'OpenCvDir', 'OpenCvBin', 'PythonExe',
                    'GTestSource', 'DefaultRuntimeConfig',
                    'DefaultDetectOutputRoot', 'DetectWriteJson',
                    'DetectWriteImage')
  foreach ($key in @('OrtRoot', 'OpenCvDir', 'OpenCvBin', 'PythonExe',
                     'GTestSource', 'DefaultRuntimeConfig',
                     'DefaultDetectOutputRoot')) {
    if ($Settings.Contains($key) -and
        -not ($Settings[$key] -is [string])) {
      Throw-ActionableError -Object "local.$key" `
        -Expected 'a string path' `
        -Actual (Get-ValueTypeName $Settings[$key]) `
        -ActionText 'use a quoted path or an empty string in the local settings file'
    }
  }
  foreach ($key in @('DetectWriteJson', 'DetectWriteImage')) {
    if ($Settings.Contains($key)) {
      Assert-BooleanSetting $Settings[$key] "local.$key"
    }
  }
}

function ConvertTo-AbsolutePath {
  param([string]$Value, [string]$BaseDirectory)
  if ([IO.Path]::IsPathRooted($Value)) {
    return [IO.Path]::GetFullPath($Value)
  }
  return [IO.Path]::GetFullPath((Join-Path $BaseDirectory $Value))
}

function Invoke-NativeStep {
  param(
    [string]$Name,
    [string]$FilePath,
    [string[]]$Arguments,
    [switch]$Quiet
  )

  Write-Host "[run] $Name"
  if ($Quiet) {
    & $FilePath @Arguments | Out-Null
  } else {
    & $FilePath @Arguments
  }
  $exitCode = $LASTEXITCODE
  if ($exitCode -ne 0) {
    Throw-ActionableError -Object $Name -Expected 'exit code 0' `
      -Actual "exit code $exitCode" `
      -ActionText 'read the first failing diagnostic above, correct it, and rerun the same action'
  }
}

function Resolve-RequiredDirectory {
  param([string]$Value, [string]$Object, [string]$ActionText)
  if ([string]::IsNullOrWhiteSpace($Value)) {
    Throw-ActionableError -Object $Object -Expected 'a configured directory path' `
      -Actual 'empty' -ActionText $ActionText
  }
  try {
    $resolved = (Resolve-Path -LiteralPath $Value -ErrorAction Stop).Path
  } catch {
    Throw-ActionableError -Object $Object -Expected 'an existing directory' `
      -Actual $Value -ActionText $ActionText
  }
  if (-not (Test-Path -LiteralPath $resolved -PathType Container)) {
    Throw-ActionableError -Object $Object -Expected 'an existing directory' `
      -Actual $resolved -ActionText $ActionText
  }
  return $resolved
}

function Resolve-RequiredFile {
  param([string]$Value, [string]$Object, [string]$ActionText)
  if ([string]::IsNullOrWhiteSpace($Value)) {
    Throw-ActionableError -Object $Object -Expected 'a configured file path' `
      -Actual 'empty' -ActionText $ActionText
  }
  try {
    $resolved = (Resolve-Path -LiteralPath $Value -ErrorAction Stop).Path
  } catch {
    Throw-ActionableError -Object $Object -Expected 'an existing regular file' `
      -Actual $Value -ActionText $ActionText
  }
  if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) {
    Throw-ActionableError -Object $Object -Expected 'an existing regular file' `
      -Actual $resolved -ActionText $ActionText
  }
  return $resolved
}

function Assert-RequiredFile {
  param([string]$Path, [string]$Object, [string]$ActionText)
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    Throw-ActionableError -Object $Object -Expected 'an existing regular file' `
      -Actual $Path -ActionText $ActionText
  }
}

function Get-PathSetting {
  param(
    [string]$ExplicitValue,
    [string]$Key,
    [string]$EnvironmentName,
    [string]$Fallback,
    [string]$FallbackBaseDirectory
  )

  if (-not [string]::IsNullOrWhiteSpace($ExplicitValue)) {
    $script:SettingSources[$Key] = 'command parameter'
    return ConvertTo-AbsolutePath $ExplicitValue $script:InvocationDirectory
  }
  if ($script:LocalSettings.ContainsKey($Key) -and
      -not [string]::IsNullOrWhiteSpace([string]$script:LocalSettings[$Key])) {
    $script:SettingSources[$Key] = 'local settings'
    return ConvertTo-AbsolutePath ([string]$script:LocalSettings[$Key]) `
      $script:LocalSettingsDirectory
  }
  $environmentValue = [Environment]::GetEnvironmentVariable(
      $EnvironmentName, [EnvironmentVariableTarget]::Process)
  if (-not [string]::IsNullOrWhiteSpace($environmentValue)) {
    $script:SettingSources[$Key] = "environment $EnvironmentName"
    return ConvertTo-AbsolutePath $environmentValue `
      $script:InvocationDirectory
  }
  if ([string]::IsNullOrWhiteSpace($Fallback)) {
    $script:SettingSources[$Key] = 'not configured'
    return ''
  }
  $script:SettingSources[$Key] = 'portable fallback'
  return ConvertTo-AbsolutePath $Fallback $FallbackBaseDirectory
}

function Get-DetectPathDefault {
  param(
    [string]$ExplicitValue,
    [string]$LocalKey,
    [string]$WorkflowValue
  )

  if (-not [string]::IsNullOrWhiteSpace($ExplicitValue)) {
    return ConvertTo-AbsolutePath $ExplicitValue $script:InvocationDirectory
  }
  if ($script:LocalSettings.ContainsKey($LocalKey) -and
      -not [string]::IsNullOrWhiteSpace(
          [string]$script:LocalSettings[$LocalKey])) {
    return ConvertTo-AbsolutePath `
      ([string]$script:LocalSettings[$LocalKey]) `
      $script:LocalSettingsDirectory
  }
  return ConvertTo-AbsolutePath $WorkflowValue `
    $script:WorkflowSettingsDirectory
}

function Get-DetectBooleanDefault {
  param([string]$LocalKey, [bool]$WorkflowValue)
  if ($script:LocalSettings.ContainsKey($LocalKey)) {
    return [bool]$script:LocalSettings[$LocalKey]
  }
  return $WorkflowValue
}

function Assert-SafeBuildDirectory {
  param([string]$Path)

  $tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\', '/')
  $target = [IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
  $tempPrefix = $tempRoot + [IO.Path]::DirectorySeparatorChar
  $leaf = Split-Path -Leaf $target

  if (-not $target.StartsWith(
      $tempPrefix, [StringComparison]::OrdinalIgnoreCase) -or
      -not $leaf.StartsWith(
      'yolo_defect_stage1_', [StringComparison]::OrdinalIgnoreCase)) {
    Throw-ActionableError -Object 'build_dir clean boundary' `
      -Expected "a directory named yolo_defect_stage1_* below $tempRoot" `
      -Actual $target `
      -ActionText 'use the default build directory or choose a dedicated Stage-1 path below TEMP'
  }
  if (Test-Path -LiteralPath $target) {
    $targetItem = Get-Item -LiteralPath $target -Force
    if (($targetItem.Attributes -band
        [IO.FileAttributes]::ReparsePoint) -ne 0) {
      Throw-ActionableError -Object 'build_dir clean boundary' `
        -Expected 'a real temporary directory rather than a junction or symbolic link' `
        -Actual $target `
        -ActionText 'remove the reparse point manually after inspecting its target, then use the default real TEMP directory'
    }
  }
  return $target
}

function Assert-GTestConfigurePolicy {
  if (-not [string]::IsNullOrWhiteSpace($script:ResolvedGTestSource)) {
    Assert-RequiredFile -Path (Join-Path $script:ResolvedGTestSource `
        'CMakeLists.txt') -Object 'GoogleTest CMakeLists.txt' `
      -ActionText 'point GTestSource at the root of the verified v1.17.0 source tree'
    return
  }
  if (-not $AllowGTestDownload) {
    Throw-ActionableError -Object 'GoogleTest source' `
      -Expected 'a separately verified v1.17.0 source directory' `
      -Actual 'not configured' `
      -ActionText 'set GTestSource in cpp_infer/.stage1.local.psd1, or explicitly pass -AllowGTestDownload to let pinned CMake FetchContent use the network'
  }
}

function Invoke-CleanBuild {
  Write-Stage 'clean Release configure and build'
  Assert-GTestConfigurePolicy
  if (Test-Path -LiteralPath $script:ResolvedBuildDir) {
    Write-Host "[clean] $script:ResolvedBuildDir"
    Remove-Item -LiteralPath $script:ResolvedBuildDir -Recurse -Force
  }

  $configureArguments = @(
    '-S', $script:CppInferDir,
    '-B', $script:ResolvedBuildDir,
    '-G', ([string]$script:WorkflowSettings.Build.Generator),
    "-DOpenCV_DIR=$script:ResolvedOpenCvDir",
    "-DONNXRUNTIME_ROOT=$script:ResolvedOrtRoot",
    "-DPython3_EXECUTABLE=$script:ResolvedPythonExe",
    ("-DCMAKE_BUILD_TYPE={0}" -f $script:WorkflowSettings.Build.Type),
    ("-DBUILD_TESTING={0}" -f $(
        if ($script:WorkflowSettings.Build.Testing) { 'ON' } else { 'OFF' }))
  )

  if (-not [string]::IsNullOrWhiteSpace($script:ResolvedGTestSource)) {
    $configureArguments +=
        "-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=$script:ResolvedGTestSource"
  }

  Invoke-NativeStep -Name 'CMake configure' -FilePath $script:CMakeExe `
    -Arguments $configureArguments
  Invoke-NativeStep -Name 'Release build' -FilePath $script:CMakeExe `
    -Arguments @('--build', $script:ResolvedBuildDir)
  Assert-BuildOutputs
}

function Assert-BuildOutputs {
  Assert-RequiredFile -Path (Join-Path $script:ResolvedBuildDir 'CMakeCache.txt') `
    -Object 'CMakeCache.txt' -ActionText 'run stage1.cmd clean-build'
  Assert-RequiredFile -Path $script:CliPath -Object 'yolo_defect_cpp.exe' `
    -ActionText 'inspect the Release build failure and rebuild'
  Assert-RequiredFile -Path $script:ImageProbePath `
    -Object 'yolo_defect_image_probe.exe' `
    -ActionText 'configure with BUILD_TESTING=ON and rebuild'
}

function Invoke-IncrementalBuild {
  Assert-RequiredFile -Path (Join-Path $script:ResolvedBuildDir 'CMakeCache.txt') `
    -Object 'existing Stage-1 build' `
    -ActionText 'run stage1.cmd clean-build before this action'
  Write-Stage 'build current source changes'
  Invoke-NativeStep -Name 'incremental Release build' -FilePath $script:CMakeExe `
    -Arguments @('--build', $script:ResolvedBuildDir)
  Assert-BuildOutputs
}

function Invoke-EnsureBuild {
  if (Test-Path -LiteralPath (Join-Path $script:ResolvedBuildDir `
      'CMakeCache.txt') -PathType Leaf) {
    Invoke-IncrementalBuild
  } else {
    Write-Host '[build] no configured tree found; configuring it now'
    Invoke-CleanBuild
  }
}

function Invoke-Doctor {
  Write-Stage 'read-only environment and workflow doctor'
  $ortVersion = (Get-Content -LiteralPath (Join-Path `
      $script:ResolvedOrtRoot 'VERSION_NUMBER') -Raw).Trim()
  if ($ortVersion -ne '1.19.2') {
    Throw-ActionableError -Object 'ORT SDK version' -Expected '1.19.2' `
      -Actual $ortVersion `
      -ActionText 'select the pinned official Windows x64 CPU SDK'
  }

  Write-Host "[pass] workflow schema: $($script:WorkflowFilePath) (v1)"
  Write-Host "[pass] x64 MSVC:       $((Get-Command cl.exe).Source)"
  Write-Host "[pass] NMake:          $((Get-Command nmake.exe).Source)"
  Write-Host "[pass] CMake:          $script:CMakeExe"
  Write-Host "[pass] CTest:          $script:CTestExe"
  Write-Host "[pass] ORT C++ SDK:    $script:ResolvedOrtRoot ($ortVersion)"
  Write-Host "[pass] OpenCV C++:     $script:ResolvedOpenCvDir ($script:CppOpenCvVersion)"
  Write-Host "[pass] Python:         $script:ResolvedPythonExe"
  if (-not [string]::IsNullOrWhiteSpace($script:ResolvedGTestSource)) {
    Assert-RequiredFile -Path (Join-Path $script:ResolvedGTestSource `
        'CMakeLists.txt') -Object 'GoogleTest CMakeLists.txt' `
      -ActionText 'point GTestSource at the root of the verified v1.17.0 source tree'
    Write-Host "[pass] GoogleTest:     $script:ResolvedGTestSource"
  } elseif ($AllowGTestDownload) {
    Write-Host '[warn] GoogleTest:     explicit pinned network download allowed'
  } else {
    Write-Host '[warn] GoogleTest:     not configured; a new configure will stop'
  }
  Write-Host "[pass] Runtime config: $script:DetectConfigPath"
  Write-Host "[pass] default output: $script:DetectOutputRoot"
  Write-Host ("[pass] detect outputs: JSON={0}, image={1}" -f
      $script:DetectWriteJson, $script:DetectWriteImage)
  Write-Host ("[pass] benchmark:      warmup={0}, repeat={1}" -f
      $script:ResolvedWarmup, $script:ResolvedRepeat)
  Write-Host '[pass] doctor is read-only; no build or evidence was created'
}

function Invoke-FullTests {
  Write-Stage 'complete CTest gate'
  Invoke-NativeStep -Name 'CTest inventory' -FilePath $script:CTestExe `
    -Arguments @('--test-dir', $script:ResolvedBuildDir, '-N')
  Invoke-NativeStep -Name 'complete CTest' -FilePath $script:CTestExe `
    -Arguments @('--test-dir', $script:ResolvedBuildDir,
                 '--output-on-failure')
}

function Invoke-Demo {
  Write-Stage 'fixed single-image Demo'
  $demoJson = Join-Path $script:RunDir 'demo\crazing_241.json'
  $demoImage = Join-Path $script:RunDir 'demo\crazing_241.png'

  Invoke-NativeStep -Name 'single-image CLI Demo' -FilePath $script:CliPath `
    -Arguments @('--config', $script:ConfigPath,
                 '--image', $script:ImagePath,
                 '--output-json', $demoJson,
                 '--output-image', $demoImage)
  Assert-RequiredFile -Path $demoJson -Object 'Demo JSON' `
    -ActionText 'inspect output path creation and JSON serialization'
  Assert-RequiredFile -Path $demoImage -Object 'Demo visualization' `
    -ActionText 'inspect OpenCV image encoding and output path creation'

  Invoke-NativeStep -Name 'Demo JSON parse' -FilePath $script:ResolvedPythonExe `
    -Arguments @('-m', 'json.tool', $demoJson) -Quiet
  Invoke-NativeStep -Name 'Demo JSON contract validation' `
    -FilePath $script:ResolvedPythonExe `
    -Arguments @($script:DetectionValidator, $demoJson,
                 '--expected-image', $script:ImagePath)
  Invoke-NativeStep -Name 'Demo PNG OpenCV probe' `
    -FilePath $script:ImageProbePath -Arguments @($demoImage)

  $demo = Get-Content -LiteralPath $demoJson -Raw | ConvertFrom-Json
  if ($demo.detections.Count -ne 3) {
    Throw-ActionableError -Object 'fixed Demo detection count' `
      -Expected '3' -Actual ([string]$demo.detections.Count) `
      -ActionText 'inspect the model/config identity, preprocess, raw output, and postprocess contract'
  }
  Write-Host "[pass] Demo: 3 detections; JSON=$demoJson; PNG=$demoImage"
}

function Invoke-Detect {
  Write-Stage 'arbitrary single-image full pipeline'
  $baseName = [IO.Path]::GetFileNameWithoutExtension(
      $script:DetectImagePath)
  $jsonPath = Join-Path $script:DetectOutputDir `
    ($baseName + '.detections.json')
  $visualizationPath = Join-Path $script:DetectOutputDir `
    ($baseName + '.visualized.png')
  $arguments = @('--config', $script:DetectConfigPath,
                 '--image', $script:DetectImagePath)
  if ($script:DetectWriteJson) {
    $arguments += @('--output-json', $jsonPath)
  }
  if ($script:DetectWriteImage) {
    $arguments += @('--output-image', $visualizationPath)
  }
  if ($Overwrite) {
    $arguments += '--overwrite'
  }

  Write-Host "[detect] config: $script:DetectConfigPath"
  Write-Host "[detect] input:  $script:DetectImagePath"
  Write-Host "[detect] output: $script:DetectOutputDir"
  Invoke-NativeStep -Name 'single-image full pipeline' `
    -FilePath $script:CliPath -Arguments $arguments

  $detectionCount = 'not read (JSON disabled)'
  $actualProvider = 'not read (JSON disabled)'
  if ($script:DetectWriteJson) {
    Assert-RequiredFile -Path $jsonPath -Object 'single-image JSON' `
      -ActionText 'inspect the result writer and selected output directory'
    Invoke-NativeStep -Name 'single-image JSON parse' `
      -FilePath $script:ResolvedPythonExe `
      -Arguments @('-m', 'json.tool', $jsonPath) -Quiet
    $document = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json
    $detectionCount = [string]$document.detections.Count
    $actualProvider = [string]$document.runtime.actual_provider
  }
  if ($script:DetectWriteImage) {
    Assert-RequiredFile -Path $visualizationPath `
      -Object 'single-image visualization' `
      -ActionText 'inspect OpenCV encoding and the selected output directory'
    Invoke-NativeStep -Name 'single-image PNG OpenCV probe' `
      -FilePath $script:ImageProbePath -Arguments @($visualizationPath)
  }

  Write-Host ("[pass] Detect: detections={0}; actual_provider={1}" -f
      $detectionCount, $actualProvider)
  if ($script:DetectWriteJson) {
    Write-Host "JSON:  $jsonPath"
  }
  if ($script:DetectWriteImage) {
    Write-Host "Image: $visualizationPath"
  }
}

function Invoke-BatchRun {
  param(
    [string]$InputPath,
    [ValidateSet('directory', 'manifest')]
    [string]$InputKind,
    [string]$OutputPath,
    [string]$ConfigPath,
    [int]$WorkerCount,
    [int]$Capacity,
    [bool]$WriteImages,
    [bool]$AllowOverwrite
  )

  Write-Stage ("bounded multi-image batch (workers={0}, queue={1})" -f
      $WorkerCount, $Capacity)
  $summaryPath = Join-Path $OutputPath 'batch_summary.json'
  $inputOption = if ($InputKind -eq 'directory') {
    '--input-dir'
  } else {
    '--manifest'
  }
  $arguments = @('--config', $ConfigPath,
                 '--batch', $inputOption, $InputPath,
                 '--output-dir', $OutputPath,
                 '--batch-summary', $summaryPath,
                 '--workers', ([string]$WorkerCount),
                 '--queue-capacity', ([string]$Capacity))
  if ($WriteImages) {
    $arguments += '--output-images'
  }
  if ($AllowOverwrite) {
    $arguments += '--overwrite'
  }

  Invoke-NativeStep -Name 'bounded batch CLI' -FilePath $script:CliPath `
    -Arguments $arguments | Out-Host
  Assert-RequiredFile -Path $summaryPath -Object 'BatchSummary JSON' `
    -ActionText 'inspect deterministic discovery, worker processing, and summary serialization'
  Invoke-NativeStep -Name 'BatchSummary JSON parse' `
    -FilePath $script:ResolvedPythonExe `
    -Arguments @('-m', 'json.tool', $summaryPath) -Quiet
  Invoke-NativeStep -Name 'BatchSummary strict validation' `
    -FilePath $script:ResolvedPythonExe `
    -Arguments @($script:BatchSummaryValidator, $summaryPath,
                 '--expected-status', 'succeeded',
                 '--expected-input-kind', $InputKind,
                 '--expected-requested-workers', ([string]$WorkerCount)) |
      Out-Host
  Write-Host "[pass] Batch: summary=$summaryPath; output=$OutputPath"
  return $summaryPath
}

function Invoke-Batch {
  [void](Invoke-BatchRun -InputPath $script:BatchInputPath `
    -InputKind $script:BatchInputKind `
    -OutputPath $script:BatchOutputDir `
    -ConfigPath $script:BatchConfigPath `
    -WorkerCount $script:ResolvedWorkers `
    -Capacity $script:ResolvedQueueCapacity `
    -WriteImages ([bool]$OutputImages) `
    -AllowOverwrite ([bool]$Overwrite))
}

function Invoke-BatchComparison {
  Write-Stage 'formal S2-03 workers=1 versus workers=4 comparison'
  $workers1Output = Join-Path $script:RunDir 'batch_workers_1'
  $workers4Output = Join-Path $script:RunDir 'batch_workers_4'
  $workers1Summary = Invoke-BatchRun `
    -InputPath $script:BatchPerformanceInput `
    -InputKind 'directory' -OutputPath $workers1Output `
    -ConfigPath $script:ConfigPath -WorkerCount 1 -Capacity 8 `
    -WriteImages $false -AllowOverwrite $false
  $workers4Summary = Invoke-BatchRun `
    -InputPath $script:BatchPerformanceInput `
    -InputKind 'directory' -OutputPath $workers4Output `
    -ConfigPath $script:ConfigPath -WorkerCount 4 -Capacity 8 `
    -WriteImages $false -AllowOverwrite $false
  $comparisonPath = Join-Path $script:RunDir 'batch_comparison.json'
  Invoke-NativeStep -Name 'batch correctness/throughput/memory comparison' `
    -FilePath $script:ResolvedPythonExe `
    -Arguments @($script:BatchComparisonTool,
                 '--workers-1-summary', $workers1Summary,
                 '--workers-4-summary', $workers4Summary,
                 '--output', $comparisonPath)
  Assert-RequiredFile -Path $comparisonPath -Object 'batch comparison JSON' `
    -ActionText 'inspect the first comparability or per-image equality failure'
  Invoke-NativeStep -Name 'batch comparison JSON parse' `
    -FilePath $script:ResolvedPythonExe `
    -Arguments @('-m', 'json.tool', $comparisonPath) -Quiet
  Write-Host "[pass] Batch comparison: $comparisonPath"
}

function Invoke-Consistency {
  Write-Stage '30-image Python ORT versus C++ ORT consistency'
  $consistencyDir = Join-Path $script:RunDir 'consistency'
  if (Test-Path -LiteralPath $consistencyDir) {
    Throw-ActionableError -Object 'consistency output directory' `
      -Expected 'a fresh path' -Actual $consistencyDir `
      -ActionText 'use the new run directory printed by this script'
  }

  Invoke-NativeStep -Name '30-image consistency comparison' `
    -FilePath $script:ResolvedPythonExe `
    -Arguments @($script:ConsistencyTool,
                 '--manifest', $script:ManifestPath,
                 '--cpp-cli', $script:CliPath,
                 '--output-dir', $consistencyDir,
                 '--cpp-opencv-version', $script:CppOpenCvVersion)

  $perImageJson = Join-Path $consistencyDir 'per_image.json'
  $summaryJson = Join-Path $consistencyDir 'summary.json'
  Assert-RequiredFile -Path $perImageJson -Object 'consistency per_image.json' `
    -ActionText 'inspect the comparison diagnostic above'
  Assert-RequiredFile -Path $summaryJson -Object 'consistency summary.json' `
    -ActionText 'inspect the comparison diagnostic above'
  Invoke-NativeStep -Name 'per-image JSON parse' `
    -FilePath $script:ResolvedPythonExe `
    -Arguments @('-m', 'json.tool', $perImageJson) -Quiet
  Invoke-NativeStep -Name 'consistency summary JSON parse' `
    -FilePath $script:ResolvedPythonExe `
    -Arguments @('-m', 'json.tool', $summaryJson) -Quiet

  $summary = Get-Content -LiteralPath $summaryJson -Raw | ConvertFrom-Json
  $perImage = Get-Content -LiteralPath $perImageJson -Raw | ConvertFrom-Json
  if (-not $summary.passed -or
      $summary.result.images_total -ne 30 -or
      $summary.result.images_passed -ne 30 -or
      $summary.result.python_detections_total -ne 62 -or
      $summary.result.cpp_detections_total -ne 62 -or
      $summary.result.matched_detections_total -ne 62 -or
      $summary.result.max_confidence_abs_error -gt 1.0e-4 -or
      $summary.result.max_bbox_coordinate_abs_error_pixels -gt 1.0e-2 -or
      $summary.result.min_matching_iou -lt 0.999 -or
      $summary.source_class_results.Count -ne 6 -or
      @($summary.source_class_results | Where-Object {
        $_.images_total -ne 5 -or $_.images_passed -ne 5
      }).Count -ne 0 -or
      $perImage.images.Count -ne 30 -or
      @($perImage.images | Where-Object { -not $_.passed }).Count -ne 0) {
    Throw-ActionableError -Object 'frozen consistency gate' `
      -Expected 'passed=true, images=30/30, matched detections=62' `
      -Actual ("passed={0}, images={1}/{2}, matches={3}" -f
          $summary.passed, $summary.result.images_passed,
          $summary.result.images_total,
          $summary.result.matched_detections_total) `
      -ActionText 'read per_image.json and trace the first mismatch from image/hash through preprocess, raw output, postprocess, and IoU matching'
  }
  Write-Host "[pass] Consistency: 30/30 images, 62/62 matches; summary=$summaryJson"
}

function Invoke-Benchmark {
  Write-Stage ("Release benchmark (warmup={0}, repeat={1})" -f
      $script:ResolvedWarmup, $script:ResolvedRepeat)
  $benchmarkJson = Join-Path $script:RunDir `
    'benchmark\yolov8_neu_det_cpu_release.json'

  Invoke-NativeStep -Name 'formal C++ benchmark' -FilePath $script:CliPath `
    -Arguments @('--config', $script:ConfigPath,
                 '--image', $script:ImagePath,
                 '--benchmark',
                 '--warmup', ([string]$script:ResolvedWarmup),
                 '--repeat', ([string]$script:ResolvedRepeat),
                 '--benchmark-json', $benchmarkJson)
  Assert-RequiredFile -Path $benchmarkJson -Object 'benchmark JSON' `
    -ActionText 'inspect the benchmark diagnostic and output path'
  Invoke-NativeStep -Name 'benchmark JSON parse' `
    -FilePath $script:ResolvedPythonExe `
    -Arguments @('-m', 'json.tool', $benchmarkJson) -Quiet
  Invoke-NativeStep -Name 'benchmark JSON strict validation' `
    -FilePath $script:ResolvedPythonExe `
    -Arguments @($script:BenchmarkValidator, $benchmarkJson,
                 '--expected-image', $script:ImagePath,
                 '--expected-warmup', ([string]$script:ResolvedWarmup),
                 '--expected-repeat', ([string]$script:ResolvedRepeat))

  $benchmark = Get-Content -LiteralPath $benchmarkJson -Raw |
      ConvertFrom-Json
  Write-Host ("[pass] Benchmark: pipeline mean={0} ms, throughput={1} img/s; JSON={2}" -f
      $benchmark.latency_ms.pipeline.mean,
      $benchmark.throughput_images_per_second.pipeline,
      $benchmarkJson)
}

function Invoke-Profile {
  Write-Stage ("separate ORT profiling run (runs={0})" -f
      $script:ResolvedProfileRuns)
  Write-Host "[profile] config: $script:ProfileConfigPath"
  Write-Host "[profile] image:  $script:ProfileImagePath"
  Write-Host "[profile] prefix: $script:ResolvedProfilePrefix"
  Write-Host '[profile] benchmark separation: profiling overhead is diagnostic only'

  $arguments = @('--config', $script:ProfileConfigPath,
                 '--image', $script:ProfileImagePath,
                 '--profile',
                 '--profile-prefix', $script:ResolvedProfilePrefix,
                 '--profile-runs', ([string]$script:ResolvedProfileRuns))
  Write-Host '[run] profiling-enabled C++ CLI'
  $outputLines = @(& $script:CliPath @arguments 2>&1)
  $exitCode = $LASTEXITCODE
  $reportedTracePaths = @()
  $reportedRunCounts = @()
  $reportedProviders = @()
  foreach ($outputLine in $outputLines) {
    $lineText = [string]$outputLine
    Write-Host $lineText
    if ($lineText -match '^profile_trace_path:\s*(.+)$') {
      $reportedTracePaths += $Matches[1].Trim()
    }
    if ($lineText -match '^profile_runs:\s*([0-9]+)$') {
      $reportedRunCounts += [int]$Matches[1]
    }
    if ($lineText -match '^actual_provider:\s*(\S+)$') {
      $reportedProviders += $Matches[1]
    }
  }
  if ($exitCode -ne 0) {
    Throw-ActionableError -Object 'profiling-enabled C++ CLI' `
      -Expected 'exit code 0' -Actual "exit code $exitCode" `
      -ActionText 'read the first profiling diagnostic above; verify the config artifact/model, fixed image, prefix permissions, and run count'
  }
  if ($reportedTracePaths.Count -ne 1) {
    Throw-ActionableError -Object 'profile_trace_path CLI report' `
      -Expected 'exactly one non-empty profile_trace_path line' `
      -Actual ("count={0}" -f $reportedTracePaths.Count) `
      -ActionText 'verify that the Release CLI implements the S2-01 bounded profiling summary and rebuild cleanly'
  }
  if ($reportedRunCounts.Count -ne 1 -or
      $reportedRunCounts[0] -ne $script:ResolvedProfileRuns) {
    Throw-ActionableError -Object 'profile_runs CLI report' `
      -Expected ("exactly one value equal to {0}" -f
          $script:ResolvedProfileRuns) `
      -Actual ("values=[{0}]" -f ($reportedRunCounts -join ', ')) `
      -ActionText 'verify that the CLI executed the requested bounded profiling run count'
  }
  if ($reportedProviders.Count -ne 1 -or
      $reportedProviders[0] -ne 'CPUExecutionProvider') {
    Throw-ActionableError -Object 'profile actual_provider CLI report' `
      -Expected 'exactly one CPUExecutionProvider value' `
      -Actual ("values=[{0}]" -f ($reportedProviders -join ', ')) `
      -ActionText 'verify explicit CPU EP registration and the profiling session summary'
  }

  $reportedTracePath = [string]$reportedTracePaths[0]
  if (-not [IO.Path]::IsPathRooted($reportedTracePath)) {
    $reportedTracePath = ConvertTo-AbsolutePath $reportedTracePath `
      $script:RepoRoot
  }
  $script:ProfileTracePath = Resolve-RequiredFile `
    -Value $reportedTracePath -Object 'ORT profile trace' `
    -ActionText 'inspect EndProfiling and the selected profile-prefix directory'
  if (-not $script:ProfileTracePath.StartsWith(
      $script:ResolvedProfilePrefix,
      [StringComparison]::OrdinalIgnoreCase)) {
    Throw-ActionableError -Object 'ORT profile trace prefix' `
      -Expected "a trace beginning with $script:ResolvedProfilePrefix" `
      -Actual $script:ProfileTracePath `
      -ActionText 'inspect the ORT-returned trace path and profile prefix normalization'
  }
  $traceItem = Get-Item -LiteralPath $script:ProfileTracePath
  if ($traceItem.Length -le 0) {
    Throw-ActionableError -Object 'ORT profile trace size' `
      -Expected 'a non-empty JSON trace' -Actual '0 bytes' `
      -ActionText 'inspect ORT profiling finalization after the requested Session::Run calls'
  }
  Invoke-NativeStep -Name 'ORT profile trace JSON parse' `
    -FilePath $script:ResolvedPythonExe `
    -Arguments @('-m', 'json.tool', $script:ProfileTracePath) -Quiet

  Write-Host ("[pass] Profile: runs={0}; bytes={1}; trace={2}" -f
      $script:ResolvedProfileRuns, $traceItem.Length,
      $script:ProfileTracePath)
}

$script:InvocationDirectory = [IO.Path]::GetFullPath((Get-Location).Path)
$supportedActions = @('help', 'doctor', 'build', 'clean-build', 'test',
                      'detect', 'batch', 'batch-compare', 'demo',
                      'consistency', 'benchmark', 'profile', 'all')
if ($Action -notin $supportedActions) {
  Throw-ActionableError -Object 'workflow action' `
    -Expected ("one of [{0}]" -f ($supportedActions -join ', ')) `
    -Actual $Action -ActionText 'run stage1.cmd help and choose one action'
}
if ($Action -eq 'help') {
  Show-Usage
  exit 0
}

if ($Action -eq 'detect') {
  if ([string]::IsNullOrWhiteSpace($Image)) {
    Throw-ActionableError -Object 'detect image' `
      -Expected 'one source image path' -Actual 'missing' `
      -ActionText 'run stage1.cmd detect <image> [output-directory]'
  }
} elseif ($Action -eq 'batch') {
  if ([string]::IsNullOrWhiteSpace($Image)) {
    Throw-ActionableError -Object 'batch input' `
      -Expected 'one directory or UTF-8 manifest path' -Actual 'missing' `
      -ActionText 'run stage1.cmd batch <input-dir-or-manifest> [output-directory]'
  }
} elseif ($Action -eq 'profile') {
  if (-not [string]::IsNullOrWhiteSpace($OutputDir) -or $Overwrite) {
    Throw-ActionableError -Object 'profile arguments' `
      -Expected '-ProfilePrefix for the trace prefix and no -Overwrite' `
      -Actual 'a detect-only output/overwrite argument was supplied' `
      -ActionText 'remove the argument and use -ProfilePrefix for profiling output'
  }
} elseif (-not [string]::IsNullOrWhiteSpace($Image) -or
          -not [string]::IsNullOrWhiteSpace($OutputDir) -or
          -not [string]::IsNullOrWhiteSpace($Config) -or $Overwrite) {
  Throw-ActionableError -Object "$Action arguments" `
    -Expected 'input/config only with detect, batch, or profile; output/overwrite only with detect or batch' `
    -Actual 'an action-specific argument was supplied' `
    -ActionText 'remove the argument or use the detect/batch/profile action'
}
if (($PSBoundParameters.ContainsKey('Warmup') -or
     $PSBoundParameters.ContainsKey('Repeat')) -and
    $Action -ne 'benchmark') {
  Throw-ActionableError -Object "$Action benchmark arguments" `
    -Expected '-Warmup and -Repeat only with benchmark' `
    -Actual 'a benchmark-only argument was supplied' `
    -ActionText 'remove the argument or use the benchmark action'
}
if (($PSBoundParameters.ContainsKey('ProfilePrefix') -or
     $PSBoundParameters.ContainsKey('ProfileRuns')) -and
    $Action -ne 'profile') {
  Throw-ActionableError -Object "$Action profile arguments" `
    -Expected '-ProfilePrefix and -ProfileRuns only with profile' `
    -Actual 'a profile-only argument was supplied' `
    -ActionText 'remove the argument or use the profile action'
}
if (($PSBoundParameters.ContainsKey('Workers') -or
     $PSBoundParameters.ContainsKey('QueueCapacity') -or $OutputImages) -and
    $Action -ne 'batch') {
  Throw-ActionableError -Object "$Action batch arguments" `
    -Expected '-Workers, -QueueCapacity, and -OutputImages only with batch' `
    -Actual 'a batch-only argument was supplied' `
    -ActionText 'remove the argument or use the batch action; batch-compare uses the frozen 1/4 workers and queue=8 protocol'
}

$script:RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$script:CppInferDir = Join-Path $script:RepoRoot 'cpp_infer'
$defaultWorkflowFile = Join-Path $PSScriptRoot 'stage1.defaults.psd1'
if ([string]::IsNullOrWhiteSpace($WorkflowFile)) {
  $WorkflowFile = $defaultWorkflowFile
} else {
  $WorkflowFile = ConvertTo-AbsolutePath $WorkflowFile `
    $script:InvocationDirectory
}
$script:WorkflowFilePath = Resolve-RequiredFile -Value $WorkflowFile `
  -Object 'workflow config file' `
  -ActionText 'restore tools\stage1.defaults.psd1 or pass -WorkflowFile with a valid literal settings file'
$script:WorkflowSettingsDirectory = Split-Path -Parent `
  $script:WorkflowFilePath
$script:WorkflowSettings = Import-SafeSettingsFile `
  -Path $script:WorkflowFilePath -ObjectName 'workflow config file' `
  -RepairText 'compare it with tools\stage1.defaults.psd1 and use only literal hashtables'
Assert-WorkflowSettings $script:WorkflowSettings
if ($Action -eq 'all' -and
    ($script:WorkflowSettings.Benchmark.Warmup -ne 10 -or
     $script:WorkflowSettings.Benchmark.Repeat -ne 100)) {
  Throw-ActionableError -Object 'all formal benchmark protocol' `
    -Expected 'workflow Benchmark.Warmup=10 and Repeat=100' `
    -Actual ("Warmup={0}, Repeat={1}" -f
        $script:WorkflowSettings.Benchmark.Warmup,
        $script:WorkflowSettings.Benchmark.Repeat) `
    -ActionText 'restore the tracked formal protocol; use benchmark -Warmup/-Repeat for a non-formal exploratory run'
}

$defaultEnvironmentFile = Join-Path $script:CppInferDir '.stage1.local.psd1'
if ([string]::IsNullOrWhiteSpace($EnvironmentFile)) {
  $EnvironmentFile = $defaultEnvironmentFile
} else {
  $EnvironmentFile = ConvertTo-AbsolutePath $EnvironmentFile `
    $script:InvocationDirectory
}
$script:LocalSettingsDirectory = Split-Path -Parent $EnvironmentFile

$script:LocalSettings = @{}
if (Test-Path -LiteralPath $EnvironmentFile -PathType Leaf) {
  $script:LocalSettings = Import-SafeSettingsFile -Path $EnvironmentFile `
    -ObjectName 'local settings file' `
    -RepairText 'copy tools\stage1.local.example.psd1 and edit only literal values'
  Assert-LocalSettings $script:LocalSettings
  Write-Host "[env] local settings: $EnvironmentFile"
} elseif ($PSBoundParameters.ContainsKey('EnvironmentFile')) {
  Throw-ActionableError -Object 'local settings file' `
    -Expected 'an existing literal hashtable file' -Actual $EnvironmentFile `
    -ActionText 'correct -EnvironmentFile or omit it to use the standard optional local file'
} else {
  Write-Host "[env] local settings not found; using parameters/environment/portable discovery"
}

$baseRoot = Split-Path -Parent (Split-Path -Parent $script:RepoRoot)
$toolsRoot = Join-Path $baseRoot 'Tools'
$defaultOrtRoot = Join-Path $toolsRoot 'onnxruntime-win-x64-1.19.2'
$defaultOpenCvDir = Join-Path $toolsRoot 'opencv\build\x64\vc16\lib'
$defaultPythonExe = Join-Path `
  ([Environment]::GetFolderPath('UserProfile')) `
  '.conda\envs\TestBase\python.exe'

$script:SettingSources = @{}
$ortSetting = Get-PathSetting -ExplicitValue $OrtRoot -Key 'OrtRoot' `
  -EnvironmentName 'ONNXRUNTIME_ROOT' -Fallback $defaultOrtRoot `
  -FallbackBaseDirectory $script:InvocationDirectory
$openCvDirSetting = Get-PathSetting -ExplicitValue $OpenCvDir -Key 'OpenCvDir' `
  -EnvironmentName 'OpenCV_DIR' -Fallback $defaultOpenCvDir `
  -FallbackBaseDirectory $script:InvocationDirectory
$pythonSetting = Get-PathSetting -ExplicitValue $PythonExe -Key 'PythonExe' `
  -EnvironmentName 'YOLO_DEFECT_PYTHON' -Fallback $defaultPythonExe `
  -FallbackBaseDirectory $script:InvocationDirectory
$gtestSetting = Get-PathSetting -ExplicitValue $GTestSource -Key 'GTestSource' `
  -EnvironmentName 'YOLO_DEFECT_GTEST_SOURCE' -Fallback '' `
  -FallbackBaseDirectory $script:InvocationDirectory
$script:ResolvedOrtRoot = Resolve-RequiredDirectory -Value $ortSetting `
  -Object 'ONNXRUNTIME_ROOT' `
  -ActionText 'set OrtRoot in cpp_infer/.stage1.local.psd1 or ONNXRUNTIME_ROOT'
$script:ResolvedOpenCvDir = Resolve-RequiredDirectory `
  -Value $openCvDirSetting -Object 'OpenCV_DIR' `
  -ActionText 'set OpenCvDir in cpp_infer/.stage1.local.psd1 or OpenCV_DIR'
$script:ResolvedPythonExe = Resolve-RequiredFile -Value $pythonSetting `
  -Object 'PythonExe' `
  -ActionText 'set PythonExe to the verified TestBase interpreter in cpp_infer/.stage1.local.psd1'

$derivedOpenCvBin = Join-Path (Split-Path -Parent $script:ResolvedOpenCvDir) 'bin'
$openCvBinSetting = Get-PathSetting -ExplicitValue $OpenCvBin -Key 'OpenCvBin' `
  -EnvironmentName 'YOLO_DEFECT_OPENCV_BIN' -Fallback $derivedOpenCvBin `
  -FallbackBaseDirectory $script:InvocationDirectory
$script:ResolvedOpenCvBin = Resolve-RequiredDirectory `
  -Value $openCvBinSetting -Object 'OpenCV runtime bin' `
  -ActionText 'set OpenCvBin in cpp_infer/.stage1.local.psd1'

$script:ResolvedGTestSource = ''
if (-not [string]::IsNullOrWhiteSpace($gtestSetting)) {
  $script:ResolvedGTestSource = Resolve-RequiredDirectory `
    -Value $gtestSetting -Object 'verified GoogleTest source' `
    -ActionText 'correct GTestSource or remove it and explicitly pass -AllowGTestDownload'
}

Assert-RequiredFile -Path (Join-Path $script:ResolvedOrtRoot 'VERSION_NUMBER') `
  -Object 'ORT VERSION_NUMBER' -ActionText 'point OrtRoot at the complete official C++ SDK'
Assert-RequiredFile -Path (Join-Path $script:ResolvedOrtRoot 'include\onnxruntime_cxx_api.h') `
  -Object 'ORT C++ header' -ActionText 'do not use a Python wheel as the C++ SDK'
Assert-RequiredFile -Path (Join-Path $script:ResolvedOrtRoot 'lib\onnxruntime.lib') `
  -Object 'ORT import library' -ActionText 'point OrtRoot at the complete official Windows SDK'
Assert-RequiredFile -Path (Join-Path $script:ResolvedOrtRoot 'lib\onnxruntime.dll') `
  -Object 'ORT runtime DLL' -ActionText 'point OrtRoot at the matching official Windows SDK'
Assert-RequiredFile -Path (Join-Path $script:ResolvedOpenCvDir 'OpenCVConfig.cmake') `
  -Object 'OpenCVConfig.cmake' -ActionText 'point OpenCvDir at the x64 vc16 lib directory'
$openCvVersionFile = Join-Path $script:ResolvedOpenCvDir `
  'OpenCVConfig-version.cmake'
Assert-RequiredFile -Path $openCvVersionFile `
  -Object 'OpenCVConfig-version.cmake' `
  -ActionText 'point OpenCvDir at a complete OpenCV CMake package'
$openCvVersionLine = Get-Content -LiteralPath $openCvVersionFile |
    Where-Object { $_ -match '^set\(OpenCV_VERSION ([0-9]+\.[0-9]+\.[0-9]+)\)$' } |
    Select-Object -First 1
if ($null -eq $openCvVersionLine -or
    $openCvVersionLine -notmatch '^set\(OpenCV_VERSION ([0-9]+\.[0-9]+\.[0-9]+)\)$') {
  Throw-ActionableError -Object 'C++ OpenCV version' `
    -Expected 'a parseable set(OpenCV_VERSION x.y.z) declaration' `
    -Actual $openCvVersionFile `
    -ActionText 'verify the selected OpenCV CMake package'
}
$script:CppOpenCvVersion = $Matches[1]

$env:ONNXRUNTIME_ROOT = $script:ResolvedOrtRoot
$env:PATH = "$script:ResolvedOpenCvBin;$($script:ResolvedOrtRoot)\lib;$env:PATH"

if ($env:VSCMD_ARG_TGT_ARCH -ne 'x64') {
  Throw-ActionableError -Object 'MSVC target architecture' -Expected 'x64' `
    -Actual ([string]$env:VSCMD_ARG_TGT_ARCH) `
    -ActionText 'invoke stage1.cmd rather than stage1.ps1 directly'
}

foreach ($toolName in @('cl.exe', 'nmake.exe', 'cmake.exe', 'ctest.exe')) {
  if (-not (Get-Command $toolName -ErrorAction SilentlyContinue)) {
    Throw-ActionableError -Object $toolName -Expected 'available on PATH' `
      -Actual 'not found' `
      -ActionText 'invoke stage1.cmd so VsDevCmd and the Visual Studio CMake tools are inherited in the same process chain'
  }
}
$script:CMakeExe = (Get-Command 'cmake.exe').Source
$script:CTestExe = (Get-Command 'ctest.exe').Source

$pythonPreflight = @'
import cv2
import numpy
import onnxruntime as ort
assert tuple(int(x) for x in ort.__version__.split(chr(46))) == (1, 19, 2), ort.__version__
cpu_provider = bytes((67, 80, 85, 69, 120, 101, 99, 117, 116, 105, 111, 110, 80, 114, 111, 118, 105, 100, 101, 114)).decode()
assert cpu_provider in ort.get_available_providers(), ort.get_available_providers()
print(ort.__version__, cv2.__version__, numpy.__version__, cpu_provider)
'@
Invoke-NativeStep -Name 'Python consistency dependency preflight' `
  -FilePath $script:ResolvedPythonExe -Arguments @('-c', $pythonPreflight)

if ([string]::IsNullOrWhiteSpace($BuildDir)) {
  $BuildDir = Join-Path ([IO.Path]::GetTempPath()) `
    ([string]$script:WorkflowSettings.Build.TemporaryDirectoryName)
} else {
  $BuildDir = ConvertTo-AbsolutePath $BuildDir $script:InvocationDirectory
}
$script:ResolvedBuildDir = Assert-SafeBuildDirectory -Path $BuildDir
$script:CliPath = Join-Path $script:ResolvedBuildDir 'bin\yolo_defect_cpp.exe'
$script:ImageProbePath = Join-Path $script:ResolvedBuildDir `
  'bin\yolo_defect_image_probe.exe'
$workflowRuntimeConfig = ConvertTo-AbsolutePath `
  ([string]$script:WorkflowSettings.Detect.RuntimeConfig) `
  $script:WorkflowSettingsDirectory
$script:ConfigPath = Resolve-RequiredFile -Value $workflowRuntimeConfig `
  -Object 'workflow default Runtime config' `
  -ActionText 'correct Detect.RuntimeConfig relative to the workflow config file'
$workflowDemoImage = ConvertTo-AbsolutePath `
  ([string]$script:WorkflowSettings.Demo.Image) `
  $script:WorkflowSettingsDirectory
$script:ImagePath = Resolve-RequiredFile -Value $workflowDemoImage `
  -Object 'workflow Demo image' `
  -ActionText 'correct Demo.Image relative to the workflow config file'
$script:DetectionValidator = (Resolve-Path -LiteralPath `
  (Join-Path $script:CppInferDir 'tests\assert_detection_json.py')).Path
$workflowManifest = ConvertTo-AbsolutePath `
  ([string]$script:WorkflowSettings.Consistency.Manifest) `
  $script:WorkflowSettingsDirectory
$script:ManifestPath = Resolve-RequiredFile -Value $workflowManifest `
  -Object 'workflow consistency manifest' `
  -ActionText 'correct Consistency.Manifest relative to the workflow config file'
$script:ConsistencyTool = (Resolve-Path -LiteralPath `
  (Join-Path $script:CppInferDir 'tools\compare_consistency.py')).Path
$script:BenchmarkValidator = (Resolve-Path -LiteralPath `
  (Join-Path $script:CppInferDir 'tests\assert_benchmark_json.py')).Path
$script:BatchSummaryValidator = (Resolve-Path -LiteralPath `
  (Join-Path $script:CppInferDir 'tools\validate_batch_summary.py')).Path
$script:BatchComparisonTool = (Resolve-Path -LiteralPath `
  (Join-Path $script:CppInferDir 'tools\compare_batch_runs.py')).Path
$script:BatchManifestPath = (Resolve-Path -LiteralPath `
  (Join-Path $script:CppInferDir `
    'tests\fixtures\s2_03_consistency_manifest.txt')).Path
$script:BatchPerformanceInput = Resolve-RequiredDirectory `
  -Value (Join-Path $script:RepoRoot 'data\images\val') `
  -Object 'S2-03 performance image directory' `
  -ActionText 'restore data\images\val before running batch comparison'

$runtimeConfigExplicit = $Config
if ($Action -eq 'profile' -and
    [string]::IsNullOrWhiteSpace($runtimeConfigExplicit)) {
  $profileConfigEnvironment = [Environment]::GetEnvironmentVariable(
      'YOLO_DEFECT_PROFILE_CONFIG',
      [EnvironmentVariableTarget]::Process)
  if (-not [string]::IsNullOrWhiteSpace($profileConfigEnvironment)) {
    $runtimeConfigExplicit = $profileConfigEnvironment
  }
}
$script:DetectConfigPath = Get-DetectPathDefault `
  -ExplicitValue $runtimeConfigExplicit -LocalKey 'DefaultRuntimeConfig' `
  -WorkflowValue ([string]$script:WorkflowSettings.Detect.RuntimeConfig)
$runtimeConfigObject = 'detect Runtime config'
if ($Action -eq 'profile') {
  $runtimeConfigObject = 'profile Runtime config'
} elseif ($Action -eq 'batch') {
  $runtimeConfigObject = 'batch Runtime config'
}
$script:DetectConfigPath = Resolve-RequiredFile `
  -Value $script:DetectConfigPath -Object $runtimeConfigObject `
  -ActionText 'correct -Config/YOLO_DEFECT_PROFILE_CONFIG, local DefaultRuntimeConfig, or workflow Detect.RuntimeConfig'
$script:DetectOutputRoot = Get-DetectPathDefault -ExplicitValue '' `
  -LocalKey 'DefaultDetectOutputRoot' `
  -WorkflowValue ([string]$script:WorkflowSettings.Detect.OutputRoot)
$script:DetectWriteJson = Get-DetectBooleanDefault `
  -LocalKey 'DetectWriteJson' `
  -WorkflowValue ([bool]$script:WorkflowSettings.Detect.WriteJson)
$script:DetectWriteImage = Get-DetectBooleanDefault `
  -LocalKey 'DetectWriteImage' `
  -WorkflowValue ([bool]$script:WorkflowSettings.Detect.WriteImage)
if (-not $script:DetectWriteJson -and -not $script:DetectWriteImage) {
  Throw-ActionableError -Object 'effective detect outputs' `
    -Expected 'JSON or image output enabled' -Actual 'both disabled' `
    -ActionText 'enable DetectWriteJson or DetectWriteImage in local/workflow settings'
}

$script:ProfileConfigPath = ''
$script:ProfileImagePath = ''
$script:ResolvedProfileRuns = 10
$script:ResolvedProfilePrefix = ''
$script:ProfileTracePath = ''
if ($Action -eq 'profile') {
  $script:ProfileConfigPath = $script:DetectConfigPath
  $script:ProfileImagePath = $script:ImagePath
  $profileImageCandidate = ''
  if (-not [string]::IsNullOrWhiteSpace($Image)) {
    $profileImageCandidate = ConvertTo-AbsolutePath $Image `
      $script:InvocationDirectory
  } else {
    $profileImageEnvironment = [Environment]::GetEnvironmentVariable(
        'YOLO_DEFECT_PROFILE_IMAGE',
        [EnvironmentVariableTarget]::Process)
    if (-not [string]::IsNullOrWhiteSpace($profileImageEnvironment)) {
      $profileImageCandidate = ConvertTo-AbsolutePath `
        $profileImageEnvironment $script:InvocationDirectory
    }
  }
  if (-not [string]::IsNullOrWhiteSpace($profileImageCandidate)) {
    $script:ProfileImagePath = Resolve-RequiredFile `
      -Value $profileImageCandidate -Object 'profile fixed sample' `
      -ActionText 'pass one existing image, correct YOLO_DEFECT_PROFILE_IMAGE, or remove the override to use the frozen crazing_241 sample'
  }

  if ($PSBoundParameters.ContainsKey('ProfileRuns')) {
    $script:ResolvedProfileRuns = $ProfileRuns
  } else {
    $profileRunsEnvironment = [Environment]::GetEnvironmentVariable(
        'YOLO_DEFECT_PROFILE_RUNS',
        [EnvironmentVariableTarget]::Process)
    if (-not [string]::IsNullOrWhiteSpace($profileRunsEnvironment)) {
      $parsedProfileRuns = 0
      if (-not [int]::TryParse(
          $profileRunsEnvironment, [ref]$parsedProfileRuns)) {
        Throw-ActionableError -Object 'YOLO_DEFECT_PROFILE_RUNS' `
          -Expected 'an integer in [1,1000000]' `
          -Actual $profileRunsEnvironment `
          -ActionText 'set a bounded integer, pass -ProfileRuns, or remove the environment override to use 10'
      }
      $script:ResolvedProfileRuns = $parsedProfileRuns
    }
  }
  if ($script:ResolvedProfileRuns -lt 1 -or
      $script:ResolvedProfileRuns -gt 1000000) {
    Throw-ActionableError -Object 'profile runs' `
      -Expected 'an integer in [1,1000000]' `
      -Actual ([string]$script:ResolvedProfileRuns) `
      -ActionText 'use the frozen value 10 for formal profiling or choose a bounded exploratory count'
  }
}

$script:ResolvedWarmup = [int]$script:WorkflowSettings.Benchmark.Warmup
$script:ResolvedRepeat = [int]$script:WorkflowSettings.Benchmark.Repeat
if ($PSBoundParameters.ContainsKey('Warmup')) {
  $script:ResolvedWarmup = $Warmup
}
if ($PSBoundParameters.ContainsKey('Repeat')) {
  $script:ResolvedRepeat = $Repeat
}
if ($script:ResolvedWarmup -lt 0 -or $script:ResolvedWarmup -gt 100000) {
  Throw-ActionableError -Object 'benchmark warmup' `
    -Expected 'an integer in [0,100000]' `
    -Actual ([string]$script:ResolvedWarmup) `
    -ActionText 'correct -Warmup or workflow Benchmark.Warmup'
}
if ($script:ResolvedRepeat -lt 1 -or $script:ResolvedRepeat -gt 100000) {
  Throw-ActionableError -Object 'benchmark repeat' `
    -Expected 'an integer in [1,100000]' `
    -Actual ([string]$script:ResolvedRepeat) `
    -ActionText 'correct -Repeat or workflow Benchmark.Repeat'
}

$script:DetectImagePath = ''
$script:DetectOutputDir = ''
if ($Action -eq 'detect') {
  $detectImageCandidate = ConvertTo-AbsolutePath $Image `
    $script:InvocationDirectory
  $script:DetectImagePath = Resolve-RequiredFile `
    -Value $detectImageCandidate -Object 'detect source image' `
    -ActionText 'pass one existing OpenCV-decodable image file'

  if (-not [string]::IsNullOrWhiteSpace($OutputDir)) {
    $script:DetectOutputDir = ConvertTo-AbsolutePath $OutputDir `
      $script:InvocationDirectory
  } else {
    $safeStem = [regex]::Replace(
        [IO.Path]::GetFileNameWithoutExtension($script:DetectImagePath),
        '[^A-Za-z0-9._-]', '_')
    if ([string]::IsNullOrWhiteSpace($safeStem)) {
      $safeStem = 'image'
    }
    $detectRunId = (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' +
        [guid]::NewGuid().ToString('N').Substring(0, 8)
    $script:DetectOutputDir = Join-Path $script:DetectOutputRoot `
      ($detectRunId + '_' + $safeStem)
  }
  $script:DetectOutputDir = [IO.Path]::GetFullPath(
      $script:DetectOutputDir)
  if (Test-Path -LiteralPath $script:DetectOutputDir -PathType Leaf) {
    Throw-ActionableError -Object 'detect output directory' `
      -Expected 'a directory path or a path that does not yet exist' `
      -Actual "regular file '$script:DetectOutputDir'" `
      -ActionText 'choose a directory rather than a file'
  }
}

$script:BatchInputPath = ''
$script:BatchInputKind = ''
$script:BatchOutputDir = ''
$script:BatchConfigPath = $script:DetectConfigPath
$script:ResolvedWorkers = 1
$script:ResolvedQueueCapacity = 2
if ($Action -eq 'batch') {
  $batchInputCandidate = ConvertTo-AbsolutePath $Image `
    $script:InvocationDirectory
  if (Test-Path -LiteralPath $batchInputCandidate -PathType Container) {
    $script:BatchInputPath = (Resolve-Path -LiteralPath `
        $batchInputCandidate).Path
    $script:BatchInputKind = 'directory'
  } elseif (Test-Path -LiteralPath $batchInputCandidate -PathType Leaf) {
    $script:BatchInputPath = (Resolve-Path -LiteralPath `
        $batchInputCandidate).Path
    $script:BatchInputKind = 'manifest'
  } else {
    Throw-ActionableError -Object 'batch input' `
      -Expected 'an existing directory or UTF-8 path-list manifest' `
      -Actual $batchInputCandidate `
      -ActionText 'correct the path and pass a directory or manifest file'
  }

  if ($PSBoundParameters.ContainsKey('Workers')) {
    $script:ResolvedWorkers = $Workers
  }
  if ($script:ResolvedWorkers -lt 1 -or
      $script:ResolvedWorkers -gt 64) {
    Throw-ActionableError -Object 'batch workers' `
      -Expected 'an integer in [1,64]' `
      -Actual ([string]$script:ResolvedWorkers) `
      -ActionText 'correct -Workers'
  }
  $script:ResolvedQueueCapacity = 2 * $script:ResolvedWorkers
  if ($PSBoundParameters.ContainsKey('QueueCapacity')) {
    $script:ResolvedQueueCapacity = $QueueCapacity
  }
  if ($script:ResolvedQueueCapacity -lt 1 -or
      $script:ResolvedQueueCapacity -gt 4096) {
    Throw-ActionableError -Object 'batch queue capacity' `
      -Expected 'an integer in [1,4096]' `
      -Actual ([string]$script:ResolvedQueueCapacity) `
      -ActionText 'correct -QueueCapacity'
  }

  if (-not [string]::IsNullOrWhiteSpace($OutputDir)) {
    $script:BatchOutputDir = ConvertTo-AbsolutePath $OutputDir `
      $script:InvocationDirectory
  } else {
    $batchRunId = (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' +
        [guid]::NewGuid().ToString('N').Substring(0, 8)
    $script:BatchOutputDir = Join-Path $script:DetectOutputRoot `
      ($batchRunId + '_batch')
  }
  $script:BatchOutputDir = [IO.Path]::GetFullPath(
      $script:BatchOutputDir)
  if (Test-Path -LiteralPath $script:BatchOutputDir -PathType Leaf) {
    Throw-ActionableError -Object 'batch output directory' `
      -Expected 'a directory path or a path that does not yet exist' `
      -Actual "regular file '$script:BatchOutputDir'" `
      -ActionText 'choose a directory rather than a file'
  }
}

$needsEvidence = $Action -in @(
    'batch-compare', 'demo', 'consistency', 'benchmark', 'profile', 'all')
$script:RunDir = ''
if ($needsEvidence) {
  $runId = (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' +
      [guid]::NewGuid().ToString('N').Substring(0, 8)
  $script:RunDir = Join-Path $script:ResolvedBuildDir `
    (Join-Path 'stage1_evidence' $runId)
  New-Item -ItemType Directory -Path $script:RunDir -Force | Out-Null
}

if ($Action -eq 'profile') {
  $profilePrefixCandidate = ''
  if (-not [string]::IsNullOrWhiteSpace($ProfilePrefix)) {
    $profilePrefixCandidate = ConvertTo-AbsolutePath $ProfilePrefix `
      $script:InvocationDirectory
  } else {
    $profilePrefixEnvironment = [Environment]::GetEnvironmentVariable(
        'YOLO_DEFECT_PROFILE_PREFIX',
        [EnvironmentVariableTarget]::Process)
    if (-not [string]::IsNullOrWhiteSpace($profilePrefixEnvironment)) {
      $profilePrefixCandidate = ConvertTo-AbsolutePath `
        $profilePrefixEnvironment $script:InvocationDirectory
    }
  }
  if ([string]::IsNullOrWhiteSpace($profilePrefixCandidate)) {
    $configStem = [regex]::Replace(
        [IO.Path]::GetFileNameWithoutExtension($script:ProfileConfigPath),
        '[^A-Za-z0-9._-]', '_')
    if ([string]::IsNullOrWhiteSpace($configStem)) {
      $configStem = 'runtime'
    }
    $profilePrefixCandidate = Join-Path $script:RunDir `
      (Join-Path 'profile' ($configStem + '_ort'))
  }
  $script:ResolvedProfilePrefix = [IO.Path]::GetFullPath(
      $profilePrefixCandidate)
  $profilePrefixLeaf = Split-Path -Leaf $script:ResolvedProfilePrefix
  if ([string]::IsNullOrWhiteSpace($profilePrefixLeaf)) {
    Throw-ActionableError -Object 'profile prefix' `
      -Expected 'a file prefix with a non-empty final component' `
      -Actual $script:ResolvedProfilePrefix `
      -ActionText 'append a filename prefix such as fp32_ort or int8_ort'
  }
  if (Test-Path -LiteralPath $script:ResolvedProfilePrefix) {
    Throw-ActionableError -Object 'profile prefix' `
      -Expected 'a new file prefix rather than an existing file or directory' `
      -Actual $script:ResolvedProfilePrefix `
      -ActionText 'choose a fresh prefix; ORT appends its timestamped JSON filename'
  }
  $profilePrefixParent = Split-Path -Parent `
    $script:ResolvedProfilePrefix
  if (Test-Path -LiteralPath $profilePrefixParent -PathType Leaf) {
    Throw-ActionableError -Object 'profile prefix parent' `
      -Expected 'a directory or a path that can be created as a directory' `
      -Actual "regular file '$profilePrefixParent'" `
      -ActionText 'choose a prefix below a writable directory'
  }
}

$originalLocation = Get-Location
try {
  Set-Location $script:RepoRoot
  Write-Host "[env] branch: $(& git branch --show-current)"
  Write-Host "[env] build:  $script:ResolvedBuildDir"
  Write-Host "[env] ORT:    $script:ResolvedOrtRoot"
  Write-Host "[env] OpenCV: $script:ResolvedOpenCvDir"
  Write-Host "[env] Python: $script:ResolvedPythonExe"
  Write-Host "[env] workflow: $script:WorkflowFilePath"

  switch ($Action) {
    'doctor' {
      Invoke-Doctor
    }
    'build' {
      Invoke-EnsureBuild
    }
    'clean-build' {
      Invoke-CleanBuild
    }
    'test' {
      Invoke-EnsureBuild
      Invoke-FullTests
    }
    'detect' {
      Invoke-EnsureBuild
      Invoke-Detect
    }
    'batch' {
      Invoke-EnsureBuild
      Invoke-Batch
    }
    'batch-compare' {
      Invoke-EnsureBuild
      Invoke-BatchComparison
    }
    'demo' {
      Invoke-EnsureBuild
      Invoke-Demo
    }
    'consistency' {
      Invoke-EnsureBuild
      Invoke-Consistency
    }
    'benchmark' {
      Invoke-EnsureBuild
      Invoke-Consistency
      Invoke-Benchmark
    }
    'profile' {
      Invoke-EnsureBuild
      Invoke-Profile
    }
    'all' {
      Invoke-CleanBuild
      Invoke-FullTests
      Invoke-Demo
      Invoke-Consistency
      Invoke-Benchmark
      [void](Invoke-BatchRun -InputPath $script:BatchManifestPath `
        -InputKind 'manifest' `
        -OutputPath (Join-Path $script:RunDir 'batch') `
        -ConfigPath $script:ConfigPath -WorkerCount 2 -Capacity 4 `
        -WriteImages $false -AllowOverwrite $false)
    }
  }

  Write-Stage "$Action PASS"
  Write-Host "Build directory: $script:ResolvedBuildDir"
  if ($needsEvidence) {
    Write-Host "Fresh evidence:  $script:RunDir"
  }
  if ($Action -eq 'profile') {
    Write-Host "Profile trace:   $script:ProfileTracePath"
  }
} finally {
  Set-Location $originalLocation
}
