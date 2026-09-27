# Runs with only the packaged DLLs and the Windows system directories available.
[CmdletBinding()]
param()
Set-StrictMode -Version 3.0
$ErrorActionPreference = 'Stop'

function Test-QtStartup([string]$PackageRoot, [string]$OutputDirectory) {
  # QApplication initializes the real Windows platform before parsing --help.
  # Own the Process/handle directly: PS 5.1 Start-Process may lose ExitCode when
  # a short-lived GUI process exits before its asynchronous bookkeeping runs.
  $qt = New-Object Diagnostics.Process
  $qt.StartInfo = New-Object Diagnostics.ProcessStartInfo
  $qt.StartInfo.FileName = Join-Path $PackageRoot 'yolo_defect_qt.exe'
  $qt.StartInfo.Arguments = '--help'
  $qt.StartInfo.WorkingDirectory = $PackageRoot
  $qt.StartInfo.UseShellExecute = $false
  $qt.StartInfo.CreateNoWindow = $true
  $qt.StartInfo.RedirectStandardOutput = $true
  $qt.StartInfo.RedirectStandardError = $true
  try {
    if (-not $qt.Start()) { throw 'Qt startup process could not be created.' }
    $qtHandle = $qt.Handle
    $stdout = $qt.StandardOutput.ReadToEndAsync()
    $stderr = $qt.StandardError.ReadToEndAsync()
    if (-not $qt.WaitForExit(15000)) {
      $qt.Kill()
      throw 'Qt startup did not exit within 15 seconds.'
    }
    $qt.WaitForExit()
    $exitCode = $qt.ExitCode
    [IO.File]::WriteAllText((Join-Path $OutputDirectory 'qt_startup.txt'), $stdout.Result)
    [IO.File]::WriteAllText((Join-Path $OutputDirectory 'qt_startup_error.txt'), $stderr.Result)
    if ($null -eq $exitCode -or $exitCode -ne 0) {
      throw "Qt startup failed with code '$exitCode'."
    }
  } finally {
    $qt.Dispose()
  }
}

$packageRoot = $PSScriptRoot
$savedEnvironment = @{}
foreach ($name in @('PATH', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_QPA_PLATFORM')) {
  $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name)
}
$result = 1
Push-Location $packageRoot
try {
  $env:PATH = "$packageRoot;$env:SystemRoot\System32;$env:SystemRoot"
  $env:QT_PLUGIN_PATH = $packageRoot
  $env:QT_QPA_PLATFORM_PLUGIN_PATH = Join-Path $packageRoot 'platforms'
  $env:QT_QPA_PLATFORM = 'windows'
  $output = 'outputs/verify_' + [DateTime]::Now.ToString('yyyyMMdd_HHmmss_fff')
  New-Item -ItemType Directory -Path $output | Out-Null
  $cli = Join-Path $packageRoot 'yolo_defect_cpp.exe'
  $variants = @(@{ Name = 'fp32'; Config = 'configs/default_config.txt' })
  if (Test-Path -LiteralPath 'configs/int8_u8s8_config.txt' -PathType Leaf) {
    $variants += @{ Name = 'u8s8'; Config = 'configs/int8_u8s8_config.txt' }
  }
  foreach ($variant in $variants) {
    $json = "$output/$($variant.Name).json"
    & $cli --config $variant.Config --image samples/crazing_241.jpg `
      --output-json $json --output-image "$output/$($variant.Name).png"
    if ($LASTEXITCODE -ne 0) { throw "$($variant.Name) inference failed with code $LASTEXITCODE." }
    Get-Content -LiteralPath $json -Raw | ConvertFrom-Json | Out-Null
  }
  & $cli --batch --config configs/default_config.txt --manifest samples/manifest.txt `
    --output-dir "$output/items" --batch-summary "$output/batch_summary.json" `
    --workers 2 --queue-capacity 4 --output-images
  if ($LASTEXITCODE -ne 0) { throw "Batch inference failed with code $LASTEXITCODE." }
  $batch = Get-Content -LiteralPath "$output/batch_summary.json" -Raw | ConvertFrom-Json
  if (@($batch.items).Count -ne 6 -or @($batch.items | Where-Object { $_.status -ne 'succeeded' }).Count -ne 0) {
    throw 'Expected six successful images in the packaged manifest.'
  }
  Test-QtStartup $packageRoot (Join-Path $packageRoot $output)
  Write-Host "PASS: $($variants.Count) packaged model(s), six-image batch, and native Qt startup."
  Write-Host "Results: $packageRoot\$output"
  $result = 0
} catch {
  [Console]::Error.WriteLine("Demo verification: $($_.Exception.Message)")
} finally {
  Pop-Location
  foreach ($name in $savedEnvironment.Keys) {
    [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], 'Process')
  }
}
exit $result
