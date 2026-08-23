@{
  # Copy this file to cpp_infer/.stage1.local.psd1 and fill in the paths for
  # one machine. The local file is ignored by Git.
  OrtRoot = 'D:\path\to\onnxruntime-win-x64-1.19.2'
  OpenCvDir = 'D:\path\to\opencv\build\x64\vc16\lib'
  OpenCvBin = 'D:\path\to\opencv\build\x64\vc16\bin'
  PythonExe = 'C:\path\to\python.exe'
  GTestSource = 'D:\path\to\verified\googletest-1.17.0-source'

  # Optional single-image convenience overrides. Leave them empty to use
  # stage1.defaults.psd1. Relative paths resolve from the local file.
  DefaultRuntimeConfig = ''
  DefaultDetectOutputRoot = ''
  # DetectWriteJson = $true
  # DetectWriteImage = $true
}
