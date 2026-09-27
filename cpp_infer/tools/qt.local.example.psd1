# Copy to cpp_infer/.qt.local.psd1 (ignored by Git).
# Use your installed Qt 6 MSVC x64 kit, not the Qt Creator directory.
# Relative paths below are resolved against the local configuration file.
@{
  QtRoot = 'C:\Qt\6.8.3\msvc2022_64'

  # Shared dependency paths are read from .stage1.local.psd1 first.
  # Uncomment only when you need a Qt-workflow-specific override:
  # OrtRoot = 'C:\SDKs\onnxruntime-win-x64-1.19.2'
  # OpenCvDir = 'C:\SDKs\opencv\build\x64\vc16\lib'
  # OpenCvBin = 'C:\SDKs\opencv\build\x64\vc16\bin'
  # PythonExe = 'C:\Python\python.exe'
  # GTestSource = 'build\googletest'

  # Optional workflow defaults:
  # BuildDir = 'build\qt-msvc-release'
  # Config = 'configs\default_config.txt'
  # Image = '..\data\images\val\crazing_241.jpg'
  # OutputDir = '..\results\qt'
  # PackageDir = '..\dist\yolo-defect-qt'
}
