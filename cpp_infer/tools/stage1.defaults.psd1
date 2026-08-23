@{
  # Machine-independent workflow defaults. Relative paths are resolved from
  # this file, never from the caller's current working directory.
  SchemaVersion = 1

  Build = @{
    Generator = 'NMake Makefiles'
    Type = 'Release'
    Testing = $true
    TemporaryDirectoryName = 'yolo_defect_stage1_manual_release'
  }

  Detect = @{
    RuntimeConfig = '..\configs\default_config.txt'
    OutputRoot = '..\results\manual'
    WriteJson = $true
    WriteImage = $true
  }

  Demo = @{
    Image = '..\..\data\images\val\crazing_241.jpg'
  }

  Consistency = @{
    Manifest = '..\tests\fixtures\consistency_manifest.json'
  }

  Benchmark = @{
    Warmup = 10
    Repeat = 100
  }
}
