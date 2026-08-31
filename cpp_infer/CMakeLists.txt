cmake_minimum_required(VERSION 3.16)

project(yolo_defect_cpp
  VERSION 0.8.0
  DESCRIPTION "C++ ONNX industrial-vision runtime with bounded multi-image concurrency"
  LANGUAGES CXX
)

option(YOLO_DEFECT_CORE_ONLY
  "Build only the dependency-free project-core portability smoke"
  OFF)
option(YOLO_DEFECT_REQUIRE_TENSORRT_EP
  "Require the isolated Linux x86_64 ORT TensorRT/CUDA execution providers"
  OFF)
option(YOLO_DEFECT_ENABLE_NATIVE_TENSORRT_BACKEND
  "Enable the load-only native TensorRT correctness fallback"
  OFF)

include(CTest)

add_library(yolo_defect_project_core STATIC
  src/project_core.cpp
)
add_library(yolo_defect::project_core ALIAS yolo_defect_project_core)
target_compile_features(yolo_defect_project_core PUBLIC cxx_std_17)
target_include_directories(yolo_defect_project_core
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
)
set_target_properties(yolo_defect_project_core PROPERTIES
  CXX_EXTENSIONS OFF
)

if(BUILD_TESTING OR YOLO_DEFECT_CORE_ONLY)
  add_executable(yolo_defect_project_core_smoke
    tests/project_core_smoke.cpp
  )
  target_compile_features(yolo_defect_project_core_smoke PRIVATE cxx_std_17)
  target_link_libraries(
    yolo_defect_project_core_smoke PRIVATE yolo_defect::project_core)
  set_target_properties(yolo_defect_project_core_smoke PROPERTIES
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
  if(BUILD_TESTING)
    add_test(
      NAME yolo_defect_project_core_smoke
      COMMAND $<TARGET_FILE:yolo_defect_project_core_smoke>
    )
    set_tests_properties(yolo_defect_project_core_smoke PROPERTIES
      LABELS "project_core;portability;smoke;quality_gate")
  endif()
endif()

if(YOLO_DEFECT_CORE_ONLY)
  message(STATUS
    "YOLO_DEFECT_CORE_ONLY=ON: skipping OpenCV, ONNX Runtime, Python, and GoogleTest")
  return()
endif()

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR
    "The supported ONNX Runtime SDK packages require a 64-bit toolchain. "
    "Select an x64 Windows toolchain or a 64-bit Linux target and point "
    "ONNXRUNTIME_ROOT at the matching official SDK.")
endif()

if(YOLO_DEFECT_REQUIRE_TENSORRT_EP)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_CROSSCOMPILING OR
     NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    message(FATAL_ERROR
      "YOLO_DEFECT_REQUIRE_TENSORRT_EP=ON supports native Linux x86_64 only. "
      "Actual target is '${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}', "
      "crosscompiling=${CMAKE_CROSSCOMPILING}. Keep this option OFF for the "
      "Windows, CPU Linux, and AArch64/QEMU gates.")
  endif()
  if(BUILD_TESTING)
    message(FATAL_ERROR
      "YOLO_DEFECT_REQUIRE_TENSORRT_EP=ON uses ORT GPU 1.20.1, while the "
      "complete CTest/Python consistency gate is intentionally pinned to "
      "ORT CPU 1.19.2. Configure this dedicated GPU product build with "
      "-DBUILD_TESTING=OFF and run the full suite in a separate unchanged "
      "CPU build tree.")
  endif()
endif()

if(YOLO_DEFECT_ENABLE_NATIVE_TENSORRT_BACKEND AND
   NOT YOLO_DEFECT_REQUIRE_TENSORRT_EP)
  message(FATAL_ERROR
    "YOLO_DEFECT_ENABLE_NATIVE_TENSORRT_BACKEND=ON requires the dedicated "
    "Linux x86_64 GPU build mode YOLO_DEFECT_REQUIRE_TENSORRT_EP=ON. Keep "
    "both options OFF for Windows, Linux CPU, and AArch64/QEMU gates.")
endif()

if(CMAKE_CROSSCOMPILING)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
     NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    message(FATAL_ERROR
      "Full cross-build supports target Linux AArch64 only. Actual target is "
      "'${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}'. Use the documented "
      "linux-aarch64-gnu toolchain or build project-core only.")
  endif()
  if(YOLO_DEFECT_AARCH64_SYSROOT STREQUAL "")
    message(FATAL_ERROR
      "YOLO_DEFECT_AARCH64_SYSROOT is required for a full AArch64 build. "
      "Point it at the target-only OpenCV tree produced by "
      "tools/bootstrap_aarch64_deps.sh; do not point it at host /usr.")
  endif()
  get_filename_component(
    YOLO_DEFECT_AARCH64_SYSROOT "${YOLO_DEFECT_AARCH64_SYSROOT}" ABSOLUTE)
  set(OpenCV_INCLUDE_DIRS
    "${YOLO_DEFECT_AARCH64_SYSROOT}/usr/include/opencv4")
  set(OpenCV_VERSION "4.6.0")
  set(_opencv_version_header
    "${OpenCV_INCLUDE_DIRS}/opencv2/core/version.hpp")
  if(NOT EXISTS "${_opencv_version_header}")
    message(FATAL_ERROR
      "Target OpenCV headers are incomplete: expected "
      "'${_opencv_version_header}'. Re-run bootstrap_aarch64_deps.sh.")
  endif()
  file(READ "${_opencv_version_header}" _opencv_version_text)
  if(NOT _opencv_version_text MATCHES
       "#define[ \t]+CV_VERSION_MAJOR[ \t]+4" OR
     NOT _opencv_version_text MATCHES
       "#define[ \t]+CV_VERSION_MINOR[ \t]+6" OR
     NOT _opencv_version_text MATCHES
       "#define[ \t]+CV_VERSION_REVISION[ \t]+0")
    message(FATAL_ERROR
      "Target OpenCV headers do not match the pinned Ubuntu 4.6.0 ABI at "
      "'${_opencv_version_header}'.")
  endif()
  unset(_opencv_version_text)
  unset(_opencv_version_header)

  set(OpenCV_LIBS "")
  foreach(_opencv_component IN ITEMS core imgproc imgcodecs)
    set(_opencv_library
      "${YOLO_DEFECT_AARCH64_SYSROOT}/usr/lib/aarch64-linux-gnu/libopencv_${_opencv_component}.so.4.6.0")
    if(NOT EXISTS "${_opencv_library}")
      message(FATAL_ERROR
        "Target OpenCV ${_opencv_component} library is missing at "
        "'${_opencv_library}'. Re-run bootstrap_aarch64_deps.sh.")
    endif()
    add_library(yolo_defect_opencv_${_opencv_component} SHARED IMPORTED)
    set_target_properties(yolo_defect_opencv_${_opencv_component} PROPERTIES
      IMPORTED_LOCATION "${_opencv_library}"
      INTERFACE_INCLUDE_DIRECTORIES "${OpenCV_INCLUDE_DIRS}")
    list(APPEND OpenCV_LIBS yolo_defect_opencv_${_opencv_component})
    unset(_opencv_library)
  endforeach()
  unset(_opencv_component)
else()
  find_package(OpenCV 4 REQUIRED COMPONENTS core imgcodecs imgproc)
endif()

find_package(Threads REQUIRED)

set(_onnxruntime_root_default "")
if(DEFINED ENV{ONNXRUNTIME_ROOT})
  set(_onnxruntime_root_default "$ENV{ONNXRUNTIME_ROOT}")
endif()
set(ONNXRUNTIME_ROOT "${_onnxruntime_root_default}" CACHE PATH
  "Root of the official extracted ONNX Runtime C++ SDK")
unset(_onnxruntime_root_default)

if(ONNXRUNTIME_ROOT STREQUAL "")
  message(FATAL_ERROR
    "ONNXRUNTIME_ROOT is required. Configure with "
    "-DONNXRUNTIME_ROOT=<official extracted C++ SDK root>. "
    "A Python wheel is not a complete C++ SDK on Windows or Linux.")
endif()

get_filename_component(ONNXRUNTIME_ROOT "${ONNXRUNTIME_ROOT}" ABSOLUTE)
set(ONNXRUNTIME_VERSION_FILE "${ONNXRUNTIME_ROOT}/VERSION_NUMBER")
set(ONNXRUNTIME_C_API_HEADER
  "${ONNXRUNTIME_ROOT}/include/onnxruntime_c_api.h")
set(ONNXRUNTIME_CXX_API_HEADER
  "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
set(ONNXRUNTIME_CPU_PROVIDER_HEADER
  "${ONNXRUNTIME_ROOT}/include/cpu_provider_factory.h")
set(_onnxruntime_required_files
    ONNXRUNTIME_VERSION_FILE
    ONNXRUNTIME_C_API_HEADER
    ONNXRUNTIME_CXX_API_HEADER
    ONNXRUNTIME_CPU_PROVIDER_HEADER)

if(WIN32)
  set(ONNXRUNTIME_IMPORT_LIBRARY
    "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
  set(ONNXRUNTIME_RUNTIME_LIBRARY
    "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll")
  list(APPEND _onnxruntime_required_files
    ONNXRUNTIME_IMPORT_LIBRARY
    ONNXRUNTIME_RUNTIME_LIBRARY)
  set(_onnxruntime_platform_expectation
    "import library and runtime DLL")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(ONNXRUNTIME_RUNTIME_LIBRARY
    "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
  list(APPEND _onnxruntime_required_files
    ONNXRUNTIME_RUNTIME_LIBRARY)
  set(_onnxruntime_platform_expectation
    "shared library lib/libonnxruntime.so")
else()
  message(FATAL_ERROR
    "Unsupported full Runtime platform '${CMAKE_SYSTEM_NAME}'. Gate A supports "
    "Windows and Linux; use -DYOLO_DEFECT_CORE_ONLY=ON for the dependency-free "
    "project-core portability smoke on another target.")
endif()

foreach(required_file IN LISTS _onnxruntime_required_files)
  if(NOT EXISTS "${${required_file}}" OR IS_DIRECTORY "${${required_file}}")
    message(FATAL_ERROR
      "Incomplete ONNX Runtime C++ SDK: ${required_file} is missing at "
      "'${${required_file}}'. Expected VERSION_NUMBER, C/C++ headers, "
      "CPU provider header, and ${_onnxruntime_platform_expectation} under "
      "ONNXRUNTIME_ROOT for ${CMAKE_SYSTEM_NAME}.")
  endif()
endforeach()
unset(_onnxruntime_required_files)
unset(_onnxruntime_platform_expectation)

file(READ "${ONNXRUNTIME_VERSION_FILE}" ONNXRUNTIME_VERSION)
string(STRIP "${ONNXRUNTIME_VERSION}" ONNXRUNTIME_VERSION)
if(YOLO_DEFECT_REQUIRE_TENSORRT_EP)
  set(_onnxruntime_expected_version "1.20.1")
else()
  set(_onnxruntime_expected_version "1.19.2")
endif()

if(YOLO_DEFECT_ENABLE_NATIVE_TENSORRT_BACKEND)
  set(TENSORRT_ROOT "" CACHE PATH
    "Root of the isolated TensorRT 10.4.0.26 runtime and headers")
  if(TENSORRT_ROOT STREQUAL "")
    message(FATAL_ERROR
      "TENSORRT_ROOT is required for the native TensorRT backend. Point it "
      "at the isolated TensorRT 10.4.0.26 tree containing headers and "
      "libnvinfer.so.10; a Python wheel is not a C++ SDK.")
  endif()
  get_filename_component(TENSORRT_ROOT "${TENSORRT_ROOT}" ABSOLUTE)
  set(TENSORRT_INCLUDE_DIR
    "${TENSORRT_ROOT}/usr/include/x86_64-linux-gnu")
  set(TENSORRT_VERSION_HEADER
    "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h")
  set(TENSORRT_NVINFER_LIBRARY
    "${TENSORRT_ROOT}/usr/lib/x86_64-linux-gnu/libnvinfer.so.10")
  foreach(_native_tensorrt_file IN ITEMS
      TENSORRT_VERSION_HEADER TENSORRT_NVINFER_LIBRARY)
    if(NOT EXISTS "${${_native_tensorrt_file}}" OR
       IS_DIRECTORY "${${_native_tensorrt_file}}")
      message(FATAL_ERROR
        "Incomplete TensorRT C++ SDK: ${_native_tensorrt_file} is missing "
        "at '${${_native_tensorrt_file}}'. Extract matching "
        "libnvinfer-headers-dev and libnvinfer10 10.4.0.26 packages into "
        "TENSORRT_ROOT.")
    endif()
  endforeach()
  unset(_native_tensorrt_file)
  file(READ "${TENSORRT_VERSION_HEADER}" _tensorrt_version_text)
  foreach(_version_pair IN ITEMS "MAJOR;10" "MINOR;4" "PATCH;0" "BUILD;26")
    list(GET _version_pair 0 _version_name)
    list(GET _version_pair 1 _version_value)
    if(NOT _tensorrt_version_text MATCHES
       "#define[ \t]+NV_TENSORRT_${_version_name}[ \t]+${_version_value}")
      message(FATAL_ERROR
        "TensorRT header version mismatch at '${TENSORRT_VERSION_HEADER}'. "
        "Expected 10.4.0.26; inspect NV_TENSORRT_${_version_name} and use "
        "the exact runtime/header pair that built the frozen engine.")
    endif()
  endforeach()
  unset(_version_value)
  unset(_version_name)
  unset(_version_pair)
  unset(_tensorrt_version_text)

  if(NOT DEFINED CUDAToolkit_ROOT OR CUDAToolkit_ROOT STREQUAL "")
    message(FATAL_ERROR
      "CUDAToolkit_ROOT is required for the native TensorRT backend. Point "
      "it at the isolated CUDA 12.6 toolkit used by the frozen engine.")
  endif()
  get_filename_component(CUDAToolkit_ROOT "${CUDAToolkit_ROOT}" ABSOLUTE)
  set(CUDA_INCLUDE_DIR
    "${CUDAToolkit_ROOT}/targets/x86_64-linux/include")
  set(CUDA_RUNTIME_HEADER "${CUDA_INCLUDE_DIR}/cuda_runtime_api.h")
  set(CUDA_CUDART_LIBRARY
    "${CUDAToolkit_ROOT}/targets/x86_64-linux/lib/libcudart.so.12")
  foreach(_native_cuda_file IN ITEMS
      CUDA_RUNTIME_HEADER CUDA_CUDART_LIBRARY)
    if(NOT EXISTS "${${_native_cuda_file}}" OR
       IS_DIRECTORY "${${_native_cuda_file}}")
      message(FATAL_ERROR
        "Incomplete isolated CUDA runtime SDK: ${_native_cuda_file} is "
        "missing at '${${_native_cuda_file}}'. Point CUDAToolkit_ROOT at "
        "the CUDA 12.6 tree with headers and libcudart.so.12; ambient CUDA "
        "installations are intentionally not searched.")
    endif()
  endforeach()
  unset(_native_cuda_file)
  file(READ "${CUDA_RUNTIME_HEADER}" _cuda_runtime_header_text)
  if(NOT _cuda_runtime_header_text MATCHES
     "#define[ \t]+CUDART_VERSION[ \t]+12060")
    message(FATAL_ERROR
      "CUDA runtime header version mismatch at '${CUDA_RUNTIME_HEADER}'. "
      "Expected CUDART_VERSION 12060 (CUDA 12.6); use the exact isolated "
      "headers/runtime pair that built the frozen engine.")
  endif()
  unset(_cuda_runtime_header_text)

  add_library(TensorRT::nvinfer SHARED IMPORTED GLOBAL)
  set_target_properties(TensorRT::nvinfer PROPERTIES
    IMPORTED_LOCATION "${TENSORRT_NVINFER_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR}")
  add_library(CUDA::cudart SHARED IMPORTED GLOBAL)
  set_target_properties(CUDA::cudart PROPERTIES
    IMPORTED_LOCATION "${CUDA_CUDART_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${CUDA_INCLUDE_DIR}")
endif()
if(NOT ONNXRUNTIME_VERSION STREQUAL "${_onnxruntime_expected_version}")
  message(FATAL_ERROR
    "Unsupported ONNX Runtime SDK version for this build mode. Expected "
    "'${_onnxruntime_expected_version}', actual '${ONNXRUNTIME_VERSION}'. "
    "Use the CPU 1.19.2 SDK for existing gates or the isolated GPU 1.20.1 "
    "SDK with YOLO_DEFECT_REQUIRE_TENSORRT_EP=ON.")
endif()
unset(_onnxruntime_expected_version)

if(YOLO_DEFECT_REQUIRE_TENSORRT_EP)
  foreach(_provider_library IN ITEMS
      libonnxruntime_providers_shared.so
      libonnxruntime_providers_cuda.so
      libonnxruntime_providers_tensorrt.so)
    if(NOT EXISTS "${ONNXRUNTIME_ROOT}/lib/${_provider_library}" OR
       IS_DIRECTORY "${ONNXRUNTIME_ROOT}/lib/${_provider_library}")
      message(FATAL_ERROR
        "Incomplete ORT GPU SDK: missing lib/${_provider_library} under "
        "'${ONNXRUNTIME_ROOT}'. Use the official Linux x86_64 GPU C++ SDK; "
        "a CPU SDK or Python wheel cannot satisfy the TensorRT build.")
    endif()
  endforeach()
  unset(_provider_library)
endif()

add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)
if(WIN32)
  set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_IMPLIB "${ONNXRUNTIME_IMPORT_LIBRARY}"
    IMPORTED_LOCATION "${ONNXRUNTIME_RUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include"
  )
else()
  set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_RUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include"
  )
endif()

if(CMAKE_CROSSCOMPILING)
  foreach(_target_library IN ITEMS
      "${ONNXRUNTIME_RUNTIME_LIBRARY}"
      "${YOLO_DEFECT_AARCH64_SYSROOT}/usr/lib/aarch64-linux-gnu/libopencv_core.so.4.6.0"
      "${YOLO_DEFECT_AARCH64_SYSROOT}/usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.6.0"
      "${YOLO_DEFECT_AARCH64_SYSROOT}/usr/lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.6.0")
    execute_process(
      COMMAND "${CMAKE_READELF}" -h "${_target_library}"
      OUTPUT_VARIABLE _target_elf_header
      ERROR_VARIABLE _target_elf_error
      RESULT_VARIABLE _target_elf_result)
    if(NOT _target_elf_result EQUAL 0 OR
       NOT _target_elf_header MATCHES "Machine:[ \t]+AArch64")
      message(FATAL_ERROR
        "Target dependency architecture mismatch at '${_target_library}'. "
        "Expected AArch64 ELF; actual readelf result: "
        "${_target_elf_error}${_target_elf_header}. Select ARM64 libraries, "
        "not host x86_64 libraries.")
    endif()
  endforeach()
  unset(_target_elf_error)
  unset(_target_elf_header)
  unset(_target_elf_result)
  unset(_target_library)
endif()

add_library(yolo_defect_runtime STATIC
  src/artifact_spec.cpp
  src/batch_discovery.cpp
  src/batch_runner.cpp
  src/batch_writer.cpp
  src/benchmark_result.cpp
  src/benchmark_runner.cpp
  src/benchmark_writer.cpp
  src/config_loader.cpp
  src/detector_pipeline.cpp
  src/image_decoder.cpp
  src/image_preprocessor.cpp
  src/key_value_parser.cpp
  src/model_metadata.cpp
  src/onnx_runner.cpp
  src/platform_info.cpp
  src/postprocessor.cpp
  src/profile_runner.cpp
  src/result_writer.cpp
)
add_library(yolo_defect::runtime ALIAS yolo_defect_runtime)

target_compile_features(yolo_defect_runtime PUBLIC cxx_std_17)
target_include_directories(yolo_defect_runtime
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(yolo_defect_runtime
  PUBLIC
    yolo_defect::project_core
  PRIVATE
    ${OpenCV_LIBS}
    onnxruntime::onnxruntime
    Threads::Threads
)
if(WIN32)
  target_link_libraries(yolo_defect_runtime PRIVATE Psapi)
endif()
target_compile_definitions(yolo_defect_runtime PRIVATE
  "YOLO_DEFECT_BUILD_TYPE=\"$<CONFIG>\""
  "YOLO_DEFECT_COMPILER_ID=\"${CMAKE_CXX_COMPILER_ID}\""
  "YOLO_DEFECT_COMPILER_VERSION=\"${CMAKE_CXX_COMPILER_VERSION}\""
  "YOLO_DEFECT_OPENCV_VERSION=\"${OpenCV_VERSION}\""
  "YOLO_DEFECT_TARGET_ARCHITECTURE=\"${CMAKE_SYSTEM_PROCESSOR}\""
)
if(YOLO_DEFECT_ENABLE_NATIVE_TENSORRT_BACKEND)
  target_sources(yolo_defect_runtime PRIVATE
    src/native_tensorrt_runner.cpp)
endif()
if(YOLO_DEFECT_REQUIRE_TENSORRT_EP)
  target_compile_definitions(yolo_defect_runtime PRIVATE
    YOLO_DEFECT_TENSORRT_EP_BUILD=1)
endif()
if(YOLO_DEFECT_ENABLE_NATIVE_TENSORRT_BACKEND)
  target_compile_definitions(yolo_defect_runtime PRIVATE
    YOLO_DEFECT_NATIVE_TENSORRT_BUILD=1)
  target_link_libraries(yolo_defect_runtime PRIVATE
    TensorRT::nvinfer CUDA::cudart)
endif()
set_target_properties(yolo_defect_runtime PROPERTIES
  CXX_EXTENSIONS OFF
)

add_executable(yolo_defect_cpp
  src/main.cpp
)
target_compile_features(yolo_defect_cpp PRIVATE cxx_std_17)
target_link_libraries(yolo_defect_cpp PRIVATE yolo_defect::runtime)
if(CMAKE_CROSSCOMPILING)
  target_link_options(yolo_defect_cpp PRIVATE
    "LINKER:-rpath-link,${YOLO_DEFECT_AARCH64_SYSROOT}/usr/lib/aarch64-linux-gnu"
    "LINKER:-rpath-link,${YOLO_DEFECT_AARCH64_SYSROOT}/lib/aarch64-linux-gnu"
    "LINKER:-rpath-link,${YOLO_DEFECT_AARCH64_SYSROOT}/usr/lib/aarch64-linux-gnu/blas"
    "LINKER:-rpath-link,${YOLO_DEFECT_AARCH64_SYSROOT}/usr/lib/aarch64-linux-gnu/lapack"
    "LINKER:-rpath-link,${YOLO_DEFECT_AARCH64_SYSROOT}/usr/lib")
endif()
set_target_properties(yolo_defect_cpp PROPERTIES
  CXX_EXTENSIONS OFF
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

function(configure_onnxruntime_runtime target_name)
  if(WIN32)
    add_custom_command(TARGET "${target_name}" POST_BUILD
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "$<TARGET_FILE:onnxruntime::onnxruntime>"
        "$<TARGET_FILE_DIR:${target_name}>"
      COMMENT
        "Staging the pinned ONNX Runtime DLL for ${target_name}"
      VERBATIM
    )
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(CMAKE_CROSSCOMPILING)
      set_target_properties("${target_name}" PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "\$ORIGIN/../lib")
    else()
      set_property(TARGET "${target_name}" APPEND PROPERTY
        BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib")
    endif()
  endif()
endfunction()

configure_onnxruntime_runtime(yolo_defect_cpp)
if(YOLO_DEFECT_ENABLE_NATIVE_TENSORRT_BACKEND)
  set_property(TARGET yolo_defect_cpp APPEND PROPERTY BUILD_RPATH
    "${TENSORRT_ROOT}/usr/lib/x86_64-linux-gnu"
    "${CUDAToolkit_ROOT}/targets/x86_64-linux/lib")
endif()

if(BUILD_TESTING)
  include(FetchContent)

  if(MSVC)
    set(gtest_force_shared_crt ON CACHE BOOL
      "Use the parent MSVC runtime library for GoogleTest" FORCE)
  endif()
  set(INSTALL_GTEST OFF CACHE BOOL
    "Do not install the build-only GoogleTest dependency" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL
    "The C++ Runtime tests only need GoogleTest" FORCE)

  set(GOOGLETEST_ARCHIVE_URL
    "https://github.com/google/googletest/archive/52eb8108c5bdec04579160ae17225d66034bd723.zip")
  set(GOOGLETEST_ARCHIVE_SHA256
    "9A56A54AE784394FF664CD55E8F4C9A03B503EBF0CB99576321C78AB3D87CA84")
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
    FetchContent_Declare(
      googletest
      URL "${GOOGLETEST_ARCHIVE_URL}"
      URL_HASH "SHA256=${GOOGLETEST_ARCHIVE_SHA256}"
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
  else()
    FetchContent_Declare(
      googletest
      URL "${GOOGLETEST_ARCHIVE_URL}"
      URL_HASH "SHA256=${GOOGLETEST_ARCHIVE_SHA256}"
    )
  endif()
  FetchContent_MakeAvailable(googletest)
  include(GoogleTest)
  find_package(Python3 3.9 REQUIRED COMPONENTS Interpreter)

  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
      "import cv2,numpy,onnxruntime as ort; assert ort.__version__ == '1.19.2', ort.__version__; assert 'CPUExecutionProvider' in ort.get_available_providers(), ort.get_available_providers(); print('Python consistency dependencies:', 'onnxruntime='+ort.__version__, 'opencv='+cv2.__version__, 'numpy='+numpy.__version__)"
    RESULT_VARIABLE S1_07_PYTHON_DEPENDENCY_RESULT
    OUTPUT_VARIABLE S1_07_PYTHON_DEPENDENCY_OUTPUT
    ERROR_VARIABLE S1_07_PYTHON_DEPENDENCY_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
  )
  if(NOT S1_07_PYTHON_DEPENDENCY_RESULT EQUAL 0)
    message(FATAL_ERROR
      "S1-07 Python consistency dependencies are unavailable from "
      "Python3_EXECUTABLE='${Python3_EXECUTABLE}'. Expected importable "
      "numpy, OpenCV Python, and onnxruntime 1.19.2 with "
      "CPUExecutionProvider. Actual error: "
      "${S1_07_PYTHON_DEPENDENCY_ERROR}. Select the documented Python "
      "environment; CMake will not install packages automatically.")
  endif()
  message(STATUS "${S1_07_PYTHON_DEPENDENCY_OUTPUT}")

  set(DEFAULT_CONFIG
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/default_config.txt")
  set(DEFAULT_TENSORRT_CONFIG
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/tensorrt_fp16_config.txt")
  set(DEFAULT_NATIVE_TENSORRT_CONFIG
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/tensorrt_native_fp16_config_v4.txt")
  set(TEST_RUNTIME_FIXTURES
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/runtime")
  set(TEST_FAILURE_SCRIPT
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/assert_cli_failure.cmake")
  set(TEST_INSPECTION_SCRIPT
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/assert_model_inspection.cmake")
  set(TEST_RAW_OUTPUT_SCRIPT
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/assert_raw_output_summary.cmake")
  set(TEST_CLI_ARGUMENT_FAILURE_SCRIPT
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/assert_cli_arguments_failure.cmake")
  set(TEST_SINGLE_IMAGE_OUTPUT_SCRIPT
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/assert_single_image_outputs.cmake")
  set(TEST_DETECTION_JSON_VALIDATOR
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/assert_detection_json.py")
  set(TEST_BENCHMARK_SCRIPT
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/assert_benchmark.cmake")
  set(TEST_BENCHMARK_JSON_VALIDATOR
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/assert_benchmark_json.py")
  set(TEST_PROFILE_SCRIPT
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/assert_profile.cmake")
  set(TEST_BATCH_CLI_SCRIPT
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/assert_batch_cli.py")
  set(TEST_BATCH_SUMMARY_VALIDATOR
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/validate_batch_summary.py")
  set(CONSISTENCY_MANIFEST
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/consistency_manifest.json")
  set(CONSISTENCY_TOOL
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/compare_consistency.py")
  set(CONSISTENCY_UNIT_TEST
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_consistency.py")
  set(S2_01_PROFILE_SUMMARY_UNIT_TEST
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_s2_01_profile_summary.py")
  set(S2_01_BENCHMARK_COMPARISON_UNIT_TEST
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_s2_01_benchmark_comparison.py")
  set(S2_01_QUALITY_UNIT_TEST
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_s2_01_quality.py")
  set(S2_01_PROTOCOL_UNIT_TEST
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_s2_01_protocol.py")
  set(S2_01_EVIDENCE_UNIT_TEST
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_s2_01_evidence.py")
  set(S2_03_BATCH_TOOLS_UNIT_TEST
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_s2_03_batch_tools.py")
  set(S2_04_EVIDENCE_TOOLS_UNIT_TEST
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_s2_04_evidence_tools.py")
  set(S2_04_PRODUCT_SET_UNIT_TEST
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_run_s2_04_product_set.py")
  set(S2_01_MANIFEST_TOOL
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_s2_01_manifests.py")
  set(FIXED_IMAGE
    "${CMAKE_CURRENT_SOURCE_DIR}/../data/images/val/crazing_241.jpg")
  set(S1_06_FAULT_INPUT_ROOT
    "${CMAKE_BINARY_DIR}/test_inputs/s1_06_faults")
  file(MAKE_DIRECTORY "${S1_06_FAULT_INPUT_ROOT}")
  set(DAMAGED_IMAGE
    "${S1_06_FAULT_INPUT_ROOT}/damaged_image.jpg")
  file(WRITE "${DAMAGED_IMAGE}"
    "S1-06 synthetic damaged image: deliberately not encoded image bytes.\n")
  set(BLOCKED_OUTPUT_PARENT
    "${S1_06_FAULT_INPUT_ROOT}/blocked_output_parent")
  file(WRITE "${BLOCKED_OUTPUT_PARENT}"
    "S1-06 regular file that cannot be used as an output directory.\n")

  add_executable(yolo_defect_metadata_tests
    tests/model_metadata_validator_test.cpp
  )
  target_compile_features(yolo_defect_metadata_tests PRIVATE cxx_std_17)
  target_link_libraries(
    yolo_defect_metadata_tests PRIVATE yolo_defect::runtime)
  set_target_properties(yolo_defect_metadata_tests PROPERTIES
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )

  add_executable(yolo_defect_inference_tests
    tests/inference_runner_test.cpp
  )
  target_compile_features(yolo_defect_inference_tests PRIVATE cxx_std_17)
  target_link_libraries(
    yolo_defect_inference_tests PRIVATE yolo_defect::runtime)
  set_target_properties(yolo_defect_inference_tests PROPERTIES
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
  configure_onnxruntime_runtime(yolo_defect_inference_tests)

  add_executable(yolo_defect_postprocess_tests
    tests/postprocessor_test.cpp
  )
  target_compile_features(yolo_defect_postprocess_tests PRIVATE cxx_std_17)
  target_link_libraries(yolo_defect_postprocess_tests
    PRIVATE
      yolo_defect::runtime
      GTest::gtest_main
  )
  set_target_properties(yolo_defect_postprocess_tests PROPERTIES
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
  gtest_discover_tests(yolo_defect_postprocess_tests
    TEST_PREFIX "postprocess."
    PROPERTIES LABELS "postprocess\\;unit\\;quality_gate"
    DISCOVERY_TIMEOUT 30
  )

  add_executable(yolo_defect_preprocessor_tests
    tests/preprocessor_mat_test.cpp
  )
  target_compile_features(yolo_defect_preprocessor_tests PRIVATE cxx_std_17)
  target_include_directories(yolo_defect_preprocessor_tests
    PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(yolo_defect_preprocessor_tests
    PRIVATE
      yolo_defect::runtime
      GTest::gtest_main
      ${OpenCV_LIBS}
  )
  set_target_properties(yolo_defect_preprocessor_tests PROPERTIES
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
  gtest_discover_tests(yolo_defect_preprocessor_tests
    TEST_PREFIX "preprocess."
    PROPERTIES LABELS "preprocess\\;unit\\;quality_gate"
    DISCOVERY_TIMEOUT 30
  )

  # Compile the filesystem-path decoder directly so this regression test has
  # no ONNX Runtime dependency or session initialization.
  add_executable(yolo_defect_image_decoder_tests
    src/image_decoder.cpp
    tests/image_decoder_test.cpp
  )
  target_compile_features(yolo_defect_image_decoder_tests PRIVATE cxx_std_17)
  target_include_directories(yolo_defect_image_decoder_tests
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries(yolo_defect_image_decoder_tests
    PRIVATE
      GTest::gtest_main
      ${OpenCV_LIBS}
  )
  set_target_properties(yolo_defect_image_decoder_tests PROPERTIES
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
  gtest_discover_tests(yolo_defect_image_decoder_tests
    TEST_PREFIX "image_decoder."
    PROPERTIES LABELS "image_decoder\;filesystem\;unit\;quality_gate"
    DISCOVERY_TIMEOUT 30
  )

  add_executable(yolo_defect_output_tests
    tests/result_writer_test.cpp
  )
  target_compile_features(yolo_defect_output_tests PRIVATE cxx_std_17)
  target_link_libraries(yolo_defect_output_tests
    PRIVATE
      yolo_defect::runtime
      GTest::gtest_main
  )
  set_target_properties(yolo_defect_output_tests PROPERTIES
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
  gtest_discover_tests(yolo_defect_output_tests
    TEST_PREFIX "output."
    PROPERTIES LABELS "output\\;unit\\;quality_gate"
    DISCOVERY_TIMEOUT 30
  )

  add_executable(yolo_defect_benchmark_tests
    tests/benchmark_test.cpp
  )
  target_compile_features(yolo_defect_benchmark_tests PRIVATE cxx_std_17)
  target_link_libraries(yolo_defect_benchmark_tests
    PRIVATE
      yolo_defect::runtime
      GTest::gtest_main
  )
  set_target_properties(yolo_defect_benchmark_tests PROPERTIES
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
  gtest_discover_tests(yolo_defect_benchmark_tests
    TEST_PREFIX "benchmark."
    PROPERTIES LABELS "benchmark\;unit\;quality_gate"
    DISCOVERY_TIMEOUT 30
  )

  add_executable(yolo_defect_batch_tests
    tests/batch_test.cpp
  )
  target_compile_features(yolo_defect_batch_tests PRIVATE cxx_std_17)
  target_include_directories(yolo_defect_batch_tests
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(yolo_defect_batch_tests
    PRIVATE
      yolo_defect::runtime
      GTest::gtest_main
  )
  set_target_properties(yolo_defect_batch_tests PROPERTIES
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
  configure_onnxruntime_runtime(yolo_defect_batch_tests)
  gtest_discover_tests(yolo_defect_batch_tests
    TEST_PREFIX "batch."
    PROPERTIES LABELS "batch\;concurrency\;unit\;quality_gate"
    DISCOVERY_TIMEOUT 30
  )

  add_executable(yolo_defect_image_probe
    tests/image_file_probe.cpp
  )
  target_compile_features(yolo_defect_image_probe PRIVATE cxx_std_17)
  target_include_directories(yolo_defect_image_probe
    PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(yolo_defect_image_probe PRIVATE ${OpenCV_LIBS})
  set_target_properties(yolo_defect_image_probe PROPERTIES
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )

  add_test(
    NAME yolo_defect_cpp_help
    COMMAND $<TARGET_FILE:yolo_defect_cpp> --help
  )
  set_tests_properties(yolo_defect_cpp_help PROPERTIES
    LABELS "cli;smoke;quality_gate"
    PASS_REGULAR_EXPRESSION "S2-04 accelerator-capable Runtime CLI"
  )

  add_test(
    NAME yolo_defect_cpp_config
    COMMAND $<TARGET_FILE:yolo_defect_cpp> --config "${DEFAULT_CONFIG}"
  )
  set_tests_properties(yolo_defect_cpp_config PROPERTIES
    LABELS "contract;smoke;quality_gate"
    PASS_REGULAR_EXPRESSION "S1-01 Runtime/artifact contract summary"
  )

  add_test(
    NAME yolo_defect_cpp_tensorrt_config_contract
    COMMAND $<TARGET_FILE:yolo_defect_cpp>
      --config
      "${DEFAULT_TENSORRT_CONFIG}"
  )
  set_tests_properties(yolo_defect_cpp_tensorrt_config_contract PROPERTIES
    LABELS "contract;tensorrt;unit;quality_gate"
    PASS_REGULAR_EXPRESSION "configured_provider: tensorrt"
  )

  add_test(
    NAME yolo_defect_cpp_preprocess
    COMMAND $<TARGET_FILE:yolo_defect_cpp>
      --config "${DEFAULT_CONFIG}"
      --image "${FIXED_IMAGE}"
  )
  set_tests_properties(yolo_defect_cpp_preprocess PROPERTIES
    LABELS "preprocess;smoke;quality_gate"
    PASS_REGULAR_EXPRESSION "S1-01 Preprocess summary"
  )

  set(PATH_INDEPENDENCE_CWD
    "${CMAKE_BINARY_DIR}/path_independence_working_directory")
  file(MAKE_DIRECTORY "${PATH_INDEPENDENCE_CWD}")
  add_test(
    NAME yolo_defect_cpp_relative_paths
    COMMAND "${CMAKE_COMMAND}"
      "-DCLI=$<TARGET_FILE:yolo_defect_cpp>"
      "-DCONFIG=${DEFAULT_CONFIG}"
      "-DFIRST_CWD=${CMAKE_CURRENT_SOURCE_DIR}/.."
      "-DSECOND_CWD=${PATH_INDEPENDENCE_CWD}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/assert_path_independence.cmake"
  )
  set_tests_properties(yolo_defect_cpp_relative_paths PROPERTIES
    LABELS "contract;quality_gate")

  add_test(
    NAME yolo_defect_cpp_inspect_model
    COMMAND "${CMAKE_COMMAND}"
      "-DCLI=$<TARGET_FILE:yolo_defect_cpp>"
      "-DCONFIG=${DEFAULT_CONFIG}"
      -P "${TEST_INSPECTION_SCRIPT}"
  )
  set_tests_properties(yolo_defect_cpp_inspect_model PROPERTIES
    LABELS "metadata;integration;quality_gate")

  add_test(
    NAME yolo_defect_cpp_raw_output_summary
    COMMAND "${CMAKE_COMMAND}"
      "-DCLI=$<TARGET_FILE:yolo_defect_cpp>"
      "-DCONFIG=${DEFAULT_CONFIG}"
      "-DIMAGE=${FIXED_IMAGE}"
      -P "${TEST_RAW_OUTPUT_SCRIPT}"
  )
  set_tests_properties(yolo_defect_cpp_raw_output_summary PROPERTIES
    LABELS "inference;integration;quality_gate")

  add_test(
    NAME yolo_defect_cpp_single_image_outputs
    COMMAND "${CMAKE_COMMAND}"
      "-DCLI=$<TARGET_FILE:yolo_defect_cpp>"
      "-DCONFIG=${DEFAULT_CONFIG}"
      "-DIMAGE=${FIXED_IMAGE}"
      "-DOUTPUT_ROOT=${CMAKE_BINARY_DIR}/test_outputs/single_image"
      "-DPYTHON=${Python3_EXECUTABLE}"
      "-DJSON_VALIDATOR=${TEST_DETECTION_JSON_VALIDATOR}"
      "-DIMAGE_PROBE=$<TARGET_FILE:yolo_defect_image_probe>"
      -P "${TEST_SINGLE_IMAGE_OUTPUT_SCRIPT}"
  )
  set_tests_properties(yolo_defect_cpp_single_image_outputs PROPERTIES
    LABELS "output;integration;quality_gate"
  )

  add_test(
    NAME yolo_defect_cpp_batch_cli
    COMMAND "${Python3_EXECUTABLE}" "${TEST_BATCH_CLI_SCRIPT}"
      --cli $<TARGET_FILE:yolo_defect_cpp>
      --config "${DEFAULT_CONFIG}"
      --image "${FIXED_IMAGE}"
      --work-root "${CMAKE_BINARY_DIR}/test_outputs/batch_cli"
  )
  set_tests_properties(yolo_defect_cpp_batch_cli PROPERTIES
    LABELS "batch;concurrency;integration;quality_gate"
    TIMEOUT 300
  )

  add_test(
    NAME yolo_defect_consistency_unit
    COMMAND "${Python3_EXECUTABLE}" "${CONSISTENCY_UNIT_TEST}" -v
  )
  set_tests_properties(yolo_defect_consistency_unit PROPERTIES
    LABELS "consistency;unit;quality_gate"
  )

  add_test(
    NAME yolo_defect_s2_01_profile_summary_unit
    COMMAND "${Python3_EXECUTABLE}"
      "${S2_01_PROFILE_SUMMARY_UNIT_TEST}" -v
  )
  set_tests_properties(yolo_defect_s2_01_profile_summary_unit PROPERTIES
    LABELS "profiling;unit;quality_gate"
  )

  add_test(
    NAME yolo_defect_s2_01_benchmark_comparison_unit
    COMMAND "${Python3_EXECUTABLE}"
      "${S2_01_BENCHMARK_COMPARISON_UNIT_TEST}" -v
  )
  set_tests_properties(
    yolo_defect_s2_01_benchmark_comparison_unit PROPERTIES
    LABELS "benchmark;comparison;unit;quality_gate"
  )

  add_test(
    NAME yolo_defect_s2_01_quality_unit
    COMMAND "${Python3_EXECUTABLE}" "${S2_01_QUALITY_UNIT_TEST}" -v
  )
  set_tests_properties(yolo_defect_s2_01_quality_unit PROPERTIES
    LABELS "ptq;quality;unit;quality_gate"
  )

  add_test(
    NAME yolo_defect_s2_01_protocol_unit
    COMMAND "${Python3_EXECUTABLE}" "${S2_01_PROTOCOL_UNIT_TEST}" -v
  )
  set_tests_properties(yolo_defect_s2_01_protocol_unit PROPERTIES
    LABELS "ptq;protocol;unit;quality_gate"
  )

  add_test(
    NAME yolo_defect_s2_01_evidence_unit
    COMMAND "${Python3_EXECUTABLE}" "${S2_01_EVIDENCE_UNIT_TEST}" -v
  )
  set_tests_properties(yolo_defect_s2_01_evidence_unit PROPERTIES
    LABELS "ptq;evidence;unit;quality_gate"
  )

  add_test(
    NAME yolo_defect_s2_01_manifest_check
    COMMAND "${Python3_EXECUTABLE}" "${S2_01_MANIFEST_TOOL}" --check
  )
  set_tests_properties(yolo_defect_s2_01_manifest_check PROPERTIES
    LABELS "ptq;manifest;quality_gate"
  )

  add_test(
    NAME yolo_defect_s2_03_batch_tools_unit
    COMMAND "${Python3_EXECUTABLE}"
      "${S2_03_BATCH_TOOLS_UNIT_TEST}" -v
  )
  set_tests_properties(yolo_defect_s2_03_batch_tools_unit PROPERTIES
    LABELS "batch;comparison;unit;quality_gate"
  )

  add_test(
    NAME yolo_defect_s2_04_evidence_tools_unit
    COMMAND "${Python3_EXECUTABLE}"
      "${S2_04_EVIDENCE_TOOLS_UNIT_TEST}" -v
  )
  set_tests_properties(yolo_defect_s2_04_evidence_tools_unit PROPERTIES
    LABELS "tensorrt;comparison;profiling;benchmark;unit;quality_gate"
  )

  add_test(
    NAME yolo_defect_s2_04_product_set_unit
    COMMAND "${Python3_EXECUTABLE}"
      "${S2_04_PRODUCT_SET_UNIT_TEST}" -v
  )
  set_tests_properties(yolo_defect_s2_04_product_set_unit PROPERTIES
    LABELS "tensorrt;correctness;unit;quality_gate"
  )

  add_test(
    NAME yolo_defect_python_cpp_consistency
    COMMAND "${Python3_EXECUTABLE}" "${CONSISTENCY_TOOL}"
      --manifest "${CONSISTENCY_MANIFEST}"
      --cpp-cli $<TARGET_FILE:yolo_defect_cpp>
      --cpp-opencv-version "${OpenCV_VERSION}"
      --output-dir "${CMAKE_BINARY_DIR}/test_outputs/consistency"
  )
  set_tests_properties(yolo_defect_python_cpp_consistency PROPERTIES
    LABELS "consistency;integration;quality_gate"
    PASS_REGULAR_EXPRESSION "Consistency summary: passed=True"
    TIMEOUT 600
  )

  add_test(
    NAME yolo_defect_cpp_benchmark_smoke
    COMMAND "${CMAKE_COMMAND}"
      "-DCLI=$<TARGET_FILE:yolo_defect_cpp>"
      "-DCONFIG=${DEFAULT_CONFIG}"
      "-DIMAGE=${FIXED_IMAGE}"
      "-DOUTPUT_ROOT=${CMAKE_BINARY_DIR}/test_outputs/benchmark"
      "-DPYTHON=${Python3_EXECUTABLE}"
      "-DJSON_VALIDATOR=${TEST_BENCHMARK_JSON_VALIDATOR}"
      -P "${TEST_BENCHMARK_SCRIPT}"
  )
  set_tests_properties(yolo_defect_cpp_benchmark_smoke PROPERTIES
    LABELS "benchmark;integration;quality_gate"
    DEPENDS yolo_defect_python_cpp_consistency
    TIMEOUT 180
  )

  add_test(
    NAME yolo_defect_cpp_profile_smoke
    COMMAND "${CMAKE_COMMAND}"
      "-DCLI=$<TARGET_FILE:yolo_defect_cpp>"
      "-DCONFIG=${DEFAULT_CONFIG}"
      "-DIMAGE=${FIXED_IMAGE}"
      "-DOUTPUT_ROOT=${CMAKE_BINARY_DIR}/test_outputs/profile"
      -P "${TEST_PROFILE_SCRIPT}"
  )
  set_tests_properties(yolo_defect_cpp_profile_smoke PROPERTIES
    LABELS "profiling;integration;quality_gate"
    DEPENDS yolo_defect_python_cpp_consistency
    TIMEOUT 180
  )

  function(add_cli_argument_failure_test test_name expected_text)
    string(JOIN "|" serialized_arguments ${ARGN})
    add_test(
      NAME "${test_name}"
      COMMAND "${CMAKE_COMMAND}"
        "-DCLI=$<TARGET_FILE:yolo_defect_cpp>"
        "-DARGS=${serialized_arguments}"
        "-DEXPECTED_TEXT=${expected_text}"
        -P "${TEST_CLI_ARGUMENT_FAILURE_SCRIPT}"
    )
    set_tests_properties("${test_name}" PROPERTIES
      LABELS "output;cli;negative;quality_gate")
  endfunction()

  function(add_cli_quality_failure_test
      test_name expected_text required_texts)
    string(JOIN "|" serialized_arguments ${ARGN})
    add_test(
      NAME "${test_name}"
      COMMAND "${CMAKE_COMMAND}"
        "-DCLI=$<TARGET_FILE:yolo_defect_cpp>"
        "-DARGS=${serialized_arguments}"
        "-DEXPECTED_TEXT=${expected_text}"
        "-DREQUIRED_TEXTS=${required_texts}"
        -P "${TEST_CLI_ARGUMENT_FAILURE_SCRIPT}"
    )
    set_tests_properties("${test_name}" PROPERTIES
      LABELS "cli;negative;quality_gate")
  endfunction()

  set(CLI_FAILURE_OUTPUT_ROOT
    "${CMAKE_BINARY_DIR}/test_outputs/cli_failures")
  file(MAKE_DIRECTORY "${CLI_FAILURE_OUTPUT_ROOT}")
  set(CLI_FAILURE_DIRECTORY_TARGET
    "${CLI_FAILURE_OUTPUT_ROOT}/existing_directory")
  file(MAKE_DIRECTORY "${CLI_FAILURE_DIRECTORY_TARGET}")

  add_cli_argument_failure_test(
    yolo_defect_cpp_output_missing_value
    "missing value"
    --output-json)
  add_cli_argument_failure_test(
    yolo_defect_cpp_output_duplicate_flag
    "--output-json was provided more than once"
    --output-json "${CLI_FAILURE_OUTPUT_ROOT}/first.json"
    --output-json "${CLI_FAILURE_OUTPUT_ROOT}/second.json")
  add_cli_argument_failure_test(
    yolo_defect_cpp_output_requires_image
    "require --image"
    --config "${DEFAULT_CONFIG}"
    --output-json "${CLI_FAILURE_OUTPUT_ROOT}/missing_image.json")
  add_cli_argument_failure_test(
    yolo_defect_cpp_output_raw_conflict
    "mutually exclusive"
    --config "${DEFAULT_CONFIG}"
    --image "${FIXED_IMAGE}"
    --raw-output-summary
    --output-json "${CLI_FAILURE_OUTPUT_ROOT}/raw_conflict.json")
  add_cli_argument_failure_test(
    yolo_defect_cpp_output_inspect_conflict
    "mutually exclusive"
    --config "${DEFAULT_CONFIG}"
    --image "${FIXED_IMAGE}"
    --inspect-model
    --output-json "${CLI_FAILURE_OUTPUT_ROOT}/inspect_conflict.json")
  add_cli_argument_failure_test(
    yolo_defect_cpp_overwrite_requires_output
    "requires --output-json or --output-image"
    --overwrite)
  add_cli_argument_failure_test(
    yolo_defect_cpp_output_paths_must_differ
    "different JSON and image output files"
    --config "${DEFAULT_CONFIG}"
    --image "${FIXED_IMAGE}"
    --output-json "${CLI_FAILURE_OUTPUT_ROOT}/same_output.dat"
    --output-image "${CLI_FAILURE_OUTPUT_ROOT}/same_output.dat")
  add_cli_argument_failure_test(
    yolo_defect_cpp_output_cannot_overwrite_input
    "protected input"
    --config "${DEFAULT_CONFIG}"
    --image "${FIXED_IMAGE}"
    --output-image "${FIXED_IMAGE}"
    --overwrite)
  add_cli_argument_failure_test(
    yolo_defect_cpp_output_rejects_directory
    "not a directory"
    --config "${DEFAULT_CONFIG}"
    --image "${FIXED_IMAGE}"
    --output-json "${CLI_FAILURE_DIRECTORY_TARGET}")

  add_cli_argument_failure_test(
    yolo_defect_cpp_batch_flag_requires_mode
    "require --batch"
    --workers 2)
  add_cli_argument_failure_test(
    yolo_defect_cpp_batch_requires_config
    "--batch requires --config"
    --batch
    --input-dir "${CMAKE_CURRENT_SOURCE_DIR}/../data/images/val"
    --output-dir "${CLI_FAILURE_OUTPUT_ROOT}/batch_missing_config"
    --batch-summary "${CLI_FAILURE_OUTPUT_ROOT}/batch_missing_config.json")
  add_cli_argument_failure_test(
    yolo_defect_cpp_batch_requires_one_input
    "requires exactly one of --input-dir or --manifest"
    --config "${DEFAULT_CONFIG}"
    --batch
    --output-dir "${CLI_FAILURE_OUTPUT_ROOT}/batch_missing_input"
    --batch-summary "${CLI_FAILURE_OUTPUT_ROOT}/batch_missing_input.json")
  add_cli_argument_failure_test(
    yolo_defect_cpp_batch_rejects_two_inputs
    "requires exactly one of --input-dir or --manifest"
    --config "${DEFAULT_CONFIG}"
    --batch
    --input-dir "${CMAKE_CURRENT_SOURCE_DIR}/../data/images/val"
    --manifest "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/s2_03_consistency_manifest.txt"
    --output-dir "${CLI_FAILURE_OUTPUT_ROOT}/batch_two_inputs"
    --batch-summary "${CLI_FAILURE_OUTPUT_ROOT}/batch_two_inputs.json")
  add_cli_argument_failure_test(
    yolo_defect_cpp_batch_invalid_workers
    "object=--workers"
    --workers 65)
  add_cli_argument_failure_test(
    yolo_defect_cpp_batch_invalid_queue_capacity
    "object=--queue-capacity"
    --queue-capacity 0)
  add_cli_argument_failure_test(
    yolo_defect_cpp_batch_mode_conflict
    "mutually exclusive"
    --config "${DEFAULT_CONFIG}"
    --batch
    --input-dir "${CMAKE_CURRENT_SOURCE_DIR}/../data/images/val"
    --output-dir "${CLI_FAILURE_OUTPUT_ROOT}/batch_conflict"
    --batch-summary "${CLI_FAILURE_OUTPUT_ROOT}/batch_conflict.json"
    --inspect-model)
  add_cli_argument_failure_test(
    yolo_defect_cpp_batch_rejects_tensorrt
    "--batch currently supports provider=cpu only"
    --config "${DEFAULT_TENSORRT_CONFIG}"
    --batch
    --input-dir "${CMAKE_CURRENT_SOURCE_DIR}/../data/images/val"
    --output-dir "${CLI_FAILURE_OUTPUT_ROOT}/batch_tensorrt"
    --batch-summary "${CLI_FAILURE_OUTPUT_ROOT}/batch_tensorrt.json")
  set_tests_properties(
    yolo_defect_cpp_batch_flag_requires_mode
    yolo_defect_cpp_batch_requires_config
    yolo_defect_cpp_batch_requires_one_input
    yolo_defect_cpp_batch_rejects_two_inputs
    yolo_defect_cpp_batch_invalid_workers
    yolo_defect_cpp_batch_invalid_queue_capacity
    yolo_defect_cpp_batch_mode_conflict
    yolo_defect_cpp_batch_rejects_tensorrt
    PROPERTIES LABELS "batch;concurrency;cli;negative;quality_gate")

  add_cli_argument_failure_test(
    yolo_defect_cpp_benchmark_json_requires_mode
    "--benchmark-json requires --benchmark"
    --benchmark-json "${CLI_FAILURE_OUTPUT_ROOT}/benchmark.json")
  add_cli_argument_failure_test(
    yolo_defect_cpp_benchmark_requires_image
    "--benchmark requires --image"
    --config "${DEFAULT_CONFIG}"
    --benchmark
    --benchmark-json "${CLI_FAILURE_OUTPUT_ROOT}/benchmark.json")
  add_cli_argument_failure_test(
    yolo_defect_cpp_benchmark_requires_json
    "--benchmark requires --benchmark-json"
    --config "${DEFAULT_CONFIG}"
    --image "${FIXED_IMAGE}"
    --benchmark)
  add_cli_argument_failure_test(
    yolo_defect_cpp_benchmark_invalid_repeat
    "object=--repeat"
    --config "${DEFAULT_CONFIG}"
    --image "${FIXED_IMAGE}"
    --benchmark
    --repeat 0
    --benchmark-json "${CLI_FAILURE_OUTPUT_ROOT}/benchmark.json")
  add_cli_argument_failure_test(
    yolo_defect_cpp_benchmark_output_conflict
    "mutually exclusive"
    --config "${DEFAULT_CONFIG}"
    --image "${FIXED_IMAGE}"
    --benchmark
    --output-json "${CLI_FAILURE_OUTPUT_ROOT}/detections.json"
    --benchmark-json "${CLI_FAILURE_OUTPUT_ROOT}/benchmark.json")

  add_cli_argument_failure_test(
    yolo_defect_cpp_profile_prefix_requires_mode
    "--profile-prefix requires --profile"
    --profile-prefix "${CLI_FAILURE_OUTPUT_ROOT}/profile")
  add_cli_argument_failure_test(
    yolo_defect_cpp_profile_requires_image
    "--profile requires --image"
    --config "${DEFAULT_CONFIG}"
    --profile
    --profile-prefix "${CLI_FAILURE_OUTPUT_ROOT}/profile")
  add_cli_argument_failure_test(
    yolo_defect_cpp_profile_requires_prefix
    "--profile requires --profile-prefix"
    --config "${DEFAULT_CONFIG}"
    --image "${FIXED_IMAGE}"
    --profile)
  add_cli_argument_failure_test(
    yolo_defect_cpp_profile_invalid_runs
    "object=--profile-runs"
    --config "${DEFAULT_CONFIG}"
    --image "${FIXED_IMAGE}"
    --profile
    --profile-prefix "${CLI_FAILURE_OUTPUT_ROOT}/profile"
    --profile-runs 0)
  add_cli_argument_failure_test(
    yolo_defect_cpp_profile_benchmark_conflict
    "mutually exclusive"
    --config "${DEFAULT_CONFIG}"
    --image "${FIXED_IMAGE}"
    --profile
    --profile-prefix "${CLI_FAILURE_OUTPUT_ROOT}/profile"
    --benchmark
    --benchmark-json "${CLI_FAILURE_OUTPUT_ROOT}/benchmark.json")
  set_tests_properties(
    yolo_defect_cpp_benchmark_json_requires_mode
    yolo_defect_cpp_benchmark_requires_image
    yolo_defect_cpp_benchmark_requires_json
    yolo_defect_cpp_benchmark_invalid_repeat
    yolo_defect_cpp_benchmark_output_conflict
    PROPERTIES LABELS "benchmark;cli;negative;quality_gate")
  set_tests_properties(
    yolo_defect_cpp_profile_prefix_requires_mode
    yolo_defect_cpp_profile_requires_image
    yolo_defect_cpp_profile_requires_prefix
    yolo_defect_cpp_profile_invalid_runs
    yolo_defect_cpp_profile_benchmark_conflict
    PROPERTIES LABELS "profiling;cli;negative;quality_gate")

  add_cli_quality_failure_test(
    yolo_defect_cpp_damaged_image
    "OpenCV decoding returned an empty image"
    "object image_path^expected^actual^action:"
    --config "${DEFAULT_CONFIG}"
    --image "${DAMAGED_IMAGE}")
  add_cli_quality_failure_test(
    yolo_defect_cpp_output_unwritable_parent
    "output.json_path"
    "expected^actual^action:"
    --config "${DEFAULT_CONFIG}"
    --image "${FIXED_IMAGE}"
    --output-json "${BLOCKED_OUTPUT_PARENT}/detections.json")
  set_tests_properties(yolo_defect_cpp_damaged_image PROPERTIES
    LABELS "preprocess;cli;negative;quality_gate")
  set_tests_properties(yolo_defect_cpp_output_unwritable_parent PROPERTIES
    LABELS "output;cli;negative;quality_gate")

  add_test(
    NAME yolo_defect_runner_invalid_input_elements
    COMMAND $<TARGET_FILE:yolo_defect_inference_tests>
      "${DEFAULT_CONFIG}" "${FIXED_IMAGE}"
  )
  set_tests_properties(yolo_defect_runner_invalid_input_elements PROPERTIES
    LABELS "inference;negative;quality_gate")

  function(add_contract_failure_test test_name fixture_name expected_text)
    add_test(
      NAME "${test_name}"
      COMMAND "${CMAKE_COMMAND}"
        "-DCLI=$<TARGET_FILE:yolo_defect_cpp>"
        "-DCONFIG=${TEST_RUNTIME_FIXTURES}/${fixture_name}"
        "-DEXPECTED_TEXT=${expected_text}"
        -P "${TEST_FAILURE_SCRIPT}"
    )
    set_tests_properties("${test_name}" PROPERTIES
      LABELS "contract;negative;quality_gate")
  endfunction()

  function(add_inspection_failure_test
      test_name fixture_name expected_text)
    add_test(
      NAME "${test_name}"
      COMMAND "${CMAKE_COMMAND}"
        "-DCLI=$<TARGET_FILE:yolo_defect_cpp>"
        "-DCONFIG=${TEST_RUNTIME_FIXTURES}/${fixture_name}"
        "-DEXPECTED_TEXT=${expected_text}"
        "-DINSPECT_MODEL=ON"
        -P "${TEST_FAILURE_SCRIPT}"
    )
    set_tests_properties("${test_name}" PROPERTIES
      LABELS "metadata;negative;quality_gate")
  endfunction()

  add_contract_failure_test(
    yolo_defect_cpp_missing_field
    missing_field.txt
    "missing required field")
  add_contract_failure_test(
    yolo_defect_cpp_unknown_field
    unknown_field.txt
    "unknown field")
  add_contract_failure_test(
    yolo_defect_cpp_duplicate_field
    duplicate_field.txt
    "duplicate field")
  add_contract_failure_test(
    yolo_defect_cpp_unsupported_runtime_schema
    unsupported_schema.txt
    "unsupported schema version")
  add_contract_failure_test(
    yolo_defect_cpp_invalid_score_threshold
    invalid_score_threshold.txt
    "invalid number")
  add_contract_failure_test(
    yolo_defect_cpp_invalid_nms_threshold
    invalid_nms_threshold.txt
    "threshold is outside the valid range")
  add_contract_failure_test(
    yolo_defect_cpp_invalid_provider
    invalid_provider.txt
    "unsupported enum value")
  add_contract_failure_test(
    yolo_defect_cpp_missing_tensorrt_cache
    missing_tensorrt_cache.txt
    "missing required field")
  add_contract_failure_test(
    yolo_defect_cpp_invalid_tensorrt_device
    invalid_tensorrt_device.txt
    "device id is negative")
  add_contract_failure_test(
    yolo_defect_cpp_invalid_tensorrt_workspace
    invalid_tensorrt_workspace.txt
    "invalid unsigned integer")
  add_contract_failure_test(
    yolo_defect_cpp_cpu_with_tensorrt_field
    cpu_with_tensorrt_field.txt
    "TensorRT-only field is present for the CPU provider")
  add_contract_failure_test(
    yolo_defect_cpp_empty_class
    empty_class_artifact.txt
    "empty list item")
  add_contract_failure_test(
    yolo_defect_cpp_invalid_artifact_enum
    invalid_enum_artifact.txt
    "unsupported enum value")
  add_contract_failure_test(
    yolo_defect_cpp_missing_artifact_field
    missing_artifact_field.txt
    "missing required field")
  add_contract_failure_test(
    yolo_defect_cpp_unknown_artifact_field
    unknown_artifact_field.txt
    "unknown field")
  add_contract_failure_test(
    yolo_defect_cpp_duplicate_artifact_field
    duplicate_artifact_field.txt
    "duplicate field")
  add_contract_failure_test(
    yolo_defect_cpp_unsupported_artifact_schema
    unsupported_artifact_schema.txt
    "unsupported schema version")
  add_contract_failure_test(
    yolo_defect_cpp_duplicate_artifact_class
    duplicate_artifact_class.txt
    "duplicate class name")
  add_contract_failure_test(
    yolo_defect_cpp_invalid_output_shape
    invalid_shape_artifact.txt
    "output shape is inconsistent with class_names")
  add_contract_failure_test(
    yolo_defect_cpp_invalid_sha256
    invalid_sha_artifact.txt
    "invalid SHA-256 declaration")
  add_contract_failure_test(
    yolo_defect_cpp_missing_model
    missing_model_artifact.txt
    "model artifact does not exist")

  add_inspection_failure_test(
    yolo_defect_cpp_inspect_missing_model
    missing_model_artifact.txt
    "model artifact does not exist")
  add_inspection_failure_test(
    yolo_defect_cpp_inspect_input_shape_mismatch
    metadata_input_shape_mismatch.txt
    "input[0].shape")
  add_inspection_failure_test(
    yolo_defect_cpp_inspect_class_count_mismatch
    metadata_class_count_mismatch.txt
    "output[0].class_channels")

  function(add_metadata_validator_test case_name)
    add_test(
      NAME "yolo_defect_metadata_${case_name}"
      COMMAND $<TARGET_FILE:yolo_defect_metadata_tests>
        "${DEFAULT_CONFIG}" "${case_name}"
    )
    set_tests_properties("yolo_defect_metadata_${case_name}" PROPERTIES
      LABELS "metadata;unit;quality_gate")
  endfunction()

  add_metadata_validator_test(valid)
  add_metadata_validator_test(input_count_mismatch)
  add_metadata_validator_test(output_count_mismatch)
  add_metadata_validator_test(input_name_mismatch)
  add_metadata_validator_test(input_shape_mismatch)
  add_metadata_validator_test(input_dtype_mismatch)
  add_metadata_validator_test(output_dtype_mismatch)
  add_metadata_validator_test(output_name_mismatch)
  add_metadata_validator_test(output_shape_mismatch)
  add_metadata_validator_test(class_count_mismatch)
  add_metadata_validator_test(provider_unavailable)
  add_metadata_validator_test(session_provider_mismatch)

  add_test(
    NAME yolo_defect_metadata_tensorrt_valid
    COMMAND $<TARGET_FILE:yolo_defect_metadata_tests>
      "${DEFAULT_TENSORRT_CONFIG}" valid
  )
  set_tests_properties(yolo_defect_metadata_tensorrt_valid PROPERTIES
    LABELS "metadata;tensorrt;unit;quality_gate")

  foreach(case_name IN ITEMS
      tensorrt_missing_cuda
      tensorrt_wrong_chain
      tensorrt_precision_mismatch)
    add_test(
      NAME "yolo_defect_metadata_${case_name}"
      COMMAND $<TARGET_FILE:yolo_defect_metadata_tests>
        "${DEFAULT_TENSORRT_CONFIG}" "${case_name}"
    )
    set_tests_properties("yolo_defect_metadata_${case_name}" PROPERTIES
      LABELS "metadata;tensorrt;negative;unit;quality_gate")
  endforeach()

  foreach(case_name IN ITEMS
      native_valid
      native_wrong_chain
      native_wrong_cache
      native_wrong_engine
      native_precision_mismatch
      native_execution_policy_mismatch)
    add_test(
      NAME "yolo_defect_metadata_${case_name}"
      COMMAND $<TARGET_FILE:yolo_defect_metadata_tests>
        "${DEFAULT_NATIVE_TENSORRT_CONFIG}" "${case_name}"
    )
    set_tests_properties("yolo_defect_metadata_${case_name}" PROPERTIES
      LABELS "metadata;tensorrt_native;unit;quality_gate")
  endforeach()
endif()
