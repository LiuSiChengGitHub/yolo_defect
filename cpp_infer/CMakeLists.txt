cmake_minimum_required(VERSION 3.16)

project(yolo_defect_cpp
  VERSION 0.6.0
  DESCRIPTION "C++ ONNX single-image detection runtime and stable outputs"
  LANGUAGES CXX
)

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR
    "S1-05 requires a 64-bit toolchain for the pinned Windows x64 "
    "ONNX Runtime SDK. Re-enter the x64 Visual Studio developer environment.")
endif()

find_package(OpenCV 4.8 REQUIRED COMPONENTS core imgcodecs imgproc)

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
    "A Python wheel DLL is not a complete C++ SDK.")
endif()

get_filename_component(ONNXRUNTIME_ROOT "${ONNXRUNTIME_ROOT}" ABSOLUTE)
set(ONNXRUNTIME_VERSION_FILE "${ONNXRUNTIME_ROOT}/VERSION_NUMBER")
set(ONNXRUNTIME_C_API_HEADER
  "${ONNXRUNTIME_ROOT}/include/onnxruntime_c_api.h")
set(ONNXRUNTIME_CXX_API_HEADER
  "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
set(ONNXRUNTIME_CPU_PROVIDER_HEADER
  "${ONNXRUNTIME_ROOT}/include/cpu_provider_factory.h")
set(ONNXRUNTIME_IMPORT_LIBRARY
  "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
set(ONNXRUNTIME_RUNTIME_DLL
  "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll")

foreach(required_file
    ONNXRUNTIME_VERSION_FILE
    ONNXRUNTIME_C_API_HEADER
    ONNXRUNTIME_CXX_API_HEADER
    ONNXRUNTIME_CPU_PROVIDER_HEADER
    ONNXRUNTIME_IMPORT_LIBRARY
    ONNXRUNTIME_RUNTIME_DLL)
  if(NOT EXISTS "${${required_file}}" OR IS_DIRECTORY "${${required_file}}")
    message(FATAL_ERROR
      "Incomplete ONNX Runtime C++ SDK: ${required_file} is missing at "
      "'${${required_file}}'. Expected VERSION_NUMBER, C/C++ headers, "
      "import library, and runtime DLL under ONNXRUNTIME_ROOT.")
  endif()
endforeach()

file(READ "${ONNXRUNTIME_VERSION_FILE}" ONNXRUNTIME_VERSION)
string(STRIP "${ONNXRUNTIME_VERSION}" ONNXRUNTIME_VERSION)
if(NOT ONNXRUNTIME_VERSION STREQUAL "1.19.2")
  message(FATAL_ERROR
    "Unsupported ONNX Runtime SDK version. Expected '1.19.2', actual "
    "'${ONNXRUNTIME_VERSION}'. Point ONNXRUNTIME_ROOT at the pinned SDK.")
endif()

add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
  IMPORTED_IMPLIB "${ONNXRUNTIME_IMPORT_LIBRARY}"
  IMPORTED_LOCATION "${ONNXRUNTIME_RUNTIME_DLL}"
  INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include"
)

add_library(yolo_defect_runtime STATIC
  src/artifact_spec.cpp
  src/config_loader.cpp
  src/detector_pipeline.cpp
  src/image_preprocessor.cpp
  src/key_value_parser.cpp
  src/model_metadata.cpp
  src/onnx_runner.cpp
  src/postprocessor.cpp
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
  PRIVATE
    ${OpenCV_LIBS}
    onnxruntime::onnxruntime
)
set_target_properties(yolo_defect_runtime PROPERTIES
  CXX_EXTENSIONS OFF
)

add_executable(yolo_defect_cpp
  src/main.cpp
)
target_compile_features(yolo_defect_cpp PRIVATE cxx_std_17)
target_link_libraries(yolo_defect_cpp PRIVATE yolo_defect::runtime)
set_target_properties(yolo_defect_cpp PROPERTIES
  CXX_EXTENSIONS OFF
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

function(stage_onnxruntime_dll target_name)
  add_custom_command(TARGET "${target_name}" POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
      "$<TARGET_FILE:onnxruntime::onnxruntime>"
      "$<TARGET_FILE_DIR:${target_name}>"
    COMMENT
      "Staging the pinned ONNX Runtime DLL for ${target_name}"
    VERBATIM
  )
endfunction()

stage_onnxruntime_dll(yolo_defect_cpp)

include(CTest)

if(BUILD_TESTING)
  include(FetchContent)

  set(gtest_force_shared_crt ON CACHE BOOL
    "Use the parent MSVC runtime library for GoogleTest" FORCE)
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

  set(DEFAULT_CONFIG
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/default_config.txt")
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
  set(FIXED_IMAGE
    "${CMAKE_CURRENT_SOURCE_DIR}/../data/images/val/crazing_241.jpg")

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
  stage_onnxruntime_dll(yolo_defect_inference_tests)

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
    PROPERTIES LABELS "postprocess"
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
    PROPERTIES LABELS "preprocess"
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
    PROPERTIES LABELS "output"
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
    PASS_REGULAR_EXPRESSION "S1-05 single-image detection CLI"
  )

  add_test(
    NAME yolo_defect_cpp_config
    COMMAND $<TARGET_FILE:yolo_defect_cpp> --config "${DEFAULT_CONFIG}"
  )
  set_tests_properties(yolo_defect_cpp_config PROPERTIES
    PASS_REGULAR_EXPRESSION "S1-01 Runtime/artifact contract summary"
  )

  add_test(
    NAME yolo_defect_cpp_preprocess
    COMMAND $<TARGET_FILE:yolo_defect_cpp>
      --config "${DEFAULT_CONFIG}"
      --image "${FIXED_IMAGE}"
  )
  set_tests_properties(yolo_defect_cpp_preprocess PROPERTIES
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

  add_test(
    NAME yolo_defect_cpp_inspect_model
    COMMAND "${CMAKE_COMMAND}"
      "-DCLI=$<TARGET_FILE:yolo_defect_cpp>"
      "-DCONFIG=${DEFAULT_CONFIG}"
      -P "${TEST_INSPECTION_SCRIPT}"
  )

  add_test(
    NAME yolo_defect_cpp_raw_output_summary
    COMMAND "${CMAKE_COMMAND}"
      "-DCLI=$<TARGET_FILE:yolo_defect_cpp>"
      "-DCONFIG=${DEFAULT_CONFIG}"
      "-DIMAGE=${FIXED_IMAGE}"
      -P "${TEST_RAW_OUTPUT_SCRIPT}"
  )

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
    LABELS "output;integration"
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
    set_tests_properties("${test_name}" PROPERTIES LABELS "output;negative")
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

  add_test(
    NAME yolo_defect_runner_invalid_input_elements
    COMMAND $<TARGET_FILE:yolo_defect_inference_tests>
      "${DEFAULT_CONFIG}" "${FIXED_IMAGE}"
  )

  function(add_contract_failure_test test_name fixture_name expected_text)
    add_test(
      NAME "${test_name}"
      COMMAND "${CMAKE_COMMAND}"
        "-DCLI=$<TARGET_FILE:yolo_defect_cpp>"
        "-DCONFIG=${TEST_RUNTIME_FIXTURES}/${fixture_name}"
        "-DEXPECTED_TEXT=${expected_text}"
        -P "${TEST_FAILURE_SCRIPT}"
    )
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
    yolo_defect_cpp_empty_class
    empty_class_artifact.txt
    "empty list item")
  add_contract_failure_test(
    yolo_defect_cpp_invalid_artifact_enum
    invalid_enum_artifact.txt
    "unsupported enum value")
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
  endfunction()

  add_metadata_validator_test(valid)
  add_metadata_validator_test(input_count_mismatch)
  add_metadata_validator_test(output_count_mismatch)
  add_metadata_validator_test(input_name_mismatch)
  add_metadata_validator_test(input_shape_mismatch)
  add_metadata_validator_test(input_dtype_mismatch)
  add_metadata_validator_test(output_name_mismatch)
  add_metadata_validator_test(output_shape_mismatch)
  add_metadata_validator_test(class_count_mismatch)
  add_metadata_validator_test(provider_unavailable)
endif()
