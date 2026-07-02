cmake_minimum_required(VERSION 3.16)

project(yolo_defect_cpp
  VERSION 0.1.0
  DESCRIPTION "C++ runtime skeleton for steel surface defect detection"
  LANGUAGES CXX
)

find_package(OpenCV 4 REQUIRED)

add_executable(yolo_defect_cpp
  src/config_loader.cpp
  src/image_preprocessor.cpp
  src/main.cpp
)

target_compile_features(yolo_defect_cpp PRIVATE cxx_std_17)
target_include_directories(yolo_defect_cpp PRIVATE include)
target_link_libraries(yolo_defect_cpp PRIVATE ${OpenCV_LIBS})

set_target_properties(yolo_defect_cpp PROPERTIES
  CXX_EXTENSIONS OFF
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

include(CTest)

if(BUILD_TESTING)
  add_test(
    NAME yolo_defect_cpp_help
    COMMAND yolo_defect_cpp --help
  )

  set_tests_properties(yolo_defect_cpp_help PROPERTIES
    PASS_REGULAR_EXPRESSION "P1-01 CMake skeleton"
  )

  add_test(
    NAME yolo_defect_cpp_config
    COMMAND yolo_defect_cpp --config "${CMAKE_CURRENT_SOURCE_DIR}/configs/default_config.txt"
  )

  set_tests_properties(yolo_defect_cpp_config PROPERTIES
    PASS_REGULAR_EXPRESSION "P1-02 Config summary"
  )

  add_test(
    NAME yolo_defect_cpp_preprocess
    COMMAND yolo_defect_cpp
      --config "${CMAKE_CURRENT_SOURCE_DIR}/configs/default_config.txt"
      --image "${CMAKE_CURRENT_SOURCE_DIR}/../data/images/val/crazing_241.jpg"
  )

  set_tests_properties(yolo_defect_cpp_preprocess PROPERTIES
    PASS_REGULAR_EXPRESSION "P1-03 Preprocess summary"
  )
endif()
