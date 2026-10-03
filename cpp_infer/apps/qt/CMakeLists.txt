find_package(Qt6 6.2 REQUIRED COMPONENTS Widgets)

add_library(yolo_defect_qt_client STATIC
  batch_types.h
  batch_worker.cpp
  batch_worker.h
  batch_table_model.cpp
  batch_table_model.h
  detection_types.h
  detection_worker.cpp
  detection_worker.h
  detection_table_model.cpp
  detection_table_model.h
  image_view.cpp
  image_view.h
  main_window.cpp
  main_window.h
  model_info_panel.cpp
  model_info_panel.h
  path_edit.cpp
  path_edit.h
  preview_worker.cpp
  preview_worker.h
  table_delegates.cpp
  table_delegates.h
  task_io.cpp
  task_io.h
  theme.cpp
  theme.h
  resources/qt_client.qrc
)
target_compile_features(yolo_defect_qt_client PUBLIC cxx_std_17)
target_include_directories(yolo_defect_qt_client
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
  PRIVATE ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(yolo_defect_qt_client
  PUBLIC Qt6::Widgets yolo_defect::runtime
  PRIVATE ${OpenCV_LIBS}
)
set_target_properties(yolo_defect_qt_client PROPERTIES
  AUTOMOC ON
  AUTORCC ON
  CXX_EXTENSIONS OFF
)
if(MSVC)
  target_compile_options(yolo_defect_qt_client PUBLIC /utf-8)
endif()

add_executable(yolo_defect_qt WIN32 main.cpp)
target_link_libraries(yolo_defect_qt PRIVATE yolo_defect_qt_client)
set_target_properties(yolo_defect_qt PROPERTIES
  CXX_EXTENSIONS OFF
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
configure_onnxruntime_runtime(yolo_defect_qt)

if(BUILD_TESTING)
  find_package(Qt6 6.2 QUIET COMPONENTS Test)
  if(TARGET Qt6::Test)
    # Optional, repeatable showcase capture. It reuses the real client and
    # Runtime; it is not part of CTest or the shipped application.
    add_executable(yolo_defect_qt_capture EXCLUDE_FROM_ALL ../../tools/qt_demo_capture.cpp)
    target_link_libraries(yolo_defect_qt_capture PRIVATE yolo_defect_qt_client Qt6::Test)
    set_target_properties(yolo_defect_qt_capture PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    configure_onnxruntime_runtime(yolo_defect_qt_capture)
    add_executable(yolo_defect_qt_tests ../../tests/qt_client_test.cpp)
    target_link_libraries(yolo_defect_qt_tests
      PRIVATE yolo_defect_qt_client Qt6::Test)
    get_filename_component(_qt_test_repo_root
      "${PROJECT_SOURCE_DIR}/.." ABSOLUTE)
    target_compile_definitions(yolo_defect_qt_tests PRIVATE
      "YOLO_DEFECT_QT_TEST_REPO_ROOT=\"${_qt_test_repo_root}\""
      "YOLO_DEFECT_QT_TEST_CLI=\"$<TARGET_FILE:yolo_defect_cpp>\""
    )
    unset(_qt_test_repo_root)
    set_target_properties(yolo_defect_qt_tests PROPERTIES
      AUTOMOC ON
      CXX_EXTENSIONS OFF
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_dependencies(yolo_defect_qt_tests yolo_defect_cpp)
    configure_onnxruntime_runtime(yolo_defect_qt_tests)
    add_test(NAME yolo_defect_qt_client
      COMMAND $<TARGET_FILE:yolo_defect_qt_tests>
        -o "${CMAKE_BINARY_DIR}/qt_client_test.txt,txt" -o -,txt)
    set_tests_properties(yolo_defect_qt_client PROPERTIES
      LABELS "qt;integration"
      ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
      TIMEOUT 180
    )
  else()
    message(STATUS "Qt6::Test unavailable: skipping Qt client tests")
  endif()
endif()
