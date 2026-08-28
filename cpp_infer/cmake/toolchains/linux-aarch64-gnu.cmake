set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

find_program(YOLO_DEFECT_AARCH64_C_COMPILER
  NAMES aarch64-linux-gnu-gcc
  NO_CMAKE_FIND_ROOT_PATH)
find_program(YOLO_DEFECT_AARCH64_CXX_COMPILER
  NAMES aarch64-linux-gnu-g++
  NO_CMAKE_FIND_ROOT_PATH)
find_program(YOLO_DEFECT_AARCH64_AR
  NAMES aarch64-linux-gnu-ar
  NO_CMAKE_FIND_ROOT_PATH)
find_program(YOLO_DEFECT_AARCH64_READELF
  NAMES aarch64-linux-gnu-readelf
  NO_CMAKE_FIND_ROOT_PATH)

foreach(_yolo_defect_host_tool IN ITEMS
    YOLO_DEFECT_AARCH64_C_COMPILER
    YOLO_DEFECT_AARCH64_CXX_COMPILER
    YOLO_DEFECT_AARCH64_AR
    YOLO_DEFECT_AARCH64_READELF)
  if(NOT ${_yolo_defect_host_tool})
    message(FATAL_ERROR
      "Required AArch64 host tool '${_yolo_defect_host_tool}' was not found. "
      "Install gcc-aarch64-linux-gnu, g++-aarch64-linux-gnu, and "
      "binutils-aarch64-linux-gnu.")
  endif()
endforeach()
unset(_yolo_defect_host_tool)

set(CMAKE_C_COMPILER "${YOLO_DEFECT_AARCH64_C_COMPILER}")
set(CMAKE_CXX_COMPILER "${YOLO_DEFECT_AARCH64_CXX_COMPILER}")
set(CMAKE_AR "${YOLO_DEFECT_AARCH64_AR}")
set(CMAKE_READELF "${YOLO_DEFECT_AARCH64_READELF}")
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

execute_process(
  COMMAND "${CMAKE_CXX_COMPILER}" -dumpmachine
  OUTPUT_VARIABLE _yolo_defect_target_triple
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _yolo_defect_target_result)
if(NOT _yolo_defect_target_result EQUAL 0 OR
   NOT _yolo_defect_target_triple MATCHES "^aarch64-linux-gnu$")
  message(FATAL_ERROR
    "AArch64 toolchain mismatch: expected compiler target "
    "'aarch64-linux-gnu', actual '${_yolo_defect_target_triple}'. "
    "Install g++-aarch64-linux-gnu and reconfigure.")
endif()
unset(_yolo_defect_target_result)
unset(_yolo_defect_target_triple)

# Ubuntu's Debian cross compiler reports '/' as its compiler sysroot and already
# knows /usr/aarch64-linux-gnu. Do not set CMAKE_SYSROOT to the private OpenCV
# tree: doing so would hide the compiler's target libc and startup objects.
set(YOLO_DEFECT_AARCH64_SYSROOT "" CACHE PATH
  "Private target-only AArch64 dependency tree (OpenCV headers/libraries)")

set(CMAKE_FIND_ROOT_PATH "/usr/aarch64-linux-gnu")
if(NOT YOLO_DEFECT_AARCH64_SYSROOT STREQUAL "")
  list(PREPEND CMAKE_FIND_ROOT_PATH "${YOLO_DEFECT_AARCH64_SYSROOT}")
endif()

# CMake/Ninja/Python are host tools. Libraries, includes, and packages must
# come from target roots unless a project target names an explicit path.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
