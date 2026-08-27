#ifndef YOLO_DEFECT_CPP_PLATFORM_INFO_H_
#define YOLO_DEFECT_CPP_PLATFORM_INFO_H_

#include "yolo_defect_cpp/benchmark_result.h"

#include <cstddef>
#include <string>

namespace yolo_defect_cpp {
namespace internal {

struct PlatformInfo {
  std::string hostname;
  std::string processor;
  std::string architecture;
  std::size_t logical_cpu_count = 0;
  std::string os_name;
  std::string os_version;
};

std::string utc_timestamp();

PlatformInfo collect_platform_info();

BenchmarkMemoryEvidence query_peak_process_memory();

}  // namespace internal
}  // namespace yolo_defect_cpp

#endif  // YOLO_DEFECT_CPP_PLATFORM_INFO_H_
