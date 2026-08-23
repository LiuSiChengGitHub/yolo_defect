#ifndef YOLO_DEFECT_CPP_BENCHMARK_WRITER_H_
#define YOLO_DEFECT_CPP_BENCHMARK_WRITER_H_

#include "yolo_defect_cpp/benchmark_result.h"

#include <filesystem>
#include <string>
#include <vector>

namespace yolo_defect_cpp {

// Serializes the validated S1-08 evidence schema with stable field order,
// classic-locale finite numbers, and UTF-8-safe JSON escaping.
std::string serialize_benchmark_json(const BenchmarkResult& result);

// Writes one benchmark JSON document. Relative paths are resolved from the
// process working directory. Existing targets require overwrite_existing and
// must be regular files; directories, symlinks, special files, and protected
// input paths are always rejected.
std::filesystem::path write_benchmark_json(
    const BenchmarkResult& result,
    const std::filesystem::path& output_path,
    bool overwrite_existing,
    const std::vector<std::filesystem::path>& protected_paths);

}  // namespace yolo_defect_cpp

#endif  // YOLO_DEFECT_CPP_BENCHMARK_WRITER_H_
