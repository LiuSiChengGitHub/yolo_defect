#ifndef YOLO_DEFECT_CPP_BATCH_WRITER_H_
#define YOLO_DEFECT_CPP_BATCH_WRITER_H_

#include "yolo_defect_cpp/batch_result.h"

#include <filesystem>
#include <string>

namespace yolo_defect_cpp {

std::string serialize_batch_summary_json(const BatchSummary& summary);

void write_batch_summary_json(
    const BatchSummary& summary,
    const std::filesystem::path& output_path,
    bool overwrite_existing = false);

}  // namespace yolo_defect_cpp

#endif  // YOLO_DEFECT_CPP_BATCH_WRITER_H_
