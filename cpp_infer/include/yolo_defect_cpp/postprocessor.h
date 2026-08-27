#ifndef YOLO_DEFECT_CPP_POSTPROCESSOR_H_
#define YOLO_DEFECT_CPP_POSTPROCESSOR_H_

#include "yolo_defect_cpp/config_loader.h"
#include "yolo_defect_cpp/project_core.h"

namespace yolo_defect_cpp {

std::vector<Detection> postprocess_yolov8_raw(
    const InferenceOutput& output,
    const RuntimeContract& contract,
    const PreprocessResult& preprocess);

}  // namespace yolo_defect_cpp

#endif  // YOLO_DEFECT_CPP_POSTPROCESSOR_H_
