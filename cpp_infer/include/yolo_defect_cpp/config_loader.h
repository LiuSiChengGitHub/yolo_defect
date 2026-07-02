#ifndef YOLO_DEFECT_CPP_CONFIG_LOADER_H_
#define YOLO_DEFECT_CPP_CONFIG_LOADER_H_

#include <string>
#include <vector>

namespace yolo_defect_cpp {

struct RuntimeConfig {
  int input_width = 0;
  int input_height = 0;
  std::vector<std::string> class_names;
  double score_threshold = 0.0;
  double nms_threshold = 0.0;
  std::string backend;
};

RuntimeConfig load_config(const std::string& config_path);

}  // namespace yolo_defect_cpp

#endif  // YOLO_DEFECT_CPP_CONFIG_LOADER_H_
