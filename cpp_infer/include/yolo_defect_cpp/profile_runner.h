#ifndef YOLO_DEFECT_CPP_PROFILE_RUNNER_H_
#define YOLO_DEFECT_CPP_PROFILE_RUNNER_H_

#include "yolo_defect_cpp/config_loader.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace yolo_defect_cpp {

struct ProfileRequest {
  std::filesystem::path image_path;
  std::filesystem::path profile_file_prefix;
  std::size_t run_count = 10;
};

struct ProfileResult {
  std::filesystem::path trace_path;
  std::size_t run_count = 0;
  std::string model_id;
  std::string declared_model_sha256;
  std::string actual_provider;
  double session_initialization_ms = 0.0;
  std::vector<std::int64_t> output_shape;
  std::size_t output_element_count = 0;
  std::size_t detection_count = 0;
};

class ProfileRunner {
 public:
  explicit ProfileRunner(RuntimeContract contract);
  ~ProfileRunner();

  ProfileRunner(const ProfileRunner&) = delete;
  ProfileRunner& operator=(const ProfileRunner&) = delete;
  ProfileRunner(ProfileRunner&&) noexcept;
  ProfileRunner& operator=(ProfileRunner&&) noexcept;

  ProfileResult run(const ProfileRequest& request);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace yolo_defect_cpp

#endif  // YOLO_DEFECT_CPP_PROFILE_RUNNER_H_
