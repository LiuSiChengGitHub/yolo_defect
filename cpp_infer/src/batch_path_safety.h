#ifndef YOLO_DEFECT_CPP_BATCH_PATH_SAFETY_H_
#define YOLO_DEFECT_CPP_BATCH_PATH_SAFETY_H_

#include <filesystem>
#include <system_error>

namespace yolo_defect_cpp {
namespace internal {

struct BatchPathLocationLess {
  bool operator()(const std::filesystem::path& lhs,
                  const std::filesystem::path& rhs) const;
};

// Windows uses wide-character ordinal ignore-case comparison; POSIX keeps
// native case-sensitive path semantics. These private helpers expose the
// exact production comparison contract to focused unit tests.
bool batch_path_text_equal(const std::filesystem::path& lhs,
                           const std::filesystem::path& rhs);

bool batch_path_is_same_or_descendant(
    const std::filesystem::path& candidate,
    const std::filesystem::path& root);

bool batch_path_is_reparse_point(
    const std::filesystem::path& path,
    std::error_code& error) noexcept;

}  // namespace internal
}  // namespace yolo_defect_cpp

#endif  // YOLO_DEFECT_CPP_BATCH_PATH_SAFETY_H_
