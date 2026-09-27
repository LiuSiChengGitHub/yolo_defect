#ifndef YOLO_DEFECT_CPP_QT_TASK_IO_H_
#define YOLO_DEFECT_CPP_QT_TASK_IO_H_

#include <QImage>
#include <QString>

#include <filesystem>

namespace yolo_defect_cpp::qt {

// Decode with the same OpenCV orientation/format handling as Runtime and
// return pixels owned by QImage. QFile preserves Unicode paths on Windows.
QImage read_preview(const std::filesystem::path& path);

// Reserve a new directory for one client invocation. It is safe for both
// single-image outputs and BatchRunner's output preflight.
QString create_task_directory(const QString& output_directory);

}  // namespace yolo_defect_cpp::qt

#endif  // YOLO_DEFECT_CPP_QT_TASK_IO_H_
