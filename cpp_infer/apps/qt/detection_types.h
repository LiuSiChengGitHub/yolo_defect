#ifndef YOLO_DEFECT_CPP_QT_DETECTION_TYPES_H_
#define YOLO_DEFECT_CPP_QT_DETECTION_TYPES_H_

#include "yolo_defect_cpp/detector_pipeline.h"

#include <QImage>
#include <QMetaType>
#include <QString>

#include <filesystem>

namespace yolo_defect_cpp::qt {

inline std::filesystem::path to_path(const QString& value) {
#ifdef _WIN32
  return std::filesystem::path(value.toStdWString());
#else
  return std::filesystem::u8path(value.toUtf8().toStdString());
#endif
}

inline QString from_path(const std::filesystem::path& value) {
#ifdef _WIN32
  return QString::fromStdWString(value.wstring());
#else
  return QString::fromStdString(value.u8string());
#endif
}

struct DetectionRequest {
  QString config_path;
  QString image_path;
  QString output_directory;
};

struct DetectionResponse {
  RuntimeContract contract;
  SingleImagePipelineResult result;
  QImage original_image;
  QImage annotated_image;
  QString output_directory;
  // Wall time for this client task, including loading, writing and previews.
  // This is deliberately separate from Runtime benchmark measurements.
  double elapsed_ms = 0.0;
};

}  // namespace yolo_defect_cpp::qt

Q_DECLARE_METATYPE(yolo_defect_cpp::RuntimeContract)
Q_DECLARE_METATYPE(yolo_defect_cpp::qt::DetectionRequest)
Q_DECLARE_METATYPE(yolo_defect_cpp::qt::DetectionResponse)

#endif  // YOLO_DEFECT_CPP_QT_DETECTION_TYPES_H_
