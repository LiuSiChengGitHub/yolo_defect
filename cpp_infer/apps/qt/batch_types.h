#ifndef YOLO_DEFECT_CPP_QT_BATCH_TYPES_H_
#define YOLO_DEFECT_CPP_QT_BATCH_TYPES_H_

#include "detection_types.h"
#include "yolo_defect_cpp/batch_result.h"

#include <cstddef>
#include <vector>

namespace yolo_defect_cpp::qt {

struct BatchDetectionRequest {
  QString config_path;
  QString input_path;
  QString output_directory;
  BatchInputKind input_kind = BatchInputKind::kDirectory;
  std::size_t workers = 1;
  std::size_t queue_capacity = 2;
};

struct BatchDetectionResponse {
  RuntimeContract contract;
  BatchSummary summary;
  QString output_directory;
  QString summary_path;
  // Includes configuration loading, worker/session creation and summary I/O.
  // Runtime's processing duration remains available in summary.timing.
  double elapsed_ms = 0.0;
};

struct PreviewRequest {
  quint64 generation = 0;
  BatchItemResult item;
};

struct PreviewResponse {
  quint64 generation = 0;
  QImage original_image;
  QImage annotated_image;
  std::vector<Detection> detections;
  QString error;
};

}  // namespace yolo_defect_cpp::qt

Q_DECLARE_METATYPE(yolo_defect_cpp::qt::BatchDetectionRequest)
Q_DECLARE_METATYPE(yolo_defect_cpp::qt::BatchDetectionResponse)
Q_DECLARE_METATYPE(yolo_defect_cpp::qt::PreviewRequest)
Q_DECLARE_METATYPE(yolo_defect_cpp::qt::PreviewResponse)

#endif  // YOLO_DEFECT_CPP_QT_BATCH_TYPES_H_
