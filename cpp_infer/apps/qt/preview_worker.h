#ifndef YOLO_DEFECT_CPP_QT_PREVIEW_WORKER_H_
#define YOLO_DEFECT_CPP_QT_PREVIEW_WORKER_H_

#include "batch_types.h"

#include <QObject>

namespace yolo_defect_cpp::qt {

// Reads only existing outputs; result browsing never reruns inference.
class PreviewWorker final : public QObject {
  Q_OBJECT

 public:
  explicit PreviewWorker(QObject* parent = nullptr);

 public slots:
  void load(PreviewRequest request);

 signals:
  void completed(PreviewResponse response);
};

}  // namespace yolo_defect_cpp::qt

#endif  // YOLO_DEFECT_CPP_QT_PREVIEW_WORKER_H_
