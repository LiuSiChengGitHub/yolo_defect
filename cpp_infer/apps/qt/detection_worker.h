#ifndef YOLO_DEFECT_CPP_QT_DETECTION_WORKER_H_
#define YOLO_DEFECT_CPP_QT_DETECTION_WORKER_H_

#include "detection_types.h"

#include <QObject>

namespace yolo_defect_cpp::qt {

class DetectionWorker final : public QObject {
  Q_OBJECT

 public:
  explicit DetectionWorker(QObject* parent = nullptr);

 public slots:
  void run(DetectionRequest request);

 signals:
  void stageChanged(QString stage);
  void contractLoaded(RuntimeContract contract);
  void completed(DetectionResponse response);
  void failed(QString message);
  void finished();
};

}  // namespace yolo_defect_cpp::qt

#endif  // YOLO_DEFECT_CPP_QT_DETECTION_WORKER_H_
