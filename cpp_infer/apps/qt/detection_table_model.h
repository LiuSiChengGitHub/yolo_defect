#ifndef YOLO_DEFECT_QT_DETECTION_TABLE_MODEL_H_
#define YOLO_DEFECT_QT_DETECTION_TABLE_MODEL_H_

#include "yolo_defect_cpp/detection_result.h"

#include <QAbstractTableModel>
#include <vector>

namespace yolo_defect_cpp::qt {

class DetectionTableModel : public QAbstractTableModel {
 public:
  explicit DetectionTableModel(QObject* parent = nullptr);
  int rowCount(const QModelIndex& parent = {}) const override;
  int columnCount(const QModelIndex& parent = {}) const override;
  QVariant data(const QModelIndex& index, int role) const override;
  QVariant headerData(int section, Qt::Orientation orientation,
                      int role) const override;
  void setDetections(std::vector<Detection> detections);
  const Detection* detectionAt(int row) const;

 private:
  std::vector<Detection> detections_;
};

}  // namespace yolo_defect_cpp::qt
#endif
