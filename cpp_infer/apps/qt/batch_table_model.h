#ifndef YOLO_DEFECT_QT_BATCH_TABLE_MODEL_H_
#define YOLO_DEFECT_QT_BATCH_TABLE_MODEL_H_

#include "yolo_defect_cpp/batch_result.h"
#include <QAbstractTableModel>

namespace yolo_defect_cpp::qt {

// Preserve Runtime discovery/manifest order, including failed/cancelled items.
class BatchTableModel : public QAbstractTableModel {
 public:
  explicit BatchTableModel(QObject* parent = nullptr);
  int rowCount(const QModelIndex& parent = {}) const override;
  int columnCount(const QModelIndex& parent = {}) const override;
  QVariant data(const QModelIndex& index, int role) const override;
  QVariant headerData(int section, Qt::Orientation orientation, int role) const override;
  void setItems(std::vector<BatchItemResult> items);
  const BatchItemResult* itemAt(int row) const;

 private:
  std::vector<BatchItemResult> items_;
};

}  // namespace yolo_defect_cpp::qt
#endif
