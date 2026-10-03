#include "detection_table_model.h"

#include <QString>
#include <QStringList>
#include <utility>

namespace yolo_defect_cpp::qt {

DetectionTableModel::DetectionTableModel(QObject* parent)
    : QAbstractTableModel(parent) {}

int DetectionTableModel::rowCount(const QModelIndex& parent) const {
  return parent.isValid() ? 0 : static_cast<int>(detections_.size());
}

int DetectionTableModel::columnCount(const QModelIndex& parent) const {
  return parent.isValid() ? 0 : 7;
}

QVariant DetectionTableModel::data(const QModelIndex& index, int role) const {
  if (!index.isValid() || index.row() < 0 || index.row() >= rowCount() ||
      index.column() < 0 || index.column() >= columnCount()) {
    return {};
  }
  if (role == Qt::TextAlignmentRole) {
    return static_cast<int>(index.column() == 1 ? Qt::AlignLeft | Qt::AlignVCenter
                                              : Qt::AlignCenter);
  }
  const auto& detection = detections_[index.row()];
  // The raw score lets the confidence column draw a bar under its text.
  if (role == Qt::UserRole && index.column() == 2) return detection.confidence;
  if (role != Qt::DisplayRole) return {};
  switch (index.column()) {
    case 0: return index.row() + 1;
    case 1: return QString::fromStdString(detection.class_name);
    case 2: return QString::number(detection.confidence * 100.0, 'f', 2) + "%";
    case 3: return QString::number(detection.bbox_xyxy.x1, 'f', 1);
    case 4: return QString::number(detection.bbox_xyxy.y1, 'f', 1);
    case 5: return QString::number(detection.bbox_xyxy.x2, 'f', 1);
    case 6: return QString::number(detection.bbox_xyxy.y2, 'f', 1);
    default: return {};
  }
}

QVariant DetectionTableModel::headerData(int section, Qt::Orientation orientation,
                                       int role) const {
  if (role != Qt::DisplayRole || orientation != Qt::Horizontal) return {};
  const QStringList headers = {"#", tr("类别"), tr("置信度"),
                               "x1", "y1", "x2", "y2"};
  return section >= 0 && section < headers.size() ? headers[section] : QVariant{};
}

void DetectionTableModel::setDetections(std::vector<Detection> detections) {
  beginResetModel();
  detections_ = std::move(detections);
  endResetModel();
}

const Detection* DetectionTableModel::detectionAt(int row) const {
  return row >= 0 && row < rowCount() ? &detections_[row] : nullptr;
}

}  // namespace yolo_defect_cpp::qt
