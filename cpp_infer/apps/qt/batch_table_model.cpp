#include "batch_table_model.h"
#include "detection_types.h"
#include "theme.h"

#include <QColor>
#include <QFileInfo>
#include <QStringList>
#include <utility>

namespace yolo_defect_cpp::qt {
BatchTableModel::BatchTableModel(QObject* parent) : QAbstractTableModel(parent) {}

int BatchTableModel::rowCount(const QModelIndex& parent) const {
  return parent.isValid() ? 0 : static_cast<int>(items_.size());
}

int BatchTableModel::columnCount(const QModelIndex& parent) const {
  return parent.isValid() ? 0 : 6;
}

const BatchItemResult* BatchTableModel::itemAt(int row) const {
  return row >= 0 && row < rowCount() ? &items_[row] : nullptr;
}

QVariant BatchTableModel::data(const QModelIndex& index, int role) const {
  const auto* item = index.isValid() ? itemAt(index.row()) : nullptr;
  if (!item || index.column() < 0 || index.column() >= columnCount()) return {};
  if (role == Qt::ToolTipRole) {
    return from_path(item->source_path) + (item->error.empty() ? QString{} :
        "\n" + QString::fromStdString(item->error));
  }
  if (role == Qt::TextAlignmentRole) {
    return static_cast<int>((index.column() == 1 || index.column() == 5)
        ? Qt::AlignLeft | Qt::AlignVCenter : Qt::AlignCenter);
  }
  if (role == Qt::ForegroundRole && index.column() == 2) {
    const auto& colors = theme::palette();
    if (item->status == BatchItemStatus::kFailed) return colors.danger;
    if (item->status == BatchItemStatus::kCancelled) return colors.warning;
    return colors.success;
  }
  if (role != Qt::DisplayRole) return {};
  switch (index.column()) {
    case 0: return static_cast<qulonglong>(item->sequence_index + 1);
    case 1: return QFileInfo(from_path(item->source_path)).fileName();
    case 2:
      switch (item->status) {
        case BatchItemStatus::kSucceeded: return tr("成功");
        case BatchItemStatus::kFailed: return tr("失败");
        case BatchItemStatus::kCancelled: return tr("已取消");
      }
      return {};
    case 3: return item->status == BatchItemStatus::kSucceeded
        ? QVariant(static_cast<qulonglong>(item->detection_count)) : QVariant(QStringLiteral("—"));
    case 4: return item->status == BatchItemStatus::kCancelled ? QStringLiteral("—")
        : QString::number(item->latency_ms, 'f', 1);
    case 5: return item->error.empty() ? tr("已保存 JSON / PNG") : QString::fromStdString(item->error);
    default: return {};
  }
}

QVariant BatchTableModel::headerData(int section, Qt::Orientation orientation, int role) const {
  if (orientation != Qt::Horizontal || role != Qt::DisplayRole) return {};
  const QStringList headers = {"#", tr("图片"), tr("状态"), tr("目标数"), tr("耗时 ms"), tr("说明")};
  return section >= 0 && section < headers.size() ? QVariant(headers[section]) : QVariant{};
}

void BatchTableModel::setItems(std::vector<BatchItemResult> items) {
  beginResetModel();
  items_ = std::move(items);
  endResetModel();
}
}  // namespace yolo_defect_cpp::qt
