#ifndef YOLO_DEFECT_QT_TABLE_DELEGATES_H_
#define YOLO_DEFECT_QT_TABLE_DELEGATES_H_

#include <QStyledItemDelegate>

namespace yolo_defect_cpp::qt {

// Paints the display text as a rounded status pill tinted with the model's
// Qt::ForegroundRole color. Row background and selection stay with the style.
class StatusPillDelegate : public QStyledItemDelegate {
 public:
  using QStyledItemDelegate::QStyledItemDelegate;
  void paint(QPainter* painter, const QStyleOptionViewItem& option,
             const QModelIndex& index) const override;
  QSize sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const override;
};

// Keeps the normal percentage text and adds a thin bar for the 0..1 score
// provided by the model as Qt::UserRole.
class ConfidenceBarDelegate : public QStyledItemDelegate {
 public:
  using QStyledItemDelegate::QStyledItemDelegate;
  void paint(QPainter* painter, const QStyleOptionViewItem& option,
             const QModelIndex& index) const override;
};

}  // namespace yolo_defect_cpp::qt
#endif
