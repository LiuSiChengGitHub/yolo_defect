#include "table_delegates.h"

#include "theme.h"

#include <QApplication>
#include <QPainter>
#include <algorithm>

namespace yolo_defect_cpp::qt {
namespace {

constexpr int kPillPadding = 16;

}  // namespace

void StatusPillDelegate::paint(QPainter* painter, const QStyleOptionViewItem& option,
                               const QModelIndex& index) const {
  QStyleOptionViewItem item(option);
  initStyleOption(&item, index);
  const QString text = item.text;
  item.text.clear();
  const QWidget* widget = item.widget;
  QStyle* style = widget ? widget->style() : QApplication::style();
  style->drawControl(QStyle::CE_ItemViewItem, &item, painter, widget);
  if (text.isEmpty()) return;

  QColor color = index.data(Qt::ForegroundRole).value<QColor>();
  if (!color.isValid()) color = item.palette.color(QPalette::Text);
  QColor fill = color;
  fill.setAlpha(30);
  const QFontMetrics metrics(item.font);
  QRectF pill(0, 0, metrics.horizontalAdvance(text) + kPillPadding, metrics.height() + 4);
  pill.moveCenter(QRectF(item.rect).center());
  painter->save();
  painter->setRenderHint(QPainter::Antialiasing);
  painter->setPen(Qt::NoPen);
  painter->setBrush(fill);
  painter->drawRoundedRect(pill, pill.height() / 2, pill.height() / 2);
  painter->setFont(item.font);
  painter->setPen(color);
  painter->drawText(pill, Qt::AlignCenter, text);
  painter->restore();
}

QSize StatusPillDelegate::sizeHint(const QStyleOptionViewItem& option,
                                   const QModelIndex& index) const {
  QSize size = QStyledItemDelegate::sizeHint(option, index);
  size.rwidth() += kPillPadding;
  return size;
}

void ConfidenceBarDelegate::paint(QPainter* painter, const QStyleOptionViewItem& option,
                                  const QModelIndex& index) const {
  QStyledItemDelegate::paint(painter, option, index);
  bool valid = false;
  const double score = index.data(Qt::UserRole).toDouble(&valid);
  if (!valid) return;
  const auto& colors = theme::palette();
  // A short bar centered under the percentage reads as part of that value.
  const qreal width = std::clamp(option.rect.width() - 24.0, 0.0, 80.0);
  const QRectF track(option.rect.center().x() - width / 2, option.rect.bottom() - 4.5,
                     width, 3);
  QRectF level = track;
  level.setWidth(track.width() * std::clamp(score, 0.0, 1.0));
  painter->save();
  painter->setRenderHint(QPainter::Antialiasing);
  painter->setPen(Qt::NoPen);
  painter->setBrush(colors.border);
  painter->drawRoundedRect(track, 1.5, 1.5);
  painter->setBrush(colors.accent);
  painter->drawRoundedRect(level, 1.5, 1.5);
  painter->restore();
}

}  // namespace yolo_defect_cpp::qt
