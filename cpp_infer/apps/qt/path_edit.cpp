#include "path_edit.h"

#include <QStyleOptionFrame>
#include <QStylePainter>

namespace yolo_defect_cpp::qt {

PathEdit::PathEdit(QWidget* parent) : QLineEdit(parent) {
  connect(this, &QLineEdit::textChanged, this, [this](const QString& value) {
    setToolTip(value);
    update();
  });
}

void PathEdit::paintEvent(QPaintEvent* event) {
  // Native painting owns cursor, selection, placeholder and all editing states.
  if (hasFocus() || hasSelectedText() || text().isEmpty()) {
    QLineEdit::paintEvent(event);
    return;
  }

  QStyleOptionFrame option;
  initStyleOption(&option);
  const QRect text_rect = style()
                              ->subElementRect(QStyle::SE_LineEditContents,
                                               &option, this)
                              .marginsRemoved(textMargins())
                              .adjusted(2, 0, -2, 0);
  const QString display = fontMetrics().elidedText(
      displayText(), Qt::ElideMiddle, qMax(0, text_rect.width()));
  if (display == displayText()) {
    QLineEdit::paintEvent(event);
    return;
  }

  QStylePainter painter(this);
  // The style still paints the QSS background, border and padding.
  painter.drawPrimitive(QStyle::PE_PanelLineEdit, option);
  painter.setClipRect(text_rect);
  painter.drawItemText(
      text_rect,
      QStyle::visualAlignment(layoutDirection(), alignment()) | Qt::AlignVCenter,
      option.palette, isEnabled(), display, QPalette::Text);
}

}  // namespace yolo_defect_cpp::qt
