#include "image_view.h"

#include <QPainter>
#include <utility>

namespace yolo_defect_cpp::qt {

ImageView::ImageView(QWidget* parent) : QWidget(parent) {
  setMinimumSize(220, 200);
  setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  empty_message_ = tr("选择图片并运行检测");
}

void ImageView::setImage(QImage image) {
  image_ = std::move(image);
  update();
}

void ImageView::clear(const QString& message) {
  image_ = {};
  empty_message_ = message;
  update();
}

void ImageView::paintEvent(QPaintEvent*) {
  QPainter painter(this);
  painter.fillRect(rect(), QColor("#152334"));
  painter.setPen(QColor("#1c2c3e"));
  for (int x = 0; x < width(); x += 24) painter.drawLine(x, 0, x, height());
  for (int y = 0; y < height(); y += 24) painter.drawLine(0, y, width(), y);
  if (image_.isNull()) {
    painter.setPen(QColor("#9bacbe"));
    painter.drawText(rect().adjusted(20, 20, -20, -20),
                     Qt::AlignCenter | Qt::TextWordWrap, empty_message_);
    return;
  }
  const auto available = size() - QSize(32, 48);
  const auto scaled = image_.size().scaled(available, Qt::KeepAspectRatio);
  const QRect target(QPoint((width() - scaled.width()) / 2,
                           (height() - 20 - scaled.height()) / 2), scaled);
  painter.setRenderHint(QPainter::SmoothPixmapTransform);
  painter.drawImage(target, image_);
  painter.setPen(QColor("#9bacbe"));
  painter.drawText(rect().adjusted(12, 0, -12, -6), Qt::AlignBottom | Qt::AlignRight,
                   tr("%1 × %2 px  ·  适应窗口").arg(image_.width()).arg(image_.height()));
}

}  // namespace yolo_defect_cpp::qt
