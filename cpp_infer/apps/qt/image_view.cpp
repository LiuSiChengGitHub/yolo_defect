#include "image_view.h"

#include <QApplication>
#include <QMouseEvent>
#include <QPainter>
#include <QResizeEvent>
#include <QWheelEvent>
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace yolo_defect_cpp::qt {
namespace {

QRectF detectionRect(const Detection& detection) {
  const auto& box = detection.bbox_xyxy;
  return QRectF(QPointF(box.x1, box.y1), QPointF(box.x2, box.y2)).normalized();
}

}  // namespace

ImageView::ImageView(QWidget* parent) : QWidget(parent) {
  setMinimumSize(220, 120);
  setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  setMouseTracking(true);
  setToolTip(tr("滚轮缩放 · 拖动平移 · 点击检测框选择 · 双击适应窗口"));
  empty_message_ = tr("选择图片并运行检测");
}

void ImageView::setImage(QImage image) {
  image_ = std::move(image);
  detections_.clear();
  selected_detection_ = -1;
  pressed_ = dragging_ = false;
  fit_mode_ = true;
  center_ = QPointF(image_.width() / 2.0, image_.height() / 2.0);
  zoom_ = fitScale();
  setCursor(image_.isNull() ? Qt::ArrowCursor : Qt::OpenHandCursor);
  emit zoomChanged(zoomFactor());
  update();
}

void ImageView::clear(const QString& message) {
  empty_message_ = message;
  setImage({});
}

void ImageView::setDetections(std::vector<Detection> detections) {
  detections_ = std::move(detections);
  selected_detection_ = -1;
  update();
}

void ImageView::setSelectedDetection(int index) {
  const int selection = index >= 0 && index < static_cast<int>(detections_.size())
                            ? index : -1;
  if (selection == selected_detection_) return;
  selected_detection_ = selection;
  update();
}

QRectF ImageView::viewportRect() const {
  return QRectF(rect()).adjusted(16, 16, -16, -32);
}

QRectF ImageView::imageRect() const {
  return QRectF(viewportRect().center() - center_ * zoom_,
                QSizeF(image_.width() * zoom_, image_.height() * zoom_));
}

double ImageView::fitScale() const {
  if (image_.isNull()) return 1.0;
  const auto viewport = viewportRect();
  return std::min(viewport.width() / image_.width(),
                  viewport.height() / image_.height());
}

void ImageView::constrainCenter() {
  const auto viewport = viewportRect();
  const double half_width = viewport.width() / (2.0 * zoom_);
  const double half_height = viewport.height() / (2.0 * zoom_);
  center_.setX(half_width >= image_.width() / 2.0
                   ? image_.width() / 2.0
                   : std::clamp(center_.x(), half_width, image_.width() - half_width));
  center_.setY(half_height >= image_.height() / 2.0
                   ? image_.height() / 2.0
                   : std::clamp(center_.y(), half_height, image_.height() - half_height));
}

void ImageView::zoomAt(double factor, const QPointF& anchor) {
  if (image_.isNull()) return;
  const QPointF image_anchor = (anchor - imageRect().topLeft()) / zoom_;
  // Always include the fit scale, even for an unusually large or tiny image.
  zoom_ = std::clamp(factor, std::min(0.02, fitScale()),
                     std::max(16.0, fitScale()));
  fit_mode_ = false;
  center_ = image_anchor - (anchor - viewportRect().center()) / zoom_;
  constrainCenter();
  emit zoomChanged(zoom_);
  update();
}

void ImageView::zoomIn() { zoomAt(zoom_ * 1.25, viewportRect().center()); }
void ImageView::zoomOut() { zoomAt(zoom_ / 1.25, viewportRect().center()); }
void ImageView::actualSize() { zoomAt(1.0, viewportRect().center()); }

void ImageView::fitToWindow() {
  if (image_.isNull()) return;
  fit_mode_ = true;
  center_ = QPointF(image_.width() / 2.0, image_.height() / 2.0);
  zoom_ = fitScale();
  emit zoomChanged(zoom_);
  update();
}

void ImageView::resizeEvent(QResizeEvent* event) {
  QWidget::resizeEvent(event);
  if (image_.isNull()) return;
  if (fit_mode_) {
    fitToWindow();
  } else {
    constrainCenter();
    update();
  }
}

void ImageView::wheelEvent(QWheelEvent* event) {
  if (image_.isNull() || !viewportRect().contains(event->position())) {
    event->ignore();
    return;
  }
  const double steps = event->pixelDelta().isNull()
                           ? event->angleDelta().y() / 120.0
                           : event->pixelDelta().y() / 60.0;
  if (steps != 0.0) zoomAt(zoom_ * std::pow(1.25, steps), event->position());
  event->accept();
}

void ImageView::mousePressEvent(QMouseEvent* event) {
  if (event->button() != Qt::LeftButton || image_.isNull() ||
      !viewportRect().contains(event->position())) {
    QWidget::mousePressEvent(event);
    return;
  }
  pressed_ = true;
  dragging_ = false;
  press_position_ = last_position_ = event->position();
  event->accept();
}

void ImageView::mouseMoveEvent(QMouseEvent* event) {
  if (!pressed_) {
    setCursor(!image_.isNull() && viewportRect().contains(event->position())
                  ? Qt::OpenHandCursor : Qt::ArrowCursor);
    QWidget::mouseMoveEvent(event);
    return;
  }
  if (!dragging_ && (event->position() - press_position_).manhattanLength() >=
                        QApplication::startDragDistance()) {
    dragging_ = true;
    setCursor(Qt::ClosedHandCursor);
  }
  if (dragging_) {
    center_ -= (event->position() - last_position_) / zoom_;
    last_position_ = event->position();
    constrainCenter();
    update();
  }
  event->accept();
}

int ImageView::detectionAt(const QPointF& image_position) const {
  int selected = -1;
  double smallest_area = std::numeric_limits<double>::max();
  for (int i = 0; i < static_cast<int>(detections_.size()); ++i) {
    const auto box = detectionRect(detections_[i]);
    const double area = box.width() * box.height();
    if (box.contains(image_position) && area < smallest_area) {
      selected = i;
      smallest_area = area;
    }
  }
  return selected;
}

void ImageView::mouseReleaseEvent(QMouseEvent* event) {
  if (event->button() != Qt::LeftButton || !pressed_) {
    QWidget::mouseReleaseEvent(event);
    return;
  }
  if (!dragging_ && viewportRect().contains(event->position())) {
    const int selected = imageRect().contains(event->position())
        ? detectionAt((event->position() - imageRect().topLeft()) / zoom_) : -1;
    setSelectedDetection(selected);
    // A repeated click is still a navigation action (e.g. reopening the
    // detection tab after browsing the batch list).
    emit detectionSelected(selected);
  }
  pressed_ = dragging_ = false;
  setCursor(viewportRect().contains(event->position())
                ? Qt::OpenHandCursor : Qt::ArrowCursor);
  event->accept();
}

void ImageView::mouseDoubleClickEvent(QMouseEvent* event) {
  if (event->button() == Qt::LeftButton && !image_.isNull() &&
      viewportRect().contains(event->position())) {
    pressed_ = dragging_ = false;
    fitToWindow();
    event->accept();
    return;
  }
  QWidget::mouseDoubleClickEvent(event);
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

  const auto target = imageRect();
  painter.save();
  painter.setClipRect(viewportRect());
  painter.setRenderHint(QPainter::SmoothPixmapTransform);
  painter.drawImage(target, image_);
  if (selected_detection_ >= 0) {
    const auto box = detectionRect(detections_[selected_detection_]);
    const QRectF selected_box(target.topLeft() + box.topLeft() * zoom_,
                              box.size() * zoom_);
    painter.setClipRect(target, Qt::IntersectClip);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setPen(QPen(QColor("#152334"), 5));
    painter.drawRect(selected_box);
    painter.setPen(QPen(QColor("#66f2d3"), 2));
    painter.drawRect(selected_box);
    painter.setBrush(QColor("#66f2d3"));
    for (const auto& corner : {selected_box.topLeft(), selected_box.topRight(),
                              selected_box.bottomLeft(), selected_box.bottomRight()}) {
      painter.drawRect(QRectF(corner - QPointF(2, 2), QSizeF(4, 4)));
    }
    const QString label = tr("选中 #%1").arg(selected_detection_ + 1);
    const QSizeF label_size(painter.fontMetrics().horizontalAdvance(label) + 12,
                            painter.fontMetrics().height() + 6);
    const QRectF label_bounds = viewportRect().intersected(target).adjusted(2, 2, -2, -2);
    if (label_bounds.width() >= label_size.width() &&
        label_bounds.height() >= label_size.height()) {
      const QPointF label_position(
          std::clamp(selected_box.left() + 3, label_bounds.left(),
                     label_bounds.right() - label_size.width()),
          std::clamp(selected_box.top() + 3, label_bounds.top(),
                     label_bounds.bottom() - label_size.height()));
      const QRectF label_box(label_position, label_size);
      painter.setPen(Qt::NoPen);
      painter.drawRoundedRect(label_box, 3, 3);
      painter.setPen(QColor("#102c2b"));
      painter.drawText(label_box, Qt::AlignCenter, label);
    }
  }
  painter.restore();

  const QString mode = fit_mode_ ? tr("适应窗口")
      : (std::abs(zoom_ - 1.0) < 0.001 ? tr("原始尺寸") : tr("自由缩放"));
  const QString caption = tr("%1 × %2 px  ·  %3%  ·  %4")
      .arg(image_.width()).arg(image_.height())
      .arg(zoom_ * 100.0, 0, 'f', zoom_ < 0.1 ? 1 : 0).arg(mode);
  painter.setPen(QColor("#9bacbe"));
  const QRect caption_rect = rect().adjusted(12, height() - 28, -12, -6);
  painter.drawText(caption_rect, Qt::AlignRight | Qt::AlignVCenter,
      painter.fontMetrics().elidedText(caption, Qt::ElideRight, caption_rect.width()));
}

}  // namespace yolo_defect_cpp::qt
