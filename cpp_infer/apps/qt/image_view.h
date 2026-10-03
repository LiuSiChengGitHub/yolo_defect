#ifndef YOLO_DEFECT_QT_IMAGE_VIEW_H_
#define YOLO_DEFECT_QT_IMAGE_VIEW_H_

#include "yolo_defect_cpp/detection_result.h"

#include <QIcon>
#include <QImage>
#include <QPointF>
#include <QRectF>
#include <QWidget>
#include <vector>

namespace yolo_defect_cpp::qt {

// Images cross threads as owned QImage values. View transforms and the
// selection overlay stay on the GUI thread and never alter output images.
class ImageView : public QWidget {
  Q_OBJECT

 public:
  explicit ImageView(QWidget* parent = nullptr);
  void setImage(QImage image);
  void clear(const QString& message);
  void setDetections(std::vector<Detection> detections);
  void setSelectedDetection(int index);
  int selectedDetection() const { return selected_detection_; }
  double zoomFactor() const { return image_.isNull() ? 0.0 : zoom_; }
  QSize imageSize() const { return image_.size(); }

 public slots:
  void zoomIn();
  void zoomOut();
  void fitToWindow();
  void actualSize();

 signals:
  void detectionSelected(int index);
  void zoomChanged(double factor);

 protected:
  void paintEvent(QPaintEvent* event) override;
  void resizeEvent(QResizeEvent* event) override;
  void wheelEvent(QWheelEvent* event) override;
  void mousePressEvent(QMouseEvent* event) override;
  void mouseMoveEvent(QMouseEvent* event) override;
  void mouseReleaseEvent(QMouseEvent* event) override;
  void mouseDoubleClickEvent(QMouseEvent* event) override;

 private:
  QRectF viewportRect() const;
  QRectF imageRect() const;
  double fitScale() const;
  void zoomAt(double factor, const QPointF& anchor);
  void constrainCenter();
  int detectionAt(const QPointF& image_position) const;

  QImage image_;
  QString empty_message_;
  QIcon empty_icon_;
  std::vector<Detection> detections_;
  int selected_detection_ = -1;
  double zoom_ = 1.0;
  QPointF center_;
  bool fit_mode_ = true;
  bool pressed_ = false;
  bool dragging_ = false;
  QPointF press_position_;
  QPointF last_position_;
};

}  // namespace yolo_defect_cpp::qt
#endif
