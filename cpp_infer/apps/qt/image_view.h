#ifndef YOLO_DEFECT_QT_IMAGE_VIEW_H_
#define YOLO_DEFECT_QT_IMAGE_VIEW_H_

#include <QImage>
#include <QWidget>

namespace yolo_defect_cpp::qt {

// A fit-to-window preview. Images cross threads as owned QImage values;
// painting and all widget access stay on the GUI thread.
class ImageView : public QWidget {
 public:
  explicit ImageView(QWidget* parent = nullptr);
  void setImage(QImage image);
  void clear(const QString& message);

 protected:
  void paintEvent(QPaintEvent* event) override;

 private:
  QImage image_;
  QString empty_message_;
};

}  // namespace yolo_defect_cpp::qt
#endif
