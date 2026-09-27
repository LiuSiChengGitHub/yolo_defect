#ifndef YOLO_DEFECT_QT_PATH_EDIT_H_
#define YOLO_DEFECT_QT_PATH_EDIT_H_

#include <QLineEdit>

namespace yolo_defect_cpp::qt {

// Keep the editable value intact while showing both ends of a long path at rest.
class PathEdit : public QLineEdit {
 public:
  explicit PathEdit(QWidget* parent = nullptr);

 protected:
  void paintEvent(QPaintEvent* event) override;
};

}  // namespace yolo_defect_cpp::qt
#endif
