#ifndef YOLO_DEFECT_QT_SEGMENTED_CONTROL_H_
#define YOLO_DEFECT_QT_SEGMENTED_CONTROL_H_

#include <QFrame>
#include <QStringList>

class QButtonGroup;

namespace yolo_defect_cpp::qt {

// A row of mutually exclusive segments with a QComboBox-like index API. The
// change signal fires only when the index really changes, also for setCurrentIndex().
class SegmentedControl : public QFrame {
  Q_OBJECT
  Q_PROPERTY(int currentIndex READ currentIndex WRITE setCurrentIndex NOTIFY currentIndexChanged)

 public:
  explicit SegmentedControl(const QStringList& labels, QWidget* parent = nullptr);
  int currentIndex() const { return current_; }
  void setCurrentIndex(int index);
  void setSegmentToolTip(int index, const QString& tip);

 signals:
  void currentIndexChanged(int index);

 private:
  QButtonGroup* group_ = nullptr;
  int current_ = 0;
};

}  // namespace yolo_defect_cpp::qt
#endif
