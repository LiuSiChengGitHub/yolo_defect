#ifndef YOLO_DEFECT_CPP_QT_MODEL_INFO_PANEL_H_
#define YOLO_DEFECT_CPP_QT_MODEL_INFO_PANEL_H_

#include <QFrame>

class QLabel;
class QToolButton;

namespace yolo_defect_cpp {
struct RuntimeContract;
}

namespace yolo_defect_cpp::qt {

class ModelInfoPanel : public QFrame {
 public:
  explicit ModelInfoPanel(QWidget* parent = nullptr);

  void setContract(const RuntimeContract& contract);
  void reset();

 private:
  QLabel* model_id_ = nullptr;
  QLabel* provider_ = nullptr;
  QLabel* input_shape_ = nullptr;
  QLabel* score_threshold_ = nullptr;
  QLabel* nms_threshold_ = nullptr;
  QLabel* nms_mode_ = nullptr;
  QLabel* class_names_ = nullptr;
  QLabel* model_path_ = nullptr;
  QLabel* config_path_ = nullptr;
  QToolButton* details_toggle_ = nullptr;
  QWidget* extra_details_ = nullptr;
};

}  // namespace yolo_defect_cpp::qt

#endif  // YOLO_DEFECT_CPP_QT_MODEL_INFO_PANEL_H_
