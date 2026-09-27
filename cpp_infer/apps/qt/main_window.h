#ifndef YOLO_DEFECT_QT_MAIN_WINDOW_H_
#define YOLO_DEFECT_QT_MAIN_WINDOW_H_

#include "detection_types.h"

#include <QMainWindow>

class QLabel;
class QLineEdit;
class QProgressBar;
class QPushButton;
class QThread;

namespace yolo_defect_cpp::qt {

class DetectionTableModel;
class ImageView;

class MainWindow : public QMainWindow {
  Q_OBJECT

 public:
  explicit MainWindow(QWidget* parent = nullptr);
  ~MainWindow() override;
  bool isBusy() const { return thread_ != nullptr; }
  void setInputs(const QString& config, const QString& image,
                 const QString& output_directory);

 public slots:
  void startDetection();

 signals:
  void taskFinished(bool success);

 protected:
  void closeEvent(QCloseEvent* event) override;

 private:
  void buildUi();
  void invalidateResult();
  void setBusy(bool busy);
  void showContract(const RuntimeContract& contract);
  void showResult(const DetectionResponse& response);
  void showError(const QString& message);
  void openPath(const QString& path);

  QWidget* input_panel_ = nullptr;
  QLineEdit* config_path_ = nullptr;
  QLineEdit* image_path_ = nullptr;
  QLineEdit* output_directory_ = nullptr;
  QPushButton* run_button_ = nullptr;
  QPushButton* open_output_button_ = nullptr;
  QPushButton* open_json_button_ = nullptr;
  QLabel* model_details_ = nullptr;
  QLabel* status_message_ = nullptr;
  QLabel* state_badge_ = nullptr;
  QLabel* result_summary_ = nullptr;
  QLabel* output_details_ = nullptr;
  QProgressBar* progress_ = nullptr;
  ImageView* original_view_ = nullptr;
  ImageView* annotated_view_ = nullptr;
  DetectionTableModel* result_model_ = nullptr;
  QThread* thread_ = nullptr;
  QString completed_directory_;
  QString completed_json_;
  bool task_succeeded_ = false;
  bool close_pending_ = false;
};

}  // namespace yolo_defect_cpp::qt
#endif
