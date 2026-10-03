#ifndef YOLO_DEFECT_QT_MAIN_WINDOW_H_
#define YOLO_DEFECT_QT_MAIN_WINDOW_H_

#include "detection_types.h"
#include "batch_types.h"

#include <QMainWindow>
#include <memory>
#include <optional>

class QLabel;
class QLineEdit;
class QProgressBar;
class QPlainTextEdit;
class QPushButton;
class QThread;
class QSpinBox;
class QTableView;
class QTabWidget;

namespace yolo_defect_cpp::qt {

class DetectionTableModel;
class ImageView;
class ModelInfoPanel;
class BatchTaskControl;
class BatchTableModel;
class SegmentedControl;

class MainWindow : public QMainWindow {
  Q_OBJECT

 public:
  explicit MainWindow(QWidget* parent = nullptr);
  ~MainWindow() override;
  bool isBusy() const { return thread_ != nullptr; }
  void setInputs(const QString& config, const QString& image,
                 const QString& output_directory);
  void setBatchInputs(const QString& config, const QString& input,
                      const QString& output_directory,
                      BatchInputKind kind = BatchInputKind::kDirectory,
                      int workers = 1, int queue_capacity = 2);

 public slots:
  void startDetection();
  void stopDetection();

 signals:
  void taskFinished(bool success);
  void previewRequested(PreviewRequest request);

 protected:
  void showEvent(QShowEvent* event) override;
  void closeEvent(QCloseEvent* event) override;

 private:
  void buildUi();
  void updateInputMode();
  void invalidateResult();
  void setBusy(bool busy);
  void showContract(const RuntimeContract& contract);
  void showResult(const DetectionResponse& response);
  void showBatchResult(const BatchDetectionResponse& response);
  void selectBatchItem(int row);
  void dispatchPreview();
  void showPreview(const PreviewResponse& response);
  void setDetections(const std::vector<Detection>& detections);
  void selectDetection(int row);
  void setState(const QString& text, const char* state);
  void showError(const QString& message);
  void openPath(const QString& path);

  QWidget* input_panel_ = nullptr;
  QWidget* input_fields_ = nullptr;
  SegmentedControl* input_mode_ = nullptr;
  QLabel* input_label_ = nullptr;
  QWidget* batch_options_ = nullptr;
  QSpinBox* workers_ = nullptr;
  QSpinBox* queue_capacity_ = nullptr;
  QLineEdit* config_path_ = nullptr;
  QLineEdit* image_path_ = nullptr;
  QLineEdit* output_directory_ = nullptr;
  QPushButton* run_button_ = nullptr;
  QPushButton* stop_button_ = nullptr;
  QPushButton* open_output_button_ = nullptr;
  QPushButton* open_json_button_ = nullptr;
  QPushButton* open_summary_button_ = nullptr;
  ModelInfoPanel* model_panel_ = nullptr;
  QLabel* status_message_ = nullptr;
  QLabel* state_badge_ = nullptr;
  QLabel* result_summary_ = nullptr;
  QLabel* batch_summary_ = nullptr;
  QLabel* item_details_ = nullptr;
  QPlainTextEdit* item_error_ = nullptr;
  QTabWidget* result_tabs_ = nullptr;
  QTableView* detection_table_ = nullptr;
  QTableView* batch_table_ = nullptr;
  BatchTableModel* batch_model_ = nullptr;
  QProgressBar* progress_ = nullptr;
  ImageView* original_view_ = nullptr;
  ImageView* annotated_view_ = nullptr;
  DetectionTableModel* result_model_ = nullptr;
  QThread* thread_ = nullptr;
  QThread* preview_thread_ = nullptr;
  std::shared_ptr<BatchTaskControl> batch_control_;
  std::optional<PreviewRequest> pending_preview_;
  quint64 preview_generation_ = 0;
  bool preview_in_flight_ = false;
  QString completed_directory_;
  QString completed_json_;
  QString completed_summary_;
  bool task_succeeded_ = false;
  bool stop_requested_ = false;
  bool task_result_received_ = false;
  bool close_pending_ = false;
  bool frame_styled_ = false;
};

}  // namespace yolo_defect_cpp::qt
#endif
