#ifndef YOLO_DEFECT_CPP_QT_BATCH_WORKER_H_
#define YOLO_DEFECT_CPP_QT_BATCH_WORKER_H_

#include "batch_types.h"
#include "yolo_defect_cpp/batch_runner.h"

#include <QObject>

#include <memory>
#include <mutex>

namespace yolo_defect_cpp::qt {

// Shared by the GUI and one batch worker. This is deliberately not a QObject:
// cancellation must run immediately, even while run() occupies the worker's
// event loop. Create a new control (and runner) for each task.
class BatchTaskControl final {
 public:
  void requestStop();
  void installRunner(std::shared_ptr<BatchRunner> runner);

 private:
  std::mutex mutex_;
  std::shared_ptr<BatchRunner> runner_;
  bool stop_requested_ = false;
};

class BatchWorker final : public QObject {
  Q_OBJECT

 public:
  explicit BatchWorker(std::shared_ptr<BatchTaskControl> control,
                       QObject* parent = nullptr);

 public slots:
  void run(BatchDetectionRequest request);

 signals:
  void stageChanged(QString stage);
  void contractLoaded(RuntimeContract contract);
  void completed(BatchDetectionResponse response);
  void failed(QString message);
  void finished();

 private:
  std::shared_ptr<BatchTaskControl> control_;
};

}  // namespace yolo_defect_cpp::qt

#endif  // YOLO_DEFECT_CPP_QT_BATCH_WORKER_H_
