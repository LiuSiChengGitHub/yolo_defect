#include "batch_worker.h"

#include "task_io.h"
#include "yolo_defect_cpp/batch_writer.h"

#include <QCoreApplication>
#include <QDir>
#include <QElapsedTimer>
#include <QStringList>

#include <stdexcept>
#include <utility>

namespace yolo_defect_cpp::qt {

void BatchTaskControl::requestStop() {
  std::shared_ptr<BatchRunner> runner;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_requested_ = true;
    runner = runner_;
  }
  if (runner) {
    runner->request_stop();
  }
}

void BatchTaskControl::installRunner(std::shared_ptr<BatchRunner> runner) {
  bool stop_requested;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    runner_ = runner;
    stop_requested = stop_requested_;
  }
  // A click during configuration loading must not be lost before publication.
  if (stop_requested) {
    runner->request_stop();
  }
}

BatchWorker::BatchWorker(std::shared_ptr<BatchTaskControl> control,
                         QObject* parent)
    : QObject(parent), control_(std::move(control)) {}

void BatchWorker::run(BatchDetectionRequest request) {
  QElapsedTimer timer;
  timer.start();
  try {
    if (request.config_path.trimmed().isEmpty() ||
        request.input_path.trimmed().isEmpty() ||
        request.output_directory.trimmed().isEmpty()) {
      throw std::invalid_argument(
          "Select a Runtime configuration, batch input and output directory.");
    }

    emit stageChanged(tr("正在读取并校验配置…"));
    BatchDetectionResponse response;
    response.contract = load_runtime_contract(to_path(request.config_path));
    emit contractLoaded(response.contract);
    response.output_directory = create_task_directory(request.output_directory);
    response.summary_path = QDir(response.output_directory)
                                .filePath(QStringLiteral("batch_summary.json"));

    BatchRequest batch;
    batch.input_kind = request.input_kind;
    batch.input_path = to_path(request.input_path);
    batch.output_directory = to_path(response.output_directory);
    batch.summary_path = to_path(response.summary_path);
    batch.requested_workers = request.workers;
    batch.queue_capacity = request.queue_capacity;
    batch.output_images = true;
    for (const QString& argument : QCoreApplication::arguments()) {
      batch.command_arguments.push_back(argument.toUtf8().toStdString());
    }

    auto runner = std::make_shared<BatchRunner>(response.contract);
    control_->installRunner(runner);
    emit stageChanged(tr("正在批量检测…完成后显示逐图结果与汇总"));
    // Runtime owns bounded queueing, worker sessions and their joins. No Qt
    // per-image scheduler is involved, and no control mutex is held here.
    response.summary = runner->run(batch);
    emit stageChanged(tr("正在保存批处理汇总…"));
    write_batch_summary_json(response.summary, batch.summary_path);
    response.elapsed_ms = static_cast<double>(timer.nsecsElapsed()) / 1.0e6;
    emit completed(std::move(response));
  } catch (const std::exception& error) {
    emit failed(QString::fromUtf8(error.what()));
  } catch (...) {
    emit failed(tr("An unknown error occurred while running batch detection."));
  }
  emit finished();
}

}  // namespace yolo_defect_cpp::qt
