#include "detection_worker.h"

#include "task_io.h"

#include <QElapsedTimer>

#include <stdexcept>
#include <utility>

namespace yolo_defect_cpp::qt {

DetectionWorker::DetectionWorker(QObject* parent) : QObject(parent) {}

void DetectionWorker::run(DetectionRequest request) {
  QElapsedTimer timer;
  timer.start();
  try {
    if (request.config_path.trimmed().isEmpty() ||
        request.image_path.trimmed().isEmpty() ||
        request.output_directory.trimmed().isEmpty()) {
      throw std::invalid_argument(
          "Select a Runtime configuration, an input image and an output directory.");
    }

    emit stageChanged(tr("正在读取并校验配置…"));
    DetectionResponse response;
    response.contract = load_runtime_contract(to_path(request.config_path));
    emit contractLoaded(response.contract);

    emit stageChanged(tr("正在加载模型…"));
    DetectorPipeline pipeline(response.contract);
    response.output_directory = create_task_directory(request.output_directory);
    const auto task_directory = to_path(response.output_directory);
    DetectionOutputRequest outputs;
    outputs.json_path = task_directory / "detections.json";
    outputs.image_path = task_directory / "detections.png";

    emit stageChanged(tr("正在检测并保存结果…"));
    response.result = pipeline.run(to_path(request.image_path), outputs);

    emit stageChanged(tr("正在准备图像预览…"));
    response.original_image =
        read_preview(response.result.detection_result.image.source_path);
    response.annotated_image = read_preview(*response.result.outputs.image_path);
    response.elapsed_ms = static_cast<double>(timer.nsecsElapsed()) / 1.0e6;
    emit completed(std::move(response));
  } catch (const std::exception& error) {
    emit failed(QString::fromUtf8(error.what()));
  } catch (...) {
    emit failed(tr("An unknown error occurred while running detection."));
  }
  emit finished();
}

}  // namespace yolo_defect_cpp::qt
