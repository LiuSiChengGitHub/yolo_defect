#include "detection_worker.h"

#include <QDateTime>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QUuid>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>

#include <stdexcept>
#include <utility>
#include <vector>

namespace yolo_defect_cpp::qt {
namespace {

QImage read_preview(const std::filesystem::path& path) {
  QFile file(from_path(path));
  if (!file.open(QIODevice::ReadOnly)) {
    throw std::runtime_error(
        QStringLiteral("Cannot open image preview: %1 (%2)")
            .arg(file.fileName(), file.errorString()).toStdString());
  }
  const QByteArray encoded = file.readAll();
  if (file.error() != QFileDevice::NoError) {
    throw std::runtime_error(
        QStringLiteral("Cannot read image preview: %1 (%2)")
            .arg(file.fileName(), file.errorString()).toStdString());
  }
  // Match Runtime's decoder and orientation handling, including formats that
  // the deployed Qt image plugins may not support. QFile keeps Unicode paths.
  const std::vector<unsigned char> bytes(encoded.begin(), encoded.end());
  const cv::Mat bgr = cv::imdecode(bytes, cv::IMREAD_COLOR);
  if (bgr.empty()) {
    throw std::runtime_error(
        QStringLiteral("OpenCV could not decode image preview: %1")
            .arg(file.fileName()).toStdString());
  }
  // The copy owns its pixels after the temporary OpenCV matrix is released.
  return QImage(bgr.data, bgr.cols, bgr.rows,
                static_cast<qsizetype>(bgr.step), QImage::Format_BGR888).copy();
}

QString create_task_directory(const QString& output_directory) {
  QDir root(QDir(output_directory).absolutePath());
  if (!root.mkpath(QStringLiteral("."))) {
    throw std::runtime_error(
        QStringLiteral("Cannot create output directory: %1")
            .arg(root.absolutePath()).toStdString());
  }
  const QString name =
      QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd_HHmmss_zzz")) +
      QLatin1Char('_') + QUuid::createUuid().toString(QUuid::Id128);
  if (!root.mkdir(name)) {
    throw std::runtime_error(
        QStringLiteral("Cannot create task output directory: %1")
            .arg(root.filePath(name)).toStdString());
  }
  return root.filePath(name);
}

}  // namespace

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
