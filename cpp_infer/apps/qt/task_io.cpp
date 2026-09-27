#include "task_io.h"

#include "detection_types.h"

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QUuid>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>

#include <stdexcept>
#include <vector>

namespace yolo_defect_cpp::qt {

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

}  // namespace yolo_defect_cpp::qt
