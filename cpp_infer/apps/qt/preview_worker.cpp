#include "preview_worker.h"

#include "task_io.h"

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace yolo_defect_cpp::qt {
namespace {

float read_number(const QJsonValue& value) {
  const double number = value.toDouble();
  if (!value.isDouble() || !std::isfinite(number) ||
      std::abs(number) > std::numeric_limits<float>::max()) {
    throw std::runtime_error("Detection JSON contains an invalid numeric value.");
  }
  return static_cast<float>(number);
}

std::vector<Detection> read_detections(const BatchItemResult& item) {
  QFile file(from_path(*item.json_output_path));
  if (!file.open(QIODevice::ReadOnly)) {
    throw std::runtime_error(
        QStringLiteral("Cannot open detection JSON: %1 (%2)")
            .arg(file.fileName(), file.errorString()).toStdString());
  }
  const QByteArray encoded = file.readAll();
  if (file.error() != QFileDevice::NoError) {
    throw std::runtime_error(
        QStringLiteral("Cannot read detection JSON: %1 (%2)")
            .arg(file.fileName(), file.errorString()).toStdString());
  }
  QJsonParseError parse_error;
  const QJsonDocument document = QJsonDocument::fromJson(encoded, &parse_error);
  if (parse_error.error != QJsonParseError::NoError || !document.isObject()) {
    throw std::runtime_error(
        QStringLiteral("Invalid detection JSON: %1 (%2)")
            .arg(file.fileName(), parse_error.errorString()).toStdString());
  }
  const QJsonObject root = document.object();
  if (root.value(QStringLiteral("schema_version")).toInt(-1) != 1 ||
      !root.value(QStringLiteral("detections")).isArray()) {
    throw std::runtime_error("Unsupported or incomplete detection JSON.");
  }
  const QJsonArray items = root.value(QStringLiteral("detections")).toArray();
  if (static_cast<std::size_t>(items.size()) != item.detection_count) {
    throw std::runtime_error("Detection JSON count differs from the batch result.");
  }
  std::vector<Detection> detections;
  detections.reserve(static_cast<std::size_t>(items.size()));
  for (const QJsonValue& value : items) {
    if (!value.isObject()) {
      throw std::runtime_error("Detection JSON contains an invalid detection.");
    }
    const QJsonObject object = value.toObject();
    Detection detection;
    detection.class_id = object.value(QStringLiteral("class_id")).toInt(-1);
    const QJsonValue class_name = object.value(QStringLiteral("class_name"));
    if (detection.class_id < 0 || !class_name.isString()) {
      throw std::runtime_error("Detection JSON contains an invalid class.");
    }
    detection.class_name = class_name.toString().toUtf8().toStdString();
    detection.confidence = read_number(object.value(QStringLiteral("confidence")));
    if (detection.confidence < 0.0F || detection.confidence > 1.0F) {
      throw std::runtime_error("Detection JSON contains an invalid confidence.");
    }
    const QJsonArray box = object.value(QStringLiteral("bbox_xyxy")).toArray();
    if (box.size() != 4) {
      throw std::runtime_error("Detection JSON bbox_xyxy must contain four coordinates.");
    }
    detection.bbox_xyxy = {read_number(box[0]), read_number(box[1]),
                          read_number(box[2]), read_number(box[3])};
    if (detection.bbox_xyxy.x1 > detection.bbox_xyxy.x2 ||
        detection.bbox_xyxy.y1 > detection.bbox_xyxy.y2) {
      throw std::runtime_error("Detection JSON contains reversed box coordinates.");
    }
    detections.push_back(std::move(detection));
  }
  return detections;
}

}  // namespace

PreviewWorker::PreviewWorker(QObject* parent) : QObject(parent) {}

void PreviewWorker::load(PreviewRequest request) {
  PreviewResponse response;
  response.generation = request.generation;
  try {
    if (request.item.status != BatchItemStatus::kSucceeded) {
      response.error = QString::fromStdString(request.item.error);
      if (response.error.isEmpty()) {
        response.error = tr("该图片没有可用的检测结果。");
      }
    } else {
      if (!request.item.json_output_path || !request.item.image_output_path) {
        throw std::runtime_error("The batch item has no JSON or image output path.");
      }
      response.detections = read_detections(request.item);
      response.original_image = read_preview(request.item.source_path);
      response.annotated_image = read_preview(*request.item.image_output_path);
    }
  } catch (const std::exception& error) {
    response.error = QString::fromUtf8(error.what());
  } catch (...) {
    response.error = tr("An unknown error occurred while loading the preview.");
  }
  if (!response.error.isEmpty()) {
    response.original_image = {};
    response.annotated_image = {};
    response.detections.clear();
  }
  emit completed(std::move(response));
}

}  // namespace yolo_defect_cpp::qt
