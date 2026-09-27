// Real Qt widget captures for documentation. No synthetic inference results.
#include "main_window.h"
#include "image_view.h"
#include "batch_table_model.h"

#include <QApplication>
#include <QCommandLineParser>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QSettings>
#include <QSignalSpy>
#include <QTableView>
#include <QTabWidget>
#include <QTest>
#include <functional>
#include <stdexcept>

using yolo_defect_cpp::qt::MainWindow;
using yolo_defect_cpp::qt::ImageView;

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void waitUntil(const std::function<bool()>& condition) {
  QElapsedTimer timer;
  timer.start();
  while (!condition() && timer.elapsed() < 60000) QTest::qWait(20);
  require(condition(), "Timed out waiting for the real client state");
}
}

int main(int argc, char** argv) {
  QApplication app(argc, argv);
  QCoreApplication::setOrganizationName("YoloDefectCapture");
  QCoreApplication::setApplicationName("Showcase");
  QCommandLineParser parser;
  parser.addHelpOption();
  parser.addOption({"repo-root", "Repository root.", "path"});
  parser.addOption({"frames", "Generated capture directory.", "path"});
  parser.process(app);
  try {
    const QDir repo(QFileInfo(parser.value("repo-root")).absoluteFilePath());
    const QDir frames(QFileInfo(parser.value("frames")).absoluteFilePath());
    require(!parser.value("repo-root").isEmpty() && !parser.value("frames").isEmpty(),
            "--repo-root and --frames are required");
    require(QDir().mkpath(frames.absolutePath()), "Cannot create capture directory");
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope,
                       frames.filePath("settings"));
    QSettings().clear(); // Dedicated capture settings, never the user's settings.
    const QString input = frames.filePath("input");
    require(QDir().mkpath(input), "Cannot create input directory");
    const QString broken = QDir(input).filePath("zz_damaged.jpg");
    if (QFile::exists(broken)) require(QFile::remove(broken), "Cannot reset damaged fixture");
    const QStringList names = {"crazing_241.jpg", "inclusion_241.jpg", "patches_241.jpg",
      "pitted_surface_241.jpg", "rolled-in_scale_241.jpg", "scratches_241.jpg"};
    for (const auto& name : names) {
      const QString destination = QDir(input).filePath(name);
      if (QFile::exists(destination)) require(QFile::remove(destination), "Cannot update sample");
      require(QFile::copy(repo.filePath("data/images/val/" + name), destination), "Missing sample image");
    }
    require(QDir::setCurrent(repo.absolutePath()), "Cannot set repository working directory");
    MainWindow window;
    window.resize(1440, 960);
    if (qEnvironmentVariableIntValue("YOLO_DEFECT_QT_TEST_HIDDEN"))
      window.setAttribute(Qt::WA_DontShowOnScreen);
    const auto configure = [&] {
      window.setBatchInputs("cpp_infer/configs/default_config.txt", repo.relativeFilePath(input),
          repo.relativeFilePath(frames.filePath("outputs")),
          yolo_defect_cpp::BatchInputKind::kDirectory, 2, 4);
    };
    configure();
    window.show();
    auto* run = window.findChild<QPushButton*>("runButton");
    auto* batch = window.findChild<QTableView*>("batchTable");
    auto* details = window.findChild<QTableView*>("resultTable");
    auto* tabs = window.findChild<QTabWidget*>("resultTabs");
    auto* original = window.findChild<ImageView*>("originalView");
    auto* annotated = window.findChild<ImageView*>("annotatedView");
    auto* error = window.findChild<QPlainTextEdit*>("itemError");
    require(run && batch && details && tabs && original && annotated && error, "Missing capture widget");
    QJsonArray sequence;
    const auto capture = [&](const QString& name, int duration, const QString& state) {
      QTest::qWait(50);
      require(window.grab().save(frames.filePath(name + ".png")), "Cannot save capture");
      sequence.append(QJsonObject{{"file", name + ".png"}, {"duration_ms", duration},
                                  {"state", state}});
    };
    capture("01-ready", 1600, "ready");
    QSignalSpy finished(&window, &MainWindow::taskFinished);
    QTest::mouseClick(run, Qt::LeftButton);
    require(window.isBusy(), "Task did not start");
    capture("02-running", 1000, "running");
    waitUntil([&] { return finished.count() == 1; });
    require(finished.at(0).at(0).toBool() && batch->model()->rowCount() == 6,
            "Showcase batch failed");
    const auto* batchModel = static_cast<yolo_defect_cpp::qt::BatchTableModel*>(batch->model());
    for (int row = 0; row < batchModel->rowCount(); ++row) {
      const auto* item = batchModel->itemAt(row);
      require(item && item->status == yolo_defect_cpp::BatchItemStatus::kSucceeded
                  && item->error.empty(), "Showcase requires every image to succeed");
    }
    waitUntil([&] { return !original->imageSize().isEmpty() && details->model()->rowCount() > 0; });
    capture("03-results", 2400, "succeeded");
    tabs->setCurrentIndex(1);
    details->selectRow(0);
    original->actualSize();
    annotated->actualSize();
    original->zoomIn();
    annotated->zoomIn();
    require(annotated->selectedDetection() == 0, "Selection did not reach the image view");
    capture("04-inspect", 2200, "succeeded");
    tabs->setCurrentIndex(0);
    batch->selectRow(5);
    waitUntil([&] { return original->property("sourcePath").toString().endsWith("scratches_241.jpg")
                           && !original->imageSize().isEmpty() && !annotated->imageSize().isEmpty(); });
    require(error->isHidden(), "Successful preview must not show an error");
    capture("05-browse", 2400, "succeeded");
    batch->selectRow(0);
    waitUntil([&] { return original->property("sourcePath").toString().endsWith("crazing_241.jpg")
                           && !original->imageSize().isEmpty() && !annotated->imageSize().isEmpty(); });
    original->fitToWindow();
    annotated->fitToWindow();
    require(error->isHidden(), "Successful overview must not show an error");
    capture("06-overview", 2200, "succeeded");
    QFile manifest(frames.filePath("frames.json"));
    require(manifest.open(QIODevice::WriteOnly), "Cannot write frame manifest");
    manifest.write(QJsonDocument(QJsonObject{
      {"description", "Real Qt client states; edited presentation timing, not a realtime recording."},
      {"logical_width", window.width()}, {"logical_height", window.height()},
      {"batch", QJsonObject{{"total", 6}, {"succeeded", 6}, {"failed", 0}, {"cancelled", 0}}},
      {"device_pixel_ratio", window.devicePixelRatioF()}, {"frames", sequence}}).toJson());
    window.close();
    return 0;
  } catch (const std::exception& exception) {
    qCritical("Qt showcase capture failed: %s", exception.what());
    return 1;
  }
}
