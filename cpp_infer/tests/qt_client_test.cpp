#include "main_window.h"
#include "image_view.h"

#include "yolo_defect_cpp/batch_result.h"

#include <QApplication>
#include <QDebug>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QFont>
#include <QFontDatabase>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QPixmap>
#include <QPlainTextEdit>
#include <QProcess>
#include <QPushButton>
#include <QRegularExpression>
#include <QScreen>
#include <QSettings>
#include <QSignalSpy>
#include <QSpinBox>
#include <QTableView>
#include <QTemporaryDir>
#include <QTest>
#include <QTimer>
#include <QToolButton>
#include <QWheelEvent>

namespace {

using yolo_defect_cpp::qt::MainWindow;
using yolo_defect_cpp::qt::ImageView;
using yolo_defect_cpp::BatchInputKind;

constexpr int kTaskTimeoutMs = 60000;

QString repositoryPath(const QString& relative) {
  return QDir(QString::fromUtf8(YOLO_DEFECT_QT_TEST_REPO_ROOT))
      .absoluteFilePath(relative);
}

QString sampleImage() {
  return repositoryPath(QStringLiteral("data/images/val/crazing_241.jpg"));
}

QString fp32Config() {
  return repositoryPath(QStringLiteral("cpp_infer/configs/default_config.txt"));
}

QStringList outputFiles(const QString& directory, const QString& pattern) {
  QStringList paths;
  QDirIterator iterator(directory, {pattern}, QDir::Files,
                        QDirIterator::Subdirectories);
  while (iterator.hasNext()) {
    paths.push_back(iterator.next());
  }
  return paths;
}

QJsonDocument readJson(const QString& path) {
  QFile file(path);
  if (!file.open(QIODevice::ReadOnly)) {
    return {};
  }
  return QJsonDocument::fromJson(file.readAll());
}

void showTestWindow(MainWindow& window) {
  if (qEnvironmentVariableIntValue("YOLO_DEFECT_QT_TEST_HIDDEN") != 0) {
    window.setAttribute(Qt::WA_DontShowOnScreen);
  }
  window.show();
}

bool saveOptionalScreenshot(MainWindow& window, const QString& name) {
  const QString directory = qEnvironmentVariable("YOLO_DEFECT_QT_SCREENSHOT_DIR");
  if (directory.isEmpty()) {
    return true;
  }
  if (!QDir().mkpath(directory)) {
    return false;
  }
  // Optional Qt renders support visual inspection with offscreen or native QPA;
  // they do not replace checking interaction in a visible desktop window.
  // Resizing and folding panels posts nested layout requests to the event loop.
  QTest::qWait(40);
  QApplication::processEvents();
  return window.grab().save(QDir(directory).filePath(name + QStringLiteral(".png")));
}

bool saveOptionalLayoutScreenshots(MainWindow& window) {
  if (qEnvironmentVariableIsEmpty("YOLO_DEFECT_QT_SCREENSHOT_DIR")) {
    return true;
  }
  auto* toggle = window.findChild<QToolButton*>(QStringLiteral("modelDetailsToggle"));
  if (!toggle) {
    return false;
  }
  const bool wasExpanded = toggle->isChecked();
  toggle->setChecked(true);
  const bool expandedSaved = saveOptionalScreenshot(window, QStringLiteral("complete_FP32_details"));
  toggle->setChecked(wasExpanded);

  const QSize previousSize = window.size();
  window.resize(980, 700);
  const bool minimumSaved = saveOptionalScreenshot(window, QStringLiteral("complete_FP32_minimum"));
  window.resize(previousSize);
  QApplication::processEvents();
  return expandedSaved && minimumSaved;
}

// Moving the declaration must preserve its artifact reference: relative paths
// in RuntimeConfig belong to that declaration, not the working directory.
bool copyConfig(const QString& source, const QString& destination) {
  QFile input(source);
  if (!input.open(QIODevice::ReadOnly)) {
    return false;
  }
  QString text = QString::fromUtf8(input.readAll());
  const QRegularExpression reference(
      QStringLiteral("^artifact_spec_path\\s*=\\s*([^\\r\\n]+)"),
      QRegularExpression::MultilineOption);
  const auto match = reference.match(text);
  if (!match.hasMatch()) {
    return false;
  }
  const QString artifact = QFileInfo(source).dir().absoluteFilePath(
      match.captured(1).trimmed());
  text.replace(match.capturedStart(), match.capturedLength(),
               QStringLiteral("artifact_spec_path = ") + artifact);
  QFile output(destination);
  if (!output.open(QIODevice::WriteOnly)) {
    return false;
  }
  const QByteArray bytes = text.toUtf8();
  return output.write(bytes) == bytes.size();
}

bool writeText(const QString& path, const QByteArray& bytes) {
  QFile file(path);
  return file.open(QIODevice::WriteOnly) && file.write(bytes) == bytes.size();
}

// A small real fixture checks Unicode paths, recursive directory discovery,
// manifest declaration order and per-image failure isolation together.
struct BatchFixture {
  QString root;
  QString config;
  QString manifest;
  QStringList directoryOrder;
  QStringList manifestOrder;

  bool create(const QTemporaryDir& temporary) {
    // Keep CLI argv ASCII: its Windows batch command_arguments currently use
    // the local code page. Unicode filenames inside discovery/manifest still
    // exercise the shared Runtime path handling; GUI config paths are covered
    // separately by recoversFromFailureWithUnicodePaths().
    root = temporary.filePath(QStringLiteral("batch input"));
    config = temporary.filePath(QStringLiteral("detection config.txt"));
    manifest = temporary.filePath(QStringLiteral("input list.txt"));
    const QString nested = QDir(root).filePath(QStringLiteral("子目录"));
    if (!QDir().mkpath(nested) || !copyConfig(fp32Config(), config)) return false;
    const QString first = QDir(root).filePath(QStringLiteral("01 缺陷.jpg"));
    const QString damaged = QDir(root).filePath(QStringLiteral("02 损坏.jpg"));
    const QString last = QDir(nested).filePath(QStringLiteral("03 划痕.jpg"));
    if (!QFile::copy(sampleImage(), first) ||
        !QFile::copy(repositoryPath(QStringLiteral("data/images/val/scratches_300.jpg")), last) ||
        !writeText(damaged, "not encoded image bytes")) return false;
    directoryOrder = {first, damaged, last};
    manifestOrder = {last, first, damaged};
    QByteArray manifestText = "# Declaration order deliberately differs from directory order.\n";
    for (const QString& source : manifestOrder) {
      manifestText += QDir(temporary.path()).relativeFilePath(source).toUtf8() + '\n';
    }
    return writeText(manifest, manifestText);
  }
};

QJsonObject stableBatchSummary(QJsonObject summary) {
  // Runtime measurements and process metadata naturally differ between two
  // executions. Preserve every deterministic contract and per-image outcome.
  for (const auto* key : {"timestamp_utc", "command_arguments", "environment",
                          "timing", "latency_ms", "throughput_images_per_second", "memory"}) {
    summary.remove(QString::fromLatin1(key));
  }
  auto runtime = summary.value(QStringLiteral("runtime")).toObject();
  runtime.remove(QStringLiteral("session_initialization_ms"));
  summary.insert(QStringLiteral("runtime"), runtime);
  auto output = summary.value(QStringLiteral("output")).toObject();
  for (const auto* key : {"directory", "batch_summary_path", "item_directory"}) {
    output.remove(QString::fromLatin1(key));
  }
  summary.insert(QStringLiteral("output"), output);
  summary.insert(QStringLiteral("queue"), QJsonObject{
      {QStringLiteral("capacity"), summary.value(QStringLiteral("queue"))
                                       .toObject().value(QStringLiteral("capacity"))}});
  QJsonArray items;
  for (const auto& value : summary.value(QStringLiteral("items")).toArray()) {
    auto item = value.toObject();
    item.remove(QStringLiteral("latency_ms"));
    for (const auto* key : {"json_output_path", "image_output_path"}) {
      const QString field = QString::fromLatin1(key);
      if (item.value(field).isString()) {
        item.insert(field, QFileInfo(item.value(field).toString()).fileName());
      }
    }
    items.append(item);
  }
  summary.insert(QStringLiteral("items"), items);
  return summary;
}

bool tableMatchesDetections(QTableView* table, const QJsonArray& detections) {
  if (table->model()->rowCount() != detections.size()) return false;
  for (int row = 0; row < detections.size(); ++row) {
    const auto detection = detections.at(row).toObject();
    if (table->model()->index(row, 1).data().toString() !=
        detection.value(QStringLiteral("class_name")).toString()) return false;
    QString confidence = table->model()->index(row, 2).data().toString();
    if (!confidence.endsWith(QLatin1Char('%'))) return false;
    confidence.chop(1);
    if (qAbs(confidence.toDouble() / 100.0 -
             detection.value(QStringLiteral("confidence")).toDouble()) >= 0.000051) return false;
    const auto bounds = detection.value(QStringLiteral("bbox_xyxy")).toArray();
    if (bounds.size() != 4) return false;
    for (int coordinate = 0; coordinate < bounds.size(); ++coordinate) {
      if (qAbs(table->model()->index(row, coordinate + 3).data().toDouble() -
               bounds.at(coordinate).toDouble()) >= 0.051) return false;
    }
  }
  return true;
}

bool makeStopFixture(const QString& input) {
  if (!QDir().mkpath(input)) return false;
  for (int index = 0; index < 12; ++index) {
    if (!QFile::copy(sampleImage(), QDir(input).filePath(
            QStringLiteral("image_%1.jpg").arg(index, 2, 10, QLatin1Char('0'))))) return false;
  }
  return true;
}

class QtClientTest : public QObject {
  Q_OBJECT

 private slots:
  void initTestCase() {
    qInfo() << "Qt platform:" << QGuiApplication::platformName()
            << "screen DPR:" << QGuiApplication::primaryScreen()->devicePixelRatio();
    QVERIFY2(QFileInfo::exists(sampleImage()), "The sample image is required.");
    QVERIFY2(QFileInfo::exists(fp32Config()), "The FP32 config is required.");
    QVERIFY2(QFileInfo::exists(QString::fromUtf8(YOLO_DEFECT_QT_TEST_CLI)),
             "Build the CLI before running the Qt integration tests.");
  }

  void matchesCliWhileResponsive_data() {
    QTest::addColumn<QString>("config");
    QTest::addColumn<QString>("model");
    QTest::newRow("FP32") << fp32Config()
                          << repositoryPath(QStringLiteral("models/best.onnx"));
    QTest::newRow("U8S8")
        << repositoryPath(QStringLiteral("cpp_infer/configs/int8_u8s8_config.txt"))
        << repositoryPath(QStringLiteral("models/best.int8.qdq.u8s8.onnx"));
  }

  void matchesCliWhileResponsive() {
    QFETCH(QString, config);
    QFETCH(QString, model);
    if (!QFileInfo::exists(model)) {
      QSKIP("The selected model artifact has not been downloaded.");
    }
    QTemporaryDir temporary;
    QVERIFY(temporary.isValid());
    const QString guiOutput = temporary.filePath(QStringLiteral("GUI 结果 output"));

    MainWindow window;
    window.setInputs(config, sampleImage(), guiOutput);
    showTestWindow(window);
    QVERIFY(saveOptionalScreenshot(
        window, QStringLiteral("ready_") + QString::fromLatin1(QTest::currentDataTag())));
    auto* run = window.findChild<QPushButton*>(QStringLiteral("runButton"));
    auto* configField = window.findChild<QLineEdit*>(QStringLiteral("configPath"));
    auto* imageField = window.findChild<QLineEdit*>(QStringLiteral("imagePath"));
    auto* outputField = window.findChild<QLineEdit*>(QStringLiteral("outputDirectory"));
    auto* table = window.findChild<QTableView*>(QStringLiteral("resultTable"));
    auto* details = window.findChild<QLabel*>(QStringLiteral("modelDetails"));
    auto* scoreThreshold = window.findChild<QLabel*>(QStringLiteral("scoreThreshold"));
    auto* nmsThreshold = window.findChild<QLabel*>(QStringLiteral("nmsThreshold"));
    auto* openOutput = window.findChild<QPushButton*>(QStringLiteral("openOutputButton"));
    auto* openJson = window.findChild<QPushButton*>(QStringLiteral("openJsonButton"));
    QVERIFY(run && configField && imageField && outputField && table && details &&
            scoreThreshold && nmsThreshold &&
            openOutput && openJson);
    QVERIFY(window.findChild<QWidget*>(QStringLiteral("originalView")));
    QVERIFY(window.findChild<QWidget*>(QStringLiteral("annotatedView")));

    QSignalSpy finished(&window, &MainWindow::taskFinished);
    int eventsDuringTask = 0;
    QTimer heartbeat;
    heartbeat.setInterval(1);
    connect(&heartbeat, &QTimer::timeout, &window, [&] {
      if (window.isBusy()) {
        ++eventsDuringTask;
      }
    });
    heartbeat.start();
    QTest::mouseClick(run, Qt::LeftButton);
    QVERIFY(window.isBusy());
    QVERIFY(!run->isEnabled());
    QVERIFY(!configField->isEnabled());
    QVERIFY(!imageField->isEnabled());
    QVERIFY(!outputField->isEnabled());

    // The public slot must also reject a second start; disabling the button
    // alone does not protect keyboard actions or queued invocations.
    window.startDetection();
    QTRY_COMPARE_WITH_TIMEOUT(finished.count(), 1, kTaskTimeoutMs);
    QVERIFY(finished.at(0).at(0).toBool());
    QVERIFY(!window.isBusy());
    QVERIFY2(eventsDuringTask > 0,
             "Model loading and inference must leave the GUI event loop running.");
    QVERIFY(run->isEnabled());
    QVERIFY(configField->isEnabled());
    QVERIFY(imageField->isEnabled());
    QVERIFY(outputField->isEnabled());
    QVERIFY(openOutput->isEnabled());
    QVERIFY(openJson->isEnabled());
    QVERIFY(saveOptionalScreenshot(
        window, QStringLiteral("complete_") + QString::fromLatin1(QTest::currentDataTag())));
    if (QString::fromLatin1(QTest::currentDataTag()) == QStringLiteral("FP32")) {
      QVERIFY(saveOptionalLayoutScreenshots(window));
    }

    const QStringList jsonPaths = outputFiles(guiOutput, QStringLiteral("*.json"));
    QCOMPARE(jsonPaths.size(), 1);
    const QJsonDocument guiDocument = readJson(jsonPaths.front());
    QVERIFY(guiDocument.isObject());
    const auto detections = guiDocument.object().value(QStringLiteral("detections")).toArray();
    QVERIFY2(!detections.isEmpty(), "The real sample should populate the result table.");
    QCOMPARE(table->model()->rowCount(), detections.size());
    QCOMPARE(table->model()->columnCount(), 7);
    for (int row = 0; row < detections.size(); ++row) {
      const auto detection = detections.at(row).toObject();
      QCOMPARE(table->model()->index(row, 1).data().toString(),
               detection.value(QStringLiteral("class_name")).toString());
      QString confidence = table->model()->index(row, 2).data().toString();
      QVERIFY(confidence.endsWith(QLatin1Char('%')));
      confidence.chop(1);
      QVERIFY(qAbs(confidence.toDouble() / 100.0 -
                   detection.value(QStringLiteral("confidence")).toDouble()) < 0.000051);
      const auto bounds = detection.value(QStringLiteral("bbox_xyxy")).toArray();
      QCOMPARE(bounds.size(), 4);
      for (int coordinate = 0; coordinate < bounds.size(); ++coordinate) {
        QVERIFY(qAbs(table->model()->index(row, coordinate + 3).data().toDouble() -
                     bounds.at(coordinate).toDouble()) < 0.051);
      }
    }
    const QString modelId = guiDocument.object().value(QStringLiteral("model"))
                                .toObject().value(QStringLiteral("model_id")).toString();
    QVERIFY(details->text().contains(modelId));
    QVERIFY(scoreThreshold->text().contains(QStringLiteral("0.25")));
    QVERIFY(nmsThreshold->text().contains(QStringLiteral("0.45")));

    const QString cliJson = temporary.filePath(QStringLiteral("cli.json"));
    QProcess cli;
    cli.start(QString::fromUtf8(YOLO_DEFECT_QT_TEST_CLI),
              {QStringLiteral("--config"), config,
               QStringLiteral("--image"), sampleImage(),
               QStringLiteral("--output-json"), cliJson});
    QVERIFY2(cli.waitForStarted(), qPrintable(cli.errorString()));
    QVERIFY2(cli.waitForFinished(kTaskTimeoutMs), qPrintable(cli.errorString()));
    const QByteArray cliLog = cli.readAllStandardError() + cli.readAllStandardOutput();
    QCOMPARE(cli.exitStatus(), QProcess::NormalExit);
    QVERIFY2(cli.exitCode() == 0, cliLog.constData());
    const QJsonDocument cliDocument = readJson(cliJson);
    QVERIFY(cliDocument.isObject());
    QCOMPARE(guiDocument, cliDocument);

    // A changed declaration cannot leave the previous detections or active
    // model details looking as though they belong to the new selection.
    configField->setText(temporary.filePath(QStringLiteral("another_config.txt")));
    QCOMPARE(table->model()->rowCount(), 0);
    QVERIFY(!details->text().contains(modelId));
    QVERIFY(!openOutput->isEnabled());
    QVERIFY(!openJson->isEnabled());
    QCOMPARE(finished.count(), 1);
  }

  void recoversFromFailureWithUnicodePaths() {
    if (!QFileInfo::exists(repositoryPath(QStringLiteral("models/best.onnx")))) {
      QSKIP("The FP32 model artifact has not been downloaded.");
    }
    QTemporaryDir temporary;
    QVERIFY(temporary.isValid());
    const QString inputRoot = temporary.filePath(QStringLiteral("输入 样例"));
    QVERIFY(QDir().mkpath(inputRoot));
    const QString config = QDir(inputRoot).filePath(QStringLiteral("检测 config.txt"));
    const QString image = QDir(inputRoot).filePath(QStringLiteral("缺陷 sample.jpg"));
    const QString damaged = QDir(inputRoot).filePath(QStringLiteral("损坏 image.jpg"));
    const QString output = temporary.filePath(QStringLiteral("检测 输出"));
    QVERIFY(copyConfig(fp32Config(), config));
    QVERIFY(QFile::copy(sampleImage(), image));
    QFile invalidImage(damaged);
    QVERIFY(invalidImage.open(QIODevice::WriteOnly));
    QVERIFY(invalidImage.write("not encoded image bytes") > 0);
    invalidImage.close();

    MainWindow window;
    window.setInputs(config, damaged, output);
    showTestWindow(window);
    auto* run = window.findChild<QPushButton*>(QStringLiteral("runButton"));
    auto* status = window.findChild<QLabel*>(QStringLiteral("statusMessage"));
    auto* imageField = window.findChild<QLineEdit*>(QStringLiteral("imagePath"));
    auto* table = window.findChild<QTableView*>(QStringLiteral("resultTable"));
    auto* openJson = window.findChild<QPushButton*>(QStringLiteral("openJsonButton"));
    QVERIFY(run && status && imageField && table && openJson);
    QSignalSpy finished(&window, &MainWindow::taskFinished);
    QTest::mouseClick(run, Qt::LeftButton);
    QTRY_COMPARE_WITH_TIMEOUT(finished.count(), 1, kTaskTimeoutMs);
    QVERIFY(!finished.at(0).at(0).toBool());
    QVERIFY(!window.isBusy());
    QVERIFY(run->isEnabled());
    QVERIFY(!status->text().trimmed().isEmpty());
    QCOMPARE(status->property("state").toString(), QStringLiteral("error"));
    QVERIFY(status->text().contains(QStringLiteral("OpenCV")));
    QVERIFY(saveOptionalScreenshot(window, QStringLiteral("error_damaged_image")));
    const QString failureMessage = status->text();
    QCOMPARE(table->model()->rowCount(), 0);
    QVERIFY(!openJson->isEnabled());

    imageField->setText(image);
    QTest::mouseClick(run, Qt::LeftButton);
    QTRY_COMPARE_WITH_TIMEOUT(finished.count(), 2, kTaskTimeoutMs);
    QVERIFY(finished.at(1).at(0).toBool());
    QVERIFY(!window.isBusy());
    QVERIFY(status->text() != failureMessage);
    QVERIFY(table->model()->rowCount() > 0);
    QVERIFY(openJson->isEnabled());
    const QStringList jsonPaths = outputFiles(output, QStringLiteral("*.json"));
    QCOMPARE(jsonPaths.size(), 1);
    const QJsonDocument document = readJson(jsonPaths.front());
    QVERIFY(document.isObject());
    const QString storedImage = document.object().value(QStringLiteral("image"))
                                    .toObject().value(QStringLiteral("path")).toString();
    QCOMPARE(QFileInfo(storedImage).canonicalFilePath(), QFileInfo(image).canonicalFilePath());
    QStringList rendered = outputFiles(output, QStringLiteral("*.png"));
    rendered += outputFiles(output, QStringLiteral("*.jpg"));
    QCOMPARE(rendered.size(), 1);
    const QImage annotated(rendered.front());
    QVERIFY(!annotated.isNull());
    QCOMPARE(annotated.size(), QImage(image).size());

    if (QFileInfo::exists(repositoryPath(QStringLiteral("models/best.int8.qdq.u8s8.onnx")))) {
      // Reuse the successful window so this exercises replacing its active
      // configuration/session, rather than starting a separate INT8 client.
      const QString int8Config = repositoryPath(
          QStringLiteral("cpp_infer/configs/int8_u8s8_config.txt"));
      const auto expectedContract = yolo_defect_cpp::load_runtime_contract(
          yolo_defect_cpp::qt::to_path(int8Config));
      const QString expectedModelId = QString::fromStdString(expectedContract.artifact.model_id);
      const QString previousModelId = document.object().value(QStringLiteral("model"))
                                          .toObject().value(QStringLiteral("model_id")).toString();
      QVERIFY(expectedModelId != previousModelId);
      auto* configField = window.findChild<QLineEdit*>(QStringLiteral("configPath"));
      auto* details = window.findChild<QLabel*>(QStringLiteral("modelDetails"));
      QVERIFY(configField && details);
      configField->setText(int8Config);
      QCOMPARE(table->model()->rowCount(), 0);
      QVERIFY(!details->text().contains(previousModelId));
      QVERIFY(!openJson->isEnabled());
      QTest::mouseClick(run, Qt::LeftButton);
      QTRY_COMPARE_WITH_TIMEOUT(finished.count(), 3, kTaskTimeoutMs);
      QVERIFY(finished.at(2).at(0).toBool());
      QVERIFY(!window.isBusy());
      QVERIFY(details->text().contains(expectedModelId));
      QVERIFY(openJson->isEnabled());
      QStringList switchedOutputs = outputFiles(output, QStringLiteral("*.json"));
      for (const auto& previous : jsonPaths) {
        switchedOutputs.removeAll(previous);
      }
      QCOMPARE(switchedOutputs.size(), 1);
      const auto switchedDocument = readJson(switchedOutputs.front());
      QVERIFY(switchedDocument.isObject());
      QCOMPARE(switchedDocument.object().value(QStringLiteral("model"))
                   .toObject().value(QStringLiteral("model_id")).toString(),
               expectedModelId);
      QCOMPARE(table->model()->rowCount(), switchedDocument.object()
                   .value(QStringLiteral("detections")).toArray().size());
    }
  }

  void batchMatchesCliAndSelection_data() {
    QTest::addColumn<bool>("manifestInput");
    QTest::newRow("directory") << false;
    QTest::newRow("manifest") << true;
  }

  void batchMatchesCliAndSelection() {
    QFETCH(bool, manifestInput);
    if (!QFileInfo::exists(repositoryPath(QStringLiteral("models/best.onnx")))) {
      QSKIP("The FP32 model artifact has not been downloaded.");
    }
    QTemporaryDir temporary;
    QVERIFY(temporary.isValid());
    BatchFixture fixture;
    QVERIFY(fixture.create(temporary));
    const QString input = manifestInput ? fixture.manifest : fixture.root;
    const QStringList expectedOrder = manifestInput ? fixture.manifestOrder : fixture.directoryOrder;
    const auto kind = manifestInput ? BatchInputKind::kManifest : BatchInputKind::kDirectory;
    const QString guiOutput = temporary.filePath(QStringLiteral("GUI 批量结果"));
    MainWindow window;
    window.setBatchInputs(fixture.config, input, guiOutput, kind, 2, 1);
    showTestWindow(window);
    auto* mode = window.findChild<QWidget*>(QStringLiteral("inputMode"));
    auto* workers = window.findChild<QSpinBox*>(QStringLiteral("workersSpin"));
    auto* queue = window.findChild<QSpinBox*>(QStringLiteral("queueSpin"));
    auto* run = window.findChild<QPushButton*>(QStringLiteral("runButton"));
    auto* stop = window.findChild<QPushButton*>(QStringLiteral("stopButton"));
    auto* batchTable = window.findChild<QTableView*>(QStringLiteral("batchTable"));
    auto* table = window.findChild<QTableView*>(QStringLiteral("resultTable"));
    auto* summaryLabel = window.findChild<QLabel*>(QStringLiteral("batchSummary"));
    auto* itemDetails = window.findChild<QLabel*>(QStringLiteral("itemDetails"));
    auto* itemError = window.findChild<QPlainTextEdit*>(QStringLiteral("itemError"));
    auto* original = window.findChild<ImageView*>(QStringLiteral("originalView"));
    auto* annotated = window.findChild<ImageView*>(QStringLiteral("annotatedView"));
    QVERIFY(mode && workers && queue && run && stop && batchTable && table &&
            summaryLabel && itemDetails && itemError && original && annotated);
    QCOMPARE(mode->property("currentIndex").toInt(), manifestInput ? 2 : 1);
    QCOMPARE(workers->value(), 2);
    QCOMPARE(queue->value(), 1);
    QSignalSpy finished(&window, &MainWindow::taskFinished);
    int eventsDuringTask = 0;
    QTimer heartbeat;
    heartbeat.setInterval(1);
    connect(&heartbeat, &QTimer::timeout, &window, [&] {
      if (window.isBusy()) ++eventsDuringTask;
    });
    heartbeat.start();
    QTest::mouseClick(run, Qt::LeftButton);
    QVERIFY(window.isBusy());
    QVERIFY(stop->isEnabled());
    QVERIFY(!run->isEnabled());
    QVERIFY(!mode->isEnabled());
    QVERIFY(!workers->isEnabled());
    QVERIFY(!queue->isEnabled());
    window.startDetection();
    QTRY_COMPARE_WITH_TIMEOUT(finished.count(), 1, kTaskTimeoutMs);
    QVERIFY(!finished.at(0).at(0).toBool());  // Partial failure is not all-success.
    QVERIFY(eventsDuringTask > 0);
    QVERIFY(!window.isBusy());
    QVERIFY(run->isEnabled());
    QVERIFY(!stop->isEnabled());
    QVERIFY(mode->isEnabled());
    QVERIFY(workers->isEnabled());
    QVERIFY(queue->isEnabled());

    const auto summaries = outputFiles(guiOutput, QStringLiteral("batch_summary.json"));
    QCOMPARE(summaries.size(), 1);
    const auto guiSummary = readJson(summaries.front()).object();
    QCOMPARE(guiSummary.value(QStringLiteral("status")).toString(), QStringLiteral("partial_failure"));
    const auto counts = guiSummary.value(QStringLiteral("counts")).toObject();
    QCOMPARE(counts.value(QStringLiteral("discovered")).toInt(), 3);
    QCOMPARE(counts.value(QStringLiteral("succeeded")).toInt(), 2);
    QCOMPARE(counts.value(QStringLiteral("failed")).toInt(), 1);
    QCOMPARE(counts.value(QStringLiteral("cancelled")).toInt(), 0);
    QVERIFY(!summaryLabel->text().trimmed().isEmpty());
    const auto items = guiSummary.value(QStringLiteral("items")).toArray();
    QCOMPARE(items.size(), 3);
    QCOMPARE(batchTable->model()->rowCount(), 3);
    QCOMPARE(batchTable->model()->columnCount(), 6);

    const QString cliOutput = temporary.filePath(QStringLiteral("CLI batch output"));
    const QString cliSummaryPath = QDir(cliOutput).filePath(QStringLiteral("batch_summary.json"));
    QProcess cli;
    cli.start(QString::fromUtf8(YOLO_DEFECT_QT_TEST_CLI),
              {QStringLiteral("--config"), fixture.config, QStringLiteral("--batch"),
               manifestInput ? QStringLiteral("--manifest") : QStringLiteral("--input-dir"), input,
               QStringLiteral("--output-dir"), cliOutput,
               QStringLiteral("--batch-summary"), cliSummaryPath,
               QStringLiteral("--workers"), QStringLiteral("2"),
               QStringLiteral("--queue-capacity"), QStringLiteral("1"),
               QStringLiteral("--output-images")});
    QVERIFY2(cli.waitForStarted(), qPrintable(cli.errorString()));
    QVERIFY2(cli.waitForFinished(kTaskTimeoutMs), qPrintable(cli.errorString()));
    const QByteArray cliLog = cli.readAllStandardError() + cli.readAllStandardOutput();
    QCOMPARE(cli.exitStatus(), QProcess::NormalExit);
    QVERIFY2(cli.exitCode() == 2, cliLog.constData());
    const auto cliDocument = readJson(cliSummaryPath);
    QVERIFY(cliDocument.isObject());
    const auto cliSummary = cliDocument.object();
    QCOMPARE(stableBatchSummary(guiSummary), stableBatchSummary(cliSummary));
    const auto cliItems = cliSummary.value(QStringLiteral("items")).toArray();

    int successfulRow = -1;
    for (int row = 0; row < items.size(); ++row) {
      const auto item = items.at(row).toObject();
      const QString source = item.value(QStringLiteral("source_path")).toString();
      QCOMPARE(item.value(QStringLiteral("sequence_index")).toInt(), row);
      QCOMPARE(QFileInfo(source).canonicalFilePath(), QFileInfo(expectedOrder.at(row)).canonicalFilePath());
      QCOMPARE(batchTable->model()->index(row, 0).data().toInt(), row + 1);
      QVERIFY(batchTable->model()->index(row, 1).data().toString().contains(QFileInfo(source).fileName()));
      batchTable->selectRow(row);
      QTRY_COMPARE(QFileInfo(original->property("sourcePath").toString()).canonicalFilePath(),
                   QFileInfo(source).canonicalFilePath());
      if (item.value(QStringLiteral("status")).toString() == QStringLiteral("succeeded")) {
        const auto guiItem = readJson(item.value(QStringLiteral("json_output_path")).toString());
        const auto cliItem = readJson(cliItems.at(row).toObject()
                                         .value(QStringLiteral("json_output_path")).toString());
        QVERIFY(guiItem.isObject());
        QCOMPARE(guiItem, cliItem);
        const auto detections = guiItem.object().value(QStringLiteral("detections")).toArray();
        QCOMPARE(batchTable->model()->index(row, 3).data().toInt(), detections.size());
        QTRY_VERIFY_WITH_TIMEOUT(tableMatchesDetections(table, detections), kTaskTimeoutMs);
        QTRY_COMPARE(original->imageSize(), QImage(source).size());
        QCOMPARE(annotated->imageSize(), original->imageSize());
        successfulRow = row;
        if (!manifestInput) QVERIFY(saveOptionalScreenshot(window, QStringLiteral("batch_success")));
      } else {
        const QString error = item.value(QStringLiteral("error")).toString();
        QVERIFY(!error.isEmpty());
        QTRY_VERIFY(itemError->toPlainText().contains(error));
        QCOMPARE(table->model()->rowCount(), 0);
        QVERIFY(annotated->imageSize().isEmpty());
        if (!manifestInput) {
          QVERIFY(saveOptionalScreenshot(window, QStringLiteral("batch_failure")));
          if (!qEnvironmentVariableIsEmpty("YOLO_DEFECT_QT_SCREENSHOT_DIR")) {
            const QSize previousSize = window.size();
            window.resize(980, 700);
            QVERIFY(saveOptionalScreenshot(window, QStringLiteral("batch_failure_minimum")));
            qInfo() << "Minimum failed batch list:" << window.size()
                    << "viewport height" << batchTable->viewport()->height()
                    << "row height" << batchTable->rowHeight(0);
            window.resize(previousSize);
          }
        }
      }
    }

    QVERIFY(successfulRow >= 0);
    // Rapid selection must not allow an older asynchronous preview to replace
    // the current item after its read finishes.
    batchTable->selectRow(0);
    batchTable->selectRow(1);
    batchTable->selectRow(successfulRow);
    const auto selectedItem = items.at(successfulRow).toObject();
    const auto selectedDetections = readJson(selectedItem.value(QStringLiteral("json_output_path"))
                                                .toString()).object().value(QStringLiteral("detections")).toArray();
    QTRY_VERIFY_WITH_TIMEOUT(tableMatchesDetections(table, selectedDetections), kTaskTimeoutMs);
    QTRY_COMPARE(QFileInfo(original->property("sourcePath").toString()).canonicalFilePath(),
                 QFileInfo(selectedItem.value(QStringLiteral("source_path")).toString()).canonicalFilePath());
    QVERIFY(!selectedDetections.isEmpty());
    const int selectedDetection = selectedDetections.size() - 1;
    table->selectRow(selectedDetection);
    QCOMPARE(original->selectedDetection(), selectedDetection);
    QCOMPARE(annotated->selectedDetection(), selectedDetection);
    original->actualSize();
    QCOMPARE(original->zoomFactor(), 1.0);
    original->zoomIn();
    QVERIFY(original->zoomFactor() > 1.0);
    original->zoomOut();
    QVERIFY(qAbs(original->zoomFactor() - 1.0) < 0.000001);
    original->fitToWindow();
    QVERIFY(original->zoomFactor() > 0.0);
    // Click a real bounding box to exercise the reverse image -> table link.
    // Choosing the smallest box makes the hit unambiguous if boxes overlap.
    int smallestBox = 0;
    double smallestArea = 0.0;
    for (int index = 0; index < selectedDetections.size(); ++index) {
      const auto box = selectedDetections.at(index).toObject()
                           .value(QStringLiteral("bbox_xyxy")).toArray();
      const double area = (box.at(2).toDouble() - box.at(0).toDouble()) *
                          (box.at(3).toDouble() - box.at(1).toDouble());
      if (index == 0 || area < smallestArea) {
        smallestArea = area;
        smallestBox = index;
      }
    }
    annotated->fitToWindow();
    const auto box = selectedDetections.at(smallestBox).toObject()
                         .value(QStringLiteral("bbox_xyxy")).toArray();
    const QPointF imagePoint((box.at(0).toDouble() + box.at(2).toDouble()) / 2.0,
                             (box.at(1).toDouble() + box.at(3).toDouble()) / 2.0);
    const QPointF imageCenter(annotated->imageSize().width() / 2.0,
                              annotated->imageSize().height() / 2.0);
    const QRectF viewport = QRectF(annotated->rect()).adjusted(16, 16, -16, -32);
    const QPoint click = (viewport.center() +
                         (imagePoint - imageCenter) * annotated->zoomFactor()).toPoint();
    table->clearSelection();
    table->setCurrentIndex(QModelIndex());
    QCOMPARE(annotated->selectedDetection(), -1);
    QSignalSpy imageSelection(annotated, &ImageView::detectionSelected);
    QTest::mouseClick(annotated, Qt::LeftButton, Qt::NoModifier, click);
    QCOMPARE(imageSelection.count(), 1);
    QCOMPARE(imageSelection.at(0).at(0).toInt(), smallestBox);
    QCOMPARE(table->currentIndex().row(), smallestBox);
    QCOMPARE(original->selectedDetection(), smallestBox);
    QCOMPARE(annotated->selectedDetection(), smallestBox);
    // Exercise actual input events as well as toolbar slots. Several wheel
    // steps enlarge this 200 px image beyond the viewport so a drag can pan.
    QApplication::processEvents();
    const QRect imageViewport = QRectF(annotated->rect())
                                    .adjusted(16, 16, -16, -32).toAlignedRect();
    const QPoint dragStart = imageViewport.center();
    const double zoomBeforeWheel = annotated->zoomFactor();
    QWheelEvent wheel(dragStart, annotated->mapToGlobal(dragStart), QPoint(),
                      QPoint(0, 6 * 120), Qt::NoButton, Qt::NoModifier,
                      Qt::NoScrollPhase, false);
    QApplication::sendEvent(annotated, &wheel);
    QVERIFY(annotated->zoomFactor() > zoomBeforeWheel);
    QVERIFY(annotated->imageSize().width() * annotated->zoomFactor() > imageViewport.width());
    const QImage beforeDrag = annotated->grab(imageViewport).toImage();
    const QPoint dragEnd = dragStart + QPoint(QApplication::startDragDistance() + 30, 0);
    QTest::mousePress(annotated, Qt::LeftButton, Qt::NoModifier, dragStart);
    QTest::mouseMove(annotated, dragEnd);
    QTest::mouseRelease(annotated, Qt::LeftButton, Qt::NoModifier, dragEnd);
    QVERIFY(annotated->grab(imageViewport).toImage() != beforeDrag);
    QCOMPARE(imageSelection.count(), 1);  // A pan must not also select a box.
    QCOMPARE(table->currentIndex().row(), smallestBox);
    annotated->fitToWindow();
    if (!manifestInput) {
      window.resize(980, 700);
      QVERIFY(saveOptionalScreenshot(window, QStringLiteral("batch_minimum")));
      QVERIFY2(table->viewport()->height() >= table->rowHeight(0) * 2,
               "The minimum-size window must show data rows, not only a table header.");
    }
    QCOMPARE(finished.count(), 1);
  }

  void batchCooperativeStopAndRestart_data() {
    QTest::addColumn<QString>("stopPhase");
    QTest::newRow("before_runner_is_registered") << QStringLiteral("immediate");
    QTest::newRow("while_runner_is_processing") << QStringLiteral("running");
    QTest::newRow("close_while_runner_is_processing") << QStringLiteral("close");
  }

  void batchCooperativeStopAndRestart() {
    QFETCH(QString, stopPhase);
    if (!QFileInfo::exists(repositoryPath(QStringLiteral("models/best.onnx")))) {
      QSKIP("The FP32 model artifact has not been downloaded.");
    }
    QTemporaryDir temporary;
    QVERIFY(temporary.isValid());
    const QString input = temporary.filePath(QStringLiteral("input"));
    const QString output = temporary.filePath(QStringLiteral("output"));
    QVERIFY(makeStopFixture(input));
    MainWindow window;
    window.setBatchInputs(fp32Config(), input, output, BatchInputKind::kDirectory, 1, 1);
    showTestWindow(window);
    auto* run = window.findChild<QPushButton*>(QStringLiteral("runButton"));
    auto* stop = window.findChild<QPushButton*>(QStringLiteral("stopButton"));
    auto* table = window.findChild<QTableView*>(QStringLiteral("batchTable"));
    QVERIFY(run && stop && table);
    QSignalSpy finished(&window, &MainWindow::taskFinished);
    int eventsDuringTask = 0;
    bool requested = false;
    bool deferredClose = false;
    QTimer observer;
    observer.setInterval(1);
    connect(&observer, &QTimer::timeout, &window, [&] {
      if (window.isBusy()) ++eventsDuringTask;
      if (requested || stopPhase == QStringLiteral("immediate") || !window.isBusy()) return;
      // Observe a completed real image, not a fixed sleep or a synthetic
      // progress event. run() is still executing the remaining bounded batch.
      if (outputFiles(output, QStringLiteral("*.detections.json")).isEmpty()) return;
      requested = true;
      if (stopPhase == QStringLiteral("close")) {
        window.close();
        deferredClose = window.isVisible() && window.isBusy();
      } else {
        window.stopDetection();
        window.stopDetection();  // Idempotent while worker/session cleanup runs.
      }
    });
    observer.start();
    window.startDetection();
    QVERIFY(window.isBusy());
    if (stopPhase == QStringLiteral("immediate")) {
      requested = true;
      window.stopDetection();
      window.stopDetection();
    }
    window.startDetection();
    QTRY_COMPARE_WITH_TIMEOUT(finished.count(), 1, kTaskTimeoutMs);
    observer.stop();
    QVERIFY(requested);
    QVERIFY(!finished.at(0).at(0).toBool());
    QVERIFY(!window.isBusy());
    QVERIFY(eventsDuringTask > 0);
    QVERIFY(!stop->isEnabled());
    const auto summaries = outputFiles(output, QStringLiteral("batch_summary.json"));
    QCOMPARE(summaries.size(), 1);
    const auto summary = readJson(summaries.front()).object();
    QCOMPARE(summary.value(QStringLiteral("status")).toString(), QStringLiteral("cancelled"));
    QVERIFY(summary.value(QStringLiteral("cooperative_stop_requested")).toBool());
    const auto counts = summary.value(QStringLiteral("counts")).toObject();
    QCOMPARE(counts.value(QStringLiteral("discovered")).toInt(), 12);
    QVERIFY(counts.value(QStringLiteral("cancelled")).toInt() > 0);
    QCOMPARE(counts.value(QStringLiteral("failed")).toInt(), 0);
    QCOMPARE(counts.value(QStringLiteral("succeeded")).toInt() +
                 counts.value(QStringLiteral("cancelled")).toInt(), 12);
    if (stopPhase != QStringLiteral("immediate")) {
      QVERIFY(counts.value(QStringLiteral("succeeded")).toInt() > 0);
    }
    QCOMPARE(table->model()->rowCount(), 12);
    if (stopPhase == QStringLiteral("close")) {
      QVERIFY(deferredClose);
      QTRY_VERIFY(!window.isVisible());
      return;
    }
    QVERIFY(saveOptionalScreenshot(window, QStringLiteral("batch_cancelled")));
    if (stopPhase == QStringLiteral("running") &&
        !qEnvironmentVariableIsEmpty("YOLO_DEFECT_QT_SCREENSHOT_DIR")) {
      const QSize previousSize = window.size();
      window.resize(1280, 730);
      QVERIFY(saveOptionalScreenshot(window, QStringLiteral("batch_browse_compact")));
      qInfo() << "Compact batch list:" << window.size()
              << "viewport height" << table->viewport()->height()
              << "row height" << table->rowHeight(0);
      window.resize(980, 700);
      QVERIFY(saveOptionalScreenshot(window, QStringLiteral("batch_browse_minimum")));
      qInfo() << "Minimum batch list:" << window.size()
              << "viewport height" << table->viewport()->height()
              << "row height" << table->rowHeight(0);
      window.resize(previousSize);
    }
    QVERIFY(run->isEnabled());
    const QString restartInput = temporary.filePath(QStringLiteral("restart input"));
    const QString restartOutput = temporary.filePath(QStringLiteral("restart output"));
    QVERIFY(QDir().mkpath(restartInput));
    QVERIFY(QFile::copy(sampleImage(), QDir(restartInput).filePath(QStringLiteral("sample.jpg"))));
    window.setBatchInputs(fp32Config(), restartInput, restartOutput);
    window.startDetection();
    QTRY_COMPARE_WITH_TIMEOUT(finished.count(), 2, kTaskTimeoutMs);
    QVERIFY(finished.at(1).at(0).toBool());
    QVERIFY(!window.isBusy());
    const auto restartedSummaries = outputFiles(restartOutput, QStringLiteral("batch_summary.json"));
    QCOMPARE(restartedSummaries.size(), 1);
    const auto restarted = readJson(restartedSummaries.front()).object();
    QCOMPARE(restarted.value(QStringLiteral("status")).toString(), QStringLiteral("succeeded"));
    QVERIFY(!restarted.value(QStringLiteral("cooperative_stop_requested")).toBool());
    QCOMPARE(restarted.value(QStringLiteral("counts")).toObject().value(QStringLiteral("succeeded")).toInt(), 1);
    QCOMPARE(table->model()->rowCount(), 1);
  }

  void closeDuringTaskWaitsWithoutBlocking() {
    if (!QFileInfo::exists(repositoryPath(QStringLiteral("models/best.onnx")))) {
      QSKIP("The FP32 model artifact has not been downloaded.");
    }
    QTemporaryDir temporary;
    QVERIFY(temporary.isValid());
    MainWindow window;
    window.setInputs(fp32Config(), sampleImage(), temporary.filePath(QStringLiteral("output")));
    showTestWindow(window);
    QSignalSpy finished(&window, &MainWindow::taskFinished);
    int eventsDuringClose = 0;
    QTimer heartbeat;
    heartbeat.setInterval(1);
    connect(&heartbeat, &QTimer::timeout, &window, [&] {
      if (window.isBusy()) {
        ++eventsDuringClose;
      }
    });
    window.startDetection();
    QVERIFY(window.isBusy());
    heartbeat.start();
    window.close();
    QVERIFY2(window.isVisible(), "Closing must defer destruction until the worker exits.");
    QVERIFY(window.isBusy());
    QTRY_COMPARE_WITH_TIMEOUT(finished.count(), 1, kTaskTimeoutMs);
    QVERIFY(finished.at(0).at(0).toBool());
    QVERIFY2(eventsDuringClose > 0, "Waiting for the worker must not block the GUI thread.");
    QTRY_VERIFY(!window.isVisible());
    QVERIFY(!window.isBusy());
  }
};

}  // namespace

int main(int argc, char* argv[]) {
  QApplication application(argc, argv);
  const QString fontPath = qEnvironmentVariable("YOLO_DEFECT_QT_TEST_FONT");
  if (!fontPath.isEmpty()) {
    const int fontId = QFontDatabase::addApplicationFont(fontPath);
    const QStringList families = QFontDatabase::applicationFontFamilies(fontId);
    if (fontId < 0 || families.isEmpty()) {
      qCritical("Could not load the requested test font: %s", qPrintable(fontPath));
      return 1;
    }
    application.setFont(QFont(families.front()));
  }
  // Keep test windows from changing the user's saved geometry and directories.
  QTemporaryDir settings;
  if (!settings.isValid()) {
    return 1;
  }
  QCoreApplication::setOrganizationName(QStringLiteral("yolo_defect_tests"));
  QCoreApplication::setApplicationName(QStringLiteral("qt_client"));
  QSettings::setDefaultFormat(QSettings::IniFormat);
  QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, settings.path());
  QtClientTest test;
  return QTest::qExec(&test, argc, argv);
}

#include "qt_client_test.moc"
