#include "main_window.h"

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
#include <QProcess>
#include <QPushButton>
#include <QRegularExpression>
#include <QScreen>
#include <QSettings>
#include <QSignalSpy>
#include <QTableView>
#include <QTemporaryDir>
#include <QTest>
#include <QTimer>
#include <QToolButton>

namespace {

using yolo_defect_cpp::qt::MainWindow;

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
