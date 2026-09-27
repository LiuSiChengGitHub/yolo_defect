#include "main_window.h"

#include <QApplication>
#include <QCommandLineParser>

int main(int argc, char* argv[]) {
  QApplication app(argc, argv);
  QCoreApplication::setOrganizationName("YoloDefect");
  QCoreApplication::setApplicationName("DefectWorkbench");
  QCoreApplication::setApplicationVersion("0.1.0");
  QCommandLineParser parser;
  parser.setApplicationDescription("Industrial defect inspection workbench");
  parser.addHelpOption();
  parser.addVersionOption();
  parser.addOption({"config", "Initial RuntimeConfig file.", "path"});
  parser.addOption({"image", "Initial input image.", "path"});
  parser.addOption({"output-dir", "Directory for per-run result folders.", "path"});
  parser.process(app);
  yolo_defect_cpp::qt::MainWindow window;
  window.setInputs(parser.value("config"), parser.value("image"), parser.value("output-dir"));
  window.show();
  return app.exec();
}
