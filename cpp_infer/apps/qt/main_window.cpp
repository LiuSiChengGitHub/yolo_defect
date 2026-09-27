#include "main_window.h"

#include "detection_table_model.h"
#include "detection_worker.h"
#include "image_view.h"

#include <QCloseEvent>
#include <QDesktopServices>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFrame>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QSettings>
#include <QSplitter>
#include <QStandardPaths>
#include <QStyle>
#include <QStringList>
#include <QTableView>
#include <QThread>
#include <QTimer>
#include <QUrl>
#include <QVBoxLayout>

namespace yolo_defect_cpp::qt {
namespace {

QLabel* label(const QString& text, const QString& name, QWidget* parent) {
  auto* result = new QLabel(text, parent);
  result->setObjectName(name);
  result->setTextFormat(Qt::PlainText);
  return result;
}

QFrame* card(QWidget* parent) {
  auto* frame = new QFrame(parent);
  frame->setObjectName("card");
  return frame;
}

}  // namespace

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
  qRegisterMetaType<RuntimeContract>();
  qRegisterMetaType<DetectionRequest>();
  qRegisterMetaType<DetectionResponse>();
  setWindowTitle(tr("工业缺陷检测工作台"));
  resize(1280, 860);
  setMinimumSize(980, 700);
  buildUi();
  QSettings settings;
  restoreGeometry(settings.value("window/geometry").toByteArray());
  output_directory_->setText(settings.value(
      "paths/output", QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation)
                          + "/DefectWorkbench").toString());
  invalidateResult();
}

MainWindow::~MainWindow() {
  // Normal window closing is asynchronous. This join also protects explicit
  // owner destruction / application shutdown while a task is still running.
  if (thread_) {
    thread_->quit();
    thread_->wait();
  }
}

void MainWindow::buildUi() {
  setStyleSheet(QStringLiteral(R"(
    QMainWindow { background: #eef2f6; }
    QWidget { font-family: "Segoe UI", "Microsoft YaHei UI"; font-size: 13px; color: #24364b; }
    QFrame#header { background: #122235; border-radius: 10px; }
    QLabel#title { color: #f6f9fc; font-size: 25px; font-weight: 600; }
    QLabel#subtitle { color: #a1b4c9; font-size: 12px; }
    QLabel#brand { color: #6de2c2; font-size: 11px; font-weight: 600; }
    QLabel#stateBadge { color: #146450; background: #dff5ed; border-radius: 12px; padding: 6px 14px; font-weight: 600; }
    QLabel#stateBadge[state="busy"] { color: #825c0b; background: #fff0c9; }
    QLabel#stateBadge[state="error"] { color: #a83838; background: #ffe2e2; }
    QFrame#card { background: white; border: 1px solid #dde5ed; border-radius: 9px; }
    QLabel#sectionTitle { font-size: 15px; font-weight: 600; color: #162e48; }
    QLabel#muted, QLabel#outputDetails { color: #6a7d92; font-size: 12px; }
    QLabel#modelDetails { font-size: 12px; color: #51677e; }
    QLabel#resultSummary { color: #1e7566; font-weight: 600; }
    QLabel#statusMessage[state="error"] { color: #b13737; }
    QLineEdit { border: 1px solid #cfd9e4; background: #f8fafc; border-radius: 5px; padding: 8px; selection-background-color: #c7ebe5; }
    QLineEdit:focus { border: 1px solid #219984; }
    QLineEdit:disabled { color: #8090a3; background: #f0f3f7; }
    QPushButton { border: 1px solid #cad6e1; border-radius: 5px; background: white; padding: 8px 12px; font-weight: 600; }
    QPushButton:hover { background: #edf6f4; border-color: #42a38f; }
    QPushButton:disabled { color: #a1acb8; background: #f4f6f8; border-color: #e0e6ec; }
    QPushButton#runButton { color: white; background: #127e6b; border: none; padding: 13px; font-size: 14px; }
    QPushButton#runButton:hover { background: #0c9179; }
    QPushButton#runButton:disabled { background: #91b9b0; }
    QTableView { border: none; background: white; alternate-background-color: #f5f8fb; gridline-color: #edf1f5; selection-background-color: #ddf2ec; selection-color: #185449; }
    QHeaderView::section { border: none; border-bottom: 1px solid #e2e8ef; background: #f4f7fa; color: #607489; padding: 8px; font-size: 12px; }
    QProgressBar { border: none; background: #e8eef3; border-radius: 2px; max-height: 4px; }
    QProgressBar::chunk { background: #209d87; }
    QScrollArea { border: none; background: transparent; }
    QSplitter::handle { background: #eef2f6; }
  )"));
  auto* central = new QWidget(this);
  setCentralWidget(central);
  auto* root = new QVBoxLayout(central);
  root->setContentsMargins(22, 18, 22, 16);
  root->setSpacing(14);

  auto* header = new QFrame(central);
  header->setObjectName("header");
  auto* header_layout = new QHBoxLayout(header);
  header_layout->setContentsMargins(22, 18, 22, 18);
  auto* titles = new QVBoxLayout;
  titles->addWidget(label("VISION / INSPECTION", "brand", header));
  titles->addWidget(label(tr("工业缺陷检测工作台"), "title", header));
  titles->addWidget(label(tr("单图检测  ·  模型配置驱动  ·  本地推理"), "subtitle", header));
  header_layout->addLayout(titles, 1);
  state_badge_ = label(tr("待就绪"), "stateBadge", header);
  header_layout->addWidget(state_badge_, 0, Qt::AlignVCenter);
  root->addWidget(header);

  auto* body = new QHBoxLayout;
  body->setSpacing(16);
  auto* sidebar_scroll = new QScrollArea(central);
  sidebar_scroll->setWidgetResizable(true);
  sidebar_scroll->setFixedWidth(310);
  sidebar_scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  auto* sidebar = new QWidget;
  auto* side = new QVBoxLayout(sidebar);
  side->setContentsMargins(0, 0, 4, 0);
  side->setSpacing(12);
  sidebar_scroll->setWidget(sidebar);
  body->addWidget(sidebar_scroll);

  input_panel_ = card(sidebar);
  auto* inputs = new QVBoxLayout(input_panel_);
  inputs->setContentsMargins(16, 16, 16, 16);
  inputs->setSpacing(10);
  inputs->addWidget(label(tr("01  检测任务"), "sectionTitle", input_panel_));
  auto add_path = [&](const QString& title, const QString& object_name,
                      const QString& placeholder, QLineEdit*& field,
                      const QString& filter, bool directory) {
    inputs->addWidget(label(title, "fieldLabel", input_panel_));
    field = new QLineEdit(input_panel_);
    field->setObjectName(object_name);
    field->setPlaceholderText(placeholder);
    auto* path_row = new QHBoxLayout;
    path_row->setSpacing(6);
    path_row->addWidget(field, 1);
    auto* browse = new QPushButton(tr("浏览"), input_panel_);
    browse->setToolTip(title);
    path_row->addWidget(browse);
    inputs->addLayout(path_row);
    connect(browse, &QPushButton::clicked, this, [this, field, filter, directory, object_name] {
      QSettings settings;
      const QString initial = field->text().isEmpty()
          ? settings.value("browse/" + object_name, QDir::currentPath()).toString()
          : (directory ? field->text() : QFileInfo(field->text()).absolutePath());
      const auto path = directory
          ? QFileDialog::getExistingDirectory(this, tr("选择输出目录"), initial)
          : QFileDialog::getOpenFileName(this, tr("选择文件"), initial, filter);
      if (path.isEmpty()) return;
      field->setText(QDir::toNativeSeparators(path));
      settings.setValue("browse/" + object_name,
                        directory ? path : QFileInfo(path).absolutePath());
    });
    connect(field, &QLineEdit::textChanged, this, &MainWindow::invalidateResult);
    connect(field, &QLineEdit::textChanged, field, &QLineEdit::setToolTip);
  };
  add_path(tr("模型配置"), "configPath", tr("选择 FP32 / INT8 配置"),
           config_path_, tr("配置文件 (*.txt);;所有文件 (*)"), false);
  add_path(tr("输入图片"), "imagePath", tr("选择待检图片"), image_path_,
           tr("图片 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp);;所有文件 (*)"), false);
  add_path(tr("结果保存位置"), "outputDirectory", tr("选择保存目录"),
           output_directory_, {}, true);
  auto* output_hint = label(tr("每次检测自动建立子目录，保存 JSON 和标注图。"), "muted", input_panel_);
  output_hint->setWordWrap(true);
  output_hint->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
  inputs->addWidget(output_hint);
  run_button_ = new QPushButton(tr("运行检测"), input_panel_);
  run_button_->setObjectName("runButton");
  inputs->addWidget(run_button_);
  connect(run_button_, &QPushButton::clicked, this, &MainWindow::startDetection);
  side->addWidget(input_panel_);

  auto* model_card = card(sidebar);
  auto* model_layout = new QVBoxLayout(model_card);
  model_layout->setContentsMargins(16, 16, 16, 16);
  model_layout->addWidget(label(tr("02  生效参数"), "sectionTitle", model_card));
  model_details_ = label({}, "modelDetails", model_card);
  model_details_->setWordWrap(true);
  model_details_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
  model_details_->setTextInteractionFlags(Qt::TextSelectableByMouse);
  model_layout->addWidget(model_details_);
  side->addWidget(model_card);
  side->addStretch();

  auto* workspace = new QSplitter(Qt::Vertical, central);
  workspace->setChildrenCollapsible(false);
  auto* previews = new QWidget(workspace);
  auto* preview_layout = new QHBoxLayout(previews);
  preview_layout->setContentsMargins(0, 0, 0, 0);
  preview_layout->setSpacing(12);
  auto add_preview = [&](const QString& title, const QString& name, ImageView*& view) {
    auto* frame = card(previews);
    auto* layout = new QVBoxLayout(frame);
    layout->setContentsMargins(12, 12, 12, 12);
    layout->addWidget(label(title, "sectionTitle", frame));
    view = new ImageView(frame);
    view->setObjectName(name);
    layout->addWidget(view, 1);
    preview_layout->addWidget(frame, 1);
  };
  add_preview(tr("原始图像"), "originalView", original_view_);
  add_preview(tr("检测结果"), "annotatedView", annotated_view_);
  workspace->addWidget(previews);

  auto* results = card(workspace);
  auto* result_layout = new QVBoxLayout(results);
  result_layout->setContentsMargins(16, 14, 16, 14);
  auto* result_header = new QHBoxLayout;
  result_header->addWidget(label(tr("检测明细"), "sectionTitle", results));
  result_summary_ = label({}, "resultSummary", results);
  result_header->addWidget(result_summary_, 1, Qt::AlignRight);
  result_layout->addLayout(result_header);
  auto* table = new QTableView(results);
  table->setObjectName("resultTable");
  result_model_ = new DetectionTableModel(table);
  table->setModel(result_model_);
  table->setAlternatingRowColors(true);
  table->setSelectionBehavior(QAbstractItemView::SelectRows);
  table->setSelectionMode(QAbstractItemView::SingleSelection);
  table->setEditTriggers(QAbstractItemView::NoEditTriggers);
  table->verticalHeader()->hide();
  table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
  table->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
  table->setMinimumHeight(110);
  result_layout->addWidget(table, 1);
  output_details_ = label({}, "outputDetails", results);
  output_details_->setWordWrap(true);
  output_details_->setTextInteractionFlags(Qt::TextSelectableByMouse);
  result_layout->addWidget(output_details_);
  auto* output_actions = new QHBoxLayout;
  output_actions->addStretch();
  open_json_button_ = new QPushButton(tr("打开 JSON"), results);
  open_json_button_->setObjectName("openJsonButton");
  open_output_button_ = new QPushButton(tr("打开结果目录"), results);
  open_output_button_->setObjectName("openOutputButton");
  output_actions->addWidget(open_json_button_);
  output_actions->addWidget(open_output_button_);
  result_layout->addLayout(output_actions);
  connect(open_json_button_, &QPushButton::clicked, this,
          [this] { openPath(completed_json_); });
  connect(open_output_button_, &QPushButton::clicked, this,
          [this] { openPath(completed_directory_); });
  workspace->addWidget(results);
  workspace->setStretchFactor(0, 3);
  workspace->setStretchFactor(1, 2);
  body->addWidget(workspace, 1);
  root->addLayout(body, 1);

  progress_ = new QProgressBar(central);
  progress_->setTextVisible(false);
  progress_->setRange(0, 1);
  progress_->setValue(0);
  root->addWidget(progress_);
  status_message_ = label({}, "statusMessage", central);
  status_message_->setWordWrap(true);
  status_message_->setTextInteractionFlags(Qt::TextSelectableByMouse);
  root->addWidget(status_message_);
}

void MainWindow::setInputs(const QString& config, const QString& image,
                           const QString& output_directory) {
  if (isBusy()) return;
  config_path_->setText(config);
  image_path_->setText(image);
  if (!output_directory.isEmpty()) output_directory_->setText(output_directory);
}

void MainWindow::invalidateResult() {
  if (isBusy()) return;
  completed_directory_.clear();
  completed_json_.clear();
  result_model_->setDetections({});
  original_view_->clear(tr("选择图片并运行检测"));
  annotated_view_->clear(tr("检测完成后显示标注图"));
  result_summary_->setText(tr("尚未检测"));
  model_details_->setText(tr("运行时加载配置，并在此显示本次使用的模型、阈值和类别。"));
  output_details_->clear();
  open_json_button_->setEnabled(false);
  open_output_button_->setEnabled(false);
  status_message_->setProperty("state", "ready");
  status_message_->style()->unpolish(status_message_);
  status_message_->style()->polish(status_message_);
  status_message_->setText(tr("准备就绪。选择配置和图片后运行检测。"));
  state_badge_->setProperty("state", "ready");
  state_badge_->setText(tr("待检测"));
  state_badge_->style()->unpolish(state_badge_);
  state_badge_->style()->polish(state_badge_);
  run_button_->setEnabled(!config_path_->text().trimmed().isEmpty() &&
                           !image_path_->text().trimmed().isEmpty() &&
                           !output_directory_->text().trimmed().isEmpty());
}

void MainWindow::setBusy(bool busy) {
  input_panel_->setEnabled(!busy);
  run_button_->setText(busy ? tr("正在检测…") : tr("运行检测"));
  progress_->setRange(0, busy ? 0 : 1);
  progress_->setValue(busy ? -1 : 0);
}

void MainWindow::startDetection() {
  if (isBusy() || !run_button_->isEnabled()) return;
  invalidateResult();
  const DetectionRequest request{config_path_->text().trimmed(), image_path_->text().trimmed(),
                                  output_directory_->text().trimmed()};
  task_succeeded_ = false;
  close_pending_ = false;
  thread_ = new QThread(this);
  auto* worker = new DetectionWorker;
  worker->moveToThread(thread_);
  connect(thread_, &QThread::started, worker, [worker, request] { worker->run(request); },
          Qt::QueuedConnection);
  connect(worker, &DetectionWorker::stageChanged, this, [this](const QString& stage) {
    if (!close_pending_) status_message_->setText(stage);
  });
  connect(worker, &DetectionWorker::contractLoaded, this, &MainWindow::showContract);
  connect(worker, &DetectionWorker::completed, this, &MainWindow::showResult);
  connect(worker, &DetectionWorker::failed, this, &MainWindow::showError);
  // quit() is thread-safe. A direct call lets shutdown finish even if the GUI
  // owner is being destroyed and is joining the worker thread.
  connect(worker, &DetectionWorker::finished, thread_, &QThread::quit, Qt::DirectConnection);
  connect(thread_, &QThread::finished, worker, &QObject::deleteLater);
  connect(thread_, &QThread::finished, this, [this] {
    thread_->wait();
    thread_->deleteLater();
    thread_ = nullptr;
    setBusy(false);
    emit taskFinished(task_succeeded_);
    if (close_pending_) QTimer::singleShot(0, this, &QWidget::close);
  });
  setBusy(true);
  state_badge_->setProperty("state", "busy");
  state_badge_->setText(tr("检测中"));
  state_badge_->style()->unpolish(state_badge_);
  state_badge_->style()->polish(state_badge_);
  status_message_->setText(tr("正在加载配置与模型…"));
  thread_->start();
}

void MainWindow::showContract(const RuntimeContract& contract) {
  QStringList classes;
  for (const auto& name : contract.artifact.class_names) classes << QString::fromStdString(name);
  QStringList dimensions;
  for (const auto value : contract.artifact.input.shape) dimensions << QString::number(value);
  model_details_->setText(tr("%1\n\n执行后端  %2\n输入尺寸  %3\n置信度阈值  %4\nNMS 阈值  %5\nNMS 模式  %6\n\n类别\n%7\n\n模型文件\n%8\n\n配置\n%9")
      .arg(QString::fromStdString(contract.artifact.model_id),
           QString::fromStdString(to_string(contract.runtime.provider)), dimensions.join(" × "))
      .arg(contract.runtime.score_threshold).arg(contract.runtime.nms_threshold)
      .arg(QString::fromStdString(to_string(contract.artifact.nms_mode)), classes.join(", "),
           QDir::fromNativeSeparators(from_path(contract.artifact.model_path)),
           QDir::fromNativeSeparators(from_path(contract.runtime.declaration_path))));
}

void MainWindow::showResult(const DetectionResponse& response) {
  task_succeeded_ = true;
  showContract(response.contract);
  original_view_->setImage(response.original_image);
  annotated_view_->setImage(response.annotated_image);
  result_model_->setDetections(response.result.detection_result.detections);
  completed_directory_ = response.output_directory;
  if (response.result.outputs.json_path) completed_json_ = from_path(*response.result.outputs.json_path);
  const auto count = response.result.detection_result.detections.size();
  result_summary_->setText(count == 0 ? tr("未检出缺陷") : tr("检出 %1 个目标").arg(count));
  output_details_->setText(tr("本次输出：%1").arg(completed_directory_));
  open_output_button_->setEnabled(true);
  open_json_button_->setEnabled(!completed_json_.isEmpty());
  state_badge_->setProperty("state", "ready");
  state_badge_->setText(tr("检测完成"));
  state_badge_->style()->unpolish(state_badge_);
  state_badge_->style()->polish(state_badge_);
  if (!close_pending_) {
    status_message_->setText(tr("检测完成 · %1 · 任务总耗时 %2 s（包含模型加载、推理与文件写出）")
        .arg(QString::fromStdString(response.result.detection_result.actual_provider))
        .arg(response.elapsed_ms / 1000.0, 0, 'f', 2));
  }
}

void MainWindow::showError(const QString& message) {
  task_succeeded_ = false;
  state_badge_->setProperty("state", "error");
  state_badge_->setText(tr("检测失败"));
  state_badge_->style()->unpolish(state_badge_);
  state_badge_->style()->polish(state_badge_);
  status_message_->setProperty("state", "error");
  status_message_->style()->unpolish(status_message_);
  status_message_->style()->polish(status_message_);
  status_message_->setText(tr("检测失败，可修改输入后重试。\n%1").arg(message));
  result_summary_->setText(tr("检测未完成"));
  annotated_view_->clear(tr("本次检测失败\n请查看下方错误信息"));
}

void MainWindow::openPath(const QString& path) {
  if (!path.isEmpty() && !QDesktopServices::openUrl(QUrl::fromLocalFile(path))) {
    status_message_->setText(tr("无法打开：%1\n请在文件管理器中查看此路径。" ).arg(path));
  }
}

void MainWindow::closeEvent(QCloseEvent* event) {
  if (isBusy()) {
    close_pending_ = true;
    status_message_->setText(tr("正在完成当前检测与文件写出，完成后自动关闭…"));
    event->ignore();
    return;
  }
  QSettings settings;
  settings.setValue("window/geometry", saveGeometry());
  settings.setValue("paths/output", output_directory_->text());
  event->accept();
}

}  // namespace yolo_defect_cpp::qt
