#include "main_window.h"

#include "detection_table_model.h"
#include "batch_table_model.h"
#include "batch_worker.h"
#include "detection_worker.h"
#include "image_view.h"
#include "model_info_panel.h"
#include "path_edit.h"
#include "preview_worker.h"

#include <QCloseEvent>
#include <QComboBox>
#include <QDesktopServices>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFrame>
#include <QHeaderView>
#include <QLabel>
#include <QItemSelectionModel>
#include <QLineEdit>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QSettings>
#include <QSplitter>
#include <QSpinBox>
#include <QStandardPaths>
#include <QStyle>
#include <QTableView>
#include <QTabWidget>
#include <QThread>
#include <QTimer>
#include <QUrl>
#include <QVBoxLayout>
#include <type_traits>

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
  qRegisterMetaType<BatchDetectionResponse>();
  qRegisterMetaType<PreviewRequest>();
  qRegisterMetaType<PreviewResponse>();
  setWindowTitle(tr("工业缺陷检测工作台"));
  resize(1280, 860);
  setMinimumSize(980, 700);
  buildUi();
  updateInputMode();
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
  if (batch_control_) batch_control_->requestStop();
  if (thread_) {
    thread_->quit();
    thread_->wait();
  }
  if (preview_thread_) {
    preview_thread_->quit();
    preview_thread_->wait();
  }
}

void MainWindow::buildUi() {
  setStyleSheet(QStringLiteral(R"(
    QMainWindow { background: #eef2f6; }
    QWidget { font-family: "Segoe UI", "Microsoft YaHei UI"; font-size: 13px; color: #24364b; }
    QWidget#workbenchSurface, QWidget#sidebarViewport, QWidget#sidebarContent { background: #eef2f6; }
    QFrame#header { background: #122235; border-radius: 10px; }
    QLabel#title { color: #f6f9fc; font-size: 23px; font-weight: 600; }
    QLabel#subtitle { color: #a1b4c9; font-size: 12px; }
    QLabel#brand { color: #6de2c2; font-size: 11px; font-weight: 600; }
    QLabel#stateBadge { color: #146450; background: #dff5ed; border-radius: 12px; padding: 6px 14px; font-weight: 600; }
    QLabel#stateBadge[state="busy"] { color: #825c0b; background: #fff0c9; }
    QLabel#stateBadge[state="warning"] { color: #825c0b; background: #fff0c9; }
    QLabel#stateBadge[state="error"] { color: #a83838; background: #ffe2e2; }
    QFrame#card { background: white; border: 1px solid #dde5ed; border-radius: 10px; }
    QLabel#sectionTitle { font-size: 15px; font-weight: 600; color: #162e48; }
    QLabel#sectionNumber { color: #127e6b; background: #e6f3ef; border-radius: 6px; font-size: 11px; font-weight: 600; }
    QLabel#fieldLabel { color: #61758a; font-size: 12px; font-weight: 600; }
    QLabel#muted { color: #6a7d92; font-size: 12px; }
    QLabel#itemDetails { color: #607489; font-size: 12px; }
    QLabel#itemDetails[state="error"] { color: #b13737; }
    QLabel#batchSummary { color: #48627b; font-size: 12px; }
    QPlainTextEdit#itemError { border: 1px solid #f0ded8; border-radius: 5px; background: #fff8f5; color: #99412d; padding: 4px; font-size: 12px; }
    QLabel#modelDetails { font-size: 12px; color: #243e56; font-weight: 600; }
    QLabel#parameterLabel { color: #728396; font-size: 12px; }
    QLabel#parameterValue, QLabel#scoreThreshold, QLabel#nmsThreshold { color: #243e56; font-size: 12px; font-weight: 600; }
    QLabel#classNames, QLabel#modelPath, QLabel#runtimeConfigPath { color: #4e647a; font-size: 12px; }
    QToolButton#modelDetailsToggle { color: #347b70; border: none; background: transparent; padding: 4px 0; font-size: 12px; }
    QToolButton#modelDetailsToggle:hover { color: #0a6655; }
    QToolButton#modelDetailsToggle:disabled { color: #9dabb7; }
    QLabel#resultSummary { color: #1e7566; font-weight: 600; }
    QLabel#statusMessage[state="error"] { color: #b13737; }
    QLineEdit { border: 1px solid #d8e1e9; background: #f8fafc; border-radius: 6px; padding: 8px; selection-background-color: #c7ebe5; selection-color: #164d43; }
    QLineEdit:focus { border: 1px solid #219984; }
    QLineEdit:disabled { color: #8090a3; background: #f0f3f7; }
    QComboBox, QSpinBox { border: 1px solid #d8e1e9; border-radius: 6px; background: #f8fafc; padding: 6px; min-height: 20px; }
    QComboBox QAbstractItemView { background: white; selection-background-color: #ddf2ec; selection-color: #185449; }
    QComboBox:disabled, QSpinBox:disabled { color: #8090a3; background: #f0f3f7; }
    QPushButton { border: 1px solid #d4dfe8; border-radius: 6px; background: white; padding: 8px 12px; font-weight: 500; }
    QPushButton#browseButton { background: #f4f7fa; color: #48627b; padding: 8px 10px; font-size: 12px; }
    QPushButton:hover { background: #edf6f4; border-color: #42a38f; }
    QPushButton:disabled { color: #a1acb8; background: #f4f6f8; border-color: #e0e6ec; }
    QPushButton#runButton { color: white; background: #127e6b; border: none; padding: 11px; font-size: 14px; font-weight: 600; }
    QPushButton#runButton:hover { background: #0c9179; }
    QPushButton#runButton:disabled { background: #91b9b0; }
    QPushButton#stopButton { color: #a45830; border-color: #ead7c9; }
    QPushButton#stopButton:disabled { color: #a1acb8; border-color: #e0e6ec; }
    QPushButton#viewAction { padding: 3px 7px; font-size: 11px; }
    QPushButton#openSummaryButton, QPushButton#openJsonButton, QPushButton#openOutputButton { padding: 6px 10px; font-size: 12px; }
    QTabWidget::pane { border: none; background: white; }
    QTabBar::tab { padding: 7px 14px; color: #728396; border-bottom: 2px solid transparent; }
    QTabBar::tab:selected { color: #127e6b; border-bottom-color: #127e6b; }
    QTableView { border: none; background: white; alternate-background-color: #f5f8fb; gridline-color: #edf1f5; selection-background-color: #ddf2ec; selection-color: #185449; }
    QHeaderView::section { border: none; border-bottom: 1px solid #e2e8ef; background: #f4f7fa; color: #607489; padding: 6px 8px; font-size: 12px; }
    QProgressBar { border: none; background: #e8eef3; border-radius: 2px; max-height: 4px; }
    QProgressBar::chunk { background: #209d87; }
    QScrollArea#sidebarScroll { border: none; background: #eef2f6; }
    QScrollBar:vertical { border: none; background: transparent; width: 8px; margin: 2px 0; }
    QScrollBar::handle:vertical { background: #c4d1dc; border-radius: 4px; min-height: 36px; }
    QScrollBar::handle:vertical:hover { background: #98adbf; }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
    QSplitter::handle { background: #eef2f6; }
    QSplitter::handle:hover { background: #c4d1dc; }
  )"));
  auto* central = new QWidget(this);
  central->setObjectName("workbenchSurface");
  central->setAttribute(Qt::WA_StyledBackground);
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
  titles->addWidget(label(tr("单图 / 批处理  ·  模型配置驱动  ·  本地推理"), "subtitle", header));
  header_layout->addLayout(titles, 1);
  state_badge_ = label(tr("待就绪"), "stateBadge", header);
  header_layout->addWidget(state_badge_, 0, Qt::AlignVCenter);
  root->addWidget(header);

  auto* body = new QHBoxLayout;
  body->setSpacing(16);
  auto* sidebar_scroll = new QScrollArea(central);
  sidebar_scroll->setObjectName("sidebarScroll");
  sidebar_scroll->viewport()->setObjectName("sidebarViewport");
  sidebar_scroll->viewport()->setAttribute(Qt::WA_StyledBackground);
  sidebar_scroll->setWidgetResizable(true);
  sidebar_scroll->setFixedWidth(340);
  sidebar_scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  auto* sidebar = new QWidget;
  sidebar->setObjectName("sidebarContent");
  sidebar->setAttribute(Qt::WA_StyledBackground);
  auto* side = new QVBoxLayout(sidebar);
  side->setContentsMargins(0, 0, 6, 0);
  side->setSpacing(12);
  sidebar_scroll->setWidget(sidebar);
  body->addWidget(sidebar_scroll);

  input_panel_ = card(sidebar);
  auto* input_layout = new QVBoxLayout(input_panel_);
  input_layout->setContentsMargins(18, 16, 18, 16);
  input_layout->setSpacing(10);
  auto* task_heading = new QHBoxLayout;
  auto* task_number = label("01", "sectionNumber", input_panel_);
  task_number->setFixedSize(26, 26);
  task_number->setAlignment(Qt::AlignCenter);
  task_heading->setSpacing(10);
  task_heading->addWidget(task_number);
  task_heading->addWidget(label(tr("检测任务"), "sectionTitle", input_panel_), 1);
  input_layout->addLayout(task_heading);
  input_fields_ = new QWidget(input_panel_);
  auto* inputs = new QVBoxLayout(input_fields_);
  inputs->setContentsMargins(0, 0, 0, 0);
  inputs->setSpacing(10);
  inputs->addWidget(label(tr("输入方式"), "fieldLabel", input_fields_));
  input_mode_ = new QComboBox(input_fields_);
  input_mode_->setObjectName("inputMode");
  input_mode_->addItems({tr("单张图片"), tr("图片目录"), tr("Manifest 清单")});
  inputs->addWidget(input_mode_);
  input_layout->addWidget(input_fields_);
  auto add_path = [&](const QString& title, const QString& object_name,
                      const QString& placeholder, QLineEdit*& field,
                      const QString& filter, bool directory) {
    auto* field_group = new QVBoxLayout;
    field_group->setSpacing(5);
    auto* field_label = label(title, "fieldLabel", input_fields_);
    if (object_name == "imagePath") input_label_ = field_label;
    field_group->addWidget(field_label);
    field = new PathEdit(input_panel_);
    field->setObjectName(object_name);
    field->setPlaceholderText(placeholder);
    auto* path_row = new QHBoxLayout;
    path_row->setSpacing(6);
    path_row->addWidget(field, 1);
    auto* browse = new QPushButton(tr("浏览"), input_panel_);
    browse->setObjectName("browseButton");
    browse->setToolTip(title);
    path_row->addWidget(browse);
    field_group->addLayout(path_row);
    inputs->addLayout(field_group);
    connect(browse, &QPushButton::clicked, this, [this, field, filter, directory, object_name] {
      QSettings settings;
      const bool select_directory = directory ||
          (object_name == "imagePath" && input_mode_->currentIndex() == 1);
      const QString selected_filter = object_name == "imagePath" && input_mode_->currentIndex() == 2
          ? tr("清单文件 (*.txt);;所有文件 (*)") : filter;
      const QString initial = field->text().isEmpty()
          ? settings.value("browse/" + object_name, QDir::currentPath()).toString()
          : (select_directory ? field->text() : QFileInfo(field->text()).absolutePath());
      const auto path = select_directory
          ? QFileDialog::getExistingDirectory(this, tr("选择目录"), initial)
          : QFileDialog::getOpenFileName(this, tr("选择文件"), initial, selected_filter);
      if (path.isEmpty()) return;
      field->setText(QDir::toNativeSeparators(path));
      settings.setValue("browse/" + object_name,
                        select_directory ? path : QFileInfo(path).absolutePath());
    });
    connect(field, &QLineEdit::textChanged, this, &MainWindow::invalidateResult);
  };
  add_path(tr("模型配置"), "configPath", tr("选择 FP32 / INT8 配置"),
           config_path_, tr("配置文件 (*.txt);;所有文件 (*)"), false);
  add_path(tr("输入图片"), "imagePath", tr("选择待检图片"), image_path_,
           tr("图片 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp);;所有文件 (*)"), false);
  add_path(tr("结果保存位置"), "outputDirectory", tr("选择保存目录"),
           output_directory_, {}, true);
  batch_options_ = new QWidget(input_fields_);
  auto* batch_settings = new QHBoxLayout(batch_options_);
  batch_settings->setContentsMargins(0, 0, 0, 0);
  auto add_number = [&](const QString& title, const char* name, int maximum, int value) {
    auto* column = new QVBoxLayout;
    column->addWidget(label(title, "fieldLabel", batch_options_));
    auto* spin = new QSpinBox(batch_options_);
    spin->setObjectName(name);
    spin->setRange(1, maximum);
    spin->setValue(value);
    column->addWidget(spin);
    batch_settings->addLayout(column);
    connect(spin, &QSpinBox::valueChanged, this, &MainWindow::invalidateResult);
    return spin;
  };
  workers_ = add_number(tr("并发数量"), "workersSpin", 64, 1);
  queue_capacity_ = add_number(tr("队列容量"), "queueSpin", 4096, 2);
  workers_->setToolTip(tr("每个并发 worker 使用独立模型 session；数量越多，内存占用越高。"));
  inputs->addWidget(batch_options_);
  auto* output_hint = label(tr("自动保存 JSON 与标注图，每次检测独立归档。"), "muted", input_panel_);
  output_hint->setWordWrap(true);
  output_hint->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
  input_layout->addWidget(output_hint);
  run_button_ = new QPushButton(tr("运行检测"), input_panel_);
  run_button_->setObjectName("runButton");
  auto* task_actions = new QHBoxLayout;
  task_actions->addWidget(run_button_, 1);
  stop_button_ = new QPushButton(tr("停止"), input_panel_);
  stop_button_->setObjectName("stopButton");
  stop_button_->setToolTip(tr("停止派发新图片，等待正在处理的图片完成并保存结果。"));
  stop_button_->setEnabled(false);
  task_actions->addWidget(stop_button_);
  input_layout->addLayout(task_actions);
  connect(run_button_, &QPushButton::clicked, this, &MainWindow::startDetection);
  connect(stop_button_, &QPushButton::clicked, this, &MainWindow::stopDetection);
  side->addWidget(input_panel_);

  model_panel_ = new ModelInfoPanel(sidebar);
  side->addWidget(model_panel_);
  side->addStretch();

  auto* workspace = new QSplitter(Qt::Vertical, central);
  workspace->setObjectName("workspaceSplitter");
  workspace->setChildrenCollapsible(false);
  workspace->setHandleWidth(8);
  auto* previews = new QWidget(workspace);
  auto* preview_layout = new QHBoxLayout(previews);
  preview_layout->setContentsMargins(0, 0, 0, 0);
  preview_layout->setSpacing(12);
  auto add_preview = [&](const QString& title, const QString& name, ImageView*& view) {
    auto* frame = card(previews);
    auto* layout = new QVBoxLayout(frame);
    layout->setContentsMargins(12, 12, 12, 12);
    auto* view_heading = new QHBoxLayout;
    view_heading->addWidget(label(title, "sectionTitle", frame), 1);
    view = new ImageView(frame);
    view->setObjectName(name);
    view->setToolTip(tr("滚轮缩放 · 拖动平移 · 点击检测框选择 · 双击适应窗口"));
    auto add_action = [&](const QString& text, const QString& tip, auto action) {
      auto* button = new QPushButton(text, frame);
      button->setObjectName("viewAction");
      button->setToolTip(tip);
      button->setFocusPolicy(Qt::NoFocus);
      view_heading->addWidget(button);
      connect(button, &QPushButton::clicked, view, action);
    };
    add_action(QStringLiteral("−"), tr("缩小"), &ImageView::zoomOut);
    add_action(QStringLiteral("+"), tr("放大"), &ImageView::zoomIn);
    add_action(QStringLiteral("1:1"), tr("按原始像素查看"), &ImageView::actualSize);
    add_action(tr("适应"), tr("显示完整图片"), &ImageView::fitToWindow);
    layout->addLayout(view_heading);
    layout->addWidget(view, 1);
    connect(view, &ImageView::detectionSelected, this, &MainWindow::selectDetection);
    preview_layout->addWidget(frame, 1);
  };
  add_preview(tr("原始图像"), "originalView", original_view_);
  add_preview(tr("检测结果"), "annotatedView", annotated_view_);
  workspace->addWidget(previews);

  auto* results = card(workspace);
  auto* result_layout = new QVBoxLayout(results);
  result_layout->setContentsMargins(16, 12, 16, 12);
  result_layout->setSpacing(6);
  auto* result_header = new QHBoxLayout;
  result_header->setSpacing(12);
  result_header->addWidget(label(tr("任务结果"), "sectionTitle", results));
  batch_summary_ = label({}, "batchSummary", results);
  result_header->addWidget(batch_summary_, 1);
  result_summary_ = label({}, "resultSummary", results);
  result_header->addWidget(result_summary_, 0, Qt::AlignRight);
  result_layout->addLayout(result_header);
  result_tabs_ = new QTabWidget(results);
  result_tabs_->setObjectName("resultTabs");
  // QTabWidget can otherwise shrink its page below QTableView's minimum at
  // high DPI. Reserve space for the tabs, header and several complete rows.
  result_tabs_->setMinimumHeight(200);
  batch_table_ = new QTableView(result_tabs_);
  batch_table_->setObjectName("batchTable");
  batch_model_ = new BatchTableModel(batch_table_);
  batch_table_->setModel(batch_model_);
  batch_table_->setAlternatingRowColors(true);
  batch_table_->setSelectionBehavior(QAbstractItemView::SelectRows);
  batch_table_->setSelectionMode(QAbstractItemView::SingleSelection);
  batch_table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
  batch_table_->setTextElideMode(Qt::ElideMiddle);
  batch_table_->setWordWrap(false);
  batch_table_->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
  batch_table_->verticalHeader()->setDefaultSectionSize(28);
  batch_table_->verticalHeader()->hide();
  batch_table_->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
  batch_table_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
  batch_table_->horizontalHeader()->setSectionResizeMode(5, QHeaderView::Stretch);
  batch_table_->setMinimumHeight(100);
  result_tabs_->addTab(batch_table_, tr("逐图结果"));
  connect(batch_table_->selectionModel(), &QItemSelectionModel::currentRowChanged,
          this, [this](const QModelIndex& current) { selectBatchItem(current.row()); });
  auto* table = new QTableView(result_tabs_);
  detection_table_ = table;
  table->setObjectName("resultTable");
  result_model_ = new DetectionTableModel(table);
  table->setModel(result_model_);
  table->setAlternatingRowColors(true);
  table->setSelectionBehavior(QAbstractItemView::SelectRows);
  table->setSelectionMode(QAbstractItemView::SingleSelection);
  table->setEditTriggers(QAbstractItemView::NoEditTriggers);
  table->setWordWrap(false);
  table->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
  table->verticalHeader()->setDefaultSectionSize(28);
  table->verticalHeader()->hide();
  table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
  table->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
  table->setMinimumHeight(100);
  result_tabs_->addTab(table, tr("检测明细"));
  connect(table->selectionModel(), &QItemSelectionModel::currentRowChanged,
          this, [this](const QModelIndex& current) {
    original_view_->setSelectedDetection(current.row());
    annotated_view_->setSelectedDetection(current.row());
  });
  result_layout->addWidget(result_tabs_, 1);
  item_details_ = label({}, "itemDetails", results);
  item_details_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
  item_details_->setTextInteractionFlags(Qt::TextSelectableByMouse);
  item_error_ = new QPlainTextEdit(results);
  item_error_->setObjectName("itemError");
  item_error_->setReadOnly(true);
  item_error_->setFixedHeight(58);
  item_error_->hide();
  result_layout->addWidget(item_error_);
  auto* output_actions = new QHBoxLayout;
  output_actions->addWidget(item_details_, 1);
  open_summary_button_ = new QPushButton(tr("打开批次汇总"), results);
  open_summary_button_->setObjectName("openSummaryButton");
  open_json_button_ = new QPushButton(tr("打开 JSON"), results);
  open_json_button_->setObjectName("openJsonButton");
  open_output_button_ = new QPushButton(tr("打开结果目录"), results);
  open_output_button_->setObjectName("openOutputButton");
  output_actions->addWidget(open_summary_button_);
  output_actions->addWidget(open_json_button_);
  output_actions->addWidget(open_output_button_);
  result_layout->addLayout(output_actions);
  connect(open_json_button_, &QPushButton::clicked, this,
          [this] { openPath(completed_json_); });
  connect(open_summary_button_, &QPushButton::clicked, this,
          [this] { openPath(completed_summary_); });
  connect(open_output_button_, &QPushButton::clicked, this,
          [this] { openPath(completed_directory_); });
  workspace->addWidget(results);
  workspace->setStretchFactor(0, 1);
  workspace->setStretchFactor(1, 1);
  workspace->setSizes({220, 340});
  workspace->handle(1)->setToolTip(tr("上下拖动，调整图像与结果列表的高度"));
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
  connect(input_mode_, &QComboBox::currentIndexChanged, this, [this] {
    image_path_->clear();
    updateInputMode();
    invalidateResult();
  });
}

void MainWindow::updateInputMode() {
  const int mode = input_mode_->currentIndex();
  input_label_->setText(mode == 0 ? tr("输入图片") : mode == 1 ? tr("输入目录") : tr("Manifest 清单"));
  image_path_->setPlaceholderText(mode == 0 ? tr("选择待检图片")
      : mode == 1 ? tr("选择包含图片的目录") : tr("UTF-8 清单，每行一个图片路径"));
  batch_options_->setVisible(mode != 0);
  stop_button_->setVisible(mode != 0);
  batch_summary_->setVisible(mode != 0);
  open_summary_button_->setVisible(mode != 0);
  result_tabs_->setTabVisible(0, mode != 0);
  result_tabs_->setCurrentIndex(mode == 0 ? 1 : 0);
  run_button_->setText(mode == 0 ? tr("运行检测") : tr("开始批处理"));
}

void MainWindow::setInputs(const QString& config, const QString& image,
                           const QString& output_directory) {
  if (isBusy()) return;
  input_mode_->setCurrentIndex(0);
  config_path_->setText(config);
  image_path_->setText(image);
  if (!output_directory.isEmpty()) output_directory_->setText(output_directory);
}

void MainWindow::setBatchInputs(const QString& config, const QString& input,
                                const QString& output_directory, BatchInputKind kind,
                                int workers, int queue_capacity) {
  if (isBusy()) return;
  input_mode_->setCurrentIndex(kind == BatchInputKind::kDirectory ? 1 : 2);
  config_path_->setText(config);
  image_path_->setText(input);
  if (!output_directory.isEmpty()) output_directory_->setText(output_directory);
  workers_->setValue(workers);
  queue_capacity_->setValue(queue_capacity);
}

void MainWindow::invalidateResult() {
  if (isBusy()) return;
  ++preview_generation_;
  pending_preview_.reset();
  completed_directory_.clear();
  completed_json_.clear();
  completed_summary_.clear();
  batch_model_->setItems({});
  setDetections({});
  original_view_->setProperty("sourcePath", QString{});
  original_view_->clear(tr("选择输入并运行检测"));
  annotated_view_->clear(tr("检测完成后显示标注图"));
  result_summary_->setText(tr("尚未检测"));
  model_panel_->reset();
  open_output_button_->setToolTip({});
  item_details_->clear();
  item_details_->setToolTip({});
  item_error_->clear();
  item_error_->hide();
  batch_summary_->setText(tr("尚未处理 · 完成后按输入顺序列出每张图片"));
  open_json_button_->setEnabled(false);
  open_output_button_->setEnabled(false);
  open_summary_button_->setEnabled(false);
  status_message_->setProperty("state", "ready");
  status_message_->style()->unpolish(status_message_);
  status_message_->style()->polish(status_message_);
  status_message_->setText(tr("准备就绪。选择配置和输入后开始；图像支持滚轮缩放、拖动和检测框选择。"));
  state_badge_->setProperty("state", "ready");
  state_badge_->setText(tr("待检测"));
  state_badge_->style()->unpolish(state_badge_);
  state_badge_->style()->polish(state_badge_);
  run_button_->setEnabled(!config_path_->text().trimmed().isEmpty() &&
                           !image_path_->text().trimmed().isEmpty() &&
                           !output_directory_->text().trimmed().isEmpty());
}

void MainWindow::setBusy(bool busy) {
  input_fields_->setEnabled(!busy);
  run_button_->setEnabled(!busy);
  run_button_->setText(busy ? tr("正在检测…") : input_mode_->currentIndex() == 0 ? tr("运行检测") : tr("开始批处理"));
  stop_button_->setEnabled(busy && static_cast<bool>(batch_control_));
  stop_button_->setText(tr("停止"));
  progress_->setRange(0, busy ? 0 : 1);
  progress_->setValue(busy ? -1 : 0);
}

void MainWindow::startDetection() {
  if (isBusy() || close_pending_ || !run_button_->isEnabled()) return;
  invalidateResult();
  task_succeeded_ = false;
  stop_requested_ = false;
  task_result_received_ = false;
  thread_ = new QThread(this);
  auto wire_worker = [this](auto* worker) {
    using Worker = std::remove_pointer_t<decltype(worker)>;
    worker->moveToThread(thread_);
    connect(worker, &Worker::stageChanged, this, [this](const QString& stage) {
      if (!close_pending_ && !stop_requested_) status_message_->setText(stage);
    });
    connect(worker, &Worker::contractLoaded, this, &MainWindow::showContract);
    connect(worker, &Worker::failed, this, &MainWindow::showError);
    // A direct thread-safe quit also supports the destructor's fallback join.
    connect(worker, &Worker::finished, thread_, &QThread::quit, Qt::DirectConnection);
    connect(thread_, &QThread::finished, worker, &QObject::deleteLater);
  };
  if (input_mode_->currentIndex() == 0) {
    const DetectionRequest request{config_path_->text().trimmed(), image_path_->text().trimmed(),
                                    output_directory_->text().trimmed()};
    auto* worker = new DetectionWorker;
    wire_worker(worker);
    connect(thread_, &QThread::started, worker, [worker, request] { worker->run(request); },
            Qt::QueuedConnection);
    connect(worker, &DetectionWorker::completed, this, &MainWindow::showResult);
  } else {
    BatchDetectionRequest request;
    request.config_path = config_path_->text().trimmed();
    request.input_path = image_path_->text().trimmed();
    request.output_directory = output_directory_->text().trimmed();
    request.input_kind = input_mode_->currentIndex() == 1
        ? BatchInputKind::kDirectory : BatchInputKind::kManifest;
    request.workers = static_cast<std::size_t>(workers_->value());
    request.queue_capacity = static_cast<std::size_t>(queue_capacity_->value());
    batch_control_ = std::make_shared<BatchTaskControl>();
    auto* worker = new BatchWorker(batch_control_);
    wire_worker(worker);
    connect(thread_, &QThread::started, worker, [worker, request] { worker->run(request); },
            Qt::QueuedConnection);
    connect(worker, &BatchWorker::completed, this, &MainWindow::showBatchResult);
  }
  connect(thread_, &QThread::finished, this, [this] {
    thread_->wait();
    thread_->deleteLater();
    thread_ = nullptr;
    batch_control_.reset();
    setBusy(false);
    emit taskFinished(task_succeeded_);
    if (close_pending_) QTimer::singleShot(0, this, &QWidget::close);
  });
  setBusy(true);
  setState(tr("检测中"), "busy");
  status_message_->setText(tr("正在加载配置与模型… 批处理完成后显示最终计数。"));
  thread_->start();
}

void MainWindow::stopDetection() {
  if (!batch_control_ || !isBusy() || stop_requested_ || task_result_received_) return;
  stop_requested_ = true;
  // This call is synchronous on the GUI thread. It never depends on the
  // worker event loop, which is occupied by BatchRunner::run().
  batch_control_->requestStop();
  stop_button_->setEnabled(false);
  stop_button_->setText(tr("收尾中"));
  setState(tr("停止中"), "busy");
  status_message_->setText(tr("已请求停止。正在处理的图片完成后保存汇总，可随后开始新任务。"));
}

void MainWindow::setState(const QString& text, const char* state) {
  state_badge_->setProperty("state", state);
  state_badge_->setText(text);
  state_badge_->style()->unpolish(state_badge_);
  state_badge_->style()->polish(state_badge_);
}

void MainWindow::setDetections(const std::vector<Detection>& detections) {
  result_model_->setDetections(detections);
  original_view_->setDetections(detections);
  annotated_view_->setDetections(detections);
  result_tabs_->setTabText(1, tr("检测明细 (%1)").arg(detections.size()));
  result_summary_->setText(detections.empty() ? tr("未检出缺陷") : tr("检出 %1 个目标").arg(detections.size()));
}

void MainWindow::selectDetection(int row) {
  if (row < 0 || row >= result_model_->rowCount()) {
    detection_table_->selectionModel()->clear();
    original_view_->setSelectedDetection(-1);
    annotated_view_->setSelectedDetection(-1);
    return;
  }
  detection_table_->setCurrentIndex(result_model_->index(row, 0));
  detection_table_->selectRow(row);
  detection_table_->scrollTo(result_model_->index(row, 0));
  original_view_->setSelectedDetection(row);
  annotated_view_->setSelectedDetection(row);
  result_tabs_->setCurrentIndex(1);
}

void MainWindow::showContract(const RuntimeContract& contract) {
  model_panel_->setContract(contract);
}

void MainWindow::showResult(const DetectionResponse& response) {
  task_result_received_ = true;
  task_succeeded_ = true;
  showContract(response.contract);
  original_view_->setImage(response.original_image);
  original_view_->setProperty("sourcePath", from_path(response.result.detection_result.image.source_path));
  annotated_view_->setImage(response.annotated_image);
  setDetections(response.result.detection_result.detections);
  completed_directory_ = response.output_directory;
  if (response.result.outputs.json_path) completed_json_ = from_path(*response.result.outputs.json_path);
  const auto count = response.result.detection_result.detections.size();
  result_summary_->setText(count == 0 ? tr("未检出缺陷") : tr("检出 %1 个目标").arg(count));
  open_output_button_->setToolTip(QDir::toNativeSeparators(completed_directory_));
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

void MainWindow::showBatchResult(const BatchDetectionResponse& response) {
  task_result_received_ = true;
  stop_button_->setEnabled(false);
  task_succeeded_ = response.summary.status == BatchStatus::kSucceeded;
  showContract(response.contract);
  completed_directory_ = response.output_directory;
  completed_summary_ = response.summary_path;
  const auto& counts = response.summary.counts;
  batch_summary_->setText(tr("共 %1 张  ·  成功 %2  ·  失败 %3  ·  取消 %4")
      .arg(counts.discovered).arg(counts.succeeded).arg(counts.failed).arg(counts.cancelled));
  batch_model_->setItems(response.summary.items);
  result_tabs_->setCurrentIndex(0);
  open_output_button_->setToolTip(QDir::toNativeSeparators(completed_directory_));
  open_output_button_->setEnabled(true);
  open_summary_button_->setEnabled(true);
  QString completion;
  switch (response.summary.status) {
    case BatchStatus::kSucceeded:
      completion = tr("批处理完成"); setState(completion, "ready"); break;
    case BatchStatus::kPartialFailure:
      completion = tr("部分图片失败"); setState(completion, "warning"); break;
    case BatchStatus::kCancelled:
      completion = tr("已停止"); setState(completion, "warning"); break;
    case BatchStatus::kFatal:
      completion = tr("批处理失败"); setState(completion, "error"); break;
  }
  if (!close_pending_) {
    status_message_->setText(tr("%1 · 任务总耗时 %2 s · 选择图片浏览，点击检测框可联动明细。")
        .arg(completion).arg(response.elapsed_ms / 1000.0, 0, 'f', 2));
    if (!response.summary.fatal_error.empty()) {
      status_message_->setText(QString::fromStdString(response.summary.fatal_error));
    }
    if (batch_model_->rowCount() > 0) {
      batch_table_->setCurrentIndex(batch_model_->index(0, 0));
      batch_table_->selectRow(0);
    } else {
      result_summary_->setText(tr("没有可浏览的图片"));
      original_view_->clear(tr("本批次没有图片"));
      annotated_view_->clear(tr("无检测结果"));
    }
  }
}

void MainWindow::selectBatchItem(int row) {
  ++preview_generation_;
  pending_preview_.reset();
  setDetections({});
  completed_json_.clear();
  open_json_button_->setEnabled(false);
  original_view_->clear(tr("正在加载所选图片…"));
  annotated_view_->clear(tr("正在加载检测结果…"));
  original_view_->setProperty("sourcePath", QString{});
  item_details_->clear();
  item_details_->setToolTip({});
  item_error_->clear();
  item_error_->hide();
  const auto* item = batch_model_->itemAt(row);
  if (!item || close_pending_) return;
  const QString source = from_path(item->source_path);
  original_view_->setProperty("sourcePath", source);
  item_details_->setProperty("state", item->status == BatchItemStatus::kFailed ? "error" : "ready");
  item_details_->style()->unpolish(item_details_);
  item_details_->style()->polish(item_details_);
  item_details_->setText(QFileInfo(source).fileName());
  item_details_->setToolTip(source);
  if (item->status != BatchItemStatus::kSucceeded) {
    const QString state = item->status == BatchItemStatus::kFailed ? tr("处理失败") : tr("尚未执行 · 已取消");
    item_details_->setText(tr("%1 — %2").arg(QFileInfo(source).fileName(), state));
    item_error_->setPlainText(QString::fromStdString(item->error));
    item_error_->setVisible(!item->error.empty());
    item_details_->setToolTip(source + "\n" + QString::fromStdString(item->error));
    result_summary_->setText(state);
    original_view_->clear(state);
    annotated_view_->clear(tr("本项没有检测结果\n原因见下方，可滚动查看全文"));
    return;
  }
  if (item->json_output_path) {
    completed_json_ = from_path(*item->json_output_path);
    open_json_button_->setEnabled(true);
  }
  result_summary_->setText(tr("加载中…"));
  pending_preview_ = PreviewRequest{preview_generation_, *item};
  dispatchPreview();
}

void MainWindow::dispatchPreview() {
  if (close_pending_ || preview_in_flight_ || !pending_preview_) return;
  if (!preview_thread_) {
    preview_thread_ = new QThread(this);
    auto* worker = new PreviewWorker;
    worker->moveToThread(preview_thread_);
    connect(this, &MainWindow::previewRequested, worker, &PreviewWorker::load, Qt::QueuedConnection);
    connect(worker, &PreviewWorker::completed, this, &MainWindow::showPreview);
    connect(preview_thread_, &QThread::finished, worker, &QObject::deleteLater);
    connect(preview_thread_, &QThread::finished, this, [this] {
      preview_thread_->wait();
      preview_thread_->deleteLater();
      preview_thread_ = nullptr;
      preview_in_flight_ = false;
      if (close_pending_ && !isBusy()) QTimer::singleShot(0, this, &QWidget::close);
    });
    preview_thread_->start();
  }
  // One read in flight, at most one pending selection. Fast clicks replace the
  // pending item instead of queueing an unbounded series of image decodes.
  preview_in_flight_ = true;
  const auto request = *pending_preview_;
  pending_preview_.reset();
  emit previewRequested(request);
}

void MainWindow::showPreview(const PreviewResponse& response) {
  preview_in_flight_ = false;
  if (!close_pending_ && response.generation == preview_generation_) {
    if (response.error.isEmpty()) {
      original_view_->setImage(response.original_image);
      annotated_view_->setImage(response.annotated_image);
      setDetections(response.detections);
    } else {
      original_view_->clear(tr("预览无法加载"));
      annotated_view_->clear(tr("结果文件可能已移动或损坏\n请查看下方原因"));
      result_summary_->setText(tr("预览失败"));
      item_error_->setPlainText(response.error);
      item_error_->show();
    }
  }
  dispatchPreview();
}

void MainWindow::showError(const QString& message) {
  task_result_received_ = true;
  stop_button_->setEnabled(false);
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
  if (isBusy() || preview_thread_) {
    close_pending_ = true;
    pending_preview_.reset();
    ++preview_generation_;
    stopDetection();
    if (preview_thread_) preview_thread_->quit();
    status_message_->setText(tr("正在收尾并保存结果，后台任务退出后自动关闭…"));
    event->ignore();
    return;
  }
  QSettings settings;
  settings.setValue("window/geometry", saveGeometry());
  settings.setValue("paths/output", output_directory_->text());
  event->accept();
}

}  // namespace yolo_defect_cpp::qt
