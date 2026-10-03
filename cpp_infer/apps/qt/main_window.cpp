#include "main_window.h"

#include "detection_table_model.h"
#include "batch_table_model.h"
#include "batch_worker.h"
#include "detection_worker.h"
#include "image_view.h"
#include "model_info_panel.h"
#include "path_edit.h"
#include "preview_worker.h"
#include "segmented_control.h"
#include "table_delegates.h"
#include "theme.h"

#include <QCloseEvent>
#include <QDesktopServices>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFrame>
#include <QApplication>
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
#include <QToolButton>
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

QFrame* panel(QWidget* parent) {
  auto* frame = new QFrame(parent);
  frame->setObjectName("panel");
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
  // Fusion draws only from the palette and stylesheet. Native styles still
  // paint parts of item views themselves, which breaks the dark theme.
  if (QApplication::style()->name().compare(QLatin1String("fusion"), Qt::CaseInsensitive) != 0) {
    QApplication::setStyle(QStringLiteral("Fusion"));
  }
  setWindowTitle(tr("工业缺陷检测工作台"));
  setWindowIcon(theme::icon(QStringLiteral("app")));
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
  setStyleSheet(theme::styleSheet());
  // Dark roles for anything the stylesheet leaves to the base style; window
  // propagation passes them on to popups such as line edit context menus.
  setPalette(theme::widgetPalette());
  setAttribute(Qt::WA_WindowPropagation);
  const auto& colors = theme::palette();
  auto* central = new QWidget(this);
  central->setObjectName("workbenchSurface");
  central->setAttribute(Qt::WA_StyledBackground);
  setCentralWidget(central);
  auto* root = new QVBoxLayout(central);
  root->setContentsMargins(0, 0, 0, 0);
  root->setSpacing(0);

  // A slim title bar continuing the dark native caption from applyWindowFrame().
  auto* top_bar = new QFrame(central);
  top_bar->setObjectName("topBar");
  auto* top_layout = new QHBoxLayout(top_bar);
  top_layout->setContentsMargins(16, 10, 16, 10);
  top_layout->setSpacing(10);
  auto* logo = new QToolButton(top_bar);
  logo->setObjectName("logo");
  logo->setIcon(theme::icon(QStringLiteral("app")));
  logo->setIconSize(QSize(28, 28));
  logo->setFocusPolicy(Qt::NoFocus);
  logo->setAttribute(Qt::WA_TransparentForMouseEvents);
  top_layout->addWidget(logo);
  top_layout->addWidget(label(tr("工业缺陷检测工作台"), "title", top_bar));
  top_layout->addSpacing(6);
  top_layout->addWidget(label(tr("单图 / 批处理  ·  模型配置驱动  ·  本地推理"), "subtitle", top_bar), 1);
  state_badge_ = label({}, "stateBadge", top_bar);
  state_badge_->setTextFormat(Qt::RichText);
  top_layout->addWidget(state_badge_, 0, Qt::AlignVCenter);
  root->addWidget(top_bar);

  auto* body = new QHBoxLayout;
  body->setContentsMargins(12, 0, 12, 12);
  body->setSpacing(10);
  auto* sidebar_scroll = new QScrollArea(central);
  sidebar_scroll->setObjectName("sidebarScroll");
  sidebar_scroll->viewport()->setObjectName("sidebarViewport");
  sidebar_scroll->viewport()->setAttribute(Qt::WA_StyledBackground);
  sidebar_scroll->setWidgetResizable(true);
  sidebar_scroll->setFixedWidth(336);
  sidebar_scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  auto* sidebar = new QWidget;
  sidebar->setObjectName("sidebarContent");
  sidebar->setAttribute(Qt::WA_StyledBackground);
  auto* side = new QVBoxLayout(sidebar);
  side->setContentsMargins(0, 0, 0, 0);
  side->setSpacing(0);
  sidebar_scroll->setWidget(sidebar);
  body->addWidget(sidebar_scroll);
  // One continuous sidebar panel; its sections are separated by a hairline.
  auto* sidebar_panel = panel(sidebar);
  auto* sidebar_sections = new QVBoxLayout(sidebar_panel);
  sidebar_sections->setContentsMargins(0, 0, 0, 0);
  sidebar_sections->setSpacing(0);
  side->addWidget(sidebar_panel);

  input_panel_ = new QFrame(sidebar_panel);
  input_panel_->setObjectName("sidebarSection");
  auto* input_layout = new QVBoxLayout(input_panel_);
  input_layout->setContentsMargins(16, 16, 16, 16);
  input_layout->setSpacing(12);
  input_layout->addWidget(label(tr("检测任务"), "sectionTitle", input_panel_));
  input_fields_ = new QWidget(input_panel_);
  auto* inputs = new QVBoxLayout(input_fields_);
  inputs->setContentsMargins(0, 0, 0, 0);
  inputs->setSpacing(12);
  auto* mode_group = new QVBoxLayout;
  mode_group->setSpacing(6);
  mode_group->addWidget(label(tr("输入方式"), "fieldLabel", input_fields_));
  input_mode_ = new SegmentedControl({tr("单张图片"), tr("图片目录"), QStringLiteral("Manifest")},
                                     input_fields_);
  input_mode_->setObjectName("inputMode");
  input_mode_->setSegmentToolTip(0, tr("检测一张图片"));
  input_mode_->setSegmentToolTip(1, tr("递归发现目录中的图片并批量检测"));
  input_mode_->setSegmentToolTip(2, tr("按 UTF-8 清单逐行列出的图片路径批量检测"));
  mode_group->addWidget(input_mode_);
  inputs->addLayout(mode_group);
  input_layout->addWidget(input_fields_);
  auto add_path = [&](const QString& title, const QString& object_name,
                      const QString& placeholder, QLineEdit*& field,
                      const QString& filter, bool directory) {
    auto* field_group = new QVBoxLayout;
    field_group->setSpacing(6);
    auto* field_label = label(title, "fieldLabel", input_fields_);
    if (object_name == "imagePath") input_label_ = field_label;
    field_group->addWidget(field_label);
    field = new PathEdit(input_panel_);
    field->setObjectName(object_name);
    field->setPlaceholderText(placeholder);
    auto* path_row = new QHBoxLayout;
    path_row->setSpacing(6);
    path_row->addWidget(field, 1);
    auto* browse = new QPushButton(input_panel_);
    browse->setObjectName("browseButton");
    browse->setIcon(theme::icon(QStringLiteral("folder"), colors.text_secondary));
    browse->setIconSize(QSize(16, 16));
    browse->setToolTip(tr("浏览"));
    browse->setAccessibleName(tr("浏览%1").arg(title));
    // Icon-only and as tall as the path field, leaving more room for the path.
    browse->setFixedWidth(36);
    browse->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);
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
  batch_settings->setSpacing(10);
  auto add_number = [&](const QString& title, const char* name, int maximum, int value) {
    auto* column = new QVBoxLayout;
    column->setSpacing(6);
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
  run_button_->setIcon(theme::icon(QStringLiteral("play"), colors.on_accent));
  run_button_->setIconSize(QSize(14, 14));
  auto* task_actions = new QHBoxLayout;
  task_actions->setSpacing(8);
  task_actions->addWidget(run_button_, 1);
  stop_button_ = new QPushButton(tr("停止"), input_panel_);
  stop_button_->setObjectName("stopButton");
  stop_button_->setIcon(theme::icon(QStringLiteral("stop"), colors.danger));
  stop_button_->setIconSize(QSize(12, 12));
  stop_button_->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);
  stop_button_->setToolTip(tr("停止派发新图片，等待正在处理的图片完成并保存结果。"));
  stop_button_->setEnabled(false);
  task_actions->addWidget(stop_button_);
  input_layout->addSpacing(2);
  input_layout->addLayout(task_actions);
  connect(run_button_, &QPushButton::clicked, this, &MainWindow::startDetection);
  connect(stop_button_, &QPushButton::clicked, this, &MainWindow::stopDetection);
  sidebar_sections->addWidget(input_panel_);
  auto* divider = new QFrame(sidebar_panel);
  divider->setObjectName("divider");
  sidebar_sections->addWidget(divider);
  model_panel_ = new ModelInfoPanel(sidebar_panel);
  sidebar_sections->addWidget(model_panel_);
  sidebar_sections->addStretch();

  auto* workspace = new QSplitter(Qt::Vertical, central);
  workspace->setObjectName("workspaceSplitter");
  workspace->setChildrenCollapsible(false);
  workspace->setHandleWidth(10);
  // Both canvases share one viewer panel, like a compare view.
  auto* viewer = panel(workspace);
  auto* viewer_layout = new QHBoxLayout(viewer);
  viewer_layout->setContentsMargins(12, 10, 12, 12);
  viewer_layout->setSpacing(12);
  auto add_view = [&](const QString& title, const QString& name, ImageView*& view) {
    auto* column = new QVBoxLayout;
    column->setSpacing(8);
    auto* view_heading = new QHBoxLayout;
    view_heading->setContentsMargins(2, 0, 0, 0);
    view_heading->addWidget(label(title, "viewTitle", viewer), 1);
    view = new ImageView(viewer);
    view->setObjectName(name);
    view->setToolTip(tr("滚轮缩放 · 拖动平移 · 点击检测框选择 · 双击适应窗口"));
    // The view actions form one segmented toolbar, disabled while no image is shown.
    auto* toolbar = new QFrame(viewer);
    toolbar->setObjectName("viewToolbar");
    auto* tools = new QHBoxLayout(toolbar);
    tools->setContentsMargins(2, 2, 2, 2);
    tools->setSpacing(2);
    auto add_action = [&](const QString& icon, const QString& text, const QString& tip,
                          auto action) {
      auto* button = new QToolButton(toolbar);
      button->setObjectName("viewAction");
      button->setToolTip(tip);
      button->setAccessibleName(tip);
      button->setFocusPolicy(Qt::NoFocus);
      button->setAutoRaise(true);
      button->setMinimumWidth(28);
      button->setFixedHeight(24);
      if (icon.isEmpty()) {
        button->setText(text);
      } else {
        button->setIcon(theme::icon(icon, colors.text_secondary));
        button->setIconSize(QSize(15, 15));
      }
      tools->addWidget(button);
      connect(button, &QToolButton::clicked, view, action);
    };
    add_action(QStringLiteral("zoom-out"), {}, tr("缩小"), &ImageView::zoomOut);
    add_action(QStringLiteral("zoom-in"), {}, tr("放大"), &ImageView::zoomIn);
    add_action({}, QStringLiteral("1:1"), tr("按原始像素查看"), &ImageView::actualSize);
    add_action(QStringLiteral("fit"), {}, tr("适应窗口，显示完整图片"), &ImageView::fitToWindow);
    connect(view, &ImageView::zoomChanged, toolbar,
            [toolbar](double factor) { toolbar->setEnabled(factor > 0.0); });
    view_heading->addWidget(toolbar);
    column->addLayout(view_heading);
    column->addWidget(view, 1);
    connect(view, &ImageView::detectionSelected, this, &MainWindow::selectDetection);
    viewer_layout->addLayout(column, 1);
  };
  add_view(tr("原始图像"), "originalView", original_view_);
  add_view(tr("检测结果"), "annotatedView", annotated_view_);
  workspace->addWidget(viewer);

  auto* results = panel(workspace);
  auto* result_layout = new QVBoxLayout(results);
  result_layout->setContentsMargins(14, 12, 14, 12);
  result_layout->setSpacing(8);
  result_tabs_ = new QTabWidget(results);
  result_tabs_->setObjectName("resultTabs");
  // QTabWidget can otherwise shrink its page below QTableView's minimum at
  // high DPI. Reserve space for the tabs, header and several complete rows.
  result_tabs_->setMinimumHeight(200);
  // The summaries share the tab row, leaving the panel height to the lists.
  auto* summaries = new QWidget(result_tabs_);
  auto* summaries_layout = new QHBoxLayout(summaries);
  // Corner widgets sit on the pane edge; the bottom margin lifts the text to
  // the tab label line above the tabs' own bottom margin.
  summaries_layout->setContentsMargins(0, 0, 2, 15);
  summaries_layout->setSpacing(14);
  batch_summary_ = label({}, "batchSummary", summaries);
  summaries_layout->addWidget(batch_summary_);
  result_summary_ = label({}, "resultSummary", summaries);
  summaries_layout->addWidget(result_summary_);
  result_tabs_->setCornerWidget(summaries, Qt::TopRightCorner);
  auto configure_table = [](QTableView* table) {
    table->setAlternatingRowColors(true);
    table->setShowGrid(false);
    table->setSelectionBehavior(QAbstractItemView::SelectRows);
    table->setSelectionMode(QAbstractItemView::SingleSelection);
    table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table->setWordWrap(false);
    table->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
    table->verticalHeader()->setDefaultSectionSize(30);
    table->verticalHeader()->hide();
    table->setMinimumHeight(100);
  };
  batch_table_ = new QTableView(result_tabs_);
  batch_table_->setObjectName("batchTable");
  batch_model_ = new BatchTableModel(batch_table_);
  batch_table_->setModel(batch_model_);
  configure_table(batch_table_);
  batch_table_->setItemDelegateForColumn(2, new StatusPillDelegate(batch_table_));
  batch_table_->setTextElideMode(Qt::ElideMiddle);
  batch_table_->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
  batch_table_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
  batch_table_->horizontalHeader()->setSectionResizeMode(5, QHeaderView::Stretch);
  result_tabs_->addTab(batch_table_, tr("逐图结果"));
  connect(batch_table_->selectionModel(), &QItemSelectionModel::currentRowChanged,
          this, [this](const QModelIndex& current) { selectBatchItem(current.row()); });
  auto* table = new QTableView(result_tabs_);
  detection_table_ = table;
  table->setObjectName("resultTable");
  result_model_ = new DetectionTableModel(table);
  table->setModel(result_model_);
  configure_table(table);
  table->setItemDelegateForColumn(2, new ConfidenceBarDelegate(table));
  table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
  table->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
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
  output_actions->setSpacing(8);
  output_actions->addWidget(item_details_, 1);
  auto output_button = [&](const QString& text, const char* name, const QString& icon) {
    auto* button = new QPushButton(text, results);
    button->setObjectName(name);
    button->setIcon(theme::icon(icon, colors.text_secondary));
    button->setIconSize(QSize(14, 14));
    return button;
  };
  open_summary_button_ = output_button(tr("打开批次汇总"), "openSummaryButton", QStringLiteral("chart"));
  open_json_button_ = output_button(tr("打开 JSON"), "openJsonButton", QStringLiteral("file"));
  open_output_button_ = output_button(tr("打开结果目录"), "openOutputButton", QStringLiteral("folder"));
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
  workspace->setSizes({400, 330});
  workspace->handle(1)->setToolTip(tr("上下拖动，调整图像与结果列表的高度"));
  body->addWidget(workspace, 1);
  root->addLayout(body, 1);

  auto* status_bar = new QFrame(central);
  status_bar->setObjectName("statusBar");
  auto* status_layout = new QHBoxLayout(status_bar);
  status_layout->setContentsMargins(16, 7, 16, 7);
  status_layout->setSpacing(16);
  status_message_ = label({}, "statusMessage", status_bar);
  status_message_->setWordWrap(true);
  status_message_->setTextInteractionFlags(Qt::TextSelectableByMouse);
  status_layout->addWidget(status_message_, 1);
  // Indeterminate activity indicator, shown only while a task runs.
  progress_ = new QProgressBar(status_bar);
  progress_->setTextVisible(false);
  progress_->setRange(0, 1);
  progress_->setValue(0);
  progress_->setFixedWidth(180);
  progress_->hide();
  status_layout->addWidget(progress_, 0, Qt::AlignVCenter);
  root->addWidget(status_bar);
  connect(input_mode_, &SegmentedControl::currentIndexChanged, this, [this] {
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
  batch_summary_->setText(tr("完成后按输入顺序列出每张图片"));
  open_json_button_->setEnabled(false);
  open_output_button_->setEnabled(false);
  open_summary_button_->setEnabled(false);
  status_message_->setProperty("state", "ready");
  status_message_->style()->unpolish(status_message_);
  status_message_->style()->polish(status_message_);
  status_message_->setText(tr("准备就绪。选择配置和输入后开始；图像支持滚轮缩放、拖动和检测框选择。"));
  setState(tr("待检测"), "idle");
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
  progress_->setVisible(busy);
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
  // A neutral badge; only the leading dot carries the state color.
  const auto& colors = theme::palette();
  const QByteArray kind(state);
  const QColor dot = kind == "ready" ? colors.success
      : kind == "busy" ? colors.accent
      : kind == "warning" ? colors.warning
      : kind == "error" ? colors.danger : colors.text_muted;
  state_badge_->setText(
      QStringLiteral("<span style=\"color:%1; font-size:10px\">&#9679;</span>&nbsp;&nbsp;%2")
          .arg(dot.name(), text.toHtmlEscaped()));
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
  setState(tr("检测完成"), "ready");
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
  batch_summary_->setText(tr("共 %1 张 · 成功 %2 · 失败 %3 · 取消 %4")
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
  setState(tr("检测失败"), "error");
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

void MainWindow::showEvent(QShowEvent* event) {
  QMainWindow::showEvent(event);
  // The native window exists from the first show; style its caption once.
  if (!frame_styled_ && QGuiApplication::platformName() == QLatin1String("windows")) {
    frame_styled_ = true;
    theme::applyWindowFrame(this);
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
