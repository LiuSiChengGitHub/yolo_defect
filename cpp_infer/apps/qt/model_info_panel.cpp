#include "model_info_panel.h"

#include "detection_types.h"
#include "theme.h"

#include <QDir>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QStringList>
#include <QStyle>
#include <QToolButton>
#include <QVBoxLayout>

namespace yolo_defect_cpp::qt {
namespace {

QLabel* makeLabel(const QString& text, const QString& name, QWidget* parent) {
  auto* result = new QLabel(text, parent);
  result->setObjectName(name);
  result->setTextFormat(Qt::PlainText);
  return result;
}

void setValue(QLabel* label, const QString& value) {
  label->setText(value);
  label->setToolTip(value);
}

}  // namespace

ModelInfoPanel::ModelInfoPanel(QWidget* parent) : QFrame(parent) {
  // A section of the sidebar panel rather than a separate card.
  setObjectName("sidebarSection");
  auto* layout = new QVBoxLayout(this);
  layout->setContentsMargins(16, 16, 16, 16);
  layout->setSpacing(10);

  layout->addWidget(makeLabel(tr("生效参数"), "sectionTitle", this));

  model_id_ = makeLabel({}, "modelDetails", this);
  model_id_->setWordWrap(true);
  model_id_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
  model_id_->setTextInteractionFlags(Qt::TextSelectableByMouse);
  layout->addWidget(model_id_);

  auto* parameters = new QGridLayout;
  parameters->setHorizontalSpacing(12);
  parameters->setVerticalSpacing(8);
  parameters->setColumnStretch(1, 1);
  auto add_parameter = [&](int row, const QString& title, const QString& object_name) {
    auto* key = makeLabel(title, "parameterLabel", this);
    parameters->addWidget(key, row, 0, Qt::AlignLeft | Qt::AlignTop);
    auto* value = makeLabel({}, object_name, this);
    value->setAlignment(Qt::AlignRight | Qt::AlignTop);
    value->setWordWrap(true);
    value->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    value->setTextInteractionFlags(Qt::TextSelectableByMouse);
    parameters->addWidget(value, row, 1);
    return value;
  };
  provider_ = add_parameter(0, tr("执行后端"), "parameterValue");
  input_shape_ = add_parameter(1, tr("输入尺寸"), "parameterValue");
  score_threshold_ = add_parameter(2, tr("置信度阈值"), "scoreThreshold");
  nms_threshold_ = add_parameter(3, tr("NMS 阈值"), "nmsThreshold");
  nms_mode_ = add_parameter(4, tr("NMS 模式"), "parameterValue");
  layout->addLayout(parameters);

  const auto& colors = theme::palette();
  const QIcon collapsed_icon = theme::icon(QStringLiteral("chevron-right"), colors.text_secondary);
  const QIcon expanded_icon = theme::icon(QStringLiteral("chevron-down"), colors.text_secondary);
  details_toggle_ = new QToolButton(this);
  details_toggle_->setObjectName("modelDetailsToggle");
  details_toggle_->setText(tr("类别与文件详情"));
  details_toggle_->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
  details_toggle_->setIcon(collapsed_icon);
  details_toggle_->setIconSize(QSize(12, 12));
  details_toggle_->setAutoRaise(true);
  details_toggle_->setCheckable(true);
  details_toggle_->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Fixed);
  layout->addWidget(details_toggle_);

  extra_details_ = new QWidget(this);
  extra_details_->setObjectName("modelExtraDetails");
  auto* details = new QVBoxLayout(extra_details_);
  details->setContentsMargins(0, 0, 0, 0);
  details->setSpacing(5);
  auto add_detail = [&](const QString& title, const QString& object_name) {
    if (details->count() != 0) details->addSpacing(5);
    details->addWidget(makeLabel(title, "parameterLabel", extra_details_));
    auto* value = makeLabel({}, object_name, extra_details_);
    value->setWordWrap(true);
    value->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    value->setTextInteractionFlags(Qt::TextSelectableByMouse);
    details->addWidget(value);
    return value;
  };
  class_names_ = add_detail(tr("类别"), "classNames");
  model_path_ = add_detail(tr("模型文件"), "modelPath");
  config_path_ = add_detail(tr("配置文件"), "runtimeConfigPath");
  layout->addWidget(extra_details_);
  connect(details_toggle_, &QToolButton::toggled, this,
          [this, collapsed_icon, expanded_icon](bool expanded) {
    extra_details_->setVisible(expanded);
    details_toggle_->setIcon(expanded ? expanded_icon : collapsed_icon);
    details_toggle_->setText(expanded ? tr("收起详情") : tr("类别与文件详情"));
  });
  reset();
}

void ModelInfoPanel::setContract(const RuntimeContract& contract) {
  QStringList classes;
  for (const auto& name : contract.artifact.class_names) {
    classes << QString::fromStdString(name);
  }
  QStringList dimensions;
  for (const auto dimension : contract.artifact.input.shape) {
    dimensions << QString::number(dimension);
  }
  setValue(model_id_, QString::fromStdString(contract.artifact.model_id));
  setLoaded(true);
  setValue(provider_, QString::fromStdString(to_string(contract.runtime.provider)));
  setValue(input_shape_, dimensions.join(" × "));
  setValue(score_threshold_, QString::number(contract.runtime.score_threshold));
  setValue(nms_threshold_, QString::number(contract.runtime.nms_threshold));
  setValue(nms_mode_, QString::fromStdString(to_string(contract.artifact.nms_mode)));
  setValue(class_names_, classes.join(", "));
  setValue(model_path_, QDir::fromNativeSeparators(from_path(contract.artifact.model_path)));
  setValue(config_path_, QDir::fromNativeSeparators(from_path(contract.runtime.declaration_path)));
  details_toggle_->setEnabled(true);
}

void ModelInfoPanel::setLoaded(bool loaded) {
  // The stylesheet mutes the model chip while it only shows a placeholder.
  model_id_->setProperty("loaded", loaded);
  model_id_->style()->unpolish(model_id_);
  model_id_->style()->polish(model_id_);
}

void ModelInfoPanel::reset() {
  model_id_->setText(tr("尚未加载模型"));
  model_id_->setToolTip(tr("运行检测后显示本次使用的模型与参数。"));
  setLoaded(false);
  for (auto* value : {provider_, input_shape_, score_threshold_, nms_threshold_, nms_mode_}) {
    value->setText(QStringLiteral("—"));
    value->setToolTip({});
  }
  for (auto* value : {class_names_, model_path_, config_path_}) {
    value->clear();
    value->setToolTip({});
  }
  details_toggle_->setChecked(false);
  details_toggle_->setEnabled(false);
  extra_details_->hide();
}

}  // namespace yolo_defect_cpp::qt
