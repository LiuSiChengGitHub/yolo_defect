#include "segmented_control.h"

#include <QAbstractButton>
#include <QButtonGroup>
#include <QHBoxLayout>
#include <QToolButton>

namespace yolo_defect_cpp::qt {

SegmentedControl::SegmentedControl(const QStringList& labels, QWidget* parent)
    : QFrame(parent), group_(new QButtonGroup(this)) {
  auto* layout = new QHBoxLayout(this);
  layout->setContentsMargins(3, 3, 3, 3);
  layout->setSpacing(2);
  group_->setExclusive(true);
  for (int index = 0; index < labels.size(); ++index) {
    auto* segment = new QToolButton(this);
    segment->setObjectName("segment");
    segment->setText(labels[index]);
    segment->setCheckable(true);
    segment->setChecked(index == current_);
    segment->setToolButtonStyle(Qt::ToolButtonTextOnly);
    segment->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    group_->addButton(segment, index);
    layout->addWidget(segment);
  }
  // An exclusive group reports the old segment unchecked, then the new one checked.
  connect(group_, &QButtonGroup::idToggled, this, [this](int id, bool checked) {
    if (!checked || id == current_) return;
    current_ = id;
    emit currentIndexChanged(id);
  });
}

void SegmentedControl::setCurrentIndex(int index) {
  if (auto* segment = group_->button(index)) segment->setChecked(true);
}

void SegmentedControl::setSegmentToolTip(int index, const QString& tip) {
  if (auto* segment = group_->button(index)) segment->setToolTip(tip);
}

}  // namespace yolo_defect_cpp::qt
