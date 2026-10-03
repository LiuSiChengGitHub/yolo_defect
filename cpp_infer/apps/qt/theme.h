#ifndef YOLO_DEFECT_QT_THEME_H_
#define YOLO_DEFECT_QT_THEME_H_

#include <QColor>
#include <QIcon>
#include <QString>

namespace yolo_defect_cpp::qt::theme {

// The single light palette shared by the stylesheet, custom painting and
// table models. Visual code reads colors from here instead of hard-coding them.
struct Palette {
  QColor window, surface, surface_alt, border, border_strong, divider;
  QColor text, text_secondary, text_muted, text_disabled, on_accent;
  QColor accent, accent_hover, accent_pressed, accent_soft, accent_text, accent_disabled;
  QColor header, header_raised, header_text, header_muted, brand;
  QColor canvas, canvas_grid, canvas_text, highlight;
  QColor success, success_soft, warning, warning_soft, danger, danger_soft, danger_line;
};

const Palette& palette();

// The workbench stylesheet resource with its @token colors resolved from palette().
QString styleSheet();

// A client resource icon drawn with its own SVG colors (for example the logo).
QIcon icon(const QString& name);

// A monochrome resource icon recolored at paint time, so one SVG serves every
// state and stays sharp at any device pixel ratio. An invalid disabled color
// falls back to palette().text_disabled.
QIcon icon(const QString& name, const QColor& color, const QColor& disabled = {});

}  // namespace yolo_defect_cpp::qt::theme
#endif
