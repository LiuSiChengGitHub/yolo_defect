#ifndef YOLO_DEFECT_QT_THEME_H_
#define YOLO_DEFECT_QT_THEME_H_

#include <QColor>
#include <QIcon>
#include <QPalette>
#include <QString>

class QWidget;

namespace yolo_defect_cpp::qt::theme {

// The single warm dark palette shared by the stylesheet, custom painting and
// table models. Visual code reads colors from here instead of hard-coding them.
struct Palette {
  // Surfaces from the deepest layer (image canvas) to raised controls.
  QColor canvas, window, panel, field, raised, raised_hover;
  QColor border, border_strong, divider;
  QColor text, text_secondary, text_muted, text_disabled;
  QColor accent, accent_bright, accent_deep, accent_soft, accent_line, on_accent;
  QColor canvas_text, highlight, selection, row_alt;
  QColor success, warning, danger, danger_soft, danger_line;
};

const Palette& palette();

// The workbench stylesheet resource with its @token colors resolved from palette().
QString styleSheet();

// A matching QPalette for the few areas a stylesheet leaves to the base style,
// such as scroll area corners and popup frames.
QPalette widgetPalette();

// A client resource icon drawn with its own SVG colors (for example the logo).
QIcon icon(const QString& name);

// A monochrome resource icon recolored at paint time, so one SVG serves every
// state and stays sharp at any device pixel ratio. An invalid disabled color
// falls back to palette().text_disabled.
QIcon icon(const QString& name, const QColor& color, const QColor& disabled = {});

// Dark native title bar matching the window background. Windows only; older
// systems ignore the attributes they do not support.
void applyWindowFrame(QWidget* window);

}  // namespace yolo_defect_cpp::qt::theme
#endif
