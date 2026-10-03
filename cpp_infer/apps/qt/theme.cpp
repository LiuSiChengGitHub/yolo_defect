#include "theme.h"

#include <QFile>
#include <QHash>
#include <QIconEngine>
#include <QPainter>
#include <QPixmap>
#include <QRegularExpression>
#include <QWidget>
#include <utility>

#ifdef Q_OS_WIN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <dwmapi.h>
#endif

// Q_INIT_RESOURCE must expand at global scope. Calling it from the theme keeps
// the resource object linked although the client is built as a static library.
static void initClientResources() { Q_INIT_RESOURCE(qt_client); }

namespace yolo_defect_cpp::qt::theme {
namespace {

void ensureResources() {
  static const bool initialized = (initClientResources(), true);
  Q_UNUSED(initialized);
}

QString cssColor(const QColor& color) {
  if (color.alpha() == 255) return color.name();
  return QStringLiteral("rgba(%1, %2, %3, %4)")
      .arg(color.red()).arg(color.green()).arg(color.blue()).arg(color.alpha());
}

class TintedIconEngine : public QIconEngine {
 public:
  TintedIconEngine(QIcon source, QColor normal, QColor disabled)
      : source_(std::move(source)), normal_(std::move(normal)), disabled_(std::move(disabled)) {}

  void paint(QPainter* painter, const QRect& rect, QIcon::Mode mode, QIcon::State state) override {
    painter->drawPixmap(rect, scaledPixmap(rect.size(), mode, state,
                                           painter->device()->devicePixelRatioF()));
  }

  QPixmap pixmap(const QSize& size, QIcon::Mode mode, QIcon::State state) override {
    return scaledPixmap(size, mode, state, 1.0);
  }

  QPixmap scaledPixmap(const QSize& size, QIcon::Mode mode, QIcon::State state,
                       qreal scale) override {
    // The SVG engine renders at size * scale; QIcon then derives the final
    // device pixel ratio from the returned pixmap size.
    QPixmap result = source_.pixmap(size, scale, QIcon::Normal, state);
    if (result.isNull()) return result;
    QPainter painter(&result);
    painter.setCompositionMode(QPainter::CompositionMode_SourceIn);
    painter.fillRect(result.rect(), mode == QIcon::Disabled ? disabled_ : normal_);
    return result;
  }

  QIconEngine* clone() const override { return new TintedIconEngine(*this); }
  QString key() const override { return QStringLiteral("yolo_defect_tinted"); }

 private:
  QIcon source_;
  QColor normal_;
  QColor disabled_;
};


}  // namespace

const Palette& palette() {
  static const Palette value{
      /*canvas*/ QColor("#1a1917"), /*window*/ QColor("#1f1e1c"),
      /*panel*/ QColor("#262523"), /*field*/ QColor("#302f2c"),
      /*raised*/ QColor("#363531"), /*raised_hover*/ QColor("#403e3a"),
      /*border*/ QColor("#2d2c29"), /*border_strong*/ QColor("#4c4a45"),
      /*divider*/ QColor("#33322e"),
      /*text*/ QColor("#f1efe8"), /*text_secondary*/ QColor("#b9b5aa"),
      /*text_muted*/ QColor("#88847a"), /*text_disabled*/ QColor("#59564f"),
      /*accent*/ QColor("#d97757"), /*accent_bright*/ QColor("#e2896b"),
      /*accent_deep*/ QColor("#c4643f"), /*accent_soft*/ QColor(217, 119, 87, 40),
      /*accent_line*/ QColor(217, 119, 87, 150), /*on_accent*/ QColor("#fffaf5"),
      /*canvas_text*/ QColor("#88847a"),
      /*highlight*/ QColor("#7dd3fc"), /*selection*/ QColor(217, 119, 87, 21),
      /*row_alt*/ QColor(255, 255, 255, 5),
      /*success*/ QColor("#9cc78a"), /*warning*/ QColor("#e6b85c"),
      /*danger*/ QColor("#ee6f6a"), /*danger_soft*/ QColor(238, 111, 106, 26),
      /*danger_line*/ QColor(238, 111, 106, 64),
  };
  return value;
}

QString styleSheet() {
  ensureResources();
  QFile file(QStringLiteral(":/theme/workbench.qss"));
  if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) return {};
  const QString sheet = QString::fromUtf8(file.readAll());
  const auto& p = palette();
  const QHash<QString, QColor> tokens = {
      {"canvas", p.canvas}, {"window", p.window}, {"panel", p.panel}, {"field", p.field},
      {"raised", p.raised}, {"raised_hover", p.raised_hover},
      {"border", p.border}, {"border_strong", p.border_strong}, {"divider", p.divider},
      {"text", p.text}, {"text_secondary", p.text_secondary}, {"text_muted", p.text_muted},
      {"text_disabled", p.text_disabled},
      {"accent", p.accent}, {"accent_bright", p.accent_bright}, {"accent_deep", p.accent_deep},
      {"accent_soft", p.accent_soft}, {"accent_line", p.accent_line}, {"on_accent", p.on_accent},
      {"selection", p.selection}, {"row_alt", p.row_alt},
      {"success", p.success}, {"warning", p.warning},
      {"danger", p.danger}, {"danger_soft", p.danger_soft}, {"danger_line", p.danger_line},
  };
  static const QRegularExpression token(QStringLiteral("@([a-z_]+)"));
  QString result;
  qsizetype copied = 0;
  for (auto matches = token.globalMatch(sheet); matches.hasNext();) {
    const auto match = matches.next();
    result += QStringView(sheet).mid(copied, match.capturedStart() - copied);
    const auto color = tokens.constFind(match.captured(1));
    Q_ASSERT_X(color != tokens.constEnd(), "theme::styleSheet", "unknown palette token");
    result += color != tokens.constEnd() ? cssColor(*color) : match.captured(0);
    copied = match.capturedEnd();
  }
  result += QStringView(sheet).mid(copied);
  return result;
}

QPalette widgetPalette() {
  const auto& p = palette();
  QPalette result;
  result.setColor(QPalette::Window, p.window);
  result.setColor(QPalette::WindowText, p.text);
  result.setColor(QPalette::Base, p.field);
  result.setColor(QPalette::AlternateBase, p.panel);
  result.setColor(QPalette::Text, p.text);
  result.setColor(QPalette::Button, p.raised);
  result.setColor(QPalette::ButtonText, p.text);
  result.setColor(QPalette::ToolTipBase, p.raised);
  result.setColor(QPalette::ToolTipText, p.text);
  result.setColor(QPalette::PlaceholderText, p.text_muted);
  result.setColor(QPalette::Highlight, p.accent_deep);
  result.setColor(QPalette::HighlightedText, p.text);
  result.setColor(QPalette::Link, p.accent_bright);
  for (const auto role : {QPalette::WindowText, QPalette::Text, QPalette::ButtonText}) {
    result.setColor(QPalette::Disabled, role, p.text_disabled);
  }
  return result;
}

QIcon icon(const QString& name) {
  ensureResources();
  return QIcon(QStringLiteral(":/icons/%1.svg").arg(name));
}

QIcon icon(const QString& name, const QColor& color, const QColor& disabled) {
  return QIcon(new TintedIconEngine(icon(name), color,
                                    disabled.isValid() ? disabled : palette().text_disabled));
}


void applyWindowFrame(QWidget* window) {
#ifdef Q_OS_WIN
  const auto handle = reinterpret_cast<HWND>(window->winId());
  const auto colorRef = [](const QColor& color) {
    return static_cast<COLORREF>(RGB(color.red(), color.green(), color.blue()));
  };
  // Numeric DWMWA_* values keep this independent of the installed SDK version:
  // 20 USE_IMMERSIVE_DARK_MODE, 34 BORDER_COLOR, 35 CAPTION_COLOR, 36 TEXT_COLOR.
  const BOOL dark = TRUE;
  DwmSetWindowAttribute(handle, 20, &dark, sizeof(dark));
  const COLORREF border = colorRef(palette().border);
  const COLORREF caption = colorRef(palette().window);
  const COLORREF text = colorRef(palette().text_secondary);
  DwmSetWindowAttribute(handle, 34, &border, sizeof(border));
  DwmSetWindowAttribute(handle, 35, &caption, sizeof(caption));
  DwmSetWindowAttribute(handle, 36, &text, sizeof(text));
#else
  Q_UNUSED(window);
#endif
}

}  // namespace yolo_defect_cpp::qt::theme
