#include "theme.h"

#include <QFile>
#include <QHash>
#include <QIconEngine>
#include <QPainter>
#include <QPixmap>
#include <QRegularExpression>
#include <utility>

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
      /*window*/ QColor("#edf1f5"), /*surface*/ QColor("#ffffff"),
      /*surface_alt*/ QColor("#f6f8fb"), /*border*/ QColor("#dde4ec"),
      /*border_strong*/ QColor("#c9d4df"), /*divider*/ QColor("#edf1f5"),
      /*text*/ QColor("#1c2e42"), /*text_secondary*/ QColor("#5b6f86"),
      /*text_muted*/ QColor("#8494a6"), /*text_disabled*/ QColor("#aab6c3"),
      /*on_accent*/ QColor("#ffffff"),
      /*accent*/ QColor("#0f8a74"), /*accent_hover*/ QColor("#0c7a66"),
      /*accent_pressed*/ QColor("#096656"), /*accent_soft*/ QColor("#e2f3ee"),
      /*accent_text*/ QColor("#0b6656"), /*accent_disabled*/ QColor("#9fcbc1"),
      /*header*/ QColor("#132339"), /*header_raised*/ QColor(255, 255, 255, 22),
      /*header_text*/ QColor("#f3f7fb"), /*header_muted*/ QColor("#97abc2"),
      /*brand*/ QColor("#5fe0c0"),
      /*canvas*/ QColor("#111c29"), /*canvas_grid*/ QColor("#18263a"),
      /*canvas_text*/ QColor("#8fa2b7"), /*highlight*/ QColor("#5eead4"),
      /*success*/ QColor("#11845a"), /*success_soft*/ QColor("#def4e9"),
      /*warning*/ QColor("#956000"), /*warning_soft*/ QColor("#fff0d3"),
      /*danger*/ QColor("#c03636"), /*danger_soft*/ QColor("#fdf0ee"),
      /*danger_line*/ QColor("#f2cfc9"),
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
      {"window", p.window}, {"surface", p.surface}, {"surface_alt", p.surface_alt},
      {"border", p.border}, {"border_strong", p.border_strong}, {"divider", p.divider},
      {"text", p.text}, {"text_secondary", p.text_secondary}, {"text_muted", p.text_muted},
      {"text_disabled", p.text_disabled}, {"on_accent", p.on_accent},
      {"accent", p.accent}, {"accent_hover", p.accent_hover},
      {"accent_pressed", p.accent_pressed}, {"accent_soft", p.accent_soft},
      {"accent_text", p.accent_text}, {"accent_disabled", p.accent_disabled},
      {"header", p.header}, {"header_raised", p.header_raised},
      {"header_text", p.header_text}, {"header_muted", p.header_muted}, {"brand", p.brand},
      {"success", p.success}, {"success_soft", p.success_soft},
      {"warning", p.warning}, {"warning_soft", p.warning_soft},
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

QIcon icon(const QString& name) {
  ensureResources();
  return QIcon(QStringLiteral(":/icons/%1.svg").arg(name));
}

QIcon icon(const QString& name, const QColor& color, const QColor& disabled) {
  return QIcon(new TintedIconEngine(icon(name), color,
                                    disabled.isValid() ? disabled : palette().text_disabled));
}

}  // namespace yolo_defect_cpp::qt::theme
