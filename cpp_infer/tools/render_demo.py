"""Render both public demo pages from one template, translations and evidence.

Run after changing docs/demo/content.json, page.template.html or evidence.json.
--check reports stale outputs and validates matching translation structures.
Only the generated HTML and referenced assets are needed for offline viewing.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "docs/demo"
REPO = "https://github.com/LiuSiChengGitHub/yolo_defect/blob/HEAD/"
SHOTS = ["01-ready", "02-running", "03-results", "04-inspect", "05-browse", "06-overview"]


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def check_structure(left: object, right: object, path: str = "content") -> None:
    if type(left) is not type(right):
        raise ValueError(f"Translation type mismatch at {path}")
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError(f"Translation keys differ at {path}: {left.keys() ^ right.keys()}")
        for key in left:
            check_structure(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, list):
        if len(left) != len(right):
            raise ValueError(f"Translation item count differs at {path}")
        for index, (a, b) in enumerate(zip(left, right)):
            check_structure(a, b, f"{path}[{index}]")


def cards(items: list[dict]) -> str:
    return "".join(f'<article class="card"><div class="num">{escape(item["tag"])}</div>'
                   f'<h3>{escape(item["title"])}</h3><p>{escape(item["body"])}</p></article>'
                   for item in items)


def ordered(items: list[str]) -> str:
    return "<ol>" + "".join(f"<li>{escape(item)}</li>" for item in items) + "</ol>"


def source_links(section: dict, language: str, label: str) -> str:
    links = "".join(f'<li><a href="{escape(REPO + source["path"])}">'
                    f'{escape(source["label"][language])}</a></li>' for source in section["sources"])
    return f'<details class="sources"><summary>{escape(label)}</summary><ul>{links}</ul></details>'


def evidence_cards(evidence: dict, language: str, labels: dict) -> str:
    result = []
    for key in ("quantization", "batch"):
        section = evidence[key]
        result.append(f'<article class="evidence-card"><div class="chart-heading">'
                      f'<div class="num">{escape(labels[key + "_tag"])}</div>'
                      f'<h3>{escape(section["title"][language])}</h3>'
                      f'<p class="muted">{escape(section["environment"][language])}</p></div>'
                      f'<figure class="chart"><img src="{escape(section["chart"][language])}" '
                      f'alt="{escape(section["title"][language])}" loading="lazy">'
                      f'<figcaption>{escape(section["conclusion"][language])}</figcaption></figure>'
                      + source_links(section, language, labels["sources"]) + "</article>")
    return "".join(result)


def platform_table(evidence: dict, language: str, labels: dict) -> str:
    keys = ["single_image", "directory_manifest", "fp32_int8", "bounded_queue"]
    headings = "".join(f"<th>{escape(value)}</th>" for value in labels["columns"])
    rows = []
    for row in evidence["rows"]:
        capabilities = "".join('<td class="capability"><span aria-label="' + escape(labels["covered"]) +
                               '">✓</span></td>' if row["capabilities"][key] else
                               f'<td>{escape(labels["unverified"])}</td>' for key in keys)
        rows.append(f'<tr><th scope="row">{escape(row["name"])}'
                    f'<small>{escape(row["environment"][language])}</small></th>' + capabilities +
                    f'<td>{escape(row["note"][language])}</td></tr>')
    return '<div class="table-wrap"><table><thead><tr>' + headings + \
           '</tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>'


def render(template: str, data: dict, evidence: dict, language: str) -> str:
    computed = {
        "pills_html": "".join(f"<span>{escape(value)}</span>" for value in data["hero"]["pills"]),
        "features_html": cards(data["features"]),
        "architecture_cards_html": cards(data["architecture"]["cards"]),
        "steps_html": "".join(f'<button type="button" data-step="{i}" aria-pressed="false">'
                               f'{escape(value)}</button>' for i, value in enumerate(data["tour"]["steps"])),
        "evidence_html": evidence_cards(evidence, language, data["engineering"]),
        "platform_table_html": platform_table(evidence["platforms"], language, data["platform"]),
        "platform_sources_html": source_links(evidence["platforms"], language, data["engineering"]["sources"]),
        "platform_conclusion": evidence["platforms"]["conclusion"][language],
        "launch_steps_html": ordered(data["run"]["launch_steps"]),
        "demo_steps_html": ordered(data["run"]["demo_steps"]),
        "verification_rows_html": "".join("<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in row) + "</tr>"
                                          for row in data["verify"]["rows"]),
        "verification_headers_html": "".join(f"<th>{escape(value)}</th>" for value in data["verify"]["columns"]),
        "script_data_html": json.dumps({"shots": SHOTS, "captions": data["tour"]["captions"],
                                         "strings": data["script"]}, ensure_ascii=False).replace("<", "\\u003c"),
    }

    def substitute(match: re.Match) -> str:
        key = match.group(1)
        if key in computed:
            return str(computed[key]) if key.endswith("_html") else escape(computed[key])
        value = data
        for part in key.split("."):
            value = value[part]
        if not isinstance(value, str):
            raise ValueError(f"Expected text token: {key}")
        return escape(value)

    rendered = re.sub(r"\{\{([a-zA-Z0-9_.]+)\}\}", substitute, template)
    if "{{" in rendered:
        raise ValueError("Unresolved template token")
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check generated files without writing")
    args = parser.parse_args()
    data = json.loads((DEMO / "content.json").read_text(encoding="utf-8"))
    evidence = json.loads((DEMO / "evidence.json").read_text(encoding="utf-8"))
    check_structure(data["zh"], data["en"])
    template = (DEMO / "page.template.html").read_text(encoding="utf-8")
    stale = []
    for language, filename in (("zh", "index.html"), ("en", "index.en.html")):
        if len(data[language]["tour"]["steps"]) != len(SHOTS) or len(data[language]["tour"]["captions"]) != len(SHOTS):
            raise ValueError(f"Expected {len(SHOTS)} synchronized tour steps")
        rendered = render(template, data[language], evidence, language)
        target = DEMO / filename
        if args.check:
            if not target.exists() or target.read_text(encoding="utf-8") != rendered:
                stale.append(str(target.relative_to(ROOT)))
        else:
            with target.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(rendered)
    if stale:
        print("Stale demo pages; run python cpp_infer/tools/render_demo.py:\n" + "\n".join(stale))
        return 1
    print("Bilingual demo pages are synchronized." if args.check else "Rendered Chinese and English demo pages.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
