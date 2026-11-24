"""
Human-readable summary for the client: report.html and report.md, rendered from validation_report.json.

Every number comes from the report dict. Nothing here computes a new statistic; it only picks, orders
and formats values that are already in the JSON, so the prose and the data can never disagree. Text
that explains a tag comes from the reason-tag catalog.

Both files are rendered from one list of blocks (headings, paragraphs, tables, lists, a banner, charts),
so they say the same thing; Markdown leaves out the charts. The HTML page skeleton (head, CSS) is in
templates/report.html; Markdown needs none, so it is just the joined blocks.

Formatting: thousands separators, percentages to one decimal ('<0.1%' for a non-zero share that would
round to 0.0%). No absolute local paths: the report holds none, and links are relative to the report folder.
"""
import html
import os
import tempfile
from pathlib import Path
from string import Template

from src.photometric import BLUR_REF_MAX_SIDE
from src.report.reasons import get_reason

SUMMARY_HTML, SUMMARY_MD = "report.html", "report.md"
TEMPLATES = Path(__file__).with_name("templates")
TITLE = "Dataset validation report"

SEVERITY_HEADINGS = {
    "error": ("Unusable files", "The file cannot be used: it does not decode, or it has no content."),
    "warning": ("Quality concerns", "The image may be fine, but a person should judge it."),
    "info": ("Redundant copies", "The image is fine, but another image already stands for it."),
    "unknown": ("Unknown tags", "These tags are not in the catalog, so their meaning is not known. The manifest is inconsistent."),
}
CHART_METRICS = ("blur_score", "mean_brightness", "pixel_std", "megapixels")
DISTRIBUTION_METRICS = ("width", "height", "megapixels", "aspect_ratio", "file_size_bytes",
                        "mean_brightness", "pixel_std", "dark_clip_ratio", "bright_clip_ratio", "blur_score")
MAX_FOLDERS = 15
MAX_PAIRS = 5
MAX_RESOLUTIONS = 5


# ---- formatting --------------------------------------------------------------------------------

def fmt_int(n) -> str:
    return "–" if n is None else f"{int(n):,}"


def fmt_pct(pct, count=None) -> str:
    if pct is None:
        return "–"
    if count and pct < 0.05:
        return "<0.1%"
    return f"{pct:.1f}%"


def fmt_share(share: dict | None) -> str:
    """'30,755 (52.0%)'."""
    if not share:
        return "–"
    return f"{fmt_int(share['count'])} ({fmt_pct(share['pct'], share['count'])})"


def fmt_num(x) -> str:
    """A metric value: from 100 up (and whole numbers) with separators and no decimals, smaller ones to 4 significant digits."""
    if x is None:
        return "–"
    if abs(x) >= 100 or (float(x).is_integer() and abs(x) >= 1):
        return f"{x:,.0f}"
    return f"{x:.4g}"


def fmt_folder(folder: str) -> str:
    return "(top level)" if folder == "." else folder


def fmt_bytes(n) -> str:
    if n is None:
        return "–"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1000


def fmt_params(params: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items()) or "defaults"


# ---- blocks ------------------------------------------------------------------------------------
# Each block is a tuple whose first item is its kind:
#   ("h1", text, meta) ("h2", text) ("h3", text) ("p", text) ("meta", text)
#   ("banner", title, items) ("kpis", [(label, value)]) ("table", headers, rows, numeric_columns)
#   ("list", items) ("links", [(label, href, description)]) ("charts", [(title, histogram, threshold)])

def _header(report: dict) -> list:
    dataset = report.get("dataset") or {}
    meta = (f"Generated {report.get('generated_at', '')} · dataset fingerprint {dataset.get('fingerprint', '–')} "
            f"({dataset.get('method', '–')}) · report schema v{report.get('schema_version')}")
    blocks = [("h1", TITLE, meta)]
    if not report.get("valid", False):
        items = [f"{i['code']}: {i['message']}" for i in report.get("issues", []) if i["severity"] == "error"]
        blocks.append(("banner", "This report is not valid. The manifest or a report step has errors, so the "
                       "numbers below cannot be trusted.", items))
    return blocks


def _summary(report: dict) -> list:
    dataset, status = report.get("dataset") or {}, report.get("status_counts")
    blocks = [("h2", "Summary")]
    if not status:
        return blocks + [("p", "The flag statistics could not be built, so there is no summary.")]
    unique = _dedup_level(report)
    text = (f"We received {fmt_int(dataset.get('file_count'))} files ({fmt_bytes(dataset.get('total_bytes'))}). "
            f"{fmt_share(status['ok'])} are usable as delivered: no check flagged them. "
            f"{fmt_share(status['review'])} need a person to look at them, and we recommend excluding "
            f"{fmt_share(status['drop'])}.")
    kpis = [("Received", fmt_int(dataset.get("file_count"))), ("Usable", fmt_share(status["ok"])),
            ("Need review", fmt_share(status["review"])), ("Recommended exclusion", fmt_share(status["drop"]))]
    if unique:
        text += (f" Exact and visually identical copies account for {fmt_share(unique['redundant'])} images, "
                 f"so the delivery holds {fmt_share(unique['unique'])} unique images.")
        kpis.append(("Unique (exact + visual)", fmt_share(unique["unique"])))
    if dataset.get("missing_files"):
        text += f" {fmt_int(dataset['missing_files'])} files listed in the manifest no longer exist."
    return blocks + [("p", text), ("kpis", kpis)]


def _dedup_level(report: dict) -> dict | None:
    """The last level built from exact and perceptual grouping only (the grouping keep/drop follows)."""
    duplicates = report.get("duplicates") or {}
    levels = [l for l in duplicates.get("levels", []) if set(l["columns"]) <= {"hash", "group_id"}]
    return levels[-1] if levels else None


def _findings(report: dict) -> list:
    flags = report.get("flags")
    blocks = [("h2", "Findings")]
    if not flags:
        return blocks + [("p", "The flag statistics could not be built.")]
    blocks.append(("p", f"{fmt_share(flags['images_flagged'])} images carry at least one flag. {flags['note']}"))
    worst = flags["severity"]["worst"]
    for severity, (heading, explanation) in SEVERITY_HEADINGS.items():
        tags = [t for t in flags["tags"] if t["severity"] == severity]
        if not tags:
            continue
        rows = []
        for t in tags:
            entry = get_reason(t["tag"])
            rows.append([t["tag"], entry.description if entry else "Not in the catalog.", fmt_share(t),
                         fmt_int(t["by_status"]["review"]), fmt_int(t["by_status"]["drop"])])
        blocks += [
            ("h3", f"{heading} ({severity})"),
            ("p", f"{explanation} {fmt_share(worst.get(severity))} images have this as their most serious finding."),
            ("table", ["Tag", "What it means", "Images", "Review", "Exclude"], rows, {2, 3, 4}),
        ]
    pairs = flags["cooccurrence"][:MAX_PAIRS]
    blocks.append(("h3", "Overlap between flags"))
    if pairs:
        rows = [[f"{a} + {b}", fmt_share(p), f"{fmt_pct(p['given'][a]['pct'])} of {a}", f"{fmt_pct(p['given'][b]['pct'])} of {b}"]
                for p in pairs for a, b in [p["tags"]]]
        blocks += [("p", "Images that carry two flags at once. A large overlap can mean two checks measure the same thing."),
                   ("table", ["Flags", "Images", "Share of first", "Share of second"], rows, {1, 2, 3})]
    else:
        blocks.append(("p", "No image carries more than one flag."))
    return blocks


def _thresholds_by_column(report: dict) -> dict:
    profile = report.get("profile") or {}
    return {t["column"]: t for t in profile.get("thresholds", {}).get("thresholds", [])}


def _profile(report: dict) -> list:
    profile = report.get("profile")
    blocks = [("h2", "What is in the delivery")]
    if not profile:
        return blocks + [("p", "The dataset profile could not be built.")]
    files = profile["files"]
    blocks.append(("p", f"{fmt_int(files['file_count'])} files, {fmt_bytes(files['total_bytes'])} in total. "
                        f"{fmt_share(profile['decodable'])} decode as images; {fmt_share(profile['undecodable'])} do not."))

    rows = []
    for metric, stats in profile["categorical"]["metrics"].items():
        if stats["values"]:
            rows.append([metric, ", ".join(f"{v['value']} {fmt_pct(v['pct'], v['count'])}" for v in stats["values"][:4])])
    if rows:
        blocks += [("h3", "Formats"), ("table", ["Property", "Values (share of decodable images)"], rows, set())]

    extensions = profile["extensions"]
    if extensions.get("available"):
        blocks.append(("p", f"File extension does not match the actual format for {fmt_share(extensions['mismatched'])} "
                            f"of the files whose format could be checked."))
    else:
        blocks.append(("p", "Extension versus actual format: not available. The manifest predates the format column; "
                            "rerun the decode check to fill it."))

    resolutions = profile["resolutions"]
    if resolutions.get("available") and resolutions["top"]:
        rows = [[r["resolution"], fmt_share(r)] for r in resolutions["top"][:MAX_RESOLUTIONS]]
        blocks += [("h3", "Image sizes"),
                   ("p", f"{fmt_int(resolutions['distinct'])} distinct resolutions. The most common:"),
                   ("table", ["Resolution", "Images"], rows, {1})]

    metrics = profile["distributions"]["metrics"]
    rows = [[m, fmt_int(metrics[m]["count"]), *(fmt_num(metrics[m][k]) for k in ("min", "p5", "median", "p95", "max"))]
            for m in DISTRIBUTION_METRICS if m in metrics and metrics[m]["count"]]
    if rows:
        blocks += [("h3", "Distributions"),
                   ("table", ["Metric", "Images", "Min", "5th pct", "Median", "95th pct", "Max"], rows, {1, 2, 3, 4, 5, 6})]
    thresholds = _thresholds_by_column(report)
    charts = [(m, metrics[m]["histogram"], thresholds.get(m)) for m in CHART_METRICS if m in metrics and metrics[m]["count"]]
    if charts:
        blocks.append(("charts", charts))

    folders = profile["folders"]
    if folders["folders"]:
        rows = [[fmt_folder(f["folder"]), fmt_share(f), fmt_pct(f["flagged"]["pct"], f["flagged"]["count"]),
                 f"{f['top_tag']['tag']} ({fmt_int(f['top_tag']['count'])})" if f["top_tag"] else "–"]
                for f in folders["folders"][:MAX_FOLDERS]]
        blocks += [("h3", "Folders"),
                   ("p", "Folders are often the only metadata in a delivery (camera, date, source), so a flag that "
                         "concentrates in one folder points at that source." +
                         (f" Showing {MAX_FOLDERS} of {fmt_int(len(folders['folders']))} folders." if len(folders["folders"]) > MAX_FOLDERS else "")),
                   ("table", ["Folder", "Images", "Flagged", "Most common flag"], rows, {1, 2})]
        rows = [[c["tag"], fmt_folder(c["folder"]), f"{fmt_pct(c['pct'], c['count'])} of {fmt_int(c['of'])}", fmt_int(c["folders_with_tag"])]
                for c in folders["tag_concentration"]]
        if rows:
            blocks.append(("table", ["Flag", "Folder with most", "Share of that flag", "Folders with it"], rows, {2, 3}))
    return blocks


def _duplicates(report: dict) -> list:
    duplicates = report.get("duplicates")
    blocks = [("h2", "Duplicates")]
    if not duplicates:
        return blocks + [("p", "The duplicate statistics could not be built.")]
    labels = {"exact": "Exact (byte-identical)", "perceptual": "+ visually identical (perceptual hash)"}
    rows = [[labels.get(l["level"], f"+ similar ({l['level']})"), fmt_int(l["groups"]), fmt_share(l["redundant"]),
             fmt_share(l["unique"]), fmt_int(l["added_redundant"])] for l in duplicates["levels"]]
    if rows:
        blocks += [("p", "Each level adds a looser kind of duplicate to the ones before it. Redundant images are the copies "
                         "beyond the first in each group; unique = images minus redundant."),
                   ("table", ["Level", "Groups", "Redundant", "Unique", "Added by this level"], rows, {1, 2, 3, 4})]
    rows = []
    for column, stats in duplicates["columns"].items():
        largest = stats["largest"][0]["size"] if stats["largest"] else 0
        rows.append([column, stats["kind"], fmt_int(stats["groups"]), fmt_share(stats["cross_folder"]),
                     fmt_int(stats["chaining"]["suspects"]), fmt_int(largest)])
    if rows:
        blocks += [("h3", "Groups per grouping"),
                   ("p", "Groups that span folders mean the same content arrived from several sources (or, in a labeled "
                         "dataset, with different labels). Chaining suspects are unusually large groups that may link "
                         "unrelated images through a chain of similar pairs; check them in the review sheet."),
                   ("table", ["Grouping", "Kind", "Groups", "Spanning folders", "Chaining suspects", "Largest"], rows, {2, 3, 4, 5})]
    notes = []
    if duplicates.get("matches_split_group") is False:
        notes.append("The manifest's split_group does not match the union of the grouping columns, so it was built from "
                     "other columns or not rebuilt. Splits made from it may not keep duplicates together.")
    notes += [f"No similarity for {u['model']}: {u['reason']}." for u in duplicates.get("similarity_unavailable", [])]
    if notes:
        blocks.append(("list", notes))
    return blocks


def _methods(report: dict) -> list:
    blocks = [("h2", "Methods and thresholds")]
    rows = []
    for check, runs in report.get("checks", {}).items():
        for run in runs:
            commit = (run["git_commit"] or "unknown")[:8] + ("*" if run["git_dirty"] else "")
            rows.append([check, fmt_params(run["params"]), run["finished_at"] or "–", run["status"], commit])
    if rows:
        blocks += [("p", "Every step that flagged or grouped images, with the settings it ran with. * = the code had "
                         "uncommitted changes."),
                   ("table", ["Step", "Settings", "Finished (UTC)", "Status", "Code"], rows, set())]
    else:
        blocks.append(("p", "No step is recorded in audits.json, so the settings behind the flags are unknown."))

    thresholds = (report.get("profile") or {}).get("thresholds", {})
    rows = [[f"{t['parameter']} ({t['column']})", fmt_num(t["threshold"]), fmt_share(t["flagged"]),
             f"{fmt_int(t['borderline']['count'])}: {fmt_int(t['borderline_flagged'])} flagged, {fmt_int(t['borderline_passing'])} pass"]
            for t in thresholds.get("thresholds", [])]
    if rows:
        margin = f"{100 * thresholds['margin']:g}%"
        blocks += [("h3", "Threshold sensitivity"),
                   ("p", f"How many images each threshold flags, and how many lie within ±{margin} of it. A large "
                         "borderline count means a small change in the threshold would move many images."),
                   ("table", ["Threshold", "Value", "Flagged", f"Within ±{margin}"], rows, {1, 2})]
    if thresholds.get("unavailable"):
        blocks.append(("list", [f"{u['parameter']} ({u['column']}): {u['reason']}" for u in thresholds["unavailable"]]))
    return blocks


def _caveats(report: dict) -> list:
    items = [
        f"Blur scores are measured on images shrunk to {BLUR_REF_MAX_SIDE} px on the longest side. Smaller images "
        "are not enlarged, so their scores are not on exactly the same scale.",
        "Visual and embedding duplicates depend on the hash or model and its threshold. Groups are built by chaining "
        "similar pairs, so a loose threshold can join unrelated images; large groups should be checked by eye.",
        "Which image is kept from a duplicate group is a choice, not a quality judgment.",
        "Nothing here checks what the images show, whether they contain personal or sensitive content, or whether "
        "they fit the intended use. Labels, if any, are not verified.",
    ]
    checks = report.get("checks", {})
    if any(run["git_dirty"] for runs in checks.values() for run in runs):
        items.append("Some steps ran on code with uncommitted changes (marked * above), so they may not be reproducible from the commit.")
    mismatched = sorted({c for c, runs in checks.items() for run in runs if not run["paths_digest_matches"]})
    if mismatched:
        items.append(f"These steps ran on a different set of files than the current manifest: {', '.join(mismatched)}.")
    failed = sorted({c for c, runs in checks.items() for run in runs if run["status"] != "ok"})
    if failed:
        items.append(f"The last run of these steps did not finish: {', '.join(failed)}.")
    items += [f"{i['code']}: {i['message']}" for i in report.get("issues", []) if i["severity"] == "warning"]
    return [("h2", "Caveats"), ("list", items)]


def _files(report: dict) -> list:
    links = []
    for output in report.get("outputs", []):
        href = output["path"].split("/", 1)[1] if "/" in output["path"] else output["path"]
        rows = f" ({fmt_int(output['rows'])} rows)" if output.get("rows") is not None and href.endswith(".csv") else ""
        links.append((href, href, output["description"] + rows))
    return [("h2", "Files"), ("links", links)] if links else []


def build_blocks(report: dict) -> list:
    """The summary as a list of blocks, in page order. Uses only values in `report`."""
    return [*_header(report), *_summary(report), *_findings(report), *_profile(report),
            *_duplicates(report), *_methods(report), *_caveats(report), *_files(report)]


# ---- rendering ---------------------------------------------------------------------------------

def _e(text) -> str:
    return html.escape(str(text), quote=True)


def histogram_svg(histogram: dict, threshold: dict | None = None, width: int = 300, height: int = 110) -> str:
    """Inline SVG bar chart of a histogram, with the threshold as a dashed line when it falls in range."""
    counts, edges = histogram["counts"], histogram["edges"]
    if not counts:
        return ""
    plot_h, peak, bar_w = height - 18, max(counts) or 1, width / len(counts)
    bars = "".join(
        f'<rect class="bar" x="{i * bar_w:.1f}" y="{plot_h - c / peak * plot_h:.1f}" width="{max(bar_w - 1, 0.5):.1f}" '
        f'height="{c / peak * plot_h:.1f}"><title>{fmt_num(edges[i])}–{fmt_num(edges[i + 1])}: {fmt_int(c)}</title></rect>'
        for i, c in enumerate(counts)
    )
    line = ""
    if threshold and edges[-1] > edges[0] and edges[0] <= threshold["threshold"] <= edges[-1]:
        x = (threshold["threshold"] - edges[0]) / (edges[-1] - edges[0]) * width
        line = f'<line class="threshold" x1="{x:.1f}" x2="{x:.1f}" y1="0" y2="{plot_h}"><title>{_e(threshold["parameter"])} = {fmt_num(threshold["threshold"])}</title></line>'
    axis = (f'<text class="axis" x="0" y="{height - 4}">{fmt_num(edges[0])}</text>'
            f'<text class="axis" x="{width}" y="{height - 4}" text-anchor="end">{fmt_num(edges[-1])}</text>')
    return f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">{bars}{line}{axis}</svg>'


def render_html(blocks: list) -> str:
    parts = []
    for block in blocks:
        kind = block[0]
        if kind == "h1":
            parts.append(f"<h1>{_e(block[1])}</h1><p class=\"meta\">{_e(block[2])}</p>")
        elif kind in ("h2", "h3"):
            parts.append(f"<{kind}>{_e(block[1])}</{kind}>")
        elif kind == "p":
            parts.append(f"<p>{_e(block[1])}</p>")
        elif kind == "banner":
            items = "".join(f"<li>{_e(i)}</li>" for i in block[2])
            parts.append(f'<div class="banner"><strong>{_e(block[1])}</strong><ul>{items}</ul></div>')
        elif kind == "kpis":
            parts.append('<div class="kpis">' + "".join(
                f'<div class="kpi"><div class="value">{_e(v)}</div><div class="label">{_e(l)}</div></div>' for l, v in block[1]) + "</div>")
        elif kind == "table":
            _, headers, rows, numeric = block
            head = "".join(f'<th class="num">{_e(h)}</th>' if i in numeric else f"<th>{_e(h)}</th>" for i, h in enumerate(headers))
            body = "".join("<tr>" + "".join(
                f'<td class="num">{_e(c)}</td>' if i in numeric else f"<td>{_e(c)}</td>" for i, c in enumerate(row)) + "</tr>" for row in rows)
            parts.append(f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')
        elif kind == "list":
            parts.append("<ul>" + "".join(f"<li>{_e(i)}</li>" for i in block[1]) + "</ul>")
        elif kind == "links":
            parts.append("<ul>" + "".join(f'<li><a href="{_e(href)}">{_e(label)}</a>: {_e(desc)}</li>' for label, href, desc in block[1]) + "</ul>")
        elif kind == "charts":
            figures = []
            for title, histogram, threshold in block[1]:
                note = f" · dashed line: {threshold['parameter']} = {fmt_num(threshold['threshold'])}" if threshold else ""
                figures.append(f"<figure>{histogram_svg(histogram, threshold)}<figcaption>{_e(title)}{_e(note)}</figcaption></figure>")
            parts.append('<div class="charts">' + "".join(figures) + "</div>")
    page = Template((TEMPLATES / SUMMARY_HTML).read_text(encoding="utf-8"))
    return page.substitute(title=_e(TITLE), body="\n".join(parts))


def _md_cell(text) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_markdown(blocks: list) -> str:
    parts = []
    for block in blocks:
        kind = block[0]
        if kind == "h1":
            parts.append(f"# {block[1]}\n\n_{block[2]}_")
        elif kind == "h2":
            parts.append(f"## {block[1]}")
        elif kind == "h3":
            parts.append(f"### {block[1]}")
        elif kind == "p":
            parts.append(block[1])
        elif kind == "banner":
            parts.append(f"> **{block[1]}**\n" + "".join(f">\n> - {i}\n" for i in block[2]).rstrip("\n"))
        elif kind == "kpis":
            parts.append("\n".join(f"- **{l}:** {v}" for l, v in block[1]))
        elif kind == "table":
            _, headers, rows, numeric = block
            align = ["---:" if i in numeric else "---" for i in range(len(headers))]
            lines = ["| " + " | ".join(map(_md_cell, headers)) + " |", "| " + " | ".join(align) + " |"]
            lines += ["| " + " | ".join(map(_md_cell, row)) + " |" for row in rows]
            parts.append("\n".join(lines))
        elif kind == "list":
            parts.append("\n".join(f"- {i}" for i in block[1]))
        elif kind == "links":
            parts.append("\n".join(f"- [{label}]({href}): {desc}" for label, href, desc in block[1]))
        elif kind == "charts":
            parts.append("_Histograms are in the HTML version of this report._")
    return "\n\n".join(parts) + "\n"


def _write_text(text: str, path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".summary-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def write_summary(report: dict, out_dir: str) -> list[str]:
    """Renders report.html and report.md from `report` into out_dir. Returns the two file names."""
    blocks = build_blocks(report)
    pages = {SUMMARY_HTML: render_html(blocks), SUMMARY_MD: render_markdown(blocks)}
    for name, text in pages.items():
        _write_text(text, os.path.join(out_dir, name))
    return list(pages)


def remove_summary(out_dir: str) -> None:
    for name in (SUMMARY_HTML, SUMMARY_MD):
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            os.remove(path)
