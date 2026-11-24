"""
Visual review sheet: one self-contained HTML page of thumbnails, for judging flags by eye.

A blur or exposure flag can only be judged by looking at the image, and a grid of thumbnails is judged
in a minute where a CSV means opening files one by one. It is also the quickest check of a threshold:
if the 'blurry' section is full of sharp images, the threshold is wrong.

Sections:
- one per tag (duplicate tags excluded, the group sections cover them): every image carrying the tag,
  whatever its status, worst first (the same ranking as review_manifest.csv), captioned with its path,
  status and metric value;
- duplicate groups: per grouping column, the chaining suspects and the largest groups from the
  report's `duplicates` section, members side by side, the kept image highlighted;
- a random sample of 'ok' images (fixed seed, so a regenerated sheet shows the same ones) as a baseline
  that keeps the reviewer calibrated.

Thumbnails show the stored pixels (no EXIF rotation, i.e. what a model would get), shrunk to at most
THUMB_SIZE px and never enlarged, as JPEG data: URIs, so the file opens offline and can be emailed.
Each section shows at most max_per_section images and says how many it left out. Nothing is loaded
from outside the file and nothing in the manifest is changed.
"""
import base64
import html
import io
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

from src.report.exports import _tag_order, row_tags, tag_metric, tag_severity, worst_first_key
from src.report.reasons import get_reason, is_duplicate_tag
from src.report.sections.profile import relative_paths
from src.report.status import OK, resolve_status

REVIEW_SHEET_FILE = "review_sheet.html"
THUMB_SIZE = 128
JPEG_QUALITY = 70
MAX_PER_SECTION = 60
MAX_GROUPS_PER_COLUMN = 5
MAX_MEMBERS_PER_GROUP = 12
SEED = 0
WORKERS = 8


# ---- thumbnails --------------------------------------------------------------------------------

def _to_rgb(img: Image.Image) -> Image.Image:
    """RGB for any mode: alpha composited on white, 16-bit and float stretched to 0-255."""
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        arr = np.asarray(img, dtype=np.float64)
        low, high = float(arr.min()), float(arr.max())
        scaled = (arr - low) / (high - low) * 255 if high > low else np.zeros_like(arr)
        return Image.fromarray(scaled.astype(np.uint8)).convert("RGB")
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return img.convert("RGB")


def thumbnail_uri(path: str, size: int = THUMB_SIZE, quality: int = JPEG_QUALITY) -> str | None:
    """A JPEG data: URI of the image shrunk to at most `size` px on its longest side, or None if it can't be decoded."""
    try:
        with Image.open(path) as img:
            img.load()
            rgb = _to_rgb(img)
    except Exception:
        return None
    rgb.thumbnail((size, size))  # only ever shrinks
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def make_thumbnails(paths, workers: int = WORKERS) -> dict[str, str | None]:
    """{absolute path: data URI or None} for the distinct paths, decoded in parallel."""
    unique = sorted(set(paths))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(zip(unique, pool.map(thumbnail_uri, unique)))


# ---- what goes in each section -----------------------------------------------------------------

def _rows(manifest: pd.DataFrame, data_dir: str, base_dir: str | None) -> pd.DataFrame:
    base = base_dir if base_dir is not None else os.getcwd()
    return pd.DataFrame({
        "rel_path": relative_paths(manifest, data_dir, base_dir),
        "abs_path": manifest["path"].map(lambda p: os.path.abspath(os.path.join(base, p))),
        "status": resolve_status(manifest),
        "reason": manifest["reason"].fillna("").astype(str),
    }, index=manifest.index)


def tag_sections(manifest: pd.DataFrame, rows: pd.DataFrame, max_per_section: int = MAX_PER_SECTION) -> list[dict]:
    """
    One section per non-duplicate tag, most severe first: every row with the tag, worst first, capped.
    Each item: rel_path, abs_path, status, and the caption's metric text.
    """
    tags_per_row = rows["reason"].map(row_tags)
    all_tags = sorted({t for tags in tags_per_row for t in tags if not is_duplicate_tag(t)}, key=_tag_order)
    sections = []
    for tag in all_tags:
        mask = tags_per_row.map(lambda tags: tag in tags)
        name, values = tag_metric(manifest[mask], tag)
        chosen = rows[mask].assign(value=values, key=worst_first_key(pd.Series(tag, index=values.index), values))
        chosen = chosen.sort_values(["key", "rel_path"], na_position="last")
        entry = get_reason(tag)
        sections.append({
            "tag": tag,
            "severity": tag_severity(tag),
            "description": entry.description if entry else "Not in the reason-tag catalog.",
            "total": int(mask.sum()),
            "metric": name,
            "items": [
                {"rel_path": r.rel_path, "abs_path": r.abs_path, "status": r.status,
                 "caption": f"{name} = {r.value:.4g}" if name and pd.notna(r.value) else ""}
                for r in chosen.head(max_per_section).itertuples()
            ],
        })
    return sections


def group_sections(duplicates: dict | None, rows: pd.DataFrame, max_groups: int = MAX_GROUPS_PER_COLUMN,
                   max_members: int = MAX_MEMBERS_PER_GROUP) -> list[dict]:
    """
    Per grouping column in the report's duplicates section: its chaining suspects, then its largest
    groups that aren't suspects, up to max_groups in total. Members come from the report (already capped
    there), the kept ('ok') ones first; at most max_members are shown per group.
    """
    if not duplicates:
        return []
    by_rel = rows.set_index("rel_path")
    sections = []
    for column, stats in duplicates.get("columns", {}).items():
        groups, seen = [], set()
        for group in [*stats["chaining"]["groups"], *stats["largest"]]:
            key = tuple(group["members"])
            if key in seen:
                continue
            seen.add(key)
            groups.append(group)
        groups = groups[:max_groups]
        if not groups:
            continue
        rendered = []
        for group in groups:
            kept = set(group["kept"])
            members = sorted(group["members"], key=lambda p: (p not in kept, p))
            rendered.append({
                "size": group["size"],
                "chaining_suspect": group["chaining_suspect"],
                "similarity": group.get("similarity"),
                "folders": group["folders"],
                "hidden": group["size"] - min(len(members), max_members),
                "items": [
                    {"rel_path": p, "abs_path": by_rel.at[p, "abs_path"] if p in by_rel.index else None,
                     "status": by_rel.at[p, "status"] if p in by_rel.index else "", "kept": p in kept, "caption": ""}
                    for p in members[:max_members]
                ],
            })
        sections.append({"column": column, "kind": stats["kind"], "groups_total": stats["groups"],
                         "suspects": stats["chaining"]["suspects"], "groups": rendered})
    return sections


def baseline_section(rows: pd.DataFrame, max_per_section: int = MAX_PER_SECTION, seed: int = SEED) -> dict:
    """A random sample of 'ok' rows with a fixed seed, sorted by path."""
    ok = rows[rows["status"] == OK]
    sample = ok.sample(n=min(max_per_section, len(ok)), random_state=seed).sort_values("rel_path")
    return {
        "total": len(ok),
        "seed": seed,
        "items": [{"rel_path": r.rel_path, "abs_path": r.abs_path, "status": r.status, "caption": ""} for r in sample.itertuples()],
    }


# ---- HTML --------------------------------------------------------------------------------------

CSS = """
:root { --bg: #ffffff; --fg: #1d1d1f; --muted: #6e6e73; --card: #f5f5f7; --line: #d2d2d7;
        --kept: #1a7f37; --error: #c62828; --warning: #b26a00; --info: #2f5fa7; --suspect: #8e24aa; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #141416; --fg: #f2f2f7; --muted: #a1a1a6; --card: #1f1f23; --line: #3a3a3f;
          --kept: #4cc764; --error: #ff6b6b; --warning: #f0b44c; --info: #79a8ff; --suspect: #d58cf0; }
}
* { box-sizing: border-box; }
body { margin: 0; padding: 24px 16px 64px; background: var(--bg); color: var(--fg);
       font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1200px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 18px; margin: 40px 0 4px; }
h3 { font-size: 15px; margin: 24px 0 6px; }
p, .meta { color: var(--muted); margin: 4px 0 12px; }
nav a { color: inherit; margin-right: 12px; white-space: nowrap; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 10px; }
.tile { background: var(--card); border: 2px solid transparent; border-radius: 8px; padding: 6px; min-width: 0; }
.tile.kept { border-color: var(--kept); }
.tile img, .tile .missing { display: block; width: 100%; aspect-ratio: 1; object-fit: contain;
       image-rendering: pixelated; background: var(--bg); border-radius: 4px; }
.tile .missing { display: grid; place-items: center; color: var(--error); font-size: 12px; text-align: center; }
.cap { font-size: 11px; margin-top: 4px; overflow-wrap: anywhere; }
.cap .path { color: var(--fg); } .cap .sub { color: var(--muted); }
.badge { display: inline-block; font-size: 11px; padding: 1px 6px; border-radius: 99px; border: 1px solid currentColor; margin-left: 6px; }
.error { color: var(--error); } .warning { color: var(--warning); } .info { color: var(--info); }
.suspect { color: var(--suspect); } .kept-text { color: var(--kept); }
.group { border-top: 1px solid var(--line); padding-top: 8px; margin-top: 12px; }
"""


def _e(text) -> str:
    return html.escape(str(text), quote=True)


def _tile(item: dict, thumbs: dict) -> str:
    uri = thumbs.get(item["abs_path"]) if item.get("abs_path") else None
    image = f'<img src="{uri}" alt="{_e(item["rel_path"])}" loading="lazy">' if uri else '<div class="missing">could not be decoded</div>'
    sub = " · ".join(x for x in (item.get("status", ""), item.get("caption", ""), "kept" if item.get("kept") else "") if x)
    css = "tile kept" if item.get("kept") else "tile"
    return (f'<figure class="{css}" style="margin:0">{image}<figcaption class="cap">'
            f'<div class="path">{_e(item["rel_path"])}</div><div class="sub">{_e(sub)}</div></figcaption></figure>')


def _shown(shown: int, total: int) -> str:
    return f"{shown:,} of {total:,} shown" if shown < total else f"all {total:,} shown"


def render_html(tags: list[dict], groups: list[dict], baseline: dict, thumbs: dict, meta: dict) -> str:
    """The whole page. `meta` holds generated_at, fingerprint and file_count for the header."""
    nav = ['<a href="#tags">Flags</a>', '<a href="#groups">Duplicate groups</a>', '<a href="#baseline">Baseline</a>']
    out = [
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>Review sheet</title><style>{CSS}</style></head><body><main>",
        "<h1>Review sheet</h1>",
        f'<div class="meta">Generated {_e(meta.get("generated_at", ""))} · {meta.get("file_count", 0):,} files · '
        f'fingerprint {_e(meta.get("fingerprint", ""))}</div>',
        "<p>Thumbnails show the stored pixels (EXIF rotation not applied), shrunk to at most "
        f"{THUMB_SIZE} px. Sections are capped; each says how many images it shows.</p>",
        f"<nav>{''.join(nav)}</nav>",
        '<h2 id="tags">Flags</h2>',
    ]
    if not tags:
        out.append("<p>No image carries a quality or integrity flag.</p>")
    for section in tags:
        order = f"worst first by {section['metric']}" if section["metric"] else "sorted by path"
        out += [
            f'<h3>{_e(section["tag"])}<span class="badge {_e(section["severity"])}">{_e(section["severity"])}</span></h3>',
            f'<p>{_e(section["description"])} {_shown(len(section["items"]), section["total"])}, {_e(order)}.</p>',
            '<div class="grid">', *(_tile(i, thumbs) for i in section["items"]), "</div>",
        ]

    out.append('<h2 id="groups">Duplicate groups</h2>')
    out.append("<p>Per grouping column: chaining suspects first, then the largest groups. "
               '<span class="kept-text">Green</span> marks the image kept for the group.</p>')
    if not groups:
        out.append("<p>No duplicate groups.</p>")
    for section in groups:
        out.append(f'<h3>{_e(section["column"])} <span class="meta">({_e(section["kind"])}; '
                   f'{section["groups_total"]:,} groups, {section["suspects"]:,} chaining suspects)</span></h3>')
        for group in section["groups"]:
            similarity = group["similarity"] or {}
            details = [f"{group['size']} images", ", ".join(f"{_e(f)}: {n}" for f, n in group["folders"].items())]
            if similarity.get("mean_pairwise_cosine") is not None:
                details.append(f"mean cosine {similarity['mean_pairwise_cosine']} ({_e(similarity['model'])})")
            if group["hidden"]:
                details.append(f"{group['hidden']} more not shown")
            suspect = '<span class="badge suspect">chaining suspect</span>' if group["chaining_suspect"] else ""
            out += [f'<div class="group"><div class="meta">{" · ".join(details)}{suspect}</div>',
                    '<div class="grid">', *(_tile(i, thumbs) for i in group["items"]), "</div></div>"]

    out += [
        '<h2 id="baseline">Baseline: unflagged images</h2>',
        f"<p>A random sample of images no check flagged (seed {baseline['seed']}), to compare the flagged "
        f"sections against. {_shown(len(baseline['items']), baseline['total'])}.</p>",
        '<div class="grid">', *(_tile(i, thumbs) for i in baseline["items"]), "</div>",
        "</main></body></html>",
    ]
    return "\n".join(out)


def build_review_sheet(
    manifest: pd.DataFrame,
    report: dict,
    data_dir: str,
    base_dir: str | None = None,
    max_per_section: int = MAX_PER_SECTION,
    seed: int = SEED,
) -> tuple[str, int]:
    """
    The review sheet's HTML and the number of thumbnails in it. Groups come from report['duplicates'],
    so the sheet shows the same groups as the JSON; the header uses report['generated_at'] and ['dataset'].
    """
    rows = _rows(manifest, data_dir, base_dir)
    tags = tag_sections(manifest, rows, max_per_section)
    groups = group_sections(report.get("duplicates"), rows, max_members=MAX_MEMBERS_PER_GROUP)
    baseline = baseline_section(rows, max_per_section, seed)

    items = [i for s in tags for i in s["items"]] + [i for s in groups for g in s["groups"] for i in g["items"]] + baseline["items"]
    thumbs = make_thumbnails(i["abs_path"] for i in items if i.get("abs_path"))
    meta = {"generated_at": report.get("generated_at", ""), **(report.get("dataset") or {})}
    return render_html(tags, groups, baseline, thumbs, meta), len(items)


def write_review_sheet(page: str, path: str) -> None:
    """Writes the page to a temp file and renames it over `path`."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".review-", suffix=".html")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(page)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
