# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""HTML report generator for ir-tracker SQLite databases.

Mirrors opt-viewer's static-site model: emit one HTML page per source file
plus an index page, with no required third-party dependencies. Pygments is
used for syntax highlighting if available, otherwise source is rendered as
plain ``<pre>`` text.
"""

from __future__ import annotations

import html
import os
import re
import sqlite3
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import irtrackdb

try:
    from pygments import highlight as _pyg_highlight
    from pygments.formatters import HtmlFormatter as _PygHtmlFormatter
    from pygments.lexers import get_lexer_for_filename, guess_lexer

    _HAVE_PYGMENTS = True
except ImportError:  # pragma: no cover - optional
    _HAVE_PYGMENTS = False


_STYLE_CSS = """\
body { font-family: -apple-system, Segoe UI, sans-serif; margin: 1em; color: #222; }
h1 { font-size: 1.4em; }
h2 { font-size: 1.1em; margin-top: 1.5em; }
table { border-collapse: collapse; width: 100%; }
table.index td, table.index th { border-bottom: 1px solid #eee; padding: 4px 8px; text-align: left; }
table.source { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; }
table.source td { vertical-align: top; padding: 0 6px; }
td.lineno { color: #888; text-align: right; user-select: none; width: 4em; border-right: 1px solid #eee; }
td.badge { width: 5em; text-align: right; }
td.src { white-space: pre; }
.badge-link { display: inline-block; background: #eef; color: #225; padding: 0 6px;
              border-radius: 8px; text-decoration: none; font-size: 11px; cursor: pointer; }
.badge-link:hover { background: #dde; }
.passes { display: none; background: #fafafa; border-left: 3px solid #99a;
          padding: 6px 10px; margin: 4px 0 4px 4em; }
.passes.open { display: block; }
.pass-hdr { font-weight: bold; color: #335; margin-top: 6px; }
.func { color: #553; margin-left: 1em; }
.inst { white-space: pre; margin-left: 2em; color: #111; }
.muted { color: #888; }
"""

_SCRIPT_JS = """\
function trackerToggle(id) {
  var e = document.getElementById(id);
  if (!e) return false;
  e.classList.toggle('open');
  return false;
}
"""


def _safe_filename(path: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", path).strip("_") or "file"


def _html_escape(text: str) -> str:
    return html.escape(text, quote=False)


def _pygments_highlight_lines(filename: str, text: str) -> Optional[List[str]]:
    if not _HAVE_PYGMENTS:
        return None
    try:
        lexer = get_lexer_for_filename(filename, stripnl=False)
    except Exception:
        try:
            lexer = guess_lexer(text, stripnl=False)
        except Exception:
            return None
    formatter = _PygHtmlFormatter(nowrap=True, noclasses=True)
    rendered = _pyg_highlight(text, lexer, formatter)
    if isinstance(rendered, bytes):
        rendered = rendered.decode("utf-8", errors="replace")
    return rendered.splitlines()


def _resolve_source(file_path: str, source_dirs: Sequence[str]) -> Optional[str]:
    if os.path.isabs(file_path) and os.path.isfile(file_path):
        return file_path
    base = os.path.basename(file_path)
    for d in source_dirs:
        for cand in (os.path.join(d, file_path), os.path.join(d, base)):
            if os.path.isfile(cand):
                return cand
    if os.path.isfile(file_path):
        return file_path
    return None


def _read_source(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def _files_in_db(con: sqlite3.Connection) -> List[Tuple[int, str]]:
    rows = con.execute(
        f"SELECT f.id, f.path, COUNT(i.id) AS n "
        f"FROM {irtrackdb.T_FILES} f "
        f"LEFT JOIN {irtrackdb.T_INSTR} i ON i.file_id = f.id "
        f"GROUP BY f.id ORDER BY f.path"
    ).fetchall()
    return [(int(r["id"]), r["path"]) for r in rows if int(r["n"]) > 0]


def _instrs_for_file(
    con: sqlite3.Connection, file_id: int
) -> Dict[int, List[irtrackdb.ShowInstRow]]:
    """Return {line: [ShowInstRow ...]} ordered by seq, function, block, inst_seq."""
    by_line: Dict[int, List[irtrackdb.ShowInstRow]] = {}
    query = (
        f"SELECT i.line, p.seq, p.pass_class, p.ir_unit, "
        f"i.function, i.basicblock, i.inst_seq, i.inst_text "
        f"FROM {irtrackdb.T_INSTR} i "
        f"JOIN {irtrackdb.T_PASSES} p ON i.pass_id = p.id "
        f"WHERE i.file_id = ? "
        f"ORDER BY i.line, p.seq, i.function, i.basicblock, i.inst_seq"
    )
    for row in con.execute(query, (file_id,)):
        rec = irtrackdb.ShowInstRow(
            int(row["seq"]),
            row["pass_class"] or "",
            row["ir_unit"] or "",
            row["function"] or "",
            row["basicblock"] or "",
            row["inst_text"] or "",
        )
        by_line.setdefault(int(row["line"]), []).append(rec)
    return by_line


def _dedup_groups(
    rows: Sequence[irtrackdb.ShowInstRow], all_passes: bool
) -> List[List[irtrackdb.ShowInstRow]]:
    """Group rows by seq; drop groups whose instruction text is identical to the
    previous emitted group when not ``all_passes`` (matches ``run_show``)."""
    by_seq: Dict[int, List[irtrackdb.ShowInstRow]] = {}
    for r in rows:
        by_seq.setdefault(r.seq, []).append(r)

    out: List[List[irtrackdb.ShowInstRow]] = []
    last_fp: Optional[str] = None
    for seq in sorted(by_seq):
        group = by_seq[seq]
        fp = "\n".join(r.inst_text for r in group)
        if not all_passes and fp == last_fp:
            continue
        out.append(group)
        last_fp = fp
    return out


def _render_passes_block(
    groups: Sequence[Sequence[irtrackdb.ShowInstRow]],
) -> str:
    if not groups:
        return '<span class="muted">no recorded snapshots</span>'
    parts: List[str] = []
    for group in groups:
        head = group[0]
        parts.append(
            '<div class="pass-hdr">seq={s} {p} <span class="muted">on {u}</span></div>'.format(
                s=head.seq,
                p=_html_escape(head.pass_class),
                u=_html_escape(head.ir_unit),
            )
        )
        cur_func = ""
        cur_bb = ""
        for r in group:
            if r.function != cur_func or r.basicblock != cur_bb:
                parts.append(
                    '<div class="func">function {f}, block {b}:</div>'.format(
                        f=_html_escape(r.function), b=_html_escape(r.basicblock)
                    )
                )
                cur_func, cur_bb = r.function, r.basicblock
            parts.append(
                '<div class="inst">{t}</div>'.format(t=_html_escape(r.inst_text))
            )
    return "".join(parts)


def _render_file_html(
    file_path: str,
    by_line: Dict[int, List[irtrackdb.ShowInstRow]],
    source_text: Optional[str],
    all_passes: bool,
    no_highlight: bool,
) -> str:
    lines: List[str] = []
    if source_text is None:
        max_line = max(by_line) if by_line else 0
        lines = [""] * max_line
    else:
        if no_highlight:
            highlighted = None
        else:
            highlighted = _pygments_highlight_lines(file_path, source_text)
        if highlighted is not None:
            lines = highlighted
        else:
            lines = source_text.splitlines()

    body: List[str] = ['<table class="source">']
    src_only_note = (
        ""
        if source_text is not None
        else '<p class="muted">Source file not found; showing recorded lines only.</p>'
    )

    for idx in range(1, len(lines) + 1):
        rows = by_line.get(idx)
        if rows:
            groups = _dedup_groups(rows, all_passes)
            badge = (
                '<a class="badge-link" href="#" '
                'onclick="return trackerToggle(\'p{n}\')">{c} pass{plural}</a>'
            ).format(
                n=idx,
                c=len(groups),
                plural="es" if len(groups) != 1 else "",
            )
        else:
            badge = ""
        src = lines[idx - 1] if idx - 1 < len(lines) else ""
        body.append(
            "<tr><td class='lineno'>{n}</td><td class='badge'>{b}</td>"
            "<td class='src'>{s}</td></tr>".format(n=idx, b=badge, s=src)
        )
        if rows:
            body.append(
                "<tr><td></td><td></td><td>"
                "<div class='passes' id='p{n}'>{block}</div>"
                "</td></tr>".format(
                    n=idx,
                    block=_render_passes_block(groups),
                )
            )
    body.append("</table>")

    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>{title}</title>"
        "<link rel='stylesheet' href='style.css'>"
        "<script>{js}</script>"
        "</head><body>"
        "<p><a href='index.html'>&larr; index</a></p>"
        "<h1>{title}</h1>{note}{body}</body></html>"
    ).format(
        title=_html_escape(file_path),
        js=_SCRIPT_JS,
        note=src_only_note,
        body="".join(body),
    )


def _render_index_html(
    files: Sequence[Tuple[str, str, int, int]],
    pass_count: int,
    inst_count: int,
) -> str:
    rows = []
    for path, link, n_lines, n_rows in files:
        rows.append(
            "<tr><td><a href='{l}'>{p}</a></td>"
            "<td>{nl}</td><td>{nr}</td></tr>".format(
                l=link, p=_html_escape(path), nl=n_lines, nr=n_rows
            )
        )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>ir-tracker report</title>"
        "<link rel='stylesheet' href='style.css'></head><body>"
        "<h1>ir-tracker report</h1>"
        "<p>{p} pass snapshots, {i} instruction rows.</p>"
        "<table class='index'><thead><tr>"
        "<th>Source file</th><th>Lines tracked</th><th>Instruction rows</th>"
        "</tr></thead><tbody>{rows}</tbody></table></body></html>"
    ).format(p=pass_count, i=inst_count, rows="".join(rows))


def generate_html(
    con: sqlite3.Connection,
    output_dir: str,
    source_dirs: Sequence[str],
    all_passes: bool,
    no_highlight: bool,
    file_filter: str = "",
) -> int:
    if irtrackdb.get_schema_version(con) < 1:
        print("ir-tracker: unsupported schema version", file=sys.stderr)
        return 1

    os.makedirs(output_dir, exist_ok=True)

    files = _files_in_db(con)
    if file_filter:
        needle = file_filter.lower()
        files = [(fid, p) for fid, p in files if needle in p.lower()]
    if not files:
        print("ir-tracker: no source files with instructions in DB", file=sys.stderr)
        return 1

    pass_count = int(
        con.execute(f"SELECT COUNT(*) AS c FROM {irtrackdb.T_PASSES}").fetchone()["c"]
    )
    inst_count = int(
        con.execute(f"SELECT COUNT(*) AS c FROM {irtrackdb.T_INSTR}").fetchone()["c"]
    )

    with open(os.path.join(output_dir, "style.css"), "w", encoding="utf-8") as f:
        f.write(_STYLE_CSS)

    index_entries: List[Tuple[str, str, int, int]] = []
    for file_id, file_path in files:
        by_line = _instrs_for_file(con, file_id)
        n_rows = sum(len(v) for v in by_line.values())
        link = _safe_filename(file_path) + ".html"
        resolved = _resolve_source(file_path, source_dirs)
        src_text = _read_source(resolved) if resolved else None
        page = _render_file_html(
            file_path, by_line, src_text, all_passes, no_highlight
        )
        with open(os.path.join(output_dir, link), "w", encoding="utf-8") as f:
            f.write(page)
        index_entries.append((file_path, link, len(by_line), n_rows))

    with open(os.path.join(output_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(_render_index_html(index_entries, pass_count, inst_count))

    print(
        "ir-tracker: wrote {n} file page(s) + index to {d}".format(
            n=len(index_entries), d=output_dir
        )
    )
    return 0
