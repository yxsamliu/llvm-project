# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""SQLite helpers for llvm/tools/ir-tracker (mirrors former llvm-ir-tracker.cpp)."""

from __future__ import annotations

import os
import sqlite3
import sys
from typing import Dict, List, NamedTuple, Optional, Sequence

T_FILES = "ir_tracker_files"
T_META = "ir_tracker_meta"
T_PASSES = "ir_tracker_passes"
T_INSTR = "ir_tracker_instructions"


def lower_string(s: str) -> str:
    return s.lower()


def open_db_readonly(path: str) -> Optional[sqlite3.Connection]:
    if not path:
        print("ir-tracker: empty database path", file=sys.stderr)
        return None
    if not os.path.isfile(path):
        print(f"ir-tracker: database not found: {path}", file=sys.stderr)
        return None
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def resolve_file_ids(con: sqlite3.Connection, file_pat: str) -> List[int]:
    pat_lower = lower_string(file_pat.strip())
    cur = con.execute(f"SELECT id, path FROM {T_FILES}")
    ids: List[int] = []
    for r in cur:
        rid = int(r["id"])
        path = r["path"] or ""
        path_lower = lower_string(path)
        if pat_lower in path_lower or path.endswith(file_pat) or path == file_pat:
            ids.append(rid)
    return ids


def parse_trace_kind(s: str) -> Tuple[bool, str]:
    """Returns (all_kinds, single_kind) where single_kind is ir|mir|isa."""
    t = lower_string(s.strip())
    if not t:
        raise ValueError
    if t == "all":
        return True, ""
    m = {
        "ir": "ir",
        "mir": "mir",
        "isa": "isa",
        "assembly": "isa",
        "asm": "isa",
    }.get(t)
    if m:
        return False, m
    raise ValueError


def get_schema_version(con: sqlite3.Connection) -> int:
    cur = con.execute(
        f"SELECT value FROM {T_META} WHERE key = 'schema_version'"
    )
    row = cur.fetchone()
    if not row or row["value"] is None:
        return -1
    try:
        return int(row["value"])
    except ValueError:
        return -1


def get_max_seq_for_kind(con: sqlite3.Connection, kind: str) -> int:
    cur = con.execute(
        f"""
        SELECT MAX(p.seq) AS m FROM {T_INSTR} i
        JOIN {T_PASSES} p ON i.pass_id = p.id
        WHERE i.kind = ?
        """,
        (kind,),
    )
    row = cur.fetchone()
    if not row or row["m"] is None:
        return -1
    return int(row["m"])


def get_max_seq_for_file_line_kind(
    con: sqlite3.Connection,
    kind: str,
    file_ids: Sequence[int],
    line: int,
    trace_col: Optional[int],
    trace_opcode: str,
) -> int:
    in_clause = ",".join("?" * len(file_ids))
    col_sql = " AND i.col = ?" if trace_col is not None and trace_col >= 0 else ""
    opc_sql = " AND i.opcode = ?" if trace_opcode else ""
    q = (
        f"SELECT MAX(p.seq) AS m FROM {T_INSTR} i JOIN {T_PASSES} p "
        f"ON i.pass_id = p.id WHERE i.kind = ? AND i.file_id IN ({in_clause}) "
        f"AND i.line = ?{col_sql}{opc_sql}"
    )
    params: List[object] = [kind, *file_ids, line]
    if trace_col is not None and trace_col >= 0:
        params.append(trace_col)
    if trace_opcode:
        params.append(trace_opcode)
    cur = con.execute(q, params)
    row = cur.fetchone()
    if not row or row["m"] is None:
        return -1
    return int(row["m"])


def run_passes(con: sqlite3.Connection) -> int:
    cur = con.execute(
        f"SELECT id, seq, pass_class, ir_unit FROM {T_PASSES} ORDER BY seq ASC"
    )
    rows = cur.fetchall()
    for r in rows:
        seq = int(r["seq"])
        rid = int(r["id"])
        pc = r["pass_class"] or ""
        iu = r["ir_unit"] or ""
        print(f"{seq:5d}  id={rid:<6}  '{pc}'  on '{iu}'")
    print(f"total passes recorded: {len(rows)}")
    return 0


def run_trace(
    con: sqlite3.Connection,
    file_pat: str,
    line_s: str,
    trace_col: Optional[int],
    trace_opcode: str,
    trace_kind: str,
) -> int:
    if get_schema_version(con) < 5:
        print(
            "ir-tracker: 'trace' with representation kinds requires "
            "ir-tracker schema_version >= 5",
            file=sys.stderr,
        )
        return 1

    file_ids = resolve_file_ids(con, file_pat)
    if not file_ids:
        print(
            f"ir-tracker: no {T_FILES} rows match --file "
            "(try a basename or substring)",
            file=sys.stderr,
        )
        return 1

    try:
        line = int(line_s, 0)
    except ValueError:
        print("ir-tracker: invalid --line", file=sys.stderr)
        return 1
    if line <= 0:
        print("ir-tracker: invalid --line", file=sys.stderr)
        return 1

    try:
        all_kinds, single_kind = parse_trace_kind(trace_kind)
    except ValueError:
        print(
            "ir-tracker: invalid --kind (expected ir, mir, isa, assembly, asm, or all)",
            file=sys.stderr,
        )
        return 1

    kind_list = ["ir", "mir", "isa"] if all_kinds else [single_kind]

    any_recorded = any(
        get_max_seq_for_file_line_kind(
            con, k, file_ids, line, trace_col, trace_opcode
        )
        >= 0
        for k in kind_list
    )
    if not any_recorded:
        print(
            "ir-tracker: no rows recorded for the requested kind(s)",
            file=sys.stderr,
        )
        return 1

    in_clause = ",".join("?" * len(file_ids))
    kind_sql = " AND i.kind = ?"
    col_sql = " AND i.col = ?" if trace_col is not None and trace_col >= 0 else ""
    opc_sql = " AND i.opcode = ?" if trace_opcode else ""

    for k in kind_list:
        max_seq = get_max_seq_for_file_line_kind(
            con, k, file_ids, line, trace_col, trace_opcode
        )
        if max_seq < 0:
            continue

        if all_kinds and len(kind_list) > 1:
            print(f"kind={k}:")

        # Final pass count
        q_cnt = (
            f"SELECT COUNT(*) AS c FROM {T_INSTR} i JOIN {T_PASSES} p "
            f"ON i.pass_id = p.id WHERE p.seq = ? AND i.file_id IN ({in_clause}) "
            f"AND i.line = ?{kind_sql}{col_sql}{opc_sql}"
        )
        b_cnt: List[object] = [max_seq, *file_ids, line, k]
        if trace_col is not None and trace_col >= 0:
            b_cnt.append(trace_col)
        if trace_opcode:
            b_cnt.append(trace_opcode)
        cur = con.execute(q_cnt, b_cnt)
        final_count = int(cur.fetchone()["c"])

        ids_join = ",".join(str(x) for x in file_ids)
        print(
            f"Matches at final pass (seq={max_seq}): {final_count} "
            f"instruction(s) (file id(s) {ids_join}, line {line})"
        )
        if final_count == 0 and not all_kinds:
            print(
                "ir-tracker: no match in final IR — debug locations may have been "
                "dropped, or try different --file/--line/--col.",
                file=sys.stderr,
            )

        q_first = (
            f"SELECT p.seq, p.pass_class, p.ir_unit, COUNT(*) AS n FROM {T_INSTR} i "
            f"JOIN {T_PASSES} p ON i.pass_id = p.id WHERE i.file_id IN ({in_clause}) "
            f"AND i.line = ?{kind_sql}{col_sql}{opc_sql} "
            f"GROUP BY p.id ORDER BY p.seq ASC LIMIT 1"
        )
        b_first: List[object] = [*file_ids, line, k]
        if trace_col is not None and trace_col >= 0:
            b_first.append(trace_col)
        if trace_opcode:
            b_first.append(trace_opcode)
        cur = con.execute(q_first, b_first)
        row = cur.fetchone()
        if row:
            pc = row["pass_class"] or ""
            iu = row["ir_unit"] or ""
            n = int(row["n"])
            print(
                f"First pass with any matching instruction: seq={int(row['seq'])} "
                f"{pc} on {iu} ({n} row(s))"
            )
        else:
            print("No pass recorded any matching instruction.")

    return 0


class ShowInstRow(NamedTuple):
    seq: int
    pass_class: str
    ir_unit: str
    kind: str
    func: str
    bb: str
    repr_line: int
    inst_text: str


def fingerprint_group(group: Sequence[ShowInstRow]) -> str:
    return "\n".join(r.inst_text for r in group) + "\n"


def print_instruction_group(
    group: Sequence[ShowInstRow], print_kind_in_title: bool
) -> None:
    if not group:
        return
    head = group[0]
    title = f"seq={head.seq} '{head.pass_class}' on '{head.ir_unit}'"
    if print_kind_in_title and head.kind:
        title += f" kind={head.kind}"
    print(title)
    prev_func = ""
    prev_bb = ""
    for r in group:
        if r.func != prev_func or r.bb != prev_bb:
            print(f"  function {r.func}, block {r.bb}:")
            prev_func = r.func
            prev_bb = r.bb
        print(f"    [{r.repr_line}] {r.inst_text}")


def run_show(
    con: sqlite3.Connection,
    file_pat: str,
    line_s: str,
    trace_col: Optional[int],
    trace_opcode: str,
    trace_kind: str,
    show_seq: int,
    show_all_passes: bool,
) -> int:
    if get_schema_version(con) < 5:
        print(
            "ir-tracker: 'show' requires ir-tracker schema_version >= 5 "
            "(database stores metadata only)",
            file=sys.stderr,
        )
        return 1
    if show_all_passes and show_seq >= 0:
        print(
            "ir-tracker: --all-passes and --seq are mutually exclusive",
            file=sys.stderr,
        )
        return 1

    try:
        all_kinds, single_kind = parse_trace_kind(trace_kind)
    except ValueError:
        print(
            "ir-tracker: invalid --kind (expected ir, mir, isa, assembly, asm, or all)",
            file=sys.stderr,
        )
        return 1

    file_ids = resolve_file_ids(con, file_pat)
    if not file_ids:
        print(
            f"ir-tracker: no {T_FILES} rows match --file "
            "(try a basename or substring)",
            file=sys.stderr,
        )
        return 1

    try:
        line = int(line_s, 0)
    except ValueError:
        print("ir-tracker: invalid --line", file=sys.stderr)
        return 1
    if line <= 0:
        print("ir-tracker: invalid --line", file=sys.stderr)
        return 1

    if show_seq < -1:
        print("ir-tracker: invalid --seq", file=sys.stderr)
        return 1

    single_pass_mode = show_seq >= 0
    all_passes_unfiltered = show_all_passes
    changed_only_default = not single_pass_mode and not all_passes_unfiltered

    if changed_only_default:
        ok = False
        if all_kinds:
            for kp in ("ir", "mir", "isa"):
                if get_max_seq_for_kind(con, kp) >= 0:
                    ok = True
        else:
            if get_max_seq_for_kind(con, single_kind) >= 0:
                ok = True
        if not ok:
            print(
                "ir-tracker: no rows recorded for the requested kind(s)",
                file=sys.stderr,
            )
            return 1

    in_clause = ",".join("?" * len(file_ids))
    kind_sql = "" if all_kinds else " AND i.kind = ?"
    seq_sql = " AND p.seq = ?" if single_pass_mode else ""
    col_sql = " AND i.col = ?" if trace_col is not None and trace_col >= 0 else ""
    opc_sql = " AND i.opcode = ?" if trace_opcode else ""

    q = (
        f"SELECT p.seq, p.pass_class, p.ir_unit, i.kind, i.function, "
        f"i.basicblock, i.inst_seq, i.repr_line, i.opcode, i.inst_text "
        f"FROM {T_INSTR} i JOIN {T_PASSES} p ON i.pass_id = p.id "
        f"WHERE i.file_id IN ({in_clause}) AND i.line = ?{kind_sql}{seq_sql}"
        f"{col_sql}{opc_sql} "
        f"ORDER BY p.seq ASC, i.kind ASC, i.function ASC, i.basicblock ASC, "
        f"i.inst_seq ASC"
    )

    params: List[object] = []
    params.extend(file_ids)
    params.append(line)
    if not all_kinds:
        params.append(single_kind)
    if single_pass_mode:
        params.append(show_seq)
    if trace_col is not None and trace_col >= 0:
        params.append(trace_col)
    if trace_opcode:
        params.append(trace_opcode)

    cur = con.execute(q, params)
    rows: List[ShowInstRow] = []
    for r in cur:
        rows.append(
            ShowInstRow(
                int(r["seq"]),
                r["pass_class"] or "",
                r["ir_unit"] or "",
                r["kind"] or "",
                r["function"] or "",
                r["basicblock"] or "",
                int(r["repr_line"]),
                r["inst_text"] or "",
            )
        )

    if not rows:
        if all_passes_unfiltered or changed_only_default:
            print("ir-tracker: no matching instructions found", file=sys.stderr)
        else:
            print(
                f"ir-tracker: no matching instructions found at seq={show_seq}",
                file=sys.stderr,
            )
        return 1

    if single_pass_mode or all_passes_unfiltered:
        i = 0
        while i < len(rows):
            j = i + 1
            while j < len(rows) and rows[j].seq == rows[i].seq and rows[j].kind == rows[i].kind:
                j += 1
            print_instruction_group(rows[i:j], all_kinds)
            i = j
        return 0

    last_fp_by_kind: Dict[str, str] = {}
    i = 0
    while i < len(rows):
        j = i + 1
        while j < len(rows) and rows[j].seq == rows[i].seq and rows[j].kind == rows[i].kind:
            j += 1
        group = rows[i:j]
        fp = fingerprint_group(group)
        k = group[0].kind
        if last_fp_by_kind.get(k) == fp:
            i = j
            continue
        last_fp_by_kind[k] = fp
        print_instruction_group(group, all_kinds)
        i = j

    return 0


def run_sql(con: sqlite3.Connection, sql: str) -> int:
    try:
        cur = con.execute(sql)
    except sqlite3.Error as e:
        print(f"ir-tracker: prepare(sql): {e}", file=sys.stderr)
        return 1
    while True:
        row = cur.fetchone()
        if row is None:
            break
        parts = []
        for c in range(len(row)):
            v = row[c]
            if v is None:
                parts.append("None")
            else:
                parts.append(str(v))
        print("(" + ", ".join(parts) + ")")
    return 0
