#!/usr/bin/env python3
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""IR tracker — query SQLite databases produced by -ir-tracker-database (opt/clang).

Layout and install follow llvm/tools/opt-viewer/. This replaces the former
llvm-ir-tracker C++ driver; recording remains in LLVM libraries.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import List, Optional, Sequence

import irtrackdb


def make_absolute_db_path(db: str) -> str:
    if os.path.isabs(db):
        return db
    return os.path.abspath(os.path.join(os.getcwd(), db))


def default_opt_executable() -> Optional[str]:
    w = shutil.which("opt")
    if w:
        return w
    script = os.path.abspath(__file__)
    cand = os.path.join(os.path.dirname(script), "..", "..", "..", "bin", "opt")
    cand = os.path.normpath(cand)  # llvm/tools/ir-tracker -> llvm-project/bin/opt
    if os.path.isfile(cand):
        return cand
    return None


def cmd_build(args: argparse.Namespace) -> int:
    opt_exe = args.opt or default_opt_executable()
    if not opt_exe or not os.path.isfile(opt_exe):
        print(
            "ir-tracker: opt not found (use --opt=PATH or ensure `opt` is on PATH)",
            file=sys.stderr,
        )
        return 2

    db_abs = make_absolute_db_path(args.db)
    argv: List[str] = [opt_exe, f"-ir-tracker-database={db_abs}"]
    for a in args.opt_args:
        if len(argv) == 2 and a == "--":
            continue
        argv.append(a)
    r = subprocess.run(argv)
    return int(r.returncode)


def cmd_passes(args: argparse.Namespace) -> int:
    con = irtrackdb.open_db_readonly(args.db)
    if not con:
        return 1
    try:
        return irtrackdb.run_passes(con)
    finally:
        con.close()


def cmd_trace(args: argparse.Namespace) -> int:
    con = irtrackdb.open_db_readonly(args.db)
    if not con:
        return 1
    try:
        return irtrackdb.run_trace(
            con,
            args.file,
            args.line,
            args.col,
            args.opcode or "",
            args.kind,
        )
    finally:
        con.close()


def cmd_show(args: argparse.Namespace) -> int:
    con = irtrackdb.open_db_readonly(args.db)
    if not con:
        return 1
    try:
        return irtrackdb.run_show(
            con,
            args.file,
            args.line,
            args.col,
            args.opcode or "",
            args.kind,
            args.seq,
            args.all_passes,
        )
    finally:
        con.close()


def cmd_sql(args: argparse.Namespace) -> int:
    con = irtrackdb.open_db_readonly(args.db)
    if not con:
        return 1
    try:
        return irtrackdb.run_sql(con, args.query)
    finally:
        con.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(
        prog="ir-tracker",
        description="LLVM IR tracker (SQLite) — query databases from -ir-tracker-database",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser(
        "build", help="Run opt with -ir-tracker-database=<db> (forward args after --)"
    )
    pb.add_argument(
        "--opt",
        default=None,
        help="Path to opt (default: `opt` on PATH, else ../bin/opt near this script)",
    )
    pb.add_argument("--db", required=True, help="SQLite database path")
    pb.add_argument(
        "opt_args",
        nargs=argparse.REMAINDER,
        help="Arguments for opt (use -- before flags if needed)",
    )
    pb.set_defaults(func=cmd_build)

    pp = sub.add_parser("passes", help="List recorded passes")
    pp.add_argument("--db", required=True)
    pp.set_defaults(func=cmd_passes)

    pt = sub.add_parser(
        "trace",
        help="Find first pass with instructions matching a source line",
    )
    pt.add_argument("--db", required=True)
    pt.add_argument(
        "--file",
        required=True,
        help="Substring or basename to match file path in ir_tracker_files",
    )
    pt.add_argument("--line", required=True, help="1-based source line")
    pt.add_argument(
        "--col",
        type=int,
        default=None,
        help="Optional source column",
    )
    pt.add_argument("--opcode", default="", help="Optional opcode filter")
    pt.add_argument(
        "--kind",
        default="all",
        help="ir | mir | isa | assembly | asm | all (default: all)",
    )
    pt.set_defaults(func=cmd_trace)

    ps = sub.add_parser(
        "show",
        help="Show tracked instructions matching a source line",
    )
    ps.add_argument("--db", required=True)
    ps.add_argument("--file", required=True)
    ps.add_argument("--line", required=True)
    ps.add_argument("--col", type=int, default=None)
    ps.add_argument("--opcode", default="")
    ps.add_argument("--kind", default="all")
    ps.add_argument(
        "--seq",
        type=int,
        default=-1,
        help="Show a single pass by sequence number (0=initial). "
        "Default: only passes where instruction text changed",
    )
    ps.add_argument(
        "--all-passes",
        action="store_true",
        help="List every pass with matches (including unchanged text)",
    )
    ps.set_defaults(func=cmd_show)

    pq = sub.add_parser("sql", help="Run a read-only SQL query")
    pq.add_argument("--db", required=True)
    pq.add_argument("query", help="SQL statement")
    pq.set_defaults(func=cmd_sql)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
