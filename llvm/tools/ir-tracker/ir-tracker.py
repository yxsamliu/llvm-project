#!/usr/bin/env python3
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Build and query SQLite databases for llvm/tools/ir-tracker."""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

import irtrackdb


def cmd_build(args: argparse.Namespace) -> int:
    return irtrackdb.build_db(args.input, args.db)


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
            con, args.file, args.line, args.col, args.opcode or ""
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
    parser = argparse.ArgumentParser(
        prog="ir-tracker",
        description="Build and query IR-tracker SQLite databases",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build", help="Build a SQLite DB from tracker text output")
    build.add_argument("--input", required=True)
    build.add_argument("--db", required=True)
    build.set_defaults(func=cmd_build)

    passes = sub.add_parser("passes", help="List recorded passes")
    passes.add_argument("--db", required=True)
    passes.set_defaults(func=cmd_passes)

    trace = sub.add_parser("trace", help="Find first/final pass for a source line")
    trace.add_argument("--db", required=True)
    trace.add_argument("--file", required=True)
    trace.add_argument("--line", required=True)
    trace.add_argument("--col", type=int, default=None)
    trace.add_argument("--opcode", default="")
    trace.set_defaults(func=cmd_trace)

    show = sub.add_parser("show", help="Show tracked instructions for a source line")
    show.add_argument("--db", required=True)
    show.add_argument("--file", required=True)
    show.add_argument("--line", required=True)
    show.add_argument("--col", type=int, default=None)
    show.add_argument("--opcode", default="")
    show.add_argument("--seq", type=int, default=-1)
    show.add_argument("--all-passes", action="store_true")
    show.set_defaults(func=cmd_show)

    sql = sub.add_parser("sql", help="Run a read-only SQL query")
    sql.add_argument("--db", required=True)
    sql.add_argument("query")
    sql.set_defaults(func=cmd_sql)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
