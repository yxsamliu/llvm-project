import sqlite3
import sys


def main() -> int:
    con = sqlite3.connect(sys.argv[1])
    cur = con.cursor()

    schema = cur.execute(
        "SELECT value FROM ir_tracker_meta WHERE key = 'schema_version'"
    ).fetchone()
    assert schema == ("1",), schema

    passes = cur.execute(
        "SELECT seq, phase FROM ir_tracker_passes ORDER BY seq"
    ).fetchall()
    assert passes, passes
    assert passes[0] == (0, "initial"), passes
    assert any(phase == "after" for _, phase in passes), passes

    expected = set(sys.argv[2:])
    functions = {
        row[0]
        for row in cur.execute(
            "SELECT DISTINCT function FROM ir_tracker_instructions ORDER BY function"
        )
    }
    assert functions == expected, (functions, expected)

    rows = cur.execute(
        "SELECT function, opcode FROM ir_tracker_instructions ORDER BY function, inst_seq"
    ).fetchall()
    for function in expected:
        assert any(row[0] == function for row in rows), rows

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
