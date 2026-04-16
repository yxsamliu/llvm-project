IR Tracker (IR JSONL + SQLite query DB)
=======================================

.. contents::
   :local:

Overview
========

The **IR tracker** records how LLVM IR evolves through the new pass manager into a
compact JSON Lines file. A small Python tool can then post-process that JSON
output into a
SQLite database for indexed queries. Each instruction row is tied to a debug
location (``file``, ``line``, ``column`` from ``DILocation``), so you can ask
which passes touched code that originated at a given source line.

Recording is enabled with the hidden LLVM option
``-ir-tracker-json-output=/absolute/path.jsonl``. The hooks live in
``StandardInstrumentations`` and therefore apply to any tool that runs the new
pass manager with that instrumentation (``opt``, ``clang``, etc.).

Only instructions that carry a **non-zero** debug location are indexed.

Recording with ``opt``
======================

.. code-block:: bash

  opt -disable-output -passes='default<O2>' \
    -ir-tracker-json-output=/tmp/pipeline.jsonl input.ll

Use an absolute output path. The input IR must already contain suitable
``!dbg`` attachments (for example, compile the source with ``-g`` and use
``clang -emit-llvm -S`` to produce ``input.ll``).

Recording with ``clang``
========================

Forward the option through Clang with ``-mllvm`` so the middle-end sees the same
flag as ``opt``:

.. code-block:: bash

  clang -O1 -emit-llvm -S -g sum.c -o sum.ll \
    -mllvm -ir-tracker-json-output=/tmp/pipeline.jsonl

Here ``-g`` ensures debug locations exist on instructions; ``-O1`` (or another
``-O`` level) selects the usual optimization pipeline that ``opt`` would run for
that tier.

SQLite build step
=================

The Python driver can convert the JSONL output into a SQLite database:

.. code-block:: bash

  python3 llvm/tools/ir-tracker/ir-tracker.py build \
    --input /tmp/pipeline.jsonl --db /tmp/pipeline.db

The resulting database uses ``schema_version = 1`` in ``ir_tracker_meta``. The
main tables are:

* ``ir_tracker_meta`` — key/value metadata (including ``schema_version``)
* ``ir_tracker_files`` — deduplicated paths from ``DIFile`` (often a basename
  such as ``sum.c``)
* ``ir_tracker_passes`` — one row per snapshot: ``seq``, ``phase`` (``initial``
  or ``after``), ``pass_class``, ``ir_unit``
* ``ir_tracker_instructions`` — instruction text and opcode per pass, keyed by
  ``file_id``, ``line``, ``col``

Query tool
==========

The Python driver lives at ``llvm/tools/ir-tracker/ir-tracker.py`` (installed
under ``<prefix>/share/ir-tracker/`` when the ``ir-tracker`` install component
is enabled). It can build the SQLite DB from tracker JSONL output and then query
that DB. Subcommands:

* ``build`` — convert tracker JSONL output into a SQLite database
* ``passes`` — list recorded passes in ``seq`` order
* ``trace`` — summarize the first and last pass that still have instructions
  matching a source location
* ``show`` — print the instructions matching ``--file`` / ``--line`` (and
  optional ``--col`` / ``--opcode``) across passes; by default only passes where
  the printed IR **changed** are shown; use ``--all-passes`` for every pass, or
  ``--seq N`` for one pass
* ``sql`` — run a single read-only SQL statement

The ``--file`` argument is matched against the path stored in
``ir_tracker_files`` (substring match, case-insensitive). Clang usually records
the ``DIFile`` basename, so prefer ``--file sum.c`` rather than a full host path.

Example: following one source line through ``clang -O1``
========================================================

Source file ``sum.c``:

.. code-block:: c

  /* Example: trivial fold (x + 0) -> x */
  int bump(int x) {
    return x + 0;
  }

Recording (same command as in *Recording with ``clang``*):

.. code-block:: bash

  clang -O1 -emit-llvm -S -g sum.c -o sum.ll \
    -mllvm -ir-tracker-json-output=/tmp/pipeline.jsonl

Then build the query database:

.. code-block:: bash

  python3 llvm/tools/ir-tracker/ir-tracker.py build \
    --input /tmp/pipeline.jsonl --db /tmp/pipeline.db

The following excerpts come from a real ``ir-tracker`` run against the database
produced that way. **Pass names and sequence numbers depend on your Clang/LLVM
version, target, and optimization level**; treat pass sequence numbers as
illustrative, not a stable ABI.

List passes (truncated):

.. code-block:: text

      0  id=1       initial  '<initial>'  on '[module]'
      1  id=2       after  'memprof-remove-attributes'  on '[module]'
      2  id=3       after  'annotation2metadata'  on '[module]'
      …
     10  id=11      after  'sroa'  on 'bump'
     11  id=12      after  'early-cse'  on 'bump'
     …

Trace line ``3`` (the ``return x + 0;`` line in ``sum.c``):

.. code-block:: bash

  python3 llvm/tools/ir-tracker/ir-tracker.py trace \
    --db /tmp/pipeline.db --file sum.c --line 3

.. code-block:: text

  Matches at final pass (seq=94): 1 instruction(s)
  First pass with any matching instruction: seq=0 <initial> on [module] (3 row(s))

``show`` without ``--all-passes`` prints only passes where the matched IR text
changed: here the load/add/return cluster simplifies until ``early-cse`` folds
``x + 0`` to ``x``:

.. code-block:: bash

  python3 llvm/tools/ir-tracker/ir-tracker.py show \
    --db /tmp/pipeline.db --file sum.c --line 3

.. code-block:: text

  seq=0 '<initial>' on '[module]'
    function bump, block entry:
        %0 = load i32, ptr %x.addr, align 4
        %add = add nsw i32 %0, 0
        ret i32 %add
  seq=10 'sroa' on 'bump'
    function bump, block entry:
        %add = add nsw i32 %x, 0
        ret i32 %add
  seq=11 'early-cse' on 'bump'
    function bump, block entry:
        ret i32 %x

The initial snapshot for the same line (``--seq 0``) recovers the unoptimized
cluster before any pass runs:

.. code-block:: bash

  python3 llvm/tools/ir-tracker/ir-tracker.py show \
    --db /tmp/pipeline.db --file sum.c --line 3 --seq 0

.. code-block:: text

  seq=0 '<initial>' on '[module]'
    function bump, block entry:
        %0 = load i32, ptr %x.addr, align 4
        %add = add nsw i32 %0, 0
        ret i32 %add

Tests
=====

* Recorder: ``llvm/test/Other/ir-tracker-db.ll``
* Query tool: ``llvm/test/tools/llvm-ir-tracker/``

Limitations
===========

* **IR only** — there is no MIR, object, or assembly capture in this schema.
* **Debug info required** — instructions without a non-zero ``!dbg`` location are
  not recorded. There is no built-in mode yet to fabricate locations for plain
  ``.ll`` or ``.bc`` without debug metadata.
* **Locations are keys, not proofs** — optimizations can merge, clone, or drop
  instructions; the database lists what survived each pass with a given
  location, not a formal def-use proof.
