====================================
IR Tracker (pass pipeline + SQLite)
====================================

.. contents::
   :local:

Overview
========

The **IR tracker** records a time-ordered view of the compiler pipeline into a
single **SQLite** database so you can ask, offline, how instructions tied to a
**source file / line** (via ``DILocation``) evolve across passes.

Typical uses:

* Find the **first pass** where an instruction appears at a given source line.
* Compare **printed instruction text** after each pass (default ``ir-tracker
  show`` mode surfaces only passes where that text **changed**).
* For the same database, inspect **LLVM IR**, **final MIR**, or **final textual
  ISA** using the ``kind`` column (when the producing pipeline recorded those
  representations).

Recording is enabled with the hidden LLVM flag **``-ir-tracker-database=PATH``**
(absolute paths recommended). **Clang** forwards it as **``-mllvm
-ir-tracker-database=…``**.

The companion **query** tool lives under **``llvm/tools/ir-tracker/``** (Python,
same install layout as :doc:`Remarks` discusses for **``opt-viewer``**): run
**``ir-tracker.py``** with subcommands ``build``, ``passes``, ``trace``,
``show``, and ``sql``. After ``ninja install-ir-tracker``, scripts are installed
under **``share/ir-tracker/``**. **Lit** uses the substitution **``%ir-tracker``**.

.. note::

   The feature is still evolving; behavior, defaults, and schema version may
   change between releases. The authoritative **schema** for a given build is
   whatever **``ir_tracker_meta.schema_version``** contains after recording.

Build requirements
===================

* **SQLite** must be available when **configuring** LLVM so that **``opt``** /
  **``clang``** (and other binaries that link the relevant libraries) are built
  with the tracker backend enabled.
* The database file is created (or reset) when a tool that enables the tracker
  starts a new recording session.

Enabling recording
===================

**``opt``** (new pass manager)
   Pass **``-ir-tracker-database=/absolute/path.db``** together with your usual
   **``-passes=…``** pipeline. The tracker hooks live in the shared pass
   instrumentation used by **``StandardInstrumentations``**.

**``clang`` / ``clang++``**
   Use **``-mllvm -ir-tracker-database=/absolute/path.db``**. For device or
   whole-program flows, combine with your usual offload and output flags.

**Plain ``.ll`` without ``!dbg``**
   Use **``--add-ir-tracker-locs``** or **``--add-ir-tracker-locs-force``** so
   ``opt`` can synthesize ``DILocation``\ s from **physical line numbers** in the
   ``.ll`` file (per function). See **``llvm/test/tools/opt/add-ir-tracker-locs.ll``**.

**Bitcode or unreadable inputs**
   The same flags may assign **synthetic ordinal** “lines” from a deterministic IR
   walk; those ordinals are **not** human source lines.

Optional tuning
----------------

**``-mllvm -ir-tracker-insert-batch=N``** (hidden, default ``1``) batches
multi-row ``INSERT``\ s into **``ir_tracker_instructions``**. Larger ``N`` can
reduce SQLite overhead on huge modules; re-measure for your workload.

Design (high level)
===================

**IR pipeline (``opt`` / Clang middle-end)**  
When the database path is set, **``IRTrackerState``** (in
``llvm/lib/Passes/StandardInstrumentations.cpp``) opens SQLite, applies the
schema, and registers **before/after pass** callbacks. After each relevant pass it:

* Inserts a row into **``ir_tracker_passes``** (monotonic ``seq``, pass name,
  IR unit).
* Walks instructions with non-null **``DILocation``**, resolves the **file**
  row in **``ir_tracker_files``**, and appends rows to **``ir_tracker_instructions``**
  with ``kind='ir'`` (and MIR/ISA rows when the producing pipeline attaches them
  through the same machinery).

The session uses a **single long transaction** (``BEGIN`` … ``COMMIT`` at
shutdown) to reduce WAL/fsync churn; secondary **indexes** on
``ir_tracker_instructions`` are created **after** ``COMMIT`` for the IR-only
recorder path.

**Machine pipeline / final ISA (Clang ``-S`` path, etc.)**  
**``llvm/lib/CodeGen/IRTrackerCodeGen.cpp``** attaches to codegen so **final MIR**
and **final assembly** text can be stored in the **same** database with
``kind='mir'`` and ``kind='isa'``, reusing the same table layout. Opening an
**existing** DB bumps ``ir_tracker_passes.seq`` past the IR pipeline so codegen
rows do not collide with IR ``seq`` values.

**Query CLI**  
**``llvm/tools/ir-tracker/ir-tracker.py``** opens the database **read-only** and
implements ``passes``, ``trace``, ``show`` (including change-only fingerprints per
``kind``), and raw ``sql``. The ``build`` subcommand is a thin wrapper around
``opt`` that injects **``-ir-tracker-database=``** before forwarded arguments.

Database format
===============

Metadata
--------

**``ir_tracker_meta``** — key/value pairs.

.. code-block:: sql

  CREATE TABLE ir_tracker_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
  );

The row **``('schema_version','5')``** marks the layout described here. Query
tools may require a minimum schema version for multi-kind ``trace``/``show``.

Files
-----

**``ir_tracker_files``** — deduplicated paths for ``DILocation`` file identity.

.. code-block:: sql

  CREATE TABLE ir_tracker_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE
  );

``path`` is typically **directory + filename** when debug info provides both.

Passes
------

**``ir_tracker_passes``** — one row per recorded pipeline step (IR pass, codegen
hook, synthetic ``<final-isa>`` marker, etc.).

.. code-block:: sql

  CREATE TABLE ir_tracker_passes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    seq INTEGER NOT NULL,
    phase TEXT NOT NULL,
    pass_class TEXT NOT NULL,
    ir_unit TEXT NOT NULL
  );

* ``seq`` — monotonic ordering within the database (``0`` is the initial IR
  snapshot when recorded).
* ``phase`` / ``pass_class`` / ``ir_unit`` — human-readable pass context (exact
  strings are tool-defined; treat them as opaque labels for diffing and display).

There is a **unique index** on ``seq`` (``ir_tracker_idx_passes_seq``).

Instructions
------------

**``ir_tracker_instructions``** — flattened instruction snapshots.

.. code-block:: sql

  CREATE TABLE ir_tracker_instructions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pass_id INTEGER NOT NULL REFERENCES ir_tracker_passes(id),
    function TEXT NOT NULL,
    basicblock TEXT NOT NULL,
    inst_seq INTEGER NOT NULL,
    repr_line INTEGER NOT NULL,
    kind TEXT NOT NULL,
    opcode TEXT NOT NULL,
    inst_text TEXT NOT NULL,
    file_id INTEGER NOT NULL REFERENCES ir_tracker_files(id),
    line INTEGER NOT NULL,
    col INTEGER NOT NULL
  );

* ``kind`` — ``'ir'``, ``'mir'``, or ``'isa'`` depending on how the row was
  captured (aliases **``assembly``** / **``asm``** are accepted only by the
  **query** tool for ``isa``).
* ``inst_text`` — stable printable form of the instruction at that pass (IR
  pretty-print, MIR print, or asm line text). Metadata suffixes may be stripped
  for fingerprint stability in the IR recorder.
* ``line`` / ``col`` / ``file_id`` — come from the ``DILocation`` (or synthetic
  location) used as the tracker **key**; they are what ``ir-tracker trace`` and
  ``show`` filter on.
* ``repr_line`` — line number used when **printing** the instruction in
  ``inst_text`` (useful when the textual form spans multiple physical lines).

Indexes (created after recording for the IR path)
--------------------------------------------------

.. code-block:: sql

  CREATE INDEX ir_tracker_idx_instr_file_loc
    ON ir_tracker_instructions(kind, file_id, line, col);
  CREATE INDEX ir_tracker_idx_instr_pass
    ON ir_tracker_instructions(pass_id);
  CREATE INDEX ir_tracker_idx_instr_repr
    ON ir_tracker_instructions(kind, pass_id, function, repr_line);

Query tool usage (summary)
==========================

Run **``ir-tracker.py --help``** and **``ir-tracker.py show --help``** for the
full flag list. Short examples:

.. code-block:: bash

  # Record (wraps opt):
  python3 llvm/tools/ir-tracker/ir-tracker.py build --db /tmp/p.db --opt /path/to/opt -- \\
    input.ll --add-ir-tracker-locs -o /dev/null -passes='default<O3>'

  # List passes:
  python3 llvm/tools/ir-tracker/ir-tracker.py passes --db /tmp/p.db

  # First / final pass info for a source line (schema v5; default --kind all):
  python3 llvm/tools/ir-tracker/ir-tracker.py trace --db /tmp/p.db \\
    --file input.ll --line 42

  # IR text for that line (change-only default; add --all-passes for everything):
  python3 llvm/tools/ir-tracker/ir-tracker.py show --db /tmp/p.db \\
    --file input.ll --line 42 --kind ir

Lit tests
=========

* **``llvm/test/tools/opt/add-ir-tracker-locs.ll``** — ``--add-ir-tracker-locs*``
  behavior.
* **``llvm/test/tools/llvm-ir-tracker/*.ll``** and **``help.test``** — query CLI
  and schema v5 ``trace``/``show`` output (**``%ir-tracker``**).

Limitations and caveats
=======================

* **Debug locations are not a stable unique ID** across SSA rewrites; ``trace``
  reports **candidates**, not a proof of def-use provenance.
* **``llc``** still has separate caveats for **``-ir-tracker-database``** and
  NewPM **``-filetype=obj``**; the validated **MIR/ISA-in-one-DB** story today is
  typically **Clang ``-S``** plus **``-mllvm -ir-tracker-database=…``**, not a
  documented **``llc``**-first workflow.
* **``-emit-llvm``** stops before codegen — expect **IR-only** rows unless you
  run a path that executes the machine pipeline with the same database.

See also
========

* :doc:`Remarks` — optimization **remarks** YAML and **``opt-viewer``** (closest
  analogue for “LLVM data file + Python viewer” documentation).
* :doc:`NewPassManager` — pass instrumentation model used by the IR tracker.
* :doc:`SourceLevelDebugging` — how ``DILocation`` and ``DIFile`` relate to real
  source.
* :doc:`HowToUpdateDebugInfo` — how transforms should preserve debug metadata.
