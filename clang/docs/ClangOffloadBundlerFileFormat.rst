Bundled Device Binaries
=======================

The ``clang-offload-bundler`` tool can be used to combine multiple device
binaries into a single bundled device binary file. The bundled device binary
entries are identified by a bundle entry ID which is defined by the
following EBNF syntax:

.. code::

  <bundle_entry_id>   ::= <offload_kind> "-" <target_triple> "-" <target_id>

Where:

**offload_kind**
  The runtime responsible for managing the loading of the code object.
  See :ref:`offload-kind-table`.

**target_triple**
  The target triple of the device binary.

**target_id**
  The target ID of the device binary (see `Target ID Definition
  <https://llvm.org/docs/AMDGPUUsage.html#target-ids>`_).

 .. table:: Bundled Device Binary Offload Kind
     :name: offload-kind-table

     ============= ==============================================================
     Offload Kind  Description
     ============= ==============================================================
     host          This offload kind is used for the first dummy empty entry
                   in the header of the bundle, which is required by
                   clang-offload-bundler, but is not used by language runtimes.

     hip           Device binary loading is managed by the HIP language runtime.

     openmp        Device binary loading is managed by the OpenMP language runtime.
     ============= ==============================================================

The format of a bundled device binary is defined by the following table:

  .. table:: Bundled Device Binary Memory Layout
     :name: bundled-device-binary-fields-table

     ========================= ======== ========================== ===============================
     Field                     Type     Size in Bytes              Description
     ========================= ======== ========================== ===============================
     Magic String              string   24                         ``__CLANG_OFFLOAD_BUNDLE__``

     Number Of Device Binaries integer  8                          Denoted as N in this table

     Entry Offset 1            integer  8                          Byte offset from beginning of
                                                                   bundled device binary to 1st device
                                                                   binary.

     Entry Size 1              integer  8                          Byte size of 1st code object.

     Entry ID Length 1         integer  8                          Bundle entry ID character length
                                                                   of 1st device binary

     Entry ID 1                string   Byte size of entry ID 1    Bundle entry ID of 1st device
                                                                   binary. This is not NUL
                                                                   terminated.

     ...

     Entry Offset N            integer  8

     Entry Size N              integer  8

     Entry ID Length N         integer  8

     Entry ID N                string   Byte size of entry ID N

     1st Device Binary         bytes    Size Of 1st Device Binary

     ...

     N-th Device Binary        bytes    Size Of N-th Device Binary
     ========================= ======== ========================== ==============================

The ``clang-offload-bundler`` is used to bundle device binaries for different processor
and feature settings.

The rules of compatible offload targets in a single bundled device binary is defined
in `AMDGPU Embedding Bundled Code Objects
  <https://llvm.org/docs/AMDGPUUsage.html#embedding-bundled-objects>`_.