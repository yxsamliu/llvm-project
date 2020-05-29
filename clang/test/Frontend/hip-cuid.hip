// RUN: not %clang_cc1 -cuid=abc-123 -offload-arch=gfx906 %s 2>&1 \
// RUN:   | FileCheck --check-prefix=INVALID %s

// INVALID: invalid value 'abc-123' in '-cuid=abc-123' (alphanumeric characters and underscore only)

// RUN: %clang_cc1 -cuid=abc_123 -offload-arch=gfx906 %s
