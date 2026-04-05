set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  AggressiveInstCombine
  Analysis
  AsmParser
  BitWriter
  CFGuard
  CodeGen
  Core
  Coroutines
  Extensions
  IPO
  IRReader
  IRPrinter
  InstCombine
  Instrumentation
  MC
  ObjCARCOpts
  Plugins
  Remarks
  ScalarOpts
  Support
  Target
  TargetParser
  TransformUtils
  Vectorize
  Passes
  )

# We don't want to link this into libLLVM
add_llvm_library(LLVMOptDriver
  STATIC
  NewPMDriver.cpp
  optdriver.cpp
  DebugLocIndex.cpp
  LLSourceDebugLoc.cpp
  PARTIAL_SOURCES_INTENDED
  DEPENDS
  intrinsics_gen
)

# Optional: index debug locations to SQLite (-debug-loc-index-database).
find_package(SQLite3 QUIET)
if(SQLite3_FOUND)
  set(DEBUG_LOC_INDEX_SQLITE_LIB SQLite::SQLite3)
  set(DEBUG_LOC_INDEX_HAVE 1)
else()
  find_path(LLVM_SQLITE3_INCLUDE_DIR sqlite3.h)
  find_library(LLVM_SQLITE3_LIBRARY sqlite3)
  if(LLVM_SQLITE3_INCLUDE_DIR AND LLVM_SQLITE3_LIBRARY)
    set(DEBUG_LOC_INDEX_SQLITE_LIB "${LLVM_SQLITE3_LIBRARY}")
    set(DEBUG_LOC_INDEX_HAVE 1)
  endif()
endif()
if(DEBUG_LOC_INDEX_HAVE)
  target_compile_definitions(LLVMOptDriver PRIVATE LLVM_ENABLE_DEBUG_LOC_INDEX=1)
  if(LLVM_SQLITE3_INCLUDE_DIR)
    target_include_directories(LLVMOptDriver PRIVATE "${LLVM_SQLITE3_INCLUDE_DIR}")
  endif()
  target_link_libraries(LLVMOptDriver PRIVATE ${DEBUG_LOC_INDEX_SQLITE_LIB})
endif()

add_llvm_tool(opt
  PARTIAL_SOURCES_INTENDED
  opt.cpp
  DEPENDS
  intrinsics_gen
  SUPPORT_PLUGINS

  EXPORT_SYMBOLS
  )
target_link_libraries(opt PRIVATE LLVMOptDriver)

setup_host_tool(opt OPT opt_exe opt_target)
