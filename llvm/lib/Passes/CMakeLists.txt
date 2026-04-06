add_llvm_component_library(LLVMPasses
  CodeGenPassBuilder.cpp
  OptimizationLevel.cpp
  PassBuilder.cpp
  PassBuilderBindings.cpp
  PassBuilderPipelines.cpp
  StandardInstrumentations.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Passes

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  AggressiveInstCombine
  Analysis
  CFGuard
  CodeGen
  GlobalISel
  Core
  Coroutines
  HipStdPar
  IPO
  InstCombine
  IRPrinter
  ObjCARC
  Scalar
  Support
  Target
  TransformUtils
  Vectorize
  Instrumentation
  )

# Optional: index tracked IR locations to SQLite (-debug-loc-index-database).
find_package(SQLite3 QUIET)
if(SQLite3_FOUND)
  set(IR_TRACKER_SQLITE_LIB SQLite::SQLite3)
  set(IR_TRACKER_HAVE_SQLITE 1)
else()
  find_path(LLVM_SQLITE3_INCLUDE_DIR sqlite3.h)
  find_library(LLVM_SQLITE3_LIBRARY sqlite3)
  if(LLVM_SQLITE3_INCLUDE_DIR AND LLVM_SQLITE3_LIBRARY)
    set(IR_TRACKER_SQLITE_LIB "${LLVM_SQLITE3_LIBRARY}")
    set(IR_TRACKER_HAVE_SQLITE 1)
  endif()
endif()
if(IR_TRACKER_HAVE_SQLITE)
  target_compile_definitions(LLVMPasses PRIVATE LLVM_ENABLE_IR_TRACKER_SQLITE=1)
  if(LLVM_SQLITE3_INCLUDE_DIR)
    target_include_directories(LLVMPasses PRIVATE "${LLVM_SQLITE3_INCLUDE_DIR}")
  endif()
  target_link_libraries(LLVMPasses PRIVATE ${IR_TRACKER_SQLITE_LIB})
endif()
