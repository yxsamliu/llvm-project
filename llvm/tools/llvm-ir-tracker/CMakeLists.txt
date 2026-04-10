set(LLVM_LINK_COMPONENTS Support)

# Mirror llvm/lib/Passes/CMakeLists.txt so query subcommands match opt's SQLite.
set(IR_TRACKER_TOOL_SQLITE 0)
find_package(SQLite3 QUIET)
if(SQLite3_FOUND)
  set(IR_TRACKER_SQLITE_LIB SQLite::SQLite3)
  set(IR_TRACKER_TOOL_SQLITE 1)
else()
  find_path(LLVM_SQLITE3_INCLUDE_DIR sqlite3.h)
  find_library(LLVM_SQLITE3_LIBRARY sqlite3)
  if(LLVM_SQLITE3_INCLUDE_DIR AND LLVM_SQLITE3_LIBRARY)
    set(IR_TRACKER_SQLITE_LIB "${LLVM_SQLITE3_LIBRARY}")
    set(IR_TRACKER_TOOL_SQLITE 1)
  endif()
endif()

add_llvm_tool(llvm-ir-tracker
  llvm-ir-tracker.cpp
  )

if(IR_TRACKER_TOOL_SQLITE)
  target_compile_definitions(llvm-ir-tracker PRIVATE LLVM_IR_TRACKER_TOOL_SQLITE=1)
  if(NOT SQLite3_FOUND AND LLVM_SQLITE3_INCLUDE_DIR)
    target_include_directories(llvm-ir-tracker PRIVATE "${LLVM_SQLITE3_INCLUDE_DIR}")
  endif()
  target_link_libraries(llvm-ir-tracker PRIVATE ${IR_TRACKER_SQLITE_LIB})
endif()
