#ifndef LLVM_CODEGEN_WAVEWORKTRACE_H
#define LLVM_CODEGEN_WAVEWORKTRACE_H
#include "llvm/ADT/StringRef.h"
namespace llvm {
class MachineFunction;
void printWaveWorkMapping(const MachineFunction &MF, StringRef Phase);
}
#endif
