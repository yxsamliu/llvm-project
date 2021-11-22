// RUN: %clang_cc1 -emit-llvm -o - -triple=i386-pc-win32 -std=c++11 %s -fcxx-exceptions -fms-extensions | FileCheck %s

// CHECK-DAG: @"_CT??_R0?AUTemplateWithDefault@@@8??$?_OH@TemplateWithDefault@@QAEXAAU0@@Z1" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 0, i8* bitcast (%rtti.TypeDescriptor25* @"??_R0?AUTemplateWithDefault@@@8" to i8*), i32 0, i32 -1, i32 0, i32 1, i8* bitcast (void (%struct.TemplateWithDefault*, %struct.TemplateWithDefault*)* @"??$?_OH@TemplateWithDefault@@QAEXAAU0@@Z" to i8*) }, section ".xdata", comdat

struct TemplateWithDefault {
  template <typename T>
  static int f() {
    return 0;
  }
  template <typename T = int>
  TemplateWithDefault(TemplateWithDefault &, T = f<T>());
};

void j(TemplateWithDefault &twd) {
  throw twd;
}
