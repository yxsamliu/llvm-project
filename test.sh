#DBG_LAMBDA=1 bin/clang++ -v -nogpuinc -nogpulib -fgpu-rdc -c lamba.hip 2>&1 | tee out.txt
bin/clang++ -nogpuinc -nogpulib -fgpu-rdc -c -v -save-temps lamba.hip
for a in lamba*.bc; do bin/llvm-dis $a; done
