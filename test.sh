DBG_LAMBDA=1 bin/clang++ -v -nogpuinc -nogpulib -fgpu-rdc -c lamba.hip 2>&1 | tee out.txt
