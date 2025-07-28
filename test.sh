#cd /home/yaxunl/git/llvm2
cd /c/git/llvm2

#ninja -C assert install || exit -1
PATH=/c/git/llvm2/install/bin:$PATH
export ROCM_PATH=/c/hipsdk/hip.cpl.547
gfx=gfx1102

export DB_PROF=1

#assert/bin/clang++ -O2 --hip-path=/home/yaxunl/git/clr/Release/install -fprofile-instr-generate --offload-arch=gfx1100 -save-temps -v -x hip tmp_rovodev_hip_pgo_comprehensive_test.hip -o tmp_rovodev_hip_pgo_test_direct 2>&1 | tee out.txt
#assert/bin/clang++ -O2 --hip-path=/home/yaxunl/git/clr/Debug/install -fprofile-instr-generate --offload-arch=gfx1100 -save-temps -v -x hip tmp_rovodev_hip_pgo_comprehensive_test.hip -o tmp_rovodev_hip_pgo_test_direct 2>&1 | tee out.txt
#assert/bin/clang++ -O2 --hip-path=/home/yaxunl/git/clr/Debug/install --offload-arch=gfx1100 -save-temps -x hip tmp_rovodev_hip_pgo_comprehensive_test.hip -o tmp_rovodev_hip_pgo_test_direct
#cp tmp_rovodev_hip_pgo_comprehensive_test-hip-amdgcn-amd-amdhsa-gfx1100.s tmp_rovodev_hip_pgo_comprehensive_test-hip-amdgcn-amd-amdhsa-gfx1100_orig.s

#clang++ -Wno-unused-value -O2 --hip-path=/home/yaxunl/git/clr/Debug/install -fprofile-instr-generate=my.profraw --offload-arch=gfx1100 -save-temps -x hip tmp_rovodev_hip_pgo_comprehensive_test.hip -o tmp_rovodev_hip_pgo_test_direct

clang++ -mllvm -debug-only=pgo-instrumentation -Wno-unused-value -O2 -fprofile-generate=my.profraw --offload-arch=$gfx -save-temps -x hip tmp_rovodev_hip_pgo_comprehensive_test.hip -o tmp_rovodev_hip_pgo_test_direct

#readelf -sW a.out-hip-amdgcn-amd-amdhsa-gfx1100 | grep __llvm_offload_prf
#readelf -sW tmp_rovodev_hip_pgo_test_direct | grep __llvm_offload_prf
#ls tmp_rovodev_hip_pgo_comprehensive_test*
#grep -C10 hipRegisterVar tmp_rovodev_hip_pgo_comprehensive_test-host-x86_64-unknown-linux-gnu.s

#export LD_LIBRARY_PATH=/home/yaxunl/git/clr/Release/install/lib
export LD_LIBRARY_PATH=/home/yaxunl/git/clr/Debug/install/lib
#ldd tmp_rovodev_hip_pgo_test_direct
#AMD_LOG_LEVEL=3 ./tmp_rovodev_hip_pgo_test_direct
./tmp_rovodev_hip_pgo_test_direct
#AMD_LOG_LEVEL=3 /opt/rocm/bin/rocgdb ./tmp_rovodev_hip_pgo_test_direct

ls *.profraw
llvm-profdata show my.amdgcn-amd-amdhsa.profraw --text --all-functions
llvm-profdata merge -o my.profdata my.profraw
llvm-profdata merge -o my.amdgcn-amd-amdhsa.profdata my.amdgcn-amd-amdhsa.profraw

#assert/bin/clang++ -Wno-unused-value -O2 --hip-path=/home/yaxunl/git/clr/Debug/install -fprofile-instr-use=my.profdata --offload-arch=gfx1100 -save-temps -x hip tmp_rovodev_hip_pgo_comprehensive_test.hip -o tmp_rovodev_hip_pgo_test_direct
clang++ -mllvm -debug-only=pgo-instrumentation -Wno-unused-value -O2 -fprofile-use=my.profdata --offload-arch=$gfx -save-temps -x hip tmp_rovodev_hip_pgo_comprehensive_test.hip -o tmp_rovodev_hip_pgo_test_direct

#diff tmp_rovodev_hip_pgo_comprehensive_test-hip-amdgcn-amd-amdhsa-gfx1100_orig.s tmp_rovodev_hip_pgo_comprehensive_test-hip-amdgcn-amd-amdhsa-gfx1100.s

#./tmp_rovodev_hip_pgo_test_direct
