cd /home/yaxunl/git/llvm2
#ninja -C assert clang compiler-rt &&
assert/bin/clang++ -O2 --hip-path=/home/yaxunl/git/clr/Release/install -fprofile-instr-generate --offload-arch=gfx1100 -save-temps -v -x hip tmp_rovodev_hip_pgo_comprehensive_test.hip -o tmp_rovodev_hip_pgo_test_direct 2>&1 | tee out.txt
#readelf -sW a.out-hip-amdgcn-amd-amdhsa-gfx1100 | grep __llvm_offload_prf
#readelf -sW tmp_rovodev_hip_pgo_test_direct | grep __llvm_offload_prf
#ls tmp_rovodev_hip_pgo_comprehensive_test*
#grep -C10 hipRegisterVar tmp_rovodev_hip_pgo_comprehensive_test-host-x86_64-unknown-linux-gnu.s
export LD_LIBRARY_PATH=/home/yaxunl/git/clr/Release/install/lib
#ldd tmp_rovodev_hip_pgo_test_direct
#AMD_LOG_LEVEL=3 ./tmp_rovodev_hip_pgo_test_direct
AMD_LOG_LEVEL=3 /opt/rocm/bin/rocgdb ./tmp_rovodev_hip_pgo_test_direct
