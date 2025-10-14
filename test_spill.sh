#LLVM_DBG_SPILL=1 /home/yaxunl/git/llvm2/Release/bin/llc /home/yaxunl/git/llvm2/llvm/test/CodeGen/AArch64/arm64-spill-remarks.ll -mtriple=arm64-apple-ios7.0 -aarch64-neon-syntax=apple

LLVM_DBG_SPILL=1 "/home/yaxunl/git/llvm2/Release/bin/llc" "/home/yaxunl/git/llvm2/llvm/test/CodeGen/AArch64/hot-evicts-cold-asm.ll" -mtriple=arm64-apple-ios7.0 -aarch64-neon-syntax=apple
