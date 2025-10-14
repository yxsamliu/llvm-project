#LLVM_DBG_SPILL=1 /home/yaxunl/git/llvm2/Release/bin/llc /home/yaxunl/git/llvm2/llvm/test/CodeGen/AArch64/arm64-spill-remarks.ll -mtriple=arm64-apple-ios7.0 -aarch64-neon-syntax=apple

export LLVM_DBG_SPILL=1
PATH=/home/yaxunl/git/llvm2/Release/bin:$PATH

#llc /home/yaxunl/git/llvm2/llvm/test/CodeGen/AArch64/hot-evicts-cold-asm.ll" -o - -mtriple=arm64-apple-ios7.0 -aarch64-neon-syntax=apple

llc /home/yaxunl/git/llvm2/llvm/test/CodeGen/X86/hot-evicts-cold-asm.ll -o - -mtriple=x86_64-unknown-linux-gnu

