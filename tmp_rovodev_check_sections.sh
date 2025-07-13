#!/bin/bash

echo "=== Checking Profile Sections in Device Binary ==="

# Find the device binary (usually has .out extension)
#DEVICE_BINARY=$(find . -name "*.out" -type f | head -1)
DEVICE_BINARY=$(ls *.out | head -1)

if [ -z "$DEVICE_BINARY" ]; then
    echo "No device binary (.out file) found"
    exit 1
fi

echo "Examining device binary: $DEVICE_BINARY"
echo

echo "=== Section Headers (objdump) ==="
objdump -h "$DEVICE_BINARY" | grep -E "(llvm_prf|Idx|Name)"
echo

echo "=== Profile Sections Details (readelf) ==="
readelf -S "$DEVICE_BINARY" | grep -A1 -B1 "llvm_prf"
echo

echo "=== Profile Section Symbols (nm) ==="
nm "$DEVICE_BINARY" | grep -E "(llvm_prf|__start__|__stop__)" | sort
echo

echo "=== Profile Section Symbols with Sizes (readelf) ==="
readelf -sW "$DEVICE_BINARY" | grep -E "(llvm_prf|__start__|__stop__)"
echo

echo "=== Hexdump of __llvm_offload_prf structure ==="
# Get the address and size of __llvm_offload_prf
OFFLOAD_PRF_INFO=$(readelf -sW "$DEVICE_BINARY" | grep "__llvm_offload_prf" | awk '{print $2, $3}')
if [ -n "$OFFLOAD_PRF_INFO" ]; then
    ADDR=$(echo $OFFLOAD_PRF_INFO | cut -d' ' -f1)
    SIZE=$(echo $OFFLOAD_PRF_INFO | cut -d' ' -f2)
    echo "Address: 0x$ADDR, Size: $SIZE bytes"
    
    # Convert hex address to decimal for hexdump
    DECIMAL_ADDR=$((0x$ADDR))
    echo "Hexdump of __llvm_offload_prf structure:"
    hexdump -C "$DEVICE_BINARY" -s $DECIMAL_ADDR -n $SIZE
else
    echo "__llvm_offload_prf symbol not found"
fi
