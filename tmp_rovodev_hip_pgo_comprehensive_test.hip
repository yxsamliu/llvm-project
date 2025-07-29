#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

extern "C" __device__ int xxx = 123;

// Multiple kernels with different execution patterns to generate diverse profile data
__global__ void simple_kernel(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx * 2;
    }
}

#if 0
__global__ void branching_kernel(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Branch-heavy code to generate interesting profile data
        if (idx % 2 == 0) {
            data[idx] = idx * 3;
        } else if (idx % 3 == 0) {
            data[idx] = idx * 5;
        } else {
            data[idx] = idx;
        }
    }
}

__global__ void loop_kernel(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int sum = 0;
        // Loop with varying iteration counts
        for (int i = 0; i < (idx % 10) + 1; i++) {
            sum += i * idx;
        }
        data[idx] = sum;
    }
}
#endif
int main() {
    //#if 0
    std::cout << "=== HIP PGO Comprehensive Test ===" << std::endl;
    
    const int N = 2048;
    std::vector<int> h_data(N);
    int* d_data;
    
    // Initialize host data
    for (int i = 0; i < N; i++) {
        h_data[i] = i;
    }
    
    // Allocate device memory
    hipError_t err = hipMalloc(&d_data, N * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory: " << hipGetErrorString(err) << std::endl;
        return 1;
    }
    
    // Copy data to device
    err = hipMemcpy(d_data, h_data.data(), N * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy data to device: " << hipGetErrorString(err) << std::endl;
        hipFree(d_data);
        return 1;
    }
    
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::cout << "Launching kernels to generate profile data..." << std::endl;
    
    // Launch multiple kernels to generate diverse profile data
    hipLaunchKernelGGL(simple_kernel, grid, block, 0, 0, d_data, N);
    hipDeviceSynchronize();
    
    //hipLaunchKernelGGL(branching_kernel, grid, block, 0, 0, d_data, N);
    //hipDeviceSynchronize();
    
    //hipLaunchKernelGGL(loop_kernel, grid, block, 0, 0, d_data, N);
    //hipDeviceSynchronize();
    
    // Launch kernels multiple times to create hot paths
    for (int iter = 0; iter < 5; iter++) {
        hipLaunchKernelGGL(simple_kernel, grid, block, 0, 0, d_data, N);
        if (iter % 2 == 0) {
            //hipLaunchKernelGGL(branching_kernel, grid, block, 0, 0, d_data, N);
        }
    }
    hipDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Copy result back
    err = hipMemcpy(h_data.data(), d_data, N * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy data from device: " << hipGetErrorString(err) << std::endl;
        hipFree(d_data);
        return 1;
    }
    
    std::cout << "Kernel execution completed in " << duration.count() << " microseconds" << std::endl;
    
    // Verify results
    std::cout << "Sample results: ";
    for (int i = 0; i < 10; i++) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;
    
    // Cleanup
    hipFree(d_data);
    
    std::cout << "Test completed successfully. Profile data should be generated." << std::endl;
    //#endif
    return 0;
}