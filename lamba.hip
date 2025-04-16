#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __host__ __attribute__((host))

typedef unsigned long size_t;

struct dim3 {
  unsigned x, y, z;
  __host__ __device__ dim3(unsigned x, unsigned y = 1, unsigned z = 1) : x(x), y(y), z(z) {}
};

typedef struct hipStream *hipStream_t;
typedef enum hipError {} hipError_t;
int hipConfigureCall(dim3 gridSize, dim3 blockSize, size_t sharedSize = 0,
                     hipStream_t stream = 0);

extern "C" hipError_t __hipPushCallConfiguration(dim3 gridSize, dim3 blockSize,
                                                 size_t sharedSize = 0,
                                                 hipStream_t stream = 0);

extern "C" hipError_t hipLaunchKernel(const void *func, dim3 gridDim,
                                      dim3 blockDim, void **args,
                                      size_t sharedMem,
                                      hipStream_t stream);
extern "C" hipError_t __hipPopCallConfiguration(dim3 *gridSize, dim3 *blockSize,
                                                 size_t *sharedSize,
                                                 hipStream_t *stream);
__device__ float foo(float x, float y) {
  return x;
}

float foo(const float &x, const float &y) {
  return x;
}

template<typename F>
static void __global__
kernel(F f)
{
  f(1);
}

void test(float const * fl, float const * A, float * Vf)
{
  float constexpr small(1.0e-25);

  auto f = [=] __device__ __host__ (unsigned int n) {
    float const value = foo(small, fl[0]);
    Vf[0] = value * A[0];
  };
  static_assert(sizeof(f) == sizeof(fl) + sizeof(A) + sizeof(Vf));
  kernel<<<1, 1>>>(f);
}
