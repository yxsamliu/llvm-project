/*===- InstrProfilingGPU.c - GPU profile counter functions ----------------===*\
|*
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
|* See https://llvm.org/LICENSE.txt for license information.
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
|*
\*===----------------------------------------------------------------------===*/

#if defined(__AMDGPU__) || defined(__NVPTX__)

#include <gpuintrin.h>
#include <stdint.h>

#define ATOMIC_ADD(ptr, val) \
  __scoped_atomic_fetch_add(ptr, val, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE)

/*
 * Check if this block is sampled (PatternOverflow mode).
 * Samples by matching lower bits of flat block ID to 0.
 *
 * sampling_bits: 0 = all blocks (100%)
 *                1 = even blocks (50%)
 *                2 = every 4th block (25%)
 *                3 = every 8th block (12.5%)
 */
__attribute__((visibility("hidden"), used))
int __gpu_pgo_is_sampled(uint32_t sampling_bits) {
  if (sampling_bits == 0)
    return 1;

  uint32_t gdx = __gpu_num_blocks_x();
  uint32_t gdy = __gpu_num_blocks_y();
  uint32_t block_id = __gpu_block_id_x() + __gpu_block_id_y() * gdx +
                       __gpu_block_id_z() * gdx * gdy;

  uint32_t mask = (1u << sampling_bits) - 1;
  return (block_id & mask) == 0;
}

typedef uint64_t __attribute__((address_space(1))) *global_u64_ptr;

/* Full wave mask: all lanes active */
#define FULL_WAVE_MASK \
  ((__gpu_num_lanes() == 64) ? ~0ULL : 0xFFFFFFFFULL)

/*
 * Per-BB warp-aggregate counter increment using atomic add.
 * Elects one leader lane per wave, counts active lanes, leader atomically
 * adds (step * active_lanes). Also updates uniform counter when all lanes
 * in the wave are active.
 */
__attribute__((visibility("hidden"), used))
void __gpu_pgo_increment(global_u64_ptr counter, global_u64_ptr uniform_counter,
                         int64_t step) {
  uint64_t lane_mask = __gpu_lane_mask();
  uint64_t active = __gpu_ballot(lane_mask, 1);
  if (__gpu_is_first_in_lane(lane_mask)) {
    int64_t count = (int64_t)__builtin_popcountg(active) * step;
    ATOMIC_ADD(counter, count);
    if (uniform_counter && active == FULL_WAVE_MASK)
      ATOMIC_ADD(uniform_counter, count);
  }
}

#if defined(__AMDGPU__)
__attribute__((weak)) const int __oclc_ABI_version = 600;
#endif

#endif /* __AMDGPU__ || __NVPTX__ */
