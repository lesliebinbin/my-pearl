#include "pearl_gemm_sm86.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "blake3/blake3.cuh"
#include "host_signal_header.hpp"

namespace pearl_sm86_detail {

static constexpr int kSm86HashTileM = 2;
static constexpr int kSm86HashTileN = 64;
static constexpr int kSm86TileM = 128;
static constexpr int kSm86TileN = 256;
static constexpr int kJackpotSize = 16;
static constexpr int kLrotPerTile = 13;

__device__ __forceinline__ uint32_t rotl32(uint32_t value, int shift) {
  return (value << shift) | (value >> (32 - shift));
}

__device__ bool hash_le_target(uint32_t const* hash, uint32_t const* target) {
  for (int i = blake3::CHAINING_VALUE_SIZE_U32 - 1; i >= 0; --i) {
    if (hash[i] > target[i]) {
      return false;
    }
    if (hash[i] < target[i]) {
      return true;
    }
  }
  return true;
}

__device__ void compute_jackpot_hash_sm86(uint32_t const* jackpot,
                                          uint32_t const* key,
                                          uint32_t* hash) {
  auto block = cute::make_tensor<uint32_t>(cute::Int<blake3::MSG_BLOCK_SIZE_U32>{});
  auto chaining_value =
      cute::make_tensor<uint32_t>(cute::Int<blake3::CHAINING_VALUE_SIZE_U32>{});

  for (int i = 0; i < blake3::MSG_BLOCK_SIZE_U32; ++i) {
    block(i) = jackpot[i];
  }
  for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
    chaining_value(i) = key[i];
  }

  blake3::compress_msg_block_u32(block, chaining_value,
                                 blake3::COMPRESS_PARAMS_SINGLE_BLOCK_KEYED);

  for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
    hash[i] = chaining_value(i);
  }
}

__device__ void write_host_signal_header_sm86(
    HostSignalSync* host_signal_sync, HostSignalHeader* host_signal_header_pinned,
    PearlAPIParams const params, int tile_m, int tile_n,
    uint32_t const* pow_target) {
  while (atomicCAS(&host_signal_sync->global_lock, 0, 1) != 0) {
    __threadfence();
  }

  if (host_signal_sync->status != HostSignalStatus::kSignalTriggered) {
    HostSignalHeader new_header = {
        .status = HostSignalStatus::kSignalTriggered,
        .gridDim = {gridDim.x, gridDim.y, gridDim.z},
        .blockDim = {blockDim.x, blockDim.y, blockDim.z},
        .blockIdx = {blockIdx.x, blockIdx.y, blockIdx.z},
        .tileCoord = {static_cast<uint32_t>(tile_m),
                      static_cast<uint32_t>(tile_n), 0},
        .threadIdx = {threadIdx.x, threadIdx.y, threadIdx.z},
        .num_registers_per_thread =
            static_cast<uint16_t>(kSm86HashTileM + kSm86HashTileN),
        .mma_size = {params.m, params.n, params.k},
        .mma_tile_size = {kSm86TileM, kSm86TileN, params.r},
        .target = {},
    };

    for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
      new_header.target[i] = pow_target[i];
    }
    new_header.thread_rows[0] = 0;
    new_header.thread_rows[1] = 8;
    for (int i = 0; i < kSm86HashTileN; ++i) {
      new_header.thread_rows[kSm86HashTileM + i] = 0;
      int group = i / 8;
      int pair = i % 8;
      new_header.thread_cols[kSm86HashTileM + i] =
          static_cast<uint8_t>(group * 32 + pair / 2 * 8 + pair % 2);
    }
    new_header.thread_cols[0] = 0;
    new_header.thread_cols[1] = 0;

    if (new_header.block_in_bounds()) {
      *host_signal_header_pinned = new_header;
      host_signal_sync->status = HostSignalStatus::kSignalTriggered;
    }
  }

  __threadfence();
  atomicExch(&host_signal_sync->global_lock, 0);
}

__global__ void pearl_gemm_sm86_kernel(
    int8_t const* __restrict__ A, int8_t const* __restrict__ B,
    float const* __restrict__ A_scales, float const* __restrict__ B_scales,
    __nv_bfloat16* __restrict__ C, int m, int n, int k) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row >= m || col >= n) {
    return;
  }

  int32_t acc = 0;
  int a_offset = row * k;
  int b_offset = col * k;
  for (int kk = 0; kk < k; ++kk) {
    acc += static_cast<int32_t>(A[a_offset + kk]) *
           static_cast<int32_t>(B[b_offset + kk]);
  }

  float scaled = static_cast<float>(acc) * A_scales[row] * B_scales[col];
  C[row * n + col] = __float2bfloat16(scaled);
}

__global__ void sm86_noise_a_kernel(PearlAPIParams params) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = params.m * params.k;
  auto A = static_cast<int8_t const*>(params.ptr_A);
  auto EAL = static_cast<int8_t const*>(params.ptr_EAL);
  auto EAR = static_cast<int8_t const*>(params.ptr_EAR_R_major);
  auto EBL = static_cast<int8_t const*>(params.ptr_EBL_R_major);
  auto ApEA = static_cast<int8_t*>(params.ptr_ApEA);
  auto AxEBL = static_cast<__half*>(params.ptr_AxEBL_mma);

  for (int linear = idx; linear < total; linear += blockDim.x * gridDim.x) {
    int row = linear / params.k;
    int col = linear % params.k;
    int noise = 0;
    for (int rr = 0; rr < params.r; ++rr) {
      noise += static_cast<int>(EAL[row * params.r + rr]) *
               static_cast<int>(EAR[col * params.r + rr]);
    }
    ApEA[linear] = static_cast<int8_t>(static_cast<int>(A[linear]) + noise);
  }

  int mr_total = params.m * params.r;
  for (int linear = idx; linear < mr_total; linear += blockDim.x * gridDim.x) {
    int row = linear / params.r;
    int rr = linear % params.r;
    int acc = 0;
    for (int kk = 0; kk < params.k; ++kk) {
      acc += static_cast<int>(A[row * params.k + kk]) *
             static_cast<int>(EBL[kk * params.r + rr]);
    }
    AxEBL[linear] = __float2half(static_cast<float>(acc) * 0x1p-14f);
  }
}

__global__ void sm86_noise_b_kernel(PearlAPIParams params) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  auto B = static_cast<int8_t const*>(params.ptr_B);
  auto EBR = static_cast<int8_t const*>(params.ptr_EBR);
  auto EBL = static_cast<int8_t const*>(params.ptr_EBL_R_major);
  auto BpEB = static_cast<int8_t*>(params.ptr_BpEB);

  int total = params.n * params.k;
  for (int linear = idx; linear < total; linear += blockDim.x * gridDim.x) {
    int row = linear / params.k;
    int col = linear % params.k;
    int noise = 0;
    for (int rr = 0; rr < params.r; ++rr) {
      noise += static_cast<int>(EBR[row * params.r + rr]) *
               static_cast<int>(EBL[col * params.r + rr]);
    }
    BpEB[linear] = static_cast<int8_t>(static_cast<int>(B[linear]) + noise);
  }
}

__global__ void sm86_ear_x_bpeb_kernel(PearlAPIParams params) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  auto BpEB = static_cast<int8_t const*>(params.ptr_BpEB);
  auto EAR = static_cast<int8_t const*>(params.ptr_EAR_R_major);
  auto EARxBpEB = static_cast<__half*>(params.ptr_EARxBpEB_mma);
  int nr_total = params.n * params.r;
  for (int linear = idx; linear < nr_total; linear += blockDim.x * gridDim.x) {
    int row = linear / params.r;
    int rr = linear % params.r;
    int acc = 0;
    for (int kk = 0; kk < params.k; ++kk) {
      acc += static_cast<int>(BpEB[row * params.k + kk]) *
             static_cast<int>(EAR[kk * params.r + rr]);
    }
    EARxBpEB[linear] = __float2half(static_cast<float>(acc) * 0x1p-12f);
  }
}

__global__ void sm86_mining_kernel(PearlAPIParams params) {
  int tile_n = blockIdx.x;
  int tile_m = blockIdx.y;
  int row_base = tile_m * kSm86TileM;
  int col_base = tile_n * kSm86TileN;

  int rows[kSm86HashTileM] = {row_base, row_base + 8};
  int cols[kSm86HashTileN];
  for (int i = 0; i < kSm86HashTileN; ++i) {
    int group = i / 8;
    int pair = i % 8;
    cols[i] = col_base + group * 32 + pair / 2 * 8 + pair % 2;
  }

  if (rows[kSm86HashTileM - 1] >= params.m ||
      cols[kSm86HashTileN - 1] >= params.n) {
    return;
  }

  auto ApEA = static_cast<int8_t const*>(params.ptr_ApEA);
  auto BpEB = static_cast<int8_t const*>(params.ptr_BpEB);
  auto pow_target = static_cast<uint32_t const*>(params.ptr_pow_target);
  auto pow_key = static_cast<uint32_t const*>(params.ptr_pow_key);
  auto host_signal_sync =
      static_cast<HostSignalSync*>(params.host_signal_sync);
  auto host_signal_header =
      static_cast<HostSignalHeader*>(params.host_signal_header_pinned);

  int32_t jackpot_tile[kSm86HashTileM][kSm86HashTileN] = {};
  uint32_t jackpot[kJackpotSize] = {};

  for (int ll = params.r; ll <= params.k; ll += params.r) {
    for (int u = 0; u < kSm86HashTileM; ++u) {
      for (int v = 0; v < kSm86HashTileN; ++v) {
        int32_t chunk_acc = 0;
        for (int kk = ll - params.r; kk < ll; ++kk) {
          chunk_acc += static_cast<int32_t>(ApEA[rows[u] * params.k + kk]) *
                       static_cast<int32_t>(BpEB[cols[v] * params.k + kk]);
        }
        jackpot_tile[u][v] += chunk_acc;
      }
    }

    uint32_t xored_tile = 0;
    for (int u = 0; u < kSm86HashTileM; ++u) {
      for (int v = 0; v < kSm86HashTileN; ++v) {
        xored_tile ^= static_cast<uint32_t>(jackpot_tile[u][v]);
      }
    }
    int tid = (ll / params.r - 1) % kJackpotSize;
    jackpot[tid] = rotl32(jackpot[tid], kLrotPerTile) ^ xored_tile;
  }

  uint32_t hash[blake3::CHAINING_VALUE_SIZE_U32];
  compute_jackpot_hash_sm86(jackpot, pow_key, hash);
  if (hash_le_target(hash, pow_target)) {
    write_host_signal_header_sm86(host_signal_sync, host_signal_header, params,
                                  tile_m, tile_n, pow_target);
  }
}

}  // namespace pearl_sm86_detail

void run_pearl_gemm_sm86(PearlAPIParams const& params, cudaStream_t stream) {
  dim3 block(16, 16);
  dim3 grid((params.n + block.x - 1) / block.x,
            (params.m + block.y - 1) / block.y);

  pearl_sm86_detail::pearl_gemm_sm86_kernel<<<grid, block, 0, stream>>>(
      static_cast<int8_t const*>(params.ptr_ApEA),
      static_cast<int8_t const*>(params.ptr_BpEB),
      static_cast<float const*>(params.ptr_A_scales),
      static_cast<float const*>(params.ptr_B_scales),
      static_cast<__nv_bfloat16*>(params.ptr_C), params.m, params.n,
      params.k);
}

void run_pearl_noisy_gemm_sm86(PearlAPIParams const& params,
                               cudaStream_t stream) {
  int threads = 256;
  int blocks = 256;
  pearl_sm86_detail::sm86_noise_a_kernel<<<blocks, threads, 0, stream>>>(params);
  pearl_sm86_detail::sm86_noise_b_kernel<<<blocks, threads, 0, stream>>>(params);
  pearl_sm86_detail::sm86_ear_x_bpeb_kernel<<<blocks, threads, 0, stream>>>(
      params);

  PearlAPIParams output_params = params;
  output_params.ptr_ApEA = params.ptr_A;
  output_params.ptr_BpEB = params.ptr_B;
  run_pearl_gemm_sm86(output_params, stream);

  dim3 mining_grid(
      (params.n + pearl_sm86_detail::kSm86TileN - 1) /
          pearl_sm86_detail::kSm86TileN,
      (params.m + pearl_sm86_detail::kSm86TileM - 1) /
          pearl_sm86_detail::kSm86TileM);
  pearl_sm86_detail::sm86_mining_kernel<<<mining_grid, 1, 0, stream>>>(params);
}
