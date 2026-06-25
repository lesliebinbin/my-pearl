# Adapting Pearl mining to non-Hopper NVIDIA GPUs

This handoff is for an AI/code agent starting from the `development` branch. The goal is to make Pearl vLLM mining work on non-Hopper GPUs such as RTX 4090 (`sm_89`) and RTX 5090 (`sm_120`, CUDA/toolchain permitting), using the A5000 (`sm_86`) port as the reference approach.

## Why changes are needed

The original Pearl CUDA path is Hopper-specific. It assumes `sm_90a` and uses Hopper-only mechanisms:

- TMA tensor-map loads/stores and TMA descriptor prefetch.
- WGMMA/GMMA warpgroup matrix instructions.
- CTA clusters and warpgroup register allocation.
- Generated CUTLASS/CUTE kernels compiled only for `arch=compute_90a,code=sm_90a`.

On non-Hopper GPUs this fails during model startup/mining with errors like:

```text
Error: Failed to initialize the TMA descriptor 801
Trying to use TMA Descriptor Prefetch without CUTE_ARCH_TMA_SM90_ENABLED
void cutlass::arch::fence_barrier_init() not implemented
CUDA error: unspecified launch failure
```

For 4090/Ada and A5000/Ampere, the Hopper TMA/WGMMA path cannot run. For 5090/Blackwell, do not assume `sm_90a` kernels are valid either; build for the actual device architecture and avoid Hopper-`90a`-specific code unless verified.

## What needs to be modified

### 1. Build system: select GPU architecture and gate SM90 kernels

Modify `miner/pearl-gemm/setup.py` so the build does not always compile `sm_90a`.

Suggested behavior:

```python
CUDA_ARCH = os.getenv("PEARL_GEMM_CUDA_ARCH", "sm89").casefold()

if CUDA_ARCH in ("sm90", "sm90a", "90", "90a", "hopper"):
    COMPUTE_CAPABILITY = "arch=compute_90a,code=sm_90a"
    ENABLE_SM90_KERNELS = True
elif CUDA_ARCH in ("sm86", "86", "a5000", "ampere"):
    COMPUTE_CAPABILITY = "arch=compute_86,code=sm_86"
    ENABLE_SM90_KERNELS = False
elif CUDA_ARCH in ("sm89", "89", "4090", "ada"):
    COMPUTE_CAPABILITY = "arch=compute_89,code=sm_89"
    ENABLE_SM90_KERNELS = False
elif CUDA_ARCH in ("sm120", "120", "5090", "blackwell"):
    COMPUTE_CAPABILITY = "arch=compute_120,code=sm_120"
    ENABLE_SM90_KERNELS = False
else:
    raise ValueError("Unsupported PEARL_GEMM_CUDA_ARCH")
```

Then:

- Always compile a non-Hopper CUDA source, e.g. `csrc/gemm/pearl_gemm_non_hopper.cu`.
- Only generate/include SM90 GEMM/noising instantiation `.cu` files when `ENABLE_SM90_KERNELS=True`.
- Add a compile macro:

```python
feature_args.append(f"-DPEARL_GEMM_ENABLE_SM90={int(ENABLE_SM90_KERNELS)}")
```

Build examples:

```bash
# RTX A5000 / Ampere
PEARL_GEMM_CUDA_ARCH=sm86 uv pip install -e miner/pearl-gemm --no-build-isolation -v

# RTX 4090 / Ada
PEARL_GEMM_CUDA_ARCH=sm89 uv pip install -e miner/pearl-gemm --no-build-isolation -v

# RTX 5090 / Blackwell, requires CUDA/NVCC that supports sm_120
PEARL_GEMM_CUDA_ARCH=sm120 uv pip install -e miner/pearl-gemm --no-build-isolation -v
```

### 2. C++ API: route non-Hopper calls away from SM90 kernels

Modify `miner/pearl-gemm/csrc/gemm/pearl_gemm_api.cpp`.

Add a non-Hopper header:

```cpp
#include "pearl_gemm_non_hopper.h"
```

For `gemm()`:

```cpp
#if !PEARL_GEMM_ENABLE_SM90
  run_pearl_gemm_non_hopper(params, stream);
  return;
#else
  // existing SM90 dispatch
#endif
```

For `noisy_gemm()`:

```cpp
#if !PEARL_GEMM_ENABLE_SM90
  run_pearl_noisy_gemm_non_hopper(params, stream);
#else
  // existing SM90 dispatch
#endif
```

For `noise_gen()`:

```cpp
#if !PEARL_GEMM_ENABLE_SM90
  run_noise_generation_non_hopper(params, stream);
#else
  // existing run_noise_generation<R, NumThreads>()
#endif
```

Leave standalone `noise_A()` and `noise_B()` unsupported or implement them with the same non-Hopper kernels if tests require them. The vLLM mining path primarily needs `noise_gen()` and `noisy_gemm()`.

### 3. Add non-Hopper CUDA kernels

Create:

```text
miner/pearl-gemm/csrc/gemm/pearl_gemm_non_hopper.h
miner/pearl-gemm/csrc/gemm/pearl_gemm_non_hopper.cu
```

The A5000 implementation can be generalized because it uses ordinary CUDA C++ loops, not Ampere-specific instructions. It should work for `sm_86`, `sm_89`, and likely `sm_120` once compiled for that arch.

Required entry points:

```cpp
void run_pearl_gemm_non_hopper(PearlAPIParams const& params, cudaStream_t stream);
void run_pearl_noisy_gemm_non_hopper(PearlAPIParams const& params, cudaStream_t stream);
void run_noise_generation_non_hopper(Noise_gen_params const& params, cudaStream_t stream);
```

Implement these components:

1. **Vanilla GEMM**
   - Compute `C = A @ B.T`.
   - Inputs are int8; accumulation is int32; output is bf16/fp16 according to existing `C`.
   - Use `A_scales[row] * B_scales[col]`.
   - Correctness first; optimization can come later.

2. **Noise generation**
   - Replace Hopper TMA `noise_generation_kernel.h`.
   - Dense EAL/EBR:
     - Use keyed BLAKE3 chunks with seed strings `"A_tensor"` / `"B_tensor"`.
     - Map random bytes into `[-32, 32)`.
     - Also write fp16 EAL/EBR denoise tensors using `kEALScaleFactorDenoise` / `kEBRScaleFactorDenoise`.
   - Sparse EAR/EBL:
     - For each K index, derive two rank indices from hash words:
       - `r0 = u & (R - 1)`
       - `r1 = r0 ^ (1 + high32((R - 1) * u))`
     - Write `+1` and `-1` in both R-major and K-major layouts when requested.
   - Zero `aux_buffer` with `cudaMemsetAsync` if supplied.

3. **Noisy GEMM / denoising**
   - Compute `ApEA = A + EAL @ EAR.T`.
   - Compute `BpEB = B + EBR @ EBL.T`.
   - Compute denoise helpers:
     - `AxEBL = A @ EBL`
     - `EARxBpEB = BpEB @ EAR`
   - For inference output, the current correctness-first approach may compute vanilla `A @ B.T` directly because the denoised noisy path should equal it. If you need strict parity with the SM90 intermediate behavior, implement full:
     - `ApEAxBpEB - AxEB - EAxBpEB`
   - Avoid cross-kernel races: compute `BpEB` fully in one kernel, then launch a separate kernel for `EARxBpEB`.

4. **Mining jackpot scanner**
   - Mirror CPU reference logic in `zk-pow/src/ffi/mine.rs`.
   - Use the configured mining pattern:
     - rows: `[0, 8]` within each 128-row tile
     - columns: `[0, 1, 8, 9, ..., 248, 249]` within each 256-column tile
   - Accumulate jackpot tiles over chunks of `noise_rank`.
   - Rotate/XOR into the 16-word jackpot ring using the same `LROT_PER_TILE`.
   - Hash jackpot with `pow_key`.
   - Compare little-endian `uint32[8]` hash against `pow_target`.
   - On success, write `HostSignalHeader` and set `HostSignalStatus::kSignalTriggered`.

5. **Host signal header**
   - Make `extract_indices(header)` return exactly the selected proof rows and columns.
   - For the default dense mining config, expected output under tile `(0, 0)` is:
     - rows: `[0, 8]`
     - columns: 64 values from `0, 1, 8, 9, ... 248, 249`

### 4. Tensor hash fallback

The vLLM mining path calls:

```python
tensor_hash(A.to(torch.uint8), key_tensor, A_tensor_hash, scratchpad)
tensor_hash(B.to(torch.uint8), key_tensor, B_tensor_hash, scratchpad)
commitment_hash_from_merkle_roots(...)
```

The original CUDA `tensor_hash()` uses TMA Merkle-tree kernels and crashes on non-Hopper GPUs. Until a non-Hopper CUDA Merkle implementation is written, use the repo reference implementation on CPU for non-Hopper devices:

```python
from miner_base.matrix_merkle_tree import MatrixMerkleTree
from blake3 import blake3

def _use_non_hopper_hash_fallback(tensor: torch.Tensor) -> bool:
    return tensor.is_cuda and torch.cuda.get_device_capability(tensor.device)[0] < 9

def tensor_hash(...):
    if _use_non_hopper_hash_fallback(data):
        digest = MatrixMerkleTree.tensor_hash(data.detach().contiguous().cpu(), key_bytes)
        copy_digest_to_cuda(digest, out)
        return None
    return pearl_gemm_cuda.tensor_hash(...)
```

For `commitment_hash_from_merkle_roots()`, the CPU reference is:

```python
B_commitment = blake3(key + B_merkle_root).digest()
A_commitment = blake3(B_commitment + A_merkle_root).digest()
```

If MoE routing roots are present, preserve the existing routing hash folding:

```python
routing_hash = blake3(routing_root + offsets_hash).digest()
A_root = blake3(A_merkle_root + routing_hash).digest()
A_commitment = blake3(B_commitment + A_root).digest()
```

This CPU fallback is slower but removes the Hopper TMA dependency and is sufficient to get serving/mining startup working.

### 5. vLLM capability gate

Update the dense Pearl vLLM kernel gate so non-Hopper GPUs are allowed.

In `miner/vllm-miner/src/vllm_miner/vllm_kernels.py`:

```python
@classmethod
def get_min_capability(cls) -> int:
    return 8
```

Then allow mining mode to call `_apply_weights_mining()` on non-Hopper once `noisy_gemm()` works:

```python
if self.mining_enabled:
    return self._apply_weights_mining(...)
```

Keep MoE separate. MoE may still have independent Hopper-only kernels and should remain gated until ported.

## Suggested implementation sequence

Start from a clean branch:

```bash
git checkout development
git checkout -b support-non-hopper-mining
```

Recommended order:

1. **Build-system split**
   - Add `PEARL_GEMM_CUDA_ARCH`.
   - Add `PEARL_GEMM_ENABLE_SM90`.
   - Build for the target GPU arch.

2. **Vanilla GEMM**
   - Add non-Hopper `gemm()` path.
   - Run:
     ```bash
     uv run python -m pytest miner/pearl-gemm/tests/test_pearl_gemm.py::TestGEMM::test_noiseless_int7_gemm -q
     ```

3. **Noise generation**
   - Add non-Hopper `noise_gen()`.
   - Run:
     ```bash
     uv run python -m pytest miner/pearl-gemm/tests/test_noise_gen.py -q
     ```

4. **Tensor hash fallback**
   - Add CPU fallback for non-Hopper.
   - Run focused parity tests:
     ```bash
     uv run python -m pytest miner/pearl-gemm/tests/test_tensor_hash.py::TestTensorHash::test_tensor_hash_shapes -q -x
     ```

5. **Noisy GEMM**
   - Add non-Hopper `noisy_gemm()`.
   - Run a small slice first:
     ```bash
     uv run python -m pytest 'miner/pearl-gemm/tests/test_pearl_gemm.py::TestNoisyGEMM::test_int7_noisy_gemm' -q -x -k '128 and 256 and fp16'
     ```

6. **Mining trigger smoke**
   - Use an artificially easy target: `(1 << 256) - 1`.
   - Confirm:
     - `HostSignalStatus.kSignalTriggered`
     - extracted rows `[0, 8]`
     - 64 extracted columns in bounds

7. **vLLM startup**
   - Start `pearld` and `pearl-gateway`.
   - Run:
     ```bash
     uv run vllm serve pearl-ai/Llama-3.1-8B-Instruct-pearl \
       --port 8002 \
       --max-model-len 8192 \
       --gpu-memory-utilization 0.9 \
       --enforce-eager
     ```
   - Confirm:
     ```text
     Application startup complete.
     ```
   - Check:
     ```bash
     curl -sS http://127.0.0.1:8002/v1/models
     ```

## Validation expectations

Minimum successful validation for a 4090/5090 port:

- Extension builds for target arch:
  - `sm89` for RTX 4090.
  - `sm120` for RTX 5090, if NVCC/CUDA supports it.
- No `TMA descriptor`, `CUTE_ARCH_TMA_SM90_ENABLED`, or `fence_barrier_init` errors.
- Vanilla GEMM tests pass.
- `noise_gen` tests pass.
- Tensor hash parity passes, at least on representative shapes.
- Noisy GEMM small slice passes.
- Easy-target mining smoke triggers a host signal.
- `vllm serve` reaches `Application startup complete`.

## Important caveats

- The correctness-first non-Hopper GEMM/noisy kernels are not optimized. They can be slow, especially the CPU tensor-hash fallback.
- A successful easy-target mining trigger proves the path can produce a candidate proof, not that mainnet rewards are guaranteed.
- Real rewards require a live block hash below the real network target and successful gateway/`pearld` submission.
- If 5090 uses a CUDA toolkit that does not recognize `sm_120`, update CUDA/NVCC first or compile with the closest supported Blackwell target recommended by NVIDIA.
