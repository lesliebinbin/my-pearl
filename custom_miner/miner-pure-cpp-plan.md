# Native/Pure C++ Miner Plan

## Goal

Build a distributable miner for non-Hopper GPUs, especially RTX 5090D / Blackwell `sm_120`, that can run on rented cloud machines with minimal Python/source exposure. The binary should preserve current `custom_miner/standalone_miner.py` behavior: connect to Pearl Gateway, receive jobs, generate synthetic matrices, compute commitments, run noisy GEMM mining, detect hits, create proof shares, and submit.

## Current state

- `custom_miner/standalone_miner.py` works on 5090D.
- `pearl-gemm` builds for `sm_120`.
- Non-Hopper tensor hashing in the Python `pearl_gemm_interface.py` uses CPU fallback for non-Hopper devices, including `sm_120`, to avoid Hopper TMA kernels.
- A new native scaffold exists at `custom_miners_cpp/`.
- The native binary is `custom_miners_cpp/builddir/pearl-custom-miner-cpp`.
- Gateway remains external and exposes mining jobs over `/tmp/pearlgw.sock`.
- The C++ binary talks to the Python Gateway via newline-delimited JSON-RPC; it does not submit directly to mainnet.

## What has been implemented

### Build/runtime fixes

- Fixed non-Hopper `task build` by guarding SM90 static-switch includes in `miner/pearl-gemm/csrc/gemm/static_switch.h`.
- Added arch selection in `miner/pearl-gemm/setup.py` for `sm86`, `sm89`, `sm90a`, and `sm120` / 5090D.
- Changed `miner/pearl-gemm/src/pearl_gemm/pearl_gemm_interface.py` so non-Hopper devices use CPU Merkle/hash fallback instead of Hopper TMA tensor hash kernels.

### Native C++ miner scaffold

Created `custom_miners_cpp/` with:

- Meson build.
- CUDA runtime linking.
- Default CUDA target `sm_120`.
- Config parser for `custom_miner/mining-config.yml`.
- UDS/TCP JSON-RPC client compatible with `pearl-gateway`.
- `getMiningInfo` polling.
- Native `Uint256` target adjustment.
- CUDA matrix allocation and matrix generation.
- Duplicated/self-contained CUDA kernels for:
  - noise generation,
  - noised A/B construction,
  - GEMM,
  - jackpot/PoW scan.
- Native BLAKE3 helper for small-message hashing and Merkle CVs.
- Native dense matrix Merkle proof generation.
- Bincode-compatible dense `PlainProof` serialization.
- Base64 proof encoding.
- `submitPlainProof` RPC call on hits.

### Validation performed

Build and one-attempt smoke test:

```bash
cd custom_miners_cpp
meson setup builddir --wipe
meson compile -C builddir
./builddir/pearl-custom-miner-cpp --config ../custom_miner/mining-config.yml --poll-once
```

Proof serialization parser test:

```bash
./builddir/pearl-custom-miner-cpp \
  --config ../custom_miner/mining-config.yml \
  --proof-self-test /tmp/pearl_cpp_proof.b64

cd ..
uv run python - <<'PY'
from pathlib import Path
from pearl_mining import PlainProof
proof = PlainProof.from_base64(Path('/tmp/pearl_cpp_proof.b64').read_text().strip())
print(proof.m, proof.n, proof.k, proof.noise_rank, len(proof.a.row_indices), len(proof.bt.row_indices), proof.min_cert_version)
PY
```

Observed parser result:

```text
16 256 4096 128 2 64 1
```

This confirms the native dense proof blob can be parsed by the Rust-backed `pearl_mining.PlainProof.from_base64`.

## Current limitations / risk

- No real hit was observed during smoke tests, so the full live path `hit -> submitPlainProof -> Gateway accepts -> node submits block -> reward` is wired but not end-to-end proven.
- Native proof generation currently targets the default dense mining config from `custom_miner/mining-config.yml`:
  - `k=4096`
  - `rank=128`
  - rows pattern `[0, 8]`
  - cols pattern `[0, 1, 8, 9, ..., 248, 249]`
- MoE proof support is not implemented.
- General `MiningConfiguration` serialization is not implemented beyond the current dense default.
- Native BLAKE3/Merkle code should be tested more thoroughly against `MatrixMerkleTree` roots and multiproofs for large real matrices.
- The C++ GEMM/noising kernels are correctness-first duplicated non-Hopper kernels, not performance-equivalent to optimized Hopper/CUTLASS kernels.

## Recommended migration path

### Phase 1: Define the native boundary

Status: mostly done in `custom_miners_cpp`.

Keep Gateway/RPC/proof submission in Python Gateway, but move the sensitive mining attempt loop into a compiled native binary.

Native API should expose something like:

```cpp
MiningAttemptResult run_attempt(
    MiningJob job,
    MinerConfig config,
    DeviceBuffers buffers
);
```

Responsibilities inside native code:

- Matrix generation.
- Tensor root / commitment hash calculation.
- Noise generation.
- Non-Hopper noisy GEMM.
- PoW target scan.
- Host signal extraction.
- Return hit/no-hit plus proof indices and commitment hashes.

Python remains only in the external Gateway process. The C++ miner does not require importing Python packages.

### Phase 2: Replace CPU hash fallback

Status: partially done in native binary with CPU-side native Merkle roots; CUDA Merkle root generation remains future work.

Implement non-Hopper CUDA tensor hash / commitment hash so 5090D no longer copies full matrices to CPU.

Files to study:

- `miner/pearl-gemm/src/pearl_gemm/pearl_gemm_interface.py`
- `miner/pearl-gemm/csrc/tensor_hash/tensor_hash.cu`
- `miner/miner-base/src/miner_base/matrix_merkle_tree.py`
- `adapt-to-non-hopper-architect.md`

Target behavior:

- Match `MatrixMerkleTree.tensor_hash(...)` exactly.
- Support CUDA tensors on `sm_86`, `sm_89`, `sm_120`.
- Avoid Hopper TMA/CUTE SM90-only paths.

### Phase 3: Native mining worker

Status: implemented as `custom_miners_cpp/`.

Created native package/binary:

```text
custom_miners_cpp/
  meson.build
  meson_options.txt
  src/
    main.cpp
    config.cpp
    json_rpc_client.cpp
    native_miner.cpp
    cuda_kernels.cu
    mining_kernels.cu
    plain_proof.cpp
```

Design choice: duplicate needed CUDA/C++ code into `custom_miners_cpp` instead of linking against the Python extension or another subproject, so the folder can be copied to another machine and compiled independently.

### Phase 4: Python-free CLI

Status: implemented for dense mining against external Python Gateway.

```bash
./builddir/pearl-custom-miner-cpp \
  --config ../custom_miner/mining-config.yml
```

Native binary responsibilities:

- Parse config.
- Connect to Gateway.
- Poll jobs.
- Run mining attempts.
- Submit proof shares.
- Log stats.

### Phase 5: Release hardening

For cloud deployment:

```bash
strip pearl-custom-miner
```

Build flags:

- Disable debug symbols.
- Avoid embedding source paths.
- Build per CUDA/toolchain version.
- Prefer static or bundled dependencies where practical.
- Keep wallet/RPC secrets in environment variables or external config, not compiled into binary.

Example release artifact:

```text
pearl-custom-miner-linux-x86_64-cuda13-sm120.tar.gz
```

Include only:

```text
pearl-custom-miner
mining-config.example.yml
README-run.md
```

## Security note

A binary protects against casual source inspection, but not against a malicious cloud provider. They can still observe process memory, filesystem, network traffic, GPU usage, and reverse-engineer the binary. Do not embed private keys, wallet seeds, API secrets, or proprietary constants directly in the executable.

## Suggested first implementation task

Next high-value tasks:

1. Run an end-to-end forced/easy-target hit test on simnet or a controlled Gateway to prove reward path submission.
2. Add automated proof round-trip tests comparing native Merkle roots/proofs with Python `MatrixMerkleTree`.
3. Generalize `MiningConfiguration` serialization from the current default dense config.
4. Move native matrix Merkle root generation from CPU to CUDA for performance.
5. Add a release packaging script that produces a stripped `sm_120` binary plus example config.
