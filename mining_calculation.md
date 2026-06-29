# Pearl Mining Calculation Notes

This note summarizes how the current standalone miner prototype relates a local
matrix seed, gateway-provided block work, and the Pearl Proof-of-Useful-Work
calculation.

Reference: Pearl whitepaper, especially:

- Section 3.1.1, "Low-Rank Noise Paradigm"
- Section 4.2, "MatMul Framework"
- Section 4.3, "Commitment Hash"
- Section 4.5, "Tiled MatMul Algorithm"

Whitepaper URL: <https://pearlresearch.ai/>

## Key idea

The gateway/mainnet does not directly provide matrices `A` and `B`, nor does it
provide the synthetic matrix seed. The gateway provides the current block work:

```text
H = incomplete_header_bytes
T = target
cert_version
```

The miner supplies or generates matrix operands:

```text
A   shape [m, k]
B_t shape [n, k]
```

`B_t` is the transposed storage used by the existing Pearl CUDA kernel. The
logical multiplication is:

```text
C = A @ B_t.T
```

The local `matrix.seed` is only used to generate synthetic matrices when there is
no external useful workload. It is not the Pearl cryptographic noise seed.

## Calculation flow

1. Generate synthetic matrices from `matrix.seed`.

```text
matrix.seed
  -> torch.Generator
  -> A   [physical_m, k]
  -> B_t [physical_n, k]
```

For the current integer path, values should be signed 7-bit compatible, e.g.
`[-64, 63]`.

The standalone prototype supports two matrix modes:

```text
materialized:
  matrix.m / matrix.n are the full GEMM dimensions.

tiled:
  matrix.logical_m / matrix.logical_n describe an Alpha-style logical profile.
  matrix.m / matrix.n are the smaller physical tile dimensions allocated and
  mined by each attempt.
```

In tiled mode the current proof path submits the physical tile as the committed
matrix. The logical profile is used for configuration/logging and to avoid
mistaking Alpha's `131072 x 131072` profile for one full output allocation.

2. Get a mining job from the gateway.

```text
H = incomplete_header_bytes
T = target
```

3. Build the mining configuration.

```text
mu = MiningConfiguration(
  common_dim=k,
  rank=r,
  mma_type=Int7xInt7ToInt32,
  rows_pattern=...,
  cols_pattern=...,
)
```

`mining.h` and `mining.w` are the row/column pattern sizes exposed by
`MiningConfiguration.hash_tile_h` and `hash_tile_w`. The Alpha-style defaults
used by the prototype are:

```text
h = 2
w = 64
matmul_tile = 128 x 256 x 128
rows_pattern = [0, 8]
cols_pattern = [0, 1, 8, 9, ..., 248, 249]
```

4. Bind the matrix work to the current block header.

```text
key = BLAKE3(H || mu.to_bytes())
```

5. Commit to the matrices.

```text
A_root = tensor_hash(A, key)
B_root = tensor_hash(B_t, key)
```

`B_t` is committed directly because proof opening uses rows of `B_t`, which
correspond to columns of logical `B`.

6. Derive Pearl commitment/noise seeds.

```text
sB = BLAKE3(key || B_root)
sA = BLAKE3(sB || A_root)
```

In the current Python/CUDA implementation these are represented as:

```text
commitment_hash_b = sB
commitment_hash_a = sA
```

7. Generate low-rank noise.

Following the whitepaper, `E` and `F` are low-rank matrices derived
deterministically from `sA` and `sB`.

Conceptually:

```text
E = generated from sA
F = generated from sB

A' = A + E
B' = B + F
```

8. Run noisy tiled matrix multiplication.

Conceptually:

```text
C' = A' @ B'
C  = C' - correction_terms
```

The whitepaper describes the recovery identity:

```text
A · B = (A + E)(B + F) - (E · B + A · F + E · F)
```

The implementation uses an equivalent optimized form in the CUDA kernel.

9. Check PoW tile openings during tiled matmul.

The kernel maintains tile hash state while computing the noised GEMM. A tile is
a hit when its hash is below the adjusted target.

Current Python-side target adjustment:

```text
adjusted_target = gateway_target * hash_tile_h * hash_tile_w * rounded_common_dim
```

If a tile hits, the kernel writes tile coordinates and selected row/column
indices into the host signal header.

10. Build and submit proof.

On hit, the miner builds `OpenedBlockInfo` from:

```text
A row indices
B_t row indices, i.e. logical B column indices
A
B_t
sA / sB
noise_rank
```

Then it creates a `PlainProof` and submits it to the gateway:

```text
submitPlainProof(plain_proof, mining_job)
```

The gateway then:

1. Checks the proof's header matches its current block template.
2. Generates a ZK certificate.
3. Builds a complete block.
4. Calls `submitblock` on `pearld`.

## Implementation map in this repo

Standalone prototype:

- `custom_miner/mining-config.yml`
- `custom_miner/standalone_miner.py`

Gateway job listener:

- `custom_miner/mining_job_listener.py`

Reusable Pearl miner packages:

- `miner/pearl-gateway/src/pearl_gateway/miner_rpc/server.py`
  - exposes `getMiningInfo`
  - accepts `submitPlainProof`
- `miner/pearl-gateway/src/pearl_gateway/comm/dataclasses.py`
  - `MiningJob`
  - `OpenedBlockInfo`
  - target adjustment
- `miner/miner-base/src/miner_base/commitment_hash.py`
  - derives the chain-bound key and commitment hashes
- `miner/miner-base/src/miner_base/block_submission.py`
  - creates `PlainProof`
- `miner/pearl-gemm/src/pearl_gemm/pearl_gemm_interface.py`
  - Python wrapper over CUDA extension
- `miner/pearl-gemm/csrc/gemm/pearl_gemm_api.cpp`
  - C++/CUDA entry points exposed through `pearl_gemm_cuda`

Important CUDA/Python calls used by the standalone miner:

```python
tensor_hash(...)
commitment_hash_from_merkle_roots(...)
noise_gen(...)
noisy_gemm(...)
get_host_signal_header(...)
extract_indices(...)
create_proof(...)
MiningClient.submit_plain_proof(...)
```

## Matrix seed vs Pearl cryptographic seed

There are two different seed concepts:

| Seed | Source | Purpose |
| --- | --- | --- |
| `matrix.seed` | Miner local config, or auto-generated | Generates synthetic `A` and `B_t` when no useful workload exists. |
| `sA`, `sB` | Derived from `A`, `B_t`, mining config, and gateway header | Generate Pearl low-rank noise `E` and `F`; bind work to current chain state. |

Therefore, using a local `matrix.seed` does not ignore mainnet. The generated
matrices are still committed against the gateway header, and the real Pearl
noise seeds depend on the current block state.

If the gateway header changes, then:

```text
H changes
key changes
sA / sB change
PoW result changes
```

A stale proof will be rejected or skipped by the gateway because the proof's
`incomplete_header_bytes` no longer matches the current block template.

## Current prototype behavior

Run one configured dry attempt:

```bash
uv run --project custom_miner python custom_miner/standalone_miner.py \
  --config custom_miner/mining-config.yml
```

In `custom_miner/mining-config.yml`:

```yaml
runtime:
  submit: false
```

means dry run. If a hit occurs, the miner logs it but does not submit proof.

Set:

```yaml
runtime:
  submit: true
```

to submit hit proofs to the gateway.

Gateway logs are currently the source of truth for whether `pearld` accepted the
final block.
