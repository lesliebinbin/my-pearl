#!/usr/bin/env python3
"""Standalone Pearl miner prototype using synthetic matrices."""

from __future__ import annotations

import argparse
import logging
import secrets
import time

import torch
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml
from miner_base.block_submission import create_proof
from miner_base.commitment_hash import CommitmentHasher
from miner_base.gateway_client import MiningClient
from miner_base.matmul_config import MatmulConfig
from pearl_gateway.comm.dataclasses import CommitmentHash, OpenedBlockInfo
from pearl_gateway.comm.mining_configuration import MiningConfiguration, MMAType, PeriodicPattern
from pearl_gateway.config import MinerRpcConfig
from pearl_gemm import (
    HostSignalStatus,
    commitment_hash_from_merkle_roots,
    extract_indices,
    get_host_signal_header,
    get_host_signal_header_size,
    get_host_signal_sync_size,
    get_required_scratchpad_bytes,
    make_pow_target_tensor,
    noise_gen,
    noisy_gemm,
    tensor_hash,
)


LOGGER = logging.getLogger("custom_miner.standalone_miner")
DEFAULT_CONFIG_PATH = Path(__file__).with_name("mining-config.yml")
BF16_BYTES = 2
INT8_BYTES = 1
FLOAT16_BYTES = 2
FLOAT32_BYTES = 4
DEFAULT_MAX_OUTPUT_GIB = 8
DEFAULT_ROWS_PATTERN = [0, 8]
DEFAULT_COLS_PATTERN = [
    0,
    1,
    8,
    9,
    16,
    17,
    24,
    25,
    32,
    33,
    40,
    41,
    48,
    49,
    56,
    57,
    64,
    65,
    72,
    73,
    80,
    81,
    88,
    89,
    96,
    97,
    104,
    105,
    112,
    113,
    120,
    121,
    128,
    129,
    136,
    137,
    144,
    145,
    152,
    153,
    160,
    161,
    168,
    169,
    176,
    177,
    184,
    185,
    192,
    193,
    200,
    201,
    208,
    209,
    216,
    217,
    224,
    225,
    232,
    233,
    240,
    241,
    248,
    249,
]


@dataclass
class AttemptResult:
    hit: bool
    elapsed_seconds: float
    hash_tiles: int
    work_units: int
    matmul_ops: int


@dataclass
class MiningStats:
    started_at: float
    last_report_at: float
    attempts: int = 0
    hits: int = 0
    elapsed_seconds: float = 0.0
    hash_tiles: int = 0
    work_units: int = 0
    matmul_ops: int = 0

    def add(self, result: AttemptResult) -> None:
        self.attempts += 1
        self.hits += int(result.hit)
        self.elapsed_seconds += result.elapsed_seconds
        self.hash_tiles += result.hash_tiles
        self.work_units += result.work_units
        self.matmul_ops += result.matmul_ops

    def should_report(self, interval: float) -> bool:
        return time.monotonic() - self.last_report_at >= interval

    def should_report_attempt(self, report_every_attempts: int) -> bool:
        return (
            report_every_attempts > 0
            and self.attempts != 0
            and self.attempts % report_every_attempts == 0
        )

    def report(self, force: bool = False) -> None:
        now = time.monotonic()
        wall_seconds = max(now - self.started_at, 1e-9)
        kernel_seconds = max(self.elapsed_seconds, 1e-9)
        LOGGER.info(
            "status attempts=%s hits=%s hashrate_th_s=%.2f work_th_s=%.2f tmac_s=%.2f "
            "attempts_s=%.2f%s",
            self.attempts,
            self.hits,
            self.hash_tiles / wall_seconds,
            self.work_units / wall_seconds / 1e12,
            self.matmul_ops / kernel_seconds / 1e12,
            self.attempts / wall_seconds,
            " final" if force else "",
        )
        self.last_report_at = now


@dataclass
class JobState:
    mining_job: Any
    adjusted_target: int
    hash_key: bytes
    refreshed_at: float


class MiningJobCache:
    def __init__(self, client: MiningClient, refresh_seconds: float) -> None:
        self._client = client
        self._refresh_seconds = refresh_seconds
        self._state: JobState | None = None

    def get(self, matmul_config) -> JobState:
        now = time.monotonic()
        if (
            self._state is not None
            and self._refresh_seconds > 0
            and now - self._state.refreshed_at < self._refresh_seconds
        ):
            return self._state

        mining_job = self._client.get_mining_info()
        self._state = JobState(
            mining_job=mining_job,
            adjusted_target=mining_job.adjust_target(matmul_config.mining_config),
            hash_key=CommitmentHasher.get_key(
                mining_job.incomplete_header_bytes,
                matmul_config.mining_config,
            ),
            refreshed_at=now,
        )
        return self._state


@dataclass(frozen=True)
class MatrixProfile:
    mode: str
    physical_m: int
    physical_n: int
    k: int
    logical_m: int
    logical_n: int


class MiningContext:
    def __init__(self, cfg: SimpleNamespace, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.profile = get_matrix_profile(cfg)
        self.matrix_seed = cfg.matrix.seed
        if self.matrix_seed is None:
            self.matrix_seed = secrets.randbits(63)
        self.matrix_generator = torch.Generator(device=device)
        self.matrix_generator.manual_seed(self.matrix_seed)
        self.matmul_config = create_matmul_config(cfg, self.profile.k)

        m, n, k, rank = (
            self.profile.physical_m,
            self.profile.physical_n,
            self.profile.k,
            cfg.mining.rank,
        )
        self.a = torch.empty((m, k), dtype=torch.int8, device=device)
        self.b_t = torch.empty((n, k), dtype=torch.int8, device=device)
        self.a_scales = torch.ones((m,), dtype=torch.float32, device=device)
        self.b_scales = torch.ones((n,), dtype=torch.float32, device=device)
        self.c = torch.empty((m, n), dtype=torch.bfloat16, device=device)

        self.eal = torch.empty((m, rank), dtype=torch.int8, device=device)
        self.ebr = torch.empty((n, rank), dtype=torch.int8, device=device)
        self.ear_r_major = torch.empty((k, rank), dtype=torch.int8, device=device)
        self.ebl_r_major = torch.empty((k, rank), dtype=torch.int8, device=device)
        self.ear_k_major = torch.empty((rank, k), dtype=torch.int8, device=device)
        self.ebl_k_major = torch.empty((rank, k), dtype=torch.int8, device=device)
        self.eal_fp16 = torch.empty((m, rank), dtype=torch.float16, device=device)
        self.ebr_fp16 = torch.empty((n, rank), dtype=torch.float16, device=device)

        self.bp_eb = torch.empty((n, k), dtype=torch.int8, device=device)
        self.ear_x_bp_eb = torch.empty((n, rank), dtype=torch.float16, device=device)
        self.ap_ea = torch.empty((m, k), dtype=torch.int8, device=device)
        self.a_x_ebl = torch.empty((m, rank), dtype=torch.float16, device=device)
        self.host_signal_header = torch.zeros(
            (get_host_signal_header_size(),),
            dtype=torch.int8,
            pin_memory=True,
        )
        self.host_signal_sync = torch.zeros(
            (get_host_signal_sync_size(),),
            dtype=torch.int8,
            device=device,
        )

        self.key_tensor = torch.empty(32, dtype=torch.uint8, device=device)
        self.scratchpad = torch.empty(
            get_required_scratchpad_bytes(max(self.a.numel(), self.b_t.numel())),
            dtype=torch.uint8,
            device=device,
        )
        self.a_root = torch.empty(32, dtype=torch.uint8, device=device)
        self.b_root = torch.empty(32, dtype=torch.uint8, device=device)
        self.commitment_hash_a = torch.empty(32, dtype=torch.uint8, device=device)
        self.commitment_hash_b = torch.empty(32, dtype=torch.uint8, device=device)

        if cfg.matrix.reuse_b:
            generate_matrix(self.b_t, cfg, self.matrix_generator)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run synthetic matrix mining attempts against Pearl Gateway."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to mining YAML config. Default: {DEFAULT_CONFIG_PATH}",
    )
    return parser.parse_args()


def to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [to_namespace(item) for item in value]
    return value


def load_config(path: Path) -> SimpleNamespace:
    with path.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    return to_namespace(cfg)


def get_config_value(cfg: SimpleNamespace, dotted_path: str, default: Any) -> Any:
    current: Any = cfg
    for part in dotted_path.split("."):
        if not hasattr(current, part):
            return default
        current = getattr(current, part)
    return current


def get_matrix_profile(cfg: SimpleNamespace) -> MatrixProfile:
    mode = get_config_value(cfg, "matrix.mode", "materialized")
    if mode not in {"materialized", "tiled"}:
        raise ValueError("matrix.mode must be 'materialized' or 'tiled'")

    physical_m = int(cfg.matrix.m)
    physical_n = int(cfg.matrix.n)
    k = int(cfg.matrix.k)
    logical_m = int(get_config_value(cfg, "matrix.logical_m", physical_m))
    logical_n = int(get_config_value(cfg, "matrix.logical_n", physical_n))

    if mode == "materialized" and (logical_m != physical_m or logical_n != physical_n):
        raise ValueError("matrix.logical_m/logical_n are only valid when matrix.mode is 'tiled'")

    for name, value in {
        "matrix.m": physical_m,
        "matrix.n": physical_n,
        "matrix.k": k,
        "matrix.logical_m": logical_m,
        "matrix.logical_n": logical_n,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    if mode == "tiled" and (physical_m > logical_m or physical_n > logical_n):
        raise ValueError("tiled mode requires matrix.m/n physical tile size <= logical_m/logical_n")

    return MatrixProfile(
        mode=mode,
        physical_m=physical_m,
        physical_n=physical_n,
        k=k,
        logical_m=logical_m,
        logical_n=logical_n,
    )


def _namespace_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [int(item) for item in value]
    return [int(item) for item in value]


def _pattern_from_config(cfg: SimpleNamespace, key: str, size_key: str, default: list[int]) -> list[int]:
    configured = get_config_value(cfg, key, None)
    expected_size = int(get_config_value(cfg, size_key, len(default)))
    if configured is None:
        if expected_size == len(default):
            return list(default)
        return list(range(expected_size))

    pattern = _namespace_list(configured)
    if len(pattern) != expected_size:
        raise ValueError(f"{key} length must match {size_key}={expected_size}")
    return pattern


def create_matmul_config(cfg: SimpleNamespace, k: int) -> MatmulConfig:
    rows_pattern = PeriodicPattern.from_list(
        _pattern_from_config(cfg, "mining.rows_pattern", "mining.h", DEFAULT_ROWS_PATTERN)
    )
    cols_pattern = PeriodicPattern.from_list(
        _pattern_from_config(cfg, "mining.cols_pattern", "mining.w", DEFAULT_COLS_PATTERN)
    )
    mining_config = MiningConfiguration(
        common_dim=k,
        rank=cfg.mining.rank,
        mma_type=MMAType.Int7xInt7ToInt32,
        rows_pattern=rows_pattern,
        cols_pattern=cols_pattern,
    )
    return MatmulConfig(
        matmul_tile_h=int(get_config_value(cfg, "mining.tile_m", 128)),
        matmul_tile_w=int(get_config_value(cfg, "mining.tile_n", 256)),
        matmul_tile_k=int(get_config_value(cfg, "mining.tile_k", 128)),
        mining_config=mining_config,
    )


def estimate_materialized_bytes(cfg: SimpleNamespace) -> dict[str, int]:
    profile = get_matrix_profile(cfg)
    m, n, k, rank = profile.physical_m, profile.physical_n, profile.k, cfg.mining.rank
    return {
        "A": m * k * INT8_BYTES,
        "B_t": n * k * INT8_BYTES,
        "C": m * n * BF16_BYTES,
        "A_scales": m * FLOAT32_BYTES,
        "B_scales": n * FLOAT32_BYTES,
        "EAL": m * rank * INT8_BYTES,
        "EBR": n * rank * INT8_BYTES,
        "EAL_fp16": m * rank * FLOAT16_BYTES,
        "EBR_fp16": n * rank * FLOAT16_BYTES,
        "ApEA": m * k * INT8_BYTES,
        "BpEB": n * k * INT8_BYTES,
        "AxEBL": m * rank * FLOAT16_BYTES,
        "EARxBpEB": n * rank * FLOAT16_BYTES,
        "noise_common": 4 * rank * k * INT8_BYTES,
    }


def format_gib(value: int) -> str:
    return f"{value / (1024**3):.2f} GiB"


def validate_materialized_profile(cfg: SimpleNamespace, device: torch.device) -> None:
    profile = get_matrix_profile(cfg)
    estimated = estimate_materialized_bytes(cfg)
    output_bytes = estimated["C"]
    total_bytes = sum(estimated.values())
    max_output_gib = get_config_value(
        cfg,
        "runtime.max_materialized_output_gib",
        DEFAULT_MAX_OUTPUT_GIB,
    )
    max_output_bytes = int(max_output_gib * 1024**3)
    free_bytes, device_total_bytes = torch.cuda.mem_get_info(device)

    LOGGER.info(
        "estimated_physical_tile_vram total=%s output_C=%s free=%s device_total=%s",
        format_gib(total_bytes),
        format_gib(output_bytes),
        format_gib(free_bytes),
        format_gib(device_total_bytes),
    )

    if output_bytes > max_output_bytes:
        raise ValueError(
            "configured physical matrix tile is too large for this prototype: "
            f"C would be materialized as {format_gib(output_bytes)}. "
            "In tiled mode, keep matrix.logical_m/logical_n large but reduce matrix.m/n "
            "to the physical tile size allocated per attempt. "
            f"Current guard is runtime.max_materialized_output_gib={max_output_gib}."
        )

    if total_bytes > int(free_bytes * 0.85):
        raise ValueError(
            "configured matrix shape is likely too large for available VRAM: "
            f"estimated {format_gib(total_bytes)} vs free {format_gib(free_bytes)}."
        )

    if profile.mode == "tiled":
        LOGGER.info(
            "tiled_profile logical=(m=%s,n=%s,k=%s) physical_tile=(m=%s,n=%s,k=%s)",
            profile.logical_m,
            profile.logical_n,
            profile.k,
            profile.physical_m,
            profile.physical_n,
            profile.k,
        )


def generate_matrix(
    out: torch.Tensor,
    cfg: SimpleNamespace,
    generator: torch.Generator,
) -> torch.Tensor:
    # Pearl's current int path expects signed 7-bit values.
    if cfg.matrix.strategy == "uniform":
        out.random_(-64, 64, generator=generator)
        return out

    if cfg.matrix.strategy == "normal":
        out.copy_(
            torch.normal(
                mean=cfg.matrix.normal_mean,
                std=cfg.matrix.normal_std,
                size=out.shape,
                device=out.device,
                generator=generator,
            )
            .round()
            .clamp(-64, 63)
            .to(torch.int8)
        )
        return out

    if cfg.matrix.strategy == "rademacher":
        out.random_(0, 2, generator=generator).mul_(2).sub_(1)
        return out

    raise ValueError(f"unsupported matrix.strategy: {cfg.matrix.strategy}")


def estimate_hash_tiles(m: int, n: int, matmul_config) -> int:
    full_matmul_tiles_m = m // matmul_config.matmul_tile_h
    full_matmul_tiles_n = n // matmul_config.matmul_tile_w
    hash_tiles_per_matmul_tile = (
        matmul_config.matmul_tile_h // matmul_config.hash_tile_h
    ) * (matmul_config.matmul_tile_w // matmul_config.hash_tile_w)
    return full_matmul_tiles_m * full_matmul_tiles_n * hash_tiles_per_matmul_tile


def generate_noise_factors(ctx: MiningContext) -> None:
    noise_gen(
        R=ctx.cfg.mining.rank,
        EAL=ctx.eal,
        EAL_fp16=ctx.eal_fp16,
        EAR_R_major=ctx.ear_r_major,
        EAR_K_major=ctx.ear_k_major,
        EBL_R_major=ctx.ebl_r_major,
        EBL_K_major=ctx.ebl_k_major,
        EBR=ctx.ebr,
        EBR_fp16=ctx.ebr_fp16,
        key_A=ctx.commitment_hash_a,
        key_B=ctx.commitment_hash_b,
    )


def build_commitment_hashes(ctx: MiningContext, hash_key: bytes) -> None:
    ctx.key_tensor.copy_(torch.tensor(list(hash_key), dtype=torch.uint8, device=ctx.device))
    tensor_hash(ctx.a.to(torch.uint8), ctx.key_tensor, ctx.a_root, ctx.scratchpad)
    tensor_hash(ctx.b_t.to(torch.uint8), ctx.key_tensor, ctx.b_root, ctx.scratchpad)
    commitment_hash_from_merkle_roots(
        ctx.a_root,
        ctx.b_root,
        ctx.key_tensor,
        ctx.commitment_hash_a,
        ctx.commitment_hash_b,
    )


def run_attempt(
    job_cache: MiningJobCache,
    ctx: MiningContext,
    attempt: int,
) -> AttemptResult:
    cfg = ctx.cfg
    profile = ctx.profile
    job_state = job_cache.get(ctx.matmul_config)
    generate_matrix(ctx.a, cfg, ctx.matrix_generator)
    if not cfg.matrix.reuse_b:
        generate_matrix(ctx.b_t, cfg, ctx.matrix_generator)
    ctx.host_signal_header.zero_()

    build_commitment_hashes(ctx, job_state.hash_key)
    generate_noise_factors(ctx)
    pow_target = make_pow_target_tensor(job_state.adjusted_target, device=ctx.device)

    started = time.monotonic()
    noisy_gemm(
        A=ctx.a,
        B=ctx.b_t,
        EAL=ctx.eal,
        EAL_fp16=ctx.eal_fp16,
        EBR=ctx.ebr,
        EBR_fp16=ctx.ebr_fp16,
        EAR_R_major=ctx.ear_r_major,
        EBL_R_major=ctx.ebl_r_major,
        EAR_K_major=ctx.ear_k_major,
        EBL_K_major=ctx.ebl_k_major,
        AxEBL_fp16=ctx.a_x_ebl,
        EARxBpEB_fp16=ctx.ear_x_bp_eb,
        ApEA=ctx.ap_ea,
        BpEB=ctx.bp_eb,
        A_scales=ctx.a_scales,
        B_scales=ctx.b_scales,
        C=ctx.c,
        host_signal_header_pinned=ctx.host_signal_header,
        host_signal_sync=ctx.host_signal_sync,
        pow_target=pow_target,
        pow_key=ctx.commitment_hash_a.view(torch.uint32),
        tile_size_m=ctx.matmul_config.matmul_tile_h,
        tile_size_n=ctx.matmul_config.matmul_tile_w,
        tile_size_k=ctx.matmul_config.matmul_tile_k,
        run_noising_A=True,
        run_noising_B=True,
        skip_reduction=False,
        skip_denoising=False,
    )
    torch.cuda.synchronize(ctx.device)
    elapsed = time.monotonic() - started
    hash_tiles = estimate_hash_tiles(profile.physical_m, profile.physical_n, ctx.matmul_config)
    work_units_per_hash_tile = job_state.mining_job._get_difficulty_adjustment_factor(
        ctx.matmul_config.mining_config
    )
    work_units = hash_tiles * work_units_per_hash_tile
    matmul_ops = 2 * profile.physical_m * profile.physical_n * profile.k

    header = get_host_signal_header(ctx.host_signal_header)
    if header.status != HostSignalStatus.kSignalTriggered:
        LOGGER.debug(
            "attempt=%s no hit elapsed=%.3fs shape=(%s,%s,%s) strategy=%s",
            attempt,
            elapsed,
            profile.physical_m,
            profile.physical_n,
            profile.k,
            cfg.matrix.strategy,
        )
        return AttemptResult(False, elapsed, hash_tiles, work_units, matmul_ops)

    indices = extract_indices(header)
    LOGGER.info(
        "attempt=%s found hit elapsed=%.3fs rows=%s cols=%s",
        attempt,
        elapsed,
        indices.A_row_indices,
        indices.B_column_indices,
    )

    if not cfg.runtime.submit:
        LOGGER.info("submit skipped attempt=%s reason=submit_disabled", attempt)
        return AttemptResult(True, elapsed, hash_tiles, work_units, matmul_ops)

    opened_block_info = OpenedBlockInfo(
        A_row_indices=indices.A_row_indices,
        B_column_indices=indices.B_column_indices,
        A=ctx.a.detach().cpu(),
        B_t=ctx.b_t.detach().cpu(),
        commitment_hash=CommitmentHash(
            noise_seed_A=ctx.commitment_hash_a.detach().cpu().numpy().tobytes(),
            noise_seed_B=ctx.commitment_hash_b.detach().cpu().numpy().tobytes(),
        ),
        noise_rank=cfg.mining.rank,
    )
    plain_proof = create_proof(opened_block_info, job_state.mining_job.incomplete_header_bytes)
    LOGGER.info("submit attempt=%s rows=%s cols=%s", attempt, indices.A_row_indices, indices.B_column_indices)
    job_cache._client.submit_plain_proof(plain_proof, job_state.mining_job)
    LOGGER.info("submitted proof to gateway attempt=%s", attempt)
    return AttemptResult(True, elapsed, hash_tiles, work_units, matmul_ops)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    logging.basicConfig(
        level=getattr(logging, cfg.runtime.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if cfg.runtime.seed is not None:
        torch.manual_seed(cfg.runtime.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for standalone mining")

    device = torch.device("cuda")
    LOGGER.info("using device=%s capability=%s", device, torch.cuda.get_device_capability(device))
    validate_materialized_profile(cfg, device)
    miner_rpc_config = MinerRpcConfig(
        transport=cfg.gateway.transport,
        socket_path=cfg.gateway.socket_path,
        host=cfg.gateway.host,
        port=cfg.gateway.port,
    )
    ctx = MiningContext(cfg, device)
    LOGGER.info(
        "config=%s submit=%s matrix_strategy=%s matrix_seed=0x%x reuse_b=%s "
        "mode=%s logical_shape=(m=%s,n=%s,k=%s) physical_tile=(m=%s,n=%s,k=%s) "
        "rank=%s h=%s w=%s matmul_tile=(%s,%s,%s)",
        args.config,
        cfg.runtime.submit,
        cfg.matrix.strategy,
        ctx.matrix_seed,
        cfg.matrix.reuse_b,
        ctx.profile.mode,
        ctx.profile.logical_m,
        ctx.profile.logical_n,
        ctx.profile.k,
        ctx.profile.physical_m,
        ctx.profile.physical_n,
        ctx.profile.k,
        cfg.mining.rank,
        ctx.matmul_config.hash_tile_h,
        ctx.matmul_config.hash_tile_w,
        ctx.matmul_config.matmul_tile_h,
        ctx.matmul_config.matmul_tile_w,
        ctx.matmul_config.matmul_tile_k,
    )
    with MiningClient(miner_rpc_config) as client:
        job_cache = MiningJobCache(client, cfg.gateway.refresh_job_seconds)
        attempt = 1
        now = time.monotonic()
        stats = MiningStats(started_at=now, last_report_at=now)
        report_every_attempts = int(get_config_value(cfg, "runtime.report_every_attempts", 10))
        while cfg.runtime.iterations == 0 or attempt <= cfg.runtime.iterations:
            result = run_attempt(job_cache, ctx, attempt)
            stats.add(result)
            if report_every_attempts > 0:
                should_report = stats.should_report_attempt(report_every_attempts)
            else:
                should_report = cfg.runtime.report_interval_seconds == 0 or stats.should_report(
                    cfg.runtime.report_interval_seconds
                )
            if should_report:
                stats.report()
            attempt += 1
        stats.report(force=True)


if __name__ == "__main__":
    main()
