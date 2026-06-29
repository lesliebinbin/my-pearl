#!/usr/bin/env python3
"""Print Pearl mining layer names and shapes from a cached Hugging Face model."""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_START_SCRIPT = Path(__file__).resolve().parents[1] / "start_vllm_mining.sh"
DEFAULT_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
PROJECTION_WEIGHT_RE = re.compile(
    r".*\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$"
)


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: list[int]
    dtype: str
    source_file: str

    @property
    def layer_name(self) -> str:
        return self.name.removesuffix(".weight")

    @property
    def in_features(self) -> int | None:
        return self.shape[1] if len(self.shape) == 2 else None

    @property
    def out_features(self) -> int | None:
        return self.shape[0] if len(self.shape) == 2 else None


@dataclass(frozen=True)
class QuantGroup:
    name: str
    weight_bits: int | None
    input_bits: int | None
    targets: list[str]

    @property
    def is_mining(self) -> bool:
        return self.weight_bits == 7 and self.input_bits == 7

    @property
    def is_linear_fallback(self) -> bool:
        return "Linear" in self.targets

    def matches(self, layer_name: str) -> bool:
        for target in self.targets:
            if target == "Linear":
                continue
            if target.startswith("re:") and re.fullmatch(target[3:], layer_name):
                return True
            if target == layer_name or target in layer_name:
                return True
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Pearl mining layer names and weight shapes from a cached model."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model id, for example pearl-ai/Gemma-4-31B-it-pearl. Defaults to start_vllm_mining.sh.",
    )
    parser.add_argument(
        "--hf-cache",
        type=Path,
        default=Path(os.getenv("HF_HUB_CACHE", DEFAULT_HF_CACHE)),
        help="Hugging Face hub cache directory.",
    )
    parser.add_argument(
        "--all-linear",
        action="store_true",
        help="Print all projection weights instead of only Pearl 7-bit mining layers.",
    )
    return parser.parse_args()


def read_default_model_id() -> str:
    if not DEFAULT_START_SCRIPT.exists():
        raise FileNotFoundError(f"start script not found: {DEFAULT_START_SCRIPT}")

    script = DEFAULT_START_SCRIPT.read_text(encoding="utf-8")
    match = re.search(r"\bvllm\s+serve\s+([^\s\\]+)", script)
    if not match:
        raise ValueError(f"could not find `vllm serve <model>` in {DEFAULT_START_SCRIPT}")
    return match.group(1)


def model_cache_dir(model_id: str, hf_cache: Path) -> Path:
    return hf_cache / f"models--{model_id.replace('/', '--')}"


def resolve_snapshot(model_id: str, hf_cache: Path) -> Path:
    cache_dir = model_cache_dir(model_id, hf_cache)
    if not cache_dir.exists():
        raise FileNotFoundError(f"model is not cached: {model_id} ({cache_dir})")

    ref_file = cache_dir / "refs" / "main"
    if ref_file.exists():
        snapshot = cache_dir / "snapshots" / ref_file.read_text(encoding="utf-8").strip()
        if snapshot.exists():
            return snapshot

    snapshots = sorted((cache_dir / "snapshots").glob("*"), key=lambda p: p.stat().st_mtime)
    if snapshots:
        return snapshots[-1]

    raise FileNotFoundError(f"no snapshots found for cached model: {model_id}")


def load_config(snapshot: Path) -> dict[str, Any]:
    config_path = snapshot / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in snapshot: {snapshot}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_quant_groups(config: dict[str, Any]) -> list[QuantGroup]:
    groups = config.get("quantization_config", {}).get("config_groups", {})
    quant_groups = []
    for name, group in groups.items():
        weights = group.get("weights") or {}
        input_activations = group.get("input_activations") or {}
        quant_groups.append(
            QuantGroup(
                name=name,
                weight_bits=weights.get("num_bits"),
                input_bits=input_activations.get("num_bits"),
                targets=group.get("targets") or [],
            )
        )
    return quant_groups


def read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        header_size = struct.unpack("<Q", file.read(8))[0]
        return json.loads(file.read(header_size))


def iter_tensor_infos(snapshot: Path) -> list[TensorInfo]:
    tensors = []
    for path in sorted(snapshot.glob("*.safetensors")):
        header = read_safetensors_header(path)
        for name, metadata in header.items():
            if name == "__metadata__":
                continue
            tensors.append(
                TensorInfo(
                    name=name,
                    shape=metadata["shape"],
                    dtype=metadata["dtype"],
                    source_file=path.name,
                )
            )
    return tensors


def is_ignored(layer: TensorInfo, ignored_names: set[str]) -> bool:
    return layer.name in ignored_names or layer.layer_name in ignored_names


def matching_group(layer: TensorInfo, groups: list[QuantGroup]) -> QuantGroup | None:
    layer_name = layer.layer_name
    for group in groups:
        if group.matches(layer_name):
            return group

    for group in groups:
        if group.is_linear_fallback:
            return group
    return None


def sort_key(layer: TensorInfo) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", layer.name)]


def select_layers(
    tensors: list[TensorInfo],
    groups: list[QuantGroup],
    ignored_names: set[str],
    include_all_linear: bool,
) -> list[tuple[TensorInfo, QuantGroup | None]]:
    selected = []
    for tensor in tensors:
        if not PROJECTION_WEIGHT_RE.fullmatch(tensor.name):
            continue
        if is_ignored(tensor, ignored_names):
            continue

        group = matching_group(tensor, groups)
        if include_all_linear or (group is not None and group.is_mining):
            selected.append((tensor, group))

    return sorted(selected, key=lambda item: sort_key(item[0]))


def print_layers(model_id: str, snapshot: Path, layers: list[tuple[TensorInfo, QuantGroup | None]]) -> None:
    print(f"model: {model_id}")
    print(f"snapshot: {snapshot}")
    print(f"layers: {len(layers)}")
    print()
    print(
        f"{'#':>4}  {'group':<10}  {'dtype':<8}  {'weight_shape':<18}  "
        f"{'input_shape':<22}  {'output_shape':<22}  name"
    )
    for index, (layer, group) in enumerate(layers, start=1):
        input_shape = f"[tokens, {layer.in_features}]" if layer.in_features is not None else "dynamic"
        output_shape = f"[tokens, {layer.out_features}]" if layer.out_features is not None else "dynamic"
        print(
            f"{index:>4}  {(group.name if group else '-'): <10}  {layer.dtype:<8}  "
            f"{str(layer.shape):<18}  {input_shape:<22}  {output_shape:<22}  {layer.layer_name}"
        )


def main() -> None:
    args = parse_args()
    model_id = args.model or read_default_model_id()
    snapshot = resolve_snapshot(model_id, args.hf_cache)
    config = load_config(snapshot)
    groups = load_quant_groups(config)
    ignored_names = set(config.get("quantization_config", {}).get("ignore") or [])
    tensors = iter_tensor_infos(snapshot)
    layers = select_layers(tensors, groups, ignored_names, args.all_linear)
    print_layers(model_id, snapshot, layers)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        pass
