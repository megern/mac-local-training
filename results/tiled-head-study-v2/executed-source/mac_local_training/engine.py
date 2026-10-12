"""Local-only LoRA training with disk-backed, rotating Adam windows.

This is block-coordinate *adapter* optimization, not a method to fit a base model
larger than available memory. Frozen base weights and all adapter weights stay in
memory. Only one window's gradients and Adam moments stay resident.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import shutil
import sys
import time
from typing import Any
import uuid


MAX_DATASET_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class TrainingConfig:
    train_path: str
    valid_path: str
    output_path: str
    model_path: str | None = None
    tiny_model: bool = False
    strategy: str = "rotating"
    steps: int = 12
    microbatch_size: int = 1
    accumulation_steps: int = 2
    window_layers: int = 1
    window_updates: int = 1
    learning_rate: float = 1e-3
    rank: int = 4
    lora_scale: float = 1.0
    seed: int = 17
    max_seq_length: int = 128
    checkpoint_every: int = 1
    resume: bool = False
    memory_budget_gib: float = 6.0

    def validate(self) -> None:
        if self.strategy not in {"all", "rotating"}:
            raise ValueError("strategy must be all or rotating")
        for name in ("steps", "microbatch_size", "accumulation_steps", "window_layers",
                     "window_updates", "rank", "checkpoint_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 2 <= self.max_seq_length <= 4096:
            raise ValueError("max_seq_length must be between 2 and 4096")
        for name in ("learning_rate", "lora_scale", "memory_budget_gib"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.tiny_model == bool(self.model_path):
            raise ValueError("Choose exactly one: tiny_model or a local model_path")
        for name in ("train_path", "valid_path"):
            if not Path(getattr(self, name)).is_file():
                raise ValueError(f"{name} must point to an existing local JSONL file")
        dataset_bytes = sum(Path(getattr(self, name)).stat().st_size
                            for name in ("train_path", "valid_path"))
        if dataset_bytes > MAX_DATASET_BYTES:
            raise ValueError("Prototype loads token windows into memory; combined datasets exceed 64 MiB")
        if self.model_path and not Path(self.model_path).is_dir():
            raise ValueError("model_path must be an existing local directory; remote IDs are refused")
        if self.model_path:
            weight_bytes = sum(p.stat().st_size for p in Path(self.model_path).glob("*.safetensors"))
            if weight_bytes >= int(self.memory_budget_gib * 1024 ** 3):
                raise ValueError("Serialized model weights alone reach the memory budget; "
                                 "runtime needs additional memory for adapters, activations and optimizer")


def _mlx():
    # Imports are deferred so planning/help does not need a Metal device.
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map, tree_unflatten
    return mx, nn, optim, tree_flatten, tree_map, tree_unflatten


class TinyTokenizer:
    """Offline UTF-8 byte tokenizer for software checks, not a pretrained model."""
    vocab_size = 259
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def encode(self, text: str, **_: Any) -> list[int]:
        return [1, *[b + 3 for b in text.encode("utf-8")], 2]


def make_tiny_model(seed: int = 17):
    """Return a randomly initialized, four-layer Qwen2 and byte tokenizer."""
    mx, _, _, _, _, _ = _mlx()
    from mlx_lm.models.qwen2 import Model, ModelArgs
    mx.random.seed(seed)
    model = Model(ModelArgs(model_type="qwen2", hidden_size=32,
                            num_hidden_layers=4, intermediate_size=64,
                            num_attention_heads=4, num_key_value_heads=2,
                            rms_norm_eps=1e-6, vocab_size=TinyTokenizer.vocab_size,
                            max_position_embeddings=4096))
    mx.eval(model.parameters())
    return model, TinyTokenizer()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_write(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _rss_bytes() -> int:
    # ru_maxrss is bytes on macOS, KiB on Linux. This is a process high-water mark.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


class CompletionTokens(list):
    """Token sequence whose target loss starts at a supplied token index."""

    def __init__(self, tokens, loss_start):
        super().__init__(tokens)
        self.loss_start = loss_start


def _read_data(path: str, tokenizer: Any, max_seq_length: int, vocab_size: int):
    rows: list[list[int]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "tokens" in row:
                tokens = row["tokens"]
            elif isinstance(row.get("text"), str):
                tokens = tokenizer.encode(row["text"])
            else:
                raise ValueError(f"{path}:{line_no}: expected text or tokens")
            if not isinstance(tokens, list) or any(type(t) is not int for t in tokens):
                raise ValueError(f"{path}:{line_no}: tokens must be integer IDs")
            if len(tokens) < 2 or any(t < 0 or t >= vocab_size for t in tokens):
                raise ValueError(f"{path}:{line_no}: invalid token IDs or fewer than two tokens")
            if "loss_start" in row:
                loss_start = row["loss_start"]
                if type(loss_start) is not int or not 1 <= loss_start < len(tokens):
                    raise ValueError(f"{path}:{line_no}: loss_start must select a nonempty target suffix")
                # Preserve the complete prompt: silently chunking a completion can
                # change the task and discard its conditioning context.
                if len(tokens) > max_seq_length + 1:
                    raise ValueError(f"{path}:{line_no}: completion sequence exceeds max_seq_length")
                rows.append(CompletionTokens(tokens, loss_start))
                continue
            # Overlap one input token: each original next-token target appears once.
            # Context resets at each boundary; this is not equivalent to full-context loss.
            for start in range(0, len(tokens) - 1, max_seq_length):
                rows.append(tokens[start:start + max_seq_length + 1])
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    return rows


def load_local_model(path: str):
    """Load supported model files only, without Hub calls or custom remote code."""
    directory = Path(path).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError("An existing local model directory is required; remote IDs are refused")
    cfg_path = directory / "config.json"
    if not cfg_path.is_file() or not list(directory.glob("*.safetensors")):
        raise ValueError("Local model needs config.json and safetensors weights")
    config = json.loads(cfg_path.read_text())
    if config.get("model_type") not in {"qwen2", "qwen3", "llama", "mistral"}:
        raise ValueError("Supported model types: qwen2, qwen3, llama, mistral")
    for filename in ("config.json", "tokenizer_config.json"):
        file = directory / filename
        if file.is_file() and json.loads(file.read_text()).get("auto_map"):
            raise ValueError("Custom model/tokenizer code is refused")
    # Defense in depth. The path is verified before mlx-lm's get_model_path call.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    from mlx_lm import load
    return load(str(directory), trust_remote_code=False,
                tokenizer_config={"trust_remote_code": False, "local_files_only": True})


class TrainingEngine:
    """Single-process runner. Checkpoints occur only at optimizer-update boundaries."""

    def __init__(self, config: TrainingConfig):
        config.validate()
        self.config = config
        self.mx, self.nn, self.optim, self.flatten, self.map, self.unflatten = _mlx()
        self.memory_budget_bytes = int(config.memory_budget_gib * 1024 ** 3)
        # MLX allocation guard; process RSS/unified memory can still exceed this value.
        self.mx.set_memory_limit(self.memory_budget_bytes)
        self.output = Path(config.output_path).expanduser().resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        if (self.output / "latest.json").exists() and not config.resume:
            raise ValueError("Output already contains a checkpoint; use resume or a new directory")
        self.mx.random.seed(config.seed)
        self.model, self.tokenizer = (make_tiny_model(config.seed) if config.tiny_model
                                     else load_local_model(config.model_path))
        self.mx.random.seed(config.seed + 1)
        self._attach_adapters()
        self.mx.eval(self.model.parameters())
        self.vocab_size = int(self.model.args.vocab_size)
        self.train_data = _read_data(config.train_path, self.tokenizer,
                                     config.max_seq_length, self.vocab_size)
        self.valid_data = _read_data(config.valid_path, self.tokenizer,
                                     config.max_seq_length, self.vocab_size)
        self.order = list(range(len(self.train_data)))
        random.Random(config.seed).shuffle(self.order)
        self.groups = ([list(range(len(self.model.layers)))] if config.strategy == "all"
                       else [list(range(i, min(i + config.window_layers, len(self.model.layers))))
                             for i in range(0, len(self.model.layers), config.window_layers)])
        self.step = 0
        self.cursor = 0
        self.active_group: int | None = None
        self.optimizer = None
        self.state_paths: dict[int, Path] = {}
        self.history: list[dict[str, Any]] = []
        self.coverage = [0] * len(self.model.layers)
        self.training_seconds = 0.0
        self.training_tokens = 0
        self.pre_nll: float | None = None
        self.identity = self._identity()
        self._work_states = self.output / "working_states"
        self._work_states.mkdir(exist_ok=True)
        if config.resume:
            self._restore()
        self.loss_and_grad = self.nn.value_and_grad(self.model, self._loss_sum)

    def _identity(self) -> dict[str, Any]:
        values = asdict(self.config)
        # Extend total steps or move the output; all optimization settings must match.
        for name in ("steps", "resume", "output_path", "checkpoint_every"):
            values.pop(name)
        values["train_path"] = str(Path(values["train_path"]).resolve())
        values["valid_path"] = str(Path(values["valid_path"]).resolve())
        if self.config.model_path:
            directory = Path(self.config.model_path).resolve()
            values["model_path"] = str(directory)
            model_files = {str(p.relative_to(directory)): _sha(p)
                           for p in sorted(directory.rglob("*")) if p.is_file()
                           and p.suffix in {".json", ".safetensors", ".model", ".txt"}}
        else:
            model_files = {"tiny_qwen2": "4_layers_32_hidden_64_intermediate_259_vocab_v1"}
        import importlib.metadata
        return {"config": values, "train_sha256": _sha(Path(self.config.train_path)),
                "valid_sha256": _sha(Path(self.config.valid_path)), "model_files": model_files,
                "versions": {name: importlib.metadata.version(name) for name in ("mlx", "mlx-lm")}}

    def _attach_adapters(self) -> None:
        from mlx_lm.tuner.lora import LoRALinear
        for layer in self.model.layers:
            if not hasattr(layer, "self_attn"):
                raise ValueError("Model does not have supported self_attn layers")
            for name in ("q_proj", "v_proj"):
                original = getattr(layer.self_attn, name, None)
                if not isinstance(original, (self.nn.Linear, self.nn.QuantizedLinear)):
                    raise ValueError(f"Unsupported projection: {name}")
                setattr(layer.self_attn, name, LoRALinear.from_base(
                    original, r=self.config.rank, dropout=0.0, scale=self.config.lora_scale))
        self.model.freeze()

    def adapter_arrays(self) -> dict[str, Any]:
        return {k: v for k, v in self.flatten(self.model.parameters())
                if k.endswith((".lora_a", ".lora_b"))}

    def base_arrays(self) -> dict[str, Any]:
        return {k: v for k, v in self.flatten(self.model.parameters())
                if not k.endswith((".lora_a", ".lora_b"))}

    def _save_optimizer(self, path: Path) -> None:
        if self.optimizer is None:
            raise RuntimeError("No active optimizer")
        tensors = dict(self.flatten(self.optimizer.state))
        self.mx.eval(tensors)
        self.mx.save_safetensors(str(path), tensors)

    def activate_group(self, group_id: int) -> None:
        if group_id == self.active_group:
            return
        if self.optimizer is not None:
            saved = self._work_states / f"group-{self.active_group}-{uuid.uuid4().hex}.safetensors"
            self._save_optimizer(saved)
            self.state_paths[self.active_group] = saved
            # Drop every reference to inactive Adam moments; freeze alone is insufficient.
            self.optimizer = None
            gc.collect()
            self.mx.clear_cache()
        self.model.freeze()
        for layer_id in self.groups[group_id]:
            for name in ("q_proj", "v_proj"):
                getattr(self.model.layers[layer_id].self_attn, name).unfreeze(
                    keys=["lora_a", "lora_b"], recurse=False)
        self.optimizer = self.optim.Adam(learning_rate=self.config.learning_rate)
        if group_id in self.state_paths:
            arrays = self.mx.load(str(self.state_paths[group_id]))
            self.optimizer.state = self.unflatten(list(arrays.items()))
            del arrays
        self.optimizer.init(self.model.trainable_parameters())
        self.mx.eval(self.optimizer.state)
        self.active_group = group_id

    def _batch(self, rows: list[list[int]]):
        width = max(len(row) for row in rows) - 1
        # Valid tokens precede padding, so future right padding cannot affect loss.
        pad = getattr(self.tokenizer, "pad_token_id", 0)
        pad = pad if pad is not None else 0
        inputs, targets, masks = [], [], []
        for row in rows:
            length = len(row) - 1
            inputs.append(row[:-1] + [pad] * (width - length))
            targets.append(row[1:] + [pad] * (width - length))
            loss_start = getattr(row, "loss_start", 1)
            masks.append([float(index >= loss_start) for index in range(1, len(row))]
                         + [0.0] * (width - length))
        return (self.mx.array(inputs, dtype=self.mx.int32),
                self.mx.array(targets, dtype=self.mx.int32),
                self.mx.array(masks, dtype=self.mx.float32))

    def _loss_sum(self, model: Any, inputs: Any, targets: Any, mask: Any):
        logits = model(inputs).astype(self.mx.float32)
        losses = self.nn.losses.cross_entropy(logits, targets, reduction="none")
        return (losses * mask).sum()

    def evaluate(self) -> float:
        self.model.eval()
        loss = 0.0
        count = 0
        for i in range(0, len(self.valid_data), self.config.microbatch_size):
            batch = self._batch(self.valid_data[i:i + self.config.microbatch_size])
            loss += float(self._loss_sum(self.model, *batch).item())
            count += int(batch[2].sum().item())
        result = loss / count
        if not math.isfinite(result):
            raise FloatingPointError("Held-out NLL is not finite")
        return result

    def update_once(self) -> dict[str, Any]:
        group_id = (self.step // self.config.window_updates) % len(self.groups)
        self.activate_group(group_id)
        self.model.train()
        started = time.perf_counter()
        accumulated = None
        loss_sum = 0.0
        token_count = 0
        for _ in range(self.config.accumulation_steps):
            rows = [self.train_data[self.order[(self.cursor + j) % len(self.order)]]
                    for j in range(self.config.microbatch_size)]
            self.cursor += self.config.microbatch_size
            batch = self._batch(rows)
            loss, grads = self.loss_and_grad(self.model, *batch)
            # Sum *token* gradients, not unweighted means of unequal-length batches.
            accumulated = grads if accumulated is None else self.map(
                lambda a, b: a + b, accumulated, grads)
            self.mx.eval(loss, accumulated)
            loss_sum += float(loss.item())
            token_count += int(batch[2].sum().item())
            del grads, batch, loss
        gradients = self.map(lambda g: g / token_count, accumulated)
        self.mx.eval(gradients)
        if not math.isfinite(loss_sum) or any(
                not bool(self.mx.all(self.mx.isfinite(v)).item()) for _, v in self.flatten(gradients)):
            raise FloatingPointError("Non-finite training loss/gradients; update not committed")
        self.optimizer.update(self.model, gradients)
        self.mx.eval(self.model.trainable_parameters(), self.optimizer.state)
        if any(not bool(self.mx.all(self.mx.isfinite(v)).item()) for v in self.adapter_arrays().values()):
            raise FloatingPointError("Non-finite adapter weights; checkpoint not committed")
        del gradients, accumulated
        elapsed = time.perf_counter() - started
        self.step += 1
        for layer in self.groups[group_id]:
            self.coverage[layer] += 1
        self.training_seconds += elapsed
        self.training_tokens += token_count
        resident = sum(v.nbytes for _, v in self.flatten(self.optimizer.state))
        record = {"step": self.step, "group": group_id, "layers": self.groups[group_id],
                  "loss": loss_sum / token_count, "tokens": token_count,
                  "seconds": elapsed, "tokens_per_second": token_count / elapsed,
                  "optimizer_resident_bytes": resident,
                  "trainable_parameters": sum(v.size for _, v in self.flatten(self.model.trainable_parameters())),
                  "mlx_peak_bytes": int(self.mx.get_peak_memory()),
                  "process_rss_high_water_bytes": _rss_bytes()}
        self.history.append(record)
        return record

    def checkpoint(self) -> None:
        checkpoints = self.output / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        token = uuid.uuid4().hex
        pending = checkpoints / f".pending-{token}"
        pending.mkdir()
        adapter_file = pending / "adapters.safetensors"
        self.mx.save_safetensors(str(adapter_file), self.adapter_arrays())
        state_files: dict[str, str] = {}
        for group_id in range(len(self.groups)):
            name = f"optimizer-{group_id}.safetensors"
            if group_id == self.active_group:
                self._save_optimizer(pending / name)
            elif group_id in self.state_paths:
                shutil.copyfile(self.state_paths[group_id], pending / name)
            else:
                continue  # Never activated: fresh Adam state is implicit and recorded.
            state_files[str(group_id)] = name
        rng = self.mx.random.state
        self.mx.save_safetensors(str(pending / "rng.safetensors"),
                                 {str(i): item for i, item in enumerate(rng)})
        metadata = {"format_version": 1, "identity": self.identity, "step": self.step,
                    "cursor": self.cursor, "active_group": self.active_group,
                    "optimizer_files": state_files, "coverage": self.coverage,
                    "history": self.history, "training_tokens": self.training_tokens,
                    "training_seconds": self.training_seconds, "pre_nll": self.pre_nll,
                    "rng_count": len(rng)}
        _json_write(pending / "state.json", metadata)
        checksums = {p.name: _sha(p) for p in pending.iterdir() if p.is_file()}
        final = checkpoints / f"step-{self.step:08d}-{token}"
        pending.rename(final)
        latest_tmp = self.output / f".latest-{token}.json"
        _json_write(latest_tmp, {"format_version": 1,
                                 "checkpoint": str(final.relative_to(self.output)),
                                 "files_sha256": checksums})
        os.replace(latest_tmp, self.output / "latest.json")
        # Inactive tensors are not read back; state references now use immutable files.
        self.state_paths = {int(group): final / filename for group, filename in state_files.items()}
        for path in self._work_states.glob("*.safetensors"):
            path.unlink()

    def _restore(self) -> None:
        latest = self.output / "latest.json"
        if not latest.is_file():
            raise ValueError("Resume needs a committed latest.json checkpoint")
        manifest = json.loads(latest.read_text())
        directory = (self.output / manifest["checkpoint"]).resolve()
        if not directory.is_relative_to(self.output):
            raise ValueError("Invalid checkpoint path")
        for filename, expected in manifest["files_sha256"].items():
            if Path(filename).name != filename or _sha(directory / filename) != expected:
                raise ValueError("Checkpoint checksum mismatch")
        state = json.loads((directory / "state.json").read_text())
        if state["identity"] != self.identity:
            raise ValueError("Resume config, datasets, model files or MLX versions changed")
        if state["step"] > self.config.steps:
            raise ValueError("Requested steps precede the committed checkpoint")
        weights = self.mx.load(str(directory / "adapters.safetensors"))
        if set(weights) != set(self.adapter_arrays()):
            raise ValueError("Adapter checkpoint keys do not match the model")
        self.model.load_weights(list(weights.items()), strict=False)
        self.mx.eval(self.model.parameters())
        self.step, self.cursor = state["step"], state["cursor"]
        self.history, self.coverage = state["history"], state["coverage"]
        self.pre_nll = state["pre_nll"]
        self.training_tokens = state["training_tokens"]
        self.training_seconds = state["training_seconds"]
        self.state_paths = {int(g): directory / f for g, f in state["optimizer_files"].items()}
        if state["active_group"] is not None:
            self.activate_group(state["active_group"])
        rng = self.mx.load(str(directory / "rng.safetensors"))
        self.mx.random.state = [rng[str(i)] for i in range(state["rng_count"])]

    def run(self) -> dict[str, Any]:
        segment_started = time.perf_counter()
        segment_start_tokens = self.training_tokens
        self.mx.reset_peak_memory()
        if self.pre_nll is None:
            self.pre_nll = self.evaluate()
        while self.step < self.config.steps:
            self.update_once()
            if self.step % self.config.checkpoint_every == 0:
                self.checkpoint()
        self.checkpoint()
        post_nll = self.evaluate()
        end_to_end_seconds = time.perf_counter() - segment_started
        report = {"mode": "random_tiny_software_check" if self.config.tiny_model else "local_pretrained_lora",
                  "strategy": self.config.strategy, "steps": self.step,
                  "microbatches": self.step * self.config.accumulation_steps,
                  "examples_consumed": self.cursor, "tokens": self.training_tokens,
                  "training_seconds": self.training_seconds,
                  "tokens_per_second": self.training_tokens / self.training_seconds,
                  "end_to_end_seconds_this_process_segment": end_to_end_seconds,
                  "tokens_per_second_including_io_and_validation_this_process_segment":
                      (self.training_tokens - segment_start_tokens) / end_to_end_seconds,
                  "heldout_nll_before": self.pre_nll, "heldout_nll_after": post_nll,
                  "heldout_nll_change": post_nll - self.pre_nll,
                  "layer_update_coverage": self.coverage,
                  "layer_update_fraction": [count / self.step for count in self.coverage],
                  "peak_optimizer_resident_bytes": max(r["optimizer_resident_bytes"] for r in self.history),
                  "mlx_peak_bytes_this_process_segment": int(self.mx.get_peak_memory()),
                  "process_rss_high_water_bytes": _rss_bytes(),
                  "groups": self.groups, "history": self.history,
                  "mlx_memory_budget_bytes": self.memory_budget_bytes,
                  "train_token_windows": len(self.train_data),
                  "valid_token_windows": len(self.valid_data),
                  "dataset_loading": "in_memory; ordinary rows chunked, completion rows preserve full context",
                  "loss_targets": {"train_completion_rows": sum(isinstance(r, CompletionTokens) for r in self.train_data),
                                   "valid_completion_rows": sum(isinstance(r, CompletionTokens) for r in self.valid_data)},
                  "identity": self.identity,
                  "limitations": ["Frozen base and all adapters still need to fit unified memory.",
                                   "MLX memory limit is an allocation guard, not a hard process RSS cap or fit guarantee.",
                                   "Combined JSONL files are capped at 64 MiB; this prototype is not a streaming corpus loader.",
                                   "Ordinary long rows are chunked with context resets; oversized completion rows are rejected.",
                                   "Rotating windows give each layer fewer updates than the all-adapter baseline.",
                                   "Training throughput excludes optimizer-state swap I/O, checkpoint I/O and validation.",
                                   "RSS is a process high-water mark; compare strategies in fresh processes.",
                                   "Tiny random-model results validate software, not useful model quality."]}
        _json_write(self.output / "report.json", report)
        return report


def run_training(config: TrainingConfig) -> dict[str, Any]:
    """Train without downloads, remote inference, cloud compute, or generation."""
    return TrainingEngine(config).run()
