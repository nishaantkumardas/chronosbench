"""Scoring — maps benchmark results to a 0–2000 composite score.

Each subsystem contributes one component score, also capped at 2000.
The composite is a weighted average, not a flat mean, so a slow GPU
on a machine that has one doesn't tank the whole result.

CPU  — total_ops across all workers, normalised against a baseline
IO   — combined read+write throughput (MB/s)
GPU  — passes per second from the metal/cuda worker
MIXED— combined cpu+io ops from the mixed phase, rewards sustained
       performance under thermal pressure
"""

from __future__ import annotations

_CPU_BASELINE  = 5_000    # total_ops a mid-range machine should hit
_IO_BASELINE   = 500.0    # MB/s read+write combined, modern SSD ballpark
_GPU_BASELINE  = 20.0     # passes/s on MPS M1; CUDA will exceed this
_MIXED_BASELINE = 3_000   # total_ops under combined thermal load

_MAX = 2000


def _clamp(value: float) -> int:
    return min(max(int(value), 0), _MAX)


def _cpu_score(cpu: dict) -> int:
    total_ops = cpu.get("total_ops", 0)
    return _clamp((total_ops / _CPU_BASELINE) * 1000)


def _io_score(io: dict) -> int:
    read  = io.get("read_mb_s",  0.0)
    write = io.get("write_mb_s", 0.0)
    combined = read + write
    # Bonus for fsync throughput — rewards drives with low write latency
    fsyncs      = io.get("fsyncs", 0)
    fsync_bonus = min(fsyncs / 500, 200)
    return _clamp((combined / _IO_BASELINE) * 1000 + fsync_bonus)


def _gpu_score(gpu: dict) -> int | None:
    if "note" in gpu:
        return None  # GPU not available — excluded from composite
    passes   = gpu.get("passes", 0)
    duration = max(gpu.get("duration_s", 1.0), 0.1)
    passes_per_sec = passes / duration
    return _clamp((passes_per_sec / _GPU_BASELINE) * 1000)


def _mixed_score(mixed: dict) -> int:
    cpu_ops = mixed.get("cpu", {}).get("total_ops", 0)
    io_read = mixed.get("io",  {}).get("read_mb_s",  0.0)
    io_write = mixed.get("io", {}).get("write_mb_s", 0.0)
    # Normalise each component then average — prevents one saturating dimension
    # from masking a slow one
    cpu_norm = cpu_ops / _MIXED_BASELINE
    io_norm  = (io_read + io_write) / _IO_BASELINE
    return _clamp(((cpu_norm + io_norm) / 2) * 1000)


def score_report(report: dict) -> dict:
    results = report.get("results", {})
    scores: dict[str, int] = {}

    if "cpu" in results:
        scores["cpu"] = _cpu_score(results["cpu"])

    if "io" in results:
        scores["io"] = _io_score(results["io"])

    if "mixed" in results:
        gpu_raw = results["mixed"].get("gpu", {})
        gpu     = _gpu_score(gpu_raw)
        if gpu is not None:
            scores["gpu"] = gpu
        scores["mixed"] = _mixed_score(results["mixed"])

    weights = {"cpu": 2.0, "io": 1.0, "gpu": 1.5, "mixed": 1.5}
    total_weight = sum(weights[k] for k in scores if k in weights)
    weighted_sum = sum(scores[k] * weights.get(k, 1.0) for k in scores)
    composite    = _clamp(weighted_sum / total_weight) if total_weight > 0 else 0

    return {"scores": scores, "composite": composite}
