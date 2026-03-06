"""CPU stress: a gauntlet of workloads designed to hit every execution unit.

What makes this different from every other benchmark:

  CHAOS MATRIX      — workers share a memory-mapped arena and write collision
                       hashes back into it after every BLAS op. Stresses the
                       OS cache, TLB, and inter-core coherency simultaneously.

  ENTROPY MILL      — pulls os.urandom at full speed, folds it through numpy,
                       and XORs a checksum into the arena. Hits the hardware
                       RNG / AES-NI path that no other benchmark exercises.

  MANDELBROT TURBINE — randomises the viewport and escape parameters each
                       pass. The branchy, data-dependent loop defeats branch
                       predictors in a way regular matrix work never does.

  RECURSIVE TENSOR FOLD — builds a binary tree of numpy arrays and folds
                       them with alternating einsum contractions. Pointer-
                       chasing on a heap structure; pure BLAS never does this.

  PRIME SIEVE RACE  — each worker sieves a unique number range with a
                       segmented Eratosthenes, then XORs results into the
                       shared arena to force cache-line invalidation.

  FFT               — 1M-point complex FFT feeding results into the arena.
"""

import math
import multiprocessing as mp
import multiprocessing.shared_memory as shm
import os
import struct
import time
from multiprocessing.connection import Connection

import numpy as np


# Shared memory arena — forces cross-core cache coherency traffic
# Uses shared_memory (picklable) instead of mmap so it survives spawn on macOS.

ARENA_BYTES = 4096  # one page; every worker hammers it


def _arena_xor(name: str, slot: int, value: int) -> None:
    """XOR a 4-byte slot in the named shared arena."""
    mem    = shm.SharedMemory(name=name)
    offset = (slot % (ARENA_BYTES // 4)) * 4
    current = struct.unpack_from("I", mem.buf, offset)[0]
    struct.pack_into("I", mem.buf, offset, (current ^ value) & 0xFFFFFFFF)
    mem.close()


# Worker 1 — Chaos Matrix (BLAS + arena writes)

def _chaos_matrix_worker(duration: float, conn: Connection, arena_name: str) -> None:
    size = 1024
    a = np.random.rand(size, size).astype("float32")
    b = np.random.rand(size, size).astype("float32")
    ops = 0
    end = time.perf_counter() + duration

    try:
        while time.perf_counter() < end:
            c = np.dot(a, b)
            _arena_xor(arena_name, ops % 64, int(c[0, 0] * 1e6) & 0xFFFFFFFF)
            # Rotate inputs so the CPU cannot cache the answer.
            a = c[:size, :size]
            b = np.roll(b, 1, axis=0)
            ops += 1
    except Exception as exc:
        conn.send({"error": str(exc), "type": "chaos_matrix", "ops": ops})
    else:
        conn.send({"ops": ops, "type": "chaos_matrix"})
    finally:
        conn.close()


# Worker 2 — Entropy Mill (RNG + AES-NI / hardware entropy path)

def _entropy_mill_worker(duration: float, conn: Connection, arena_name: str) -> None:
    ops = 0
    chunk = 65536
    end = time.perf_counter() + duration

    try:
        while time.perf_counter() < end:
            raw = os.urandom(chunk)
            arr = np.frombuffer(raw, dtype=np.uint8).astype(np.uint32)
            checksum = int(np.bitwise_xor.reduce(arr))
            _arena_xor(arena_name, (ops + 32) % 64, checksum)
            ops += 1
    except Exception as exc:
        conn.send({"error": str(exc), "type": "entropy_mill", "ops": ops})
    else:
        conn.send({"ops": ops, "type": "entropy_mill"})
    finally:
        conn.close()


# Worker 3 — Mandelbrot Turbine (branch-heavy, data-dependent)

def _mandelbrot_worker(duration: float, conn: Connection, arena_name: str) -> None:
    ops = 0
    width, height = 512, 512
    max_iter = 256
    end = time.perf_counter() + duration
    rng = np.random.default_rng()

    try:
        while time.perf_counter() < end:
            cx = rng.uniform(-2.5, 1.0)
            cy = rng.uniform(-1.25, 1.25)
            scale = rng.uniform(0.001, 0.5)

            x = np.linspace(cx, cx + scale, width, dtype=np.float64)
            y = np.linspace(cy, cy + scale, height, dtype=np.float64)
            C = x[np.newaxis, :] + 1j * y[:, np.newaxis]
            Z = np.zeros_like(C)
            M = np.zeros(C.shape, dtype=np.int32)

            for i in range(max_iter):
                mask = np.abs(Z) <= 2.0
                Z[mask] = Z[mask] ** 2 + C[mask]
                M[mask] += 1

            _arena_xor(arena_name, ops % 64, int(M.sum()) & 0xFFFFFFFF)
            ops += 1
    except Exception as exc:
        conn.send({"error": str(exc), "type": "mandelbrot", "ops": ops})
    else:
        conn.send({"ops": ops, "type": "mandelbrot"})
    finally:
        conn.close()


# Worker 4 — Recursive Tensor Fold (pointer-chasing heap stress)

def _build_tree(depth: int, size: int):
    if depth == 0:
        return np.random.rand(size, size).astype("float32")
    return [_build_tree(depth - 1, size), _build_tree(depth - 1, size)]


def _fold_tree(node) -> np.ndarray:
    if isinstance(node, np.ndarray):
        return node
    left = _fold_tree(node[0])
    right = _fold_tree(node[1])
    if left.shape == right.shape:
        return np.einsum("ij,jk->ik", left, right)
    return left + right[: left.shape[0], : left.shape[1]]


def _tensor_fold_worker(duration: float, conn: Connection, arena_name: str) -> None:
    ops = 0
    end = time.perf_counter() + duration

    try:
        while time.perf_counter() < end:
            tree = _build_tree(depth=3, size=64)
            result = _fold_tree(tree)
            _arena_xor(arena_name, (ops + 16) % 64, int(abs(result[0, 0]) * 1e6) & 0xFFFFFFFF)
            ops += 1
    except Exception as exc:
        conn.send({"error": str(exc), "type": "tensor_fold", "ops": ops})
    else:
        conn.send({"ops": ops, "type": "tensor_fold"})
    finally:
        conn.close()


# Worker 5 — Segmented Sieve Race (integer ALU + cache-line invalidation)

def _segmented_sieve(low: int, high: int) -> int:
    limit = int(math.isqrt(high)) + 1
    small_primes: list[int] = []
    sieve = bytearray([1]) * limit
    sieve[0] = sieve[1] = 0
    for i in range(2, limit):
        if sieve[i]:
            small_primes.append(i)
            for j in range(i * i, limit, i):
                sieve[j] = 0

    segment = bytearray([1]) * (high - low + 1)
    for p in small_primes:
        start = max(p * p, ((low + p - 1) // p) * p)
        for j in range(start, high + 1, p):
            segment[j - low] = 0
    if low <= 1:
        segment[0] = 0
        if low == 0 and len(segment) > 1:
            segment[1] = 0
    return sum(segment)


def _sieve_race_worker(duration: float, conn: Connection, arena_name: str) -> None:
    ops = 0
    found = 0
    base = (os.getpid() % 1000) * 10 ** 6 + 10 ** 7
    window = 50_000
    end = time.perf_counter() + duration

    try:
        low = base
        while time.perf_counter() < end:
            count = _segmented_sieve(low, low + window)
            found += count
            _arena_xor(arena_name, ops % 64, count & 0xFFFFFFFF)
            low += window
            ops += 1
    except Exception as exc:
        conn.send({"error": str(exc), "type": "sieve_race", "ops": ops, "found": found})
    else:
        conn.send({"ops": ops, "found": found, "type": "sieve_race"})
    finally:
        conn.close()


# Worker 6 — FFT (feeds into arena)

def _fft_worker(duration: float, conn: Connection, arena_name: str) -> None:
    buf = np.random.rand(1 << 20).astype("complex64")
    ops = 0
    end = time.perf_counter() + duration

    try:
        while time.perf_counter() < end:
            result = np.fft.fft(buf)
            _arena_xor(arena_name, (ops + 48) % 64, int(abs(result[0]) * 1e6) & 0xFFFFFFFF)
            ops += 1
    except Exception as exc:
        conn.send({"error": str(exc), "type": "fft", "ops": ops})
    else:
        conn.send({"ops": ops, "type": "fft"})
    finally:
        conn.close()


# Workload registry

_WORKERS = [
    (_chaos_matrix_worker, "Chaos Matrix"),
    (_entropy_mill_worker, "Entropy Mill"),
    (_mandelbrot_worker,   "Mandelbrot Turbine"),
    (_tensor_fold_worker,  "Recursive Tensor Fold"),
    (_sieve_race_worker,   "Prime Sieve Race"),
    (_fft_worker,          "FFT"),
]

_SUBTEST_NAMES = [name for _, name in _WORKERS]


# CPUStress

class CPUStress:
    def __init__(self) -> None:
        self._processes: list[mp.Process] = []
        self._conns: list[Connection] = []
        self._arena: shm.SharedMemory | None = None
        self._started_at: float = 0.0
        self._duration: float = 0.0

    def start(self, duration: float = 60) -> None:
        self._duration = duration
        self._started_at = time.perf_counter()
        self._arena = shm.SharedMemory(create=True, size=ARENA_BYTES)
        cpu_count = mp.cpu_count()

        for i in range(cpu_count):
            worker_fn, _ = _WORKERS[i % len(_WORKERS)]
            parent, child = mp.Pipe(duplex=False)
            p = mp.Process(
                target=worker_fn,
                args=(duration, child, self._arena.name),
                daemon=True,
            )
            p.start()
            child.close()
            self._processes.append(p)
            self._conns.append(parent)

    def stop(self) -> None:
        for p in self._processes:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
                p.join(timeout=1)
        if self._arena:
            self._arena.close()
            self._arena.unlink()

    def result(self) -> dict:
        breakdown: dict[str, int] = {}
        errors: list[str] = []

        for conn in self._conns:
            try:
                if conn.poll(timeout=2):
                    r = conn.recv()
                    if "error" in r:
                        errors.append(r["error"])
                    t = r.get("type", "unknown")
                    count = r.get("ops", 0) + r.get("found", 0)
                    breakdown[t] = breakdown.get(t, 0) + count
            except EOFError:
                pass
            finally:
                conn.close()

        return {
            "breakdown": breakdown,
            "total_ops": sum(breakdown.values()),
            "processes_used": len(self._processes),
            "errors": errors or None,
        }

    @property
    def current_subtest(self) -> str:
        if self._duration <= 0:
            return _SUBTEST_NAMES[0]
        elapsed = time.perf_counter() - self._started_at
        idx = min(
            int(elapsed / self._duration * len(_SUBTEST_NAMES)),
            len(_SUBTEST_NAMES) - 1,
        )
        return _SUBTEST_NAMES[idx]
