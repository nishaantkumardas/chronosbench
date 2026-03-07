'''Metal/MPS compute: particle simulation + matrix multiply.

Uses PyTorch MPS on macOS, CUDA where available, falls back to NumPy.
Note: GPU access is via PyTorch's MPS/CUDA backends, not raw Metal shaders.
'''

import time
import numpy as np

# Device detection

gpu_available = False
backend = "numpy"
device = None
torch = None

try:
    import torch as _torch

    if hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
        device = _torch.device("mps")
        gpu_available = True
        backend = "mps"
    elif _torch.cuda.is_available():
        device = _torch.device("cuda")
        gpu_available = True
        backend = "cuda"
    else:
        device = _torch.device("cpu")
        backend = "cpu"

    torch = _torch

except ImportError:
    pass

# Keep metal_available as a public alias so existing callers don't break.
metal_available = gpu_available

# Internal state

_last: dict = {"note": "not run"}

# NumPy fallback

def _run_numpy(duration: float) -> None:
    global _last

    # Allocate once, reuse every pass.
    a = np.random.rand(1024, 1024).astype("float32")
    b = np.random.rand(1024, 1024).astype("float32")

    passes = 0
    start = time.perf_counter()

    while time.perf_counter() - start < duration:
        np.dot(a, b, out=None)  # result intentionally discarded; dot forces compute
        passes += 1

    _last = {
        "passes": passes,
        "duration_s": round(time.perf_counter() - start, 3),
        "current_test": "Matrix Multiply (numpy)",
        "backend": "numpy",
    }


# GPU path (MPS / CUDA)

def _sync() -> None:
    """Flush the device command queue so timings are honest."""
    if backend == "mps":
        torch.mps.synchronize()
    elif backend == "cuda":
        torch.cuda.synchronize()


def _run_gpu(duration: float) -> None:
    global _last

    # Allocate buffers once outside the loop.
    mat_a = torch.randn((4096, 2048), device=device)
    mat_b = torch.randn((2048, 4096), device=device)
    particles = torch.randn((4_000_000, 3), device=device)
    velocity = torch.randn_like(particles)
    noise = torch.empty_like(velocity)

    passes = 0
    particle_steps_per_pass = 8
    current_test = "init"

    try:
        start = time.perf_counter()

        while time.perf_counter() - start < duration:
            # --- matrix multiply ---
            current_test = "Matrix Multiply"
            c = torch.matmul(mat_a, mat_b)

            # --- particle sim ---
            current_test = "Particle Simulation"
            for _ in range(particle_steps_per_pass):
                torch.randn(velocity.shape, out=noise, device=device)
                velocity.add_(noise, alpha=0.01)
                particles.add_(velocity, alpha=0.001)

            passes += 1

            # Sync every 10 passes — keeps the command queue full between flushes
            # rather than stalling the GPU on every single matmul readback.
            if passes % 10 == 0:
                _sync()

        _sync()
        elapsed = time.perf_counter() - start

        _last = {
            "passes": passes,
            "particle_steps": passes * particle_steps_per_pass,
            "duration_s": round(elapsed, 3),
            "current_test": current_test,
            "backend": backend,
        }

    except Exception as exc:
        _sync()
        _last = {"error": str(exc), "backend": backend}
        raise


# Public API

def run_metal_particle(duration: float = 30) -> None:
    """Run the GPU (or NumPy fallback) stress workload for *duration* seconds."""
    if not gpu_available or torch is None:
        _run_numpy(duration)
    else:
        _run_gpu(duration)


def get_last_metal_result() -> dict:
    return _last
