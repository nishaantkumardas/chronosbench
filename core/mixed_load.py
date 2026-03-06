"""Mixed load: CPU + GPU (MPS/Metal) + I/O running simultaneously.

The point isn't to measure each subsystem in isolation — it's to see how
they degrade each other under real thermal and memory-bandwidth pressure.
CPU and I/O fight over the memory bus. GPU competes for power headroom.
That interaction is what this phase captures.
"""

import threading
import time

from core import cpu_stress, io_stress

try:
    from core.metal_compute import (
        get_last_metal_result,
        gpu_available,
        run_metal_particle,
    )
except Exception:
    gpu_available = False

    def run_metal_particle(d: float) -> None:
        time.sleep(d)

    def get_last_metal_result() -> dict:
        return {"note": "gpu not available"}


class MixedLoad:
    def __init__(self) -> None:
        self._cpu = cpu_stress.CPUStress()
        self._io  = io_stress.IOStress()
        self._gpu_thread: threading.Thread | None = None
        self._started_at: float = 0.0
        self._duration: float = 0.0

    def start(self, duration: float = 60) -> None:
        self._duration = duration
        self._started_at = time.perf_counter()

        self._cpu.start(duration=duration)
        self._io.start(duration=duration)

        if gpu_available:
            self._gpu_thread = threading.Thread(
                target=run_metal_particle,
                args=(duration,),
                daemon=True,
            )
            self._gpu_thread.start()

    def stop(self) -> None:
        self._io.stop()
        self._cpu.stop()
        if self._gpu_thread and self._gpu_thread.is_alive():
            self._gpu_thread.join(timeout=3)

    def result(self) -> dict:
        return {
            "cpu": self._cpu.result(),
            "io":  self._io.result(),
            "gpu": get_last_metal_result() if gpu_available else {"note": "gpu not available"},
        }

    @property
    def current_subtest(self) -> str:
        if gpu_available:
            return get_last_metal_result().get("current_test", "Mixed: CPU+GPU+I/O")
        return "Mixed: CPU+I/O"
