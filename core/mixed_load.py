"""Mixed load: concurrently stress CPU + GPU (MPS/Metal if available) + I/O.
"""
import threading, time
from core import cpu_stress, io_stress
try:
    from core.metal_compute import run_metal_particle, get_last_metal_result, metal_availabl
except Exception:
    def run_metal_particle(d): time.sleep(d)
    def get_last_metal_result(): return {'note':'metal not available'}
    metal_available = False

class MixedLoad:
    def __init__(self):
        self._cpu = cpu_stress.CPUStress()
        self._io = io_stress.IOStress()
        self._gpu_thread = None

    def start(self, duration=60):
        self._cpu.start(duration=duration)
        self._io.start(duration=duration)
        if metal_available:
            self._gpu_thread = threading.Thread(target=run_metal_particle, args=(duration,), daemon=True)
            self._gpu_thread.start()

    def stop(self):
        self._io.stop()
        self._cpu.stop()
        if self._gpu_thread and self._gpu_thread.is_alive():
            self._gpu_thread.join(timeout=1)

    def result(self):
        res = {'cpu': self._cpu.result(), 'io': self._io.result()}
        if metal_available:
            res['gpu'] = get_last_metal_result()
        else:
            res['gpu'] = {'note':'metal not available'}
        return res

    def current_subtest(self):
        if metal_available:
            return get_last_metal_result().get('current_test','Metal Particle+Matrix')
        return 'Mixed: CPU+I/O'
