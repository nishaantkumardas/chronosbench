#!/usr/bin/env python3
"""
ChronosBench - Telemetry
Samples CPU, GPU, memory, and I/O in the background at a fixed interval.
macOS reads GPU stats via powermetrics (requires sudo) with ioreg fallback.
"""

import re
import subprocess
import sys
import threading
import time

import psutil


class TelemetryThread(threading.Thread):
    def __init__(self, interval: float = 1.0) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self._snapshot: dict = {}
        self._last_io = psutil.disk_io_counters()
        self._last_time = time.perf_counter()
        self._mem_total_gb = psutil.virtual_memory().total / 1024 ** 3

    def run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()

    def latest_snapshot(self) -> dict:
        with self._lock:
            return dict(self._snapshot)

    def _sample(self) -> None:
        snap: dict = {
            "cpu_percent":   psutil.cpu_percent(interval=None),
            "cpu_freq":      self._cpu_freq(),
            "cpu_temp":      None,
            "cpu_power_w":   None,
            "gpu_percent":   None,
            "gpu_temp":      None,
            "gpu_power_w":   None,
            "mem_total_gb":  round(self._mem_total_gb, 2),
            "mem_used_gb":   round(psutil.virtual_memory().used / 1024 ** 3, 2),
            "io_read_mb_s":  0.0,
            "io_write_mb_s": 0.0,
        }

        snap["io_read_mb_s"], snap["io_write_mb_s"] = self._io_rates()

        if sys.platform == "darwin":
            gpu_usage, gpu_temp, gpu_power = self._gpu_mac()
            snap["gpu_percent"] = gpu_usage
            snap["gpu_temp"]    = gpu_temp
            snap["gpu_power_w"] = gpu_power

        with self._lock:
            self._snapshot = snap

    def _cpu_freq(self) -> int | None:
        try:
            freq = psutil.cpu_freq()
            return round(freq.current) if freq else None
        except Exception:
            return None

    def _io_rates(self) -> tuple[float, float]:
        try:
            now_io   = psutil.disk_io_counters()
            now_time = time.perf_counter()
            dt = now_time - self._last_time

            if dt > 0:
                read_mb_s  = (now_io.read_bytes  - self._last_io.read_bytes)  / 1024 ** 2 / dt
                write_mb_s = (now_io.write_bytes - self._last_io.write_bytes) / 1024 ** 2 / dt
            else:
                read_mb_s = write_mb_s = 0.0

            self._last_io   = now_io
            self._last_time = now_time
            return round(read_mb_s, 2), round(write_mb_s, 2)
        except Exception:
            return 0.0, 0.0

    def _gpu_mac(self) -> tuple[float | None, float | None, float | None]:
        try:
            out = subprocess.check_output(
                ["sudo", "powermetrics", "--samplers", "smc", "--show-gpu", "--show-power"],
                stderr=subprocess.DEVNULL,
                timeout=3,
            ).decode("utf-8", errors="ignore")

            usage = self._re_float(r"GPU Active residency:\s*([\d\.]+)%", out)
            temp  = self._re_float(r"GPU die temperature:\s*([\d\.]+)",    out)
            power = self._re_float(r"GPU Power:\s*([\d\.]+)\s*W",          out)
            return usage, temp, power

        except Exception:
            pass

        # ioreg fallback — power only
        try:
            out   = subprocess.check_output(["ioreg", "-l"], stderr=subprocess.DEVNULL).decode("utf-8", errors="ignore")
            power = self._re_float(r"GPU Power.*?([\d\.]+)", out)
            return None, None, power
        except Exception:
            return None, None, None

    @staticmethod
    def _re_float(pattern: str, text: str) -> float | None:
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None
