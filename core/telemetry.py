#!/usr/bin/env python3
"""
ChronosBench X - Telemetry system
Collects CPU, GPU, memory, I/O and thermal metrics in the background.
Supports macOS (Metal/MPS), Windows (CUDA/DirectML), Linux (OpenCL/Vulkan).
"""

import sys
import time
import threadin
import psutil
import subprocess
import re
import platform


class TelemetryThread(threading.Thread):
    def __init__(self, interval: float = 1.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_event = threading.Event()
        # last sampled values
        self.cpu_percent = 0.0
        self.cpu_temp = None
        self.cpu_freq = None
        self.cpu_power_w = None
        self.gpu_percent = None
        self.gpu_temp = None
        self.gpu_power_w = None
        self.mem_total_gb = psutil.virtual_memory().total / (1024**3)
        self.mem_used_gb = 0.0
        self.io_read_mb_s = 0.0
        self.io_write_mb_s = 0.0
        self.last_io = psutil.disk_io_counters()
        self.last_time = time.time()
        self.latest = {}

    # ------------------------------------------------------------------ #
    # macOS GPU telemetry (Metal / M-series)
    # ------------------------------------------------------------------ #
    def _get_gpu_stats_mac(self):
        gpu_usage = gpu_temp = gpu_power = None

        # try powermetrics first
        try:
            out = subprocess.check_output(
                ["sudo", "powermetrics", "--samplers", "smc", "--show-gpu", "--show-power"],
                stderr=subprocess.DEVNULL,
                timeout=3,
            ).decode("utf-8", errors="ignore")

            usage_match = re.search(r"GPU Active residency:\s*([\d\.]+)%", out)
            temp_match = re.search(r"GPU die temperature:\s*([\d\.]+)", out)
            power_match = re.search(r"GPU Power:\s*([\d\.]+)\s*W", out)

            if usage_match:
                gpu_usage = float(usage_match.group(1))
            if temp_match:
                gpu_temp = float(temp_match.group(1))
            if power_match:
                gpu_power = float(power_match.group(1))

        except Exception:
            # fallback to ioreg if powermetrics fails
            try:
                ioreg = subprocess.check_output(["ioreg", "-l"], stderr=subprocess.DEVNULL).decode(
                    "utf-8", errors="ignore"
                )
                power_match = re.search(r"GPU Power.*?(\d+\.\d+)", ioreg)
                if power_match:
                    gpu_power = float(power_match.group(1))
            except Exception:
                pass

        return gpu_usage, gpu_temp, gpu_power

    # ------------------------------------------------------------------ #
    # main sampler
    # ------------------------------------------------------------------ #
    def run(self):
        while not self.stop_event.is_set():
            self.sample_once()
            time.sleep(self.interval)

    def sample_once(self):
        # CPU %
        self.cpu_percent = psutil.cpu_percent(interval=None)

        # CPU frequency
        try:
            freq = psutil.cpu_freq()
            if freq:
                self.cpu_freq = round(freq.current)
        except Exception:
            self.cpu_freq = None

        # Memory
        vm = psutil.virtual_memory()
        self.mem_used_gb = vm.used / (1024**3)

        # Disk I/O MB/s
        try:
            now_io = psutil.disk_io_counters()
            now_time = time.time()
            delta_t = now_time - self.last_time
            if delta_t > 0:
                self.io_read_mb_s = (now_io.read_bytes - self.last_io.read_bytes) / (1024**2) / delta_t
                self.io_write_mb_s = (now_io.write_bytes - self.last_io.write_bytes) / (1024**2) / delta_t
            self.last_io, self.last_time = now_io, now_time
        except Exception:
            pass

        # macOS GPU telemetry
        if sys.platform == "darwin":
            g_usage, g_temp, g_power = self._get_gpu_stats_mac()
            self.gpu_percent = g_usage
            self.gpu_temp = g_temp
            self.gpu_power_w = g_power

        # package snapshot
        self.latest = {
            "cpu_percent": self.cpu_percent,
            "cpu_temp": self.cpu_temp,
            "cpu_freq": self.cpu_freq,
            "cpu_power_w": self.cpu_power_w,
            "gpu_percent": self.gpu_percent,
            "gpu_temp": self.gpu_temp,
            "gpu_power_w": self.gpu_power_w,
            "mem_total_gb": self.mem_total_gb,
            "mem_used_gb": self.mem_used_gb,
            "io_read_mb_s": self.io_read_mb_s,
            "io_write_mb_s": self.io_write_mb_s,
        }

    # ------------------------------------------------------------------ #
    def latest_snapshot(self):
        """Return most recent metrics dict."""
        return self.latest

    def stop(self):
        self.stop_event.set()
