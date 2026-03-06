"""I/O stress: hammers the disk subsystem with patterns no benchmark uses.

Four concurrent workers, each targeting a different failure mode:

  SEQUENTIAL FLOOD   — writes a single 1 GB file in large blocks, then reads
                       it back. Measures raw sustained throughput.

  RANDOM SEEK STORM  — opens the same file and issues thousands of small
                       random-offset reads/writes. Kills SSD write amplification
                       and spins up latency on HDDs.

  METADATA CHURN     — creates and deletes thousands of tiny files per second.
                       Stresses the filesystem's directory entry cache and inode
                       allocator, which sequential tests never touch.

  FSYNC GAUNTLET     — writes 4 KB blocks and calls fsync() after every write.
                       Forces the drive's write cache to flush each time.
                       Most SSDs throttle hard under this; most benchmarks skip it.
"""

import os
import random
import shutil
import struct
import tempfile
import threading
import time
from pathlib import Path


BLOCK_LARGE  = 4 * 1024 * 1024   # 4 MB — sequential flood
BLOCK_SMALL  = 4 * 1024          # 4 KB — fsync gauntlet + random seeks
FILE_SIZE_MB = 512


class IOStress:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._metrics: dict = {
            "write_bytes": 0,
            "read_bytes": 0,
            "fsyncs": 0,
            "metadata_ops": 0,
            "random_ops": 0,
            "duration_s": 0.0,
        }
        self._tempdir: str | None = None
        self._start_time: float = 0.0

    def _add(self, key: str, value: int) -> None:
        with self._lock:
            self._metrics[key] += value

    def _sequential_flood(self, path: Path, duration: float) -> None:
        end = time.perf_counter() + duration
        block = os.urandom(BLOCK_LARGE)  # random bytes — defeats compression on NVMe
        blocks_per_file = (FILE_SIZE_MB * 1024 * 1024) // BLOCK_LARGE

        while time.perf_counter() < end and not self._stop.is_set():
            with path.open("wb") as f:
                for _ in range(blocks_per_file):
                    f.write(block)
                    self._add("write_bytes", BLOCK_LARGE)

            with path.open("rb") as f:
                while chunk := f.read(BLOCK_LARGE):
                    self._add("read_bytes", len(chunk))

    def _random_seek_storm(self, path: Path, duration: float) -> None:
        # Needs a pre-existing file — wait briefly if sequential hasn't written yet.
        deadline = time.perf_counter() + 5.0
        while not path.exists() and time.perf_counter() < deadline:
            time.sleep(0.1)
        if not path.exists():
            return

        file_size = path.stat().st_size
        end = time.perf_counter() + duration
        rng = random.Random()

        with path.open("r+b") as f:
            while time.perf_counter() < end and not self._stop.is_set():
                offset = rng.randint(0, max(0, file_size - BLOCK_SMALL))
                f.seek(offset)
                if rng.random() < 0.5:
                    f.write(os.urandom(BLOCK_SMALL))
                    self._add("write_bytes", BLOCK_SMALL)
                else:
                    data = f.read(BLOCK_SMALL)
                    self._add("read_bytes", len(data))
                self._add("random_ops", 1)

    def _metadata_churn(self, base: Path, duration: float) -> None:
        churn_dir = base / "meta_churn"
        churn_dir.mkdir(exist_ok=True)
        end = time.perf_counter() + duration
        idx = 0

        while time.perf_counter() < end and not self._stop.is_set():
            batch: list[Path] = []
            for _ in range(256):
                p = churn_dir / f"{idx}.tmp"
                p.write_bytes(struct.pack("Q", idx))  # 8-byte file
                batch.append(p)
                idx += 1
            for p in batch:
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
            self._add("metadata_ops", len(batch) * 2)  # create + delete

    def _fsync_gauntlet(self, base: Path, duration: float) -> None:
        path = base / "fsync_target.bin"
        end = time.perf_counter() + duration
        block = os.urandom(BLOCK_SMALL)

        with path.open("wb") as f:
            while time.perf_counter() < end and not self._stop.is_set():
                f.write(block)
                f.flush()
                os.fsync(f.fileno())
                self._add("write_bytes", BLOCK_SMALL)
                self._add("fsyncs", 1)

        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def start(self, duration: float = 60) -> None:
        self._stop.clear()
        self._start_time = time.perf_counter()
        self._tempdir = tempfile.mkdtemp(prefix="chronos_io_")
        base = Path(self._tempdir)
        flood_path = base / "flood.bin"

        workers = [
            threading.Thread(target=self._sequential_flood,  args=(flood_path, duration), daemon=True),
            threading.Thread(target=self._random_seek_storm, args=(flood_path, duration), daemon=True),
            threading.Thread(target=self._metadata_churn,    args=(base, duration),       daemon=True),
            threading.Thread(target=self._fsync_gauntlet,    args=(base, duration),       daemon=True),
        ]

        self._threads = workers
        for t in workers:
            t.start()

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=3)
        self._metrics["duration_s"] = time.perf_counter() - self._start_time
        if self._tempdir:
            try:
                shutil.rmtree(self._tempdir)
            except Exception:
                pass

    def result(self) -> dict:
        dur = max(self._metrics["duration_s"], 1.0)
        rb  = self._metrics["read_bytes"]
        wb  = self._metrics["write_bytes"]
        return {
            "read_mb_s":    round(rb / 1024 / 1024 / dur, 2),
            "write_mb_s":   round(wb / 1024 / 1024 / dur, 2),
            "fsyncs":       self._metrics["fsyncs"],
            "metadata_ops": self._metrics["metadata_ops"],
            "random_ops":   self._metrics["random_ops"],
            "duration_s":   round(dur, 3),
        }
