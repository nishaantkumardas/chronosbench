# ChronosBench 

Stresses your CPU and Metal GPU from the terminal. Live utilization bars while it runs. JSON report when it's done.

```
pip install -r requirements.txt
python main.py
```

---

### Why this exists

Every benchmark I tried either skipped the GPU entirely, required a GUI, or couldn't be scripted. So I built this.

---

### How it works

Six worker processes run concurrently, each targeting a different execution unit — BLAS matmul, FFT, Mandelbrot (branch-heavy, defeats the predictor), recursive einsum folds, a segmented prime sieve, and an entropy mill that pulls `os.urandom` at full speed to hit the AES-NI path.

The twist: all six share a memory-mapped page and XOR collision hashes into it after every op. The point isn't peak FLOPS — it's the cache coherency traffic between cores. That's where chips actually differ under load.

GPU runs via PyTorch MPS on macOS. Falls back to CUDA, then NumPy.

---

### Phases

| Phase | What runs | Duration |
|---|---|---|
| CPU Stress | 6 worker types across all logical cores | 50% |
| I/O Stress | Sequential flood + random seeks + metadata churn + fsync gauntlet | 25% |
| Mixed Thermal Sweep | Everything simultaneously | 25% |

The I/O phase writes random bytes, not zeros — modern NVMe controllers compress repetitive data and lie about throughput.

---

### Scoring

Component scores out of 2000, weighted composite. GPU is excluded from the composite if not detected rather than penalizing the score. Baselines are calibrated against M1/NVMe — tune `scoring.py` if your numbers look off.

---

### Requirements

- Python 3.11+
- macOS, Linux, or Windows
- `sudo` on macOS for GPU telemetry via `powermetrics`

---

### Output

```
reports/chronosbenchx_2026-03-06_14-22-01.json
reports/chronosbenchx_2026-03-06_14-22-01.txt
```

Mostly so you can prove to your future self that this laptop used to be fast.
