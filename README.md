# ChronosBench X v1.1.1 — CLI Rich + Metal (Best-effort)

This package is a demanding CLI benchmark that:
- uses PyTorch MPS on macOS where available to exercise Metal,
- collects telemetry (powermetrics recommended, sudo advised),
- displays a live rich CLI UI showing current test, CPU/GPU temps, usage %, RAM in GB, IO throughput,
- cycles CPU subtests (Matrix Multiply, FFT, Prime Search) and a Mixed GPU particle+matrix workload,
- caps scores to 1000 and writes JSON/text reports into `reports/`.

Run (macOS, full telemetry):
  sudo python3 main.py

Install dependencies:
  pip install -r requirements.txt

Notes:
- MPS (PyTorch) is used for Metal compute where available. If not installed, GPU test falls back.
- powermetrics provides accurate temps/power; it requires sudo. Without sudo telemetry will be limited.
- This tool intentionally stresses hardware — use at your own risk.
