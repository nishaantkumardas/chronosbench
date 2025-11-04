# ChronosBench v1.0 — CLI + Metal

This package is a demanding CLI benchmark that:
- uses PyTorch MPS on macOS where available to exercise Metal,
- collects telemetry
- displays a live CLI UI showing current test, CPU/GPU, RAM usage %, IO throughput,
- cycles CPU subtests (Matrix Multiply, FFT, Prime Search) and a Mixed GPU particle+matrix workload,
- caps scores to 2000 and writes JSON/text reports into `reports/`.

Run (macOS, full telemetry):
  python3 main.py

Install dependencies:
  pip3 install -r requirements.txt

Notes:
- MPS (PyTorch) is used for Metal compute where available. If not installed, GPU test falls back.
- This tool intentionally stresses hardware.
....
