#!/usr/bin/env python3
"""ChronosBench — CLI entry point."""

import platform
import time

from rich import print
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.console import Group

from core.telemetry import TelemetryThread
from core.cpu_stress import CPUStress
from core.io_stress import IOStress
from core.mixed_load import MixedLoad
from core.metal_compute import gpu_available
from utils.scoring import score_report
from utils.report import save_report

VERSION = "1.1.1"


def choose_platform() -> int:
    system = platform.system()
    options = {"1": "macOS (Metal/MPS)", "2": "Windows (CUDA/DirectML)", "3": "Linux (OpenCL/Vulkan)"}
    print(Panel.fit(f"[bold cyan]ChronosBench X v{VERSION}[/bold cyan]\nSelect platform:"))
    for k, v in options.items():
        print(f"{k}) {v}")
    default = "1" if system == "Darwin" else ("2" if system == "Windows" else "3")
    return int(Prompt.ask("Choose platform", choices=list(options.keys()), default=default))


def choose_telemetry_level() -> int:
    print("Telemetry detail level:\n1) Full (live temps, usage, power)\n2) Minimal (only scores)")
    return int(Prompt.ask("Telemetry level", choices=["1", "2"], default="1"))


def choose_duration() -> int:
    choices = {"1": 60, "2": 180, "3": 480}
    print("Select test intensity:\n1) Quick (~1 min)\n2) Standard (~3 min)\n3) Extended (~8 min)")
    return choices[Prompt.ask("Choose intensity", choices=list(choices.keys()), default="2")]


def _make_layout() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body",   ratio=1),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(Layout(name="left"), Layout(name="right"))
    return layout


def _render(snap: dict, phase: str, subtest: str, elapsed: float, total: float, phase_elapsed: int) -> Layout:
    layout = _make_layout()

    header = Panel(
        f"[bold]ChronosBench X v{VERSION}[/bold] — {phase} — {elapsed:.0f}/{total:.0f}s",
        style="bold white on blue",
    )

    left = Table.grid()
    left.add_column()
    left.add_row(f"[bold]CPU:[/bold] {snap.get('cpu_percent', 'N/A')}% | {snap.get('cpu_temp', 'N/A')}°C | {snap.get('cpu_freq', 'N/A')} MHz")
    left.add_row(f"[bold]Memory:[/bold] {snap.get('mem_used_gb', 0):.2f} / {snap.get('mem_total_gb', 0):.2f} GB")
    left.add_row(f"[bold]Subtest:[/bold] {subtest}")

    right = Table.grid()
    right.add_column()
    right.add_row(f"[bold]GPU:[/bold] {snap.get('gpu_percent', 'N/A')}% | {snap.get('gpu_temp', 'N/A')}°C | {snap.get('gpu_power_w', 'N/A')} W")
    right.add_row(f"[bold]I/O:[/bold] R: {snap.get('io_read_mb_s', 0):.2f} MB/s | W: {snap.get('io_write_mb_s', 0):.2f} MB/s")

    footer = Panel("[bold]Ctrl+C to abort — thermal cutoff at 95°C[/bold]", style="red")

    layout["header"].update(header)
    layout["body"]["left"].update(Group(Panel(left, title="System"), Panel(right, title="GPU / I/O")))
    layout["body"]["right"].update(Panel(f"[bold green]{phase}[/bold green]\n{subtest}\nElapsed: {phase_elapsed}s"))
    layout["footer"].update(footer)
    return layout


def _current_subtest(module, fallback: str) -> str:
    sub = getattr(module, "current_subtest", fallback)
    return sub() if callable(sub) else (sub or fallback)


def run_phase(live: Live, tel: TelemetryThread, phase_name: str, module, phase_duration: int, total_duration: int, start_time: float) -> None:
    phase_start = time.perf_counter()
    module.start(duration=phase_duration)

    while time.perf_counter() - phase_start < phase_duration:
        elapsed      = time.perf_counter() - start_time
        phase_elapsed = int(time.perf_counter() - phase_start)
        subtest      = _current_subtest(module, phase_name)
        snap         = tel.latest_snapshot()
        live.update(_render(snap, phase_name, subtest, elapsed, total_duration, phase_elapsed))
        time.sleep(0.25)

    module.stop()


def main() -> None:
    plat            = choose_platform()
    telemetry_level = choose_telemetry_level()
    duration        = choose_duration()

    print(f"\nGPU available: [bold]{'yes' if gpu_available else 'no'}[/bold]  |  platform choice: {plat}\n")

    tel   = TelemetryThread()
    cpu   = CPUStress()
    io    = IOStress()
    mixed = MixedLoad()

    tel.start()
    start = time.perf_counter()

    try:
        with Live(refresh_per_second=4) as live:
            run_phase(live, tel, "CPU Stress",          cpu,   duration // 2, duration, start)
            run_phase(live, tel, "I/O Stress",          io,    duration // 4, duration, start)
            run_phase(live, tel, "Mixed Thermal Sweep", mixed, duration // 4, duration, start)

    except KeyboardInterrupt:
        print("\n[bold red]Aborted.[/bold red]")

    finally:
        tel.stop()
        report = {
            "meta": {
                "platform":  platform.system(),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "version":   VERSION,
            },
            "results": {
                "cpu":       cpu.result(),
                "io":        io.result(),
                "mixed":     mixed.result(),
                "telemetry": tel.latest_snapshot(),
            },
        }
        report["scores"] = score_report(report)
        save_report(report)
        print(Panel(f"Composite score: [bold]{report['scores']['composite']}[/bold] / 2000", title="Result", style="bold green"))


if __name__ == "__main__":
    main()
