#!/usr/bin/env python3
"""ChronosBench X v1.1.1 - CLI Rich + Metal
Interactive prompts for platform, telemetry level, and intensity.
Shows live rich UI with current subtest names and telemetry.
"""
import platform, time, json, sy
from rich import print
from rich.panel import Panel
from rich.prompt import Prompt
from rich.live import Live
from rich.table import Table
from rich.console import Group
from rich.layout import Layout

from core.telemetry import TelemetryThread
from core.cpu_stress import CPUStress
from core.io_stress import IOStress
from core.mixed_load import MixedLoad
from core.metal_compute import metal_available
from utils.scoring import score_report
from utils.report import save_report

def choose_platform():
    system = platform.system()
    options = {"1":"macOS (Metal/MPS)", "2":"Windows (CUDA/DirectML)", "3":"Linux (OpenCL/Vulkan)"}
    print(Panel.fit(f"[bold cyan]ChronosBench X v1.1.1 — CLI Rich[/bold cyan]\nSelect platform:"))
    for k,v in options.items():
        print(f"{k}) {v}")
    default = "1" if system == 'Darwin' else ("2" if system=='Windows' else "3")
    choice = Prompt.ask("Choose platform", choices=list(options.keys()), default=default)
    return int(choice)

def choose_telemetry_level():
    print("Telemetry detail level:\n1) Full (live temps, usage, power)\n2) Minimal (only scores)")
    choice = Prompt.ask("Telemetry level", choices=["1","2"], default="1")
    return int(choice)

def choose_duration():
    choices = {"1":60, "2":180, "3":480}
    print("Select test intensity:\n1) Quick (~1 min)\n2) Standard (~3 min)\n3) Extended (~8 min)")
    choice = Prompt.ask("Choose intensity", choices=list(choices.keys()), default="2")
    return choices[choice]

def make_layout():
    layout = Layout()
    layout.split_column(Layout(name="header", size=3), Layout(name="body", ratio=1), Layout(name="footer", size=3))
    layout['body'].split_row(Layout(name='left'), Layout(name='right'))
    return layout

def update_panel(snap, phase, subtest, elapsed, total):
    header = Panel(f"[bold]ChronosBench X v1.1.1[/bold] — Phase: {phase} — {elapsed:.0f}/{total:.0f}s", style="bold white on blue")
    left = Table.grid()
    left.add_column()
    left.add_row(f"[bold]CPU:[/bold] {snap.get('cpu_percent','N/A')}% | {snap.get('cpu_temp','N/A')}°C | {snap.get('cpu_freq','N/A')} MHz")
    left.add_row(f"[bold]Memory:[/bold] {snap.get('mem_used_gb',0):.2f} / {snap.get('mem_total_gb',0):.2f} GB")
    left.add_row(f"[bold]Current Test:[/bold] {subtest}")
    right = Table.grid()
    right.add_column()
    right.add_row(f"[bold]GPU:[/bold] {snap.get('gpu_percent','N/A')}% | {snap.get('gpu_temp','N/A')}°C | {snap.get('gpu_power_w','N/A')} W")
    right.add_row(f"[bold]I/O:[/bold] R: {snap.get('io_read_mb_s',0):.2f} MB/s | W: {snap.get('io_write_mb_s',0):.2f} MB/s")
    body = Group(Panel(left, title='System'), Panel(right, title='GPU / I/O'))
    footer = Panel('[bold]Press Ctrl+C to abort. Safety cutoff at 95°C.[/bold]', style='red')
    return header, body, footer

def run_phase(live, tel, phase_name, module, phase_duration, total_duration, subtest, start_time):
    """Run a single benchmark phase with live telemetry updates."""
    t_phase_start = time.time()
    module.start(duration=phase_duration)

    while time.time() - t_phase_start < phase_duration:
        snap = tel.latest_snapshot()
        header, body, footer = update_panel(
            snap,
            phase_name,
            module.current_subtest if hasattr(module, "current_subtest") else subtest,
            time.time() - start_time,
            total_duration,
        )
        layout = make_layout()
        layout["header"].update(header)
        layout["body"]["left"].update(body)
        layout["body"]["right"].update(
            Panel(
                f"[bold green]Running: {phase_name}\n"
                f"Test: {module.current_subtest if hasattr(module, 'current_subtest') else subtest or 'N/A'}\n"
                f"Elapsed: {int(time.time() - t_phase_start)} s"
            )
        )
        layout["footer"].update(footer)
        live.update(layout)
        time.sleep(0.5)

    module.stop()


def main():
    plat = choose_platform()
    telemetry_level = choose_telemetry_level()
    duration = choose_duration()
    print(f"Starting test for {duration} seconds... (platform choice: {plat})\nMetal available: {metal_available}")
    tel = TelemetryThread()  # background telemetry (powermetrics)
    cpu = CPUStress()
    io = IOStress()
    mixed = MixedLoad()

    start_global = time.time()
    tel.start()
    try:
        start = time.time()
        with Live(refresh_per_second=4) as live:
            run_phase(live, tel, 'CPU Stress', cpu, duration//2, duration, cpu.current_subtest, start)
            run_phase(live, tel, 'I/O Stress', io, duration//4, duration, None, start)
            run_phase(live, tel, 'Mixed Thermal Sweep', mixed, duration//4, duration, None, start)

    except KeyboardInterrupt:
        print('\n[bold red]Aborted by user[/bold red]')
    finally:
        tel.stop()
        # gather results
        report = {
            'meta': {'platform': platform.system(), 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())},
            'results': {'cpu': cpu.result(), 'io': io.result(), 'mixed': mixed.result(), 'telemetry': tel.latest_snapshot()}
        }
        report['scores'] = score_report(report)
        save_report(report)
        print(Panel(f"Composite score: {report['scores']['composite']} / 2000", title='Result', style='bold green'))

if __name__ == '__main__':
    main()
