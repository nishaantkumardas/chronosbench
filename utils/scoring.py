import mat

def score_report(report):
    results = report.get("results", {})
    scores = {}

    # CPU score
    if "cpu" in results:
        cpu = results["cpu"]
        mat_score = 1.0 / max(cpu.get("matmul_duration_s", 1e-3), 1e-3)
        part_score = 1.0 / max(cpu.get("particle_sim_s", 1e-3), 1e-3)
        scores["cpu"] = min(int((mat_score + part_score) * 100), 2000)

    # GPU score
    if "mixed" in results:
        gpu_dur = results["mixed"].get("gpu_duration_s", 1.0)
        scores["gpu"] = min(int(1000 / gpu_dur), 2000)

    # I/O score
    if "io" in results:
        io = results["io"]
        r = io.get("read_mb_s", 0)
        w = io.get("write_mb_s", 0)
        scores["io"] = min(int((r + w) / 20), 2000)

    # Responsiveness / telemetry
    if "telemetry" in results:
        tel = results["telemetry"]
        cpu_temp = tel.get("cpu_temp", 50) or 50
        cpu_percent = tel.get("cpu_percent", 50)
        therm_eff = 100 - abs(cpu_temp - 60)
        resp_score = (cpu_percent + therm_eff) / 2
        scores["responsiveness"] = min(int(resp_score * 10), 2000)

    # Compute composite after individual scores exist
    if scores:
        composite = sum(scores.values()) / len(scores)
        composite = min(int(composite), 2000)
    else:
        composite = 0

    return {
        "scores": scores,
        "composite": composite
    }
