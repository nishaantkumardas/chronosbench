"""Save JSON and human-readable report."""
import os, json, time
def save_report(report):
    os.makedirs('reports', exist_ok=True)
    ts = time.strftime('%Y-%m-%d_%H-%M-%S')
    fname = f'reports/chronosbenchx_{ts}.json'
    with open(fname,'w',encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    with open(fname.replace('.json','.txt'),'w',encoding='utf-8') as f:
        f.write('ChronosBench X v1.1.1 Report\n'
        f.write(json.dumps(report, indent=2))
    print(f'Report saved to {fname}')
