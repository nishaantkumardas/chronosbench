"""CPU stress with multiple subtests: Matrix Multiply, FFT, Prime Search.
Each worker is a process performing one workload for the requested duration.
"""
import multiprocessing as mp, time, numpy as np, mat
from multiprocessing import Pipe

def _matrix_worker(duration, conn):
    end = time.time() + duration
    ops = 0
    try:
        size = 2048
        while time.time() < end:
            a = np.random.rand(size, size).astype('float32')
            b = np.random.rand(size, size).astype('float32')
            _ = a.dot(b)
            ops += 1
    except Exception as e:
        conn.send({'error': str(e)})
    finally:
        conn.send({'ops': ops, 'type': 'matrix'})
        conn.close()

def _fft_worker(duration, conn):
    end = time.time() + duration
    ops = 0
    try:
        while time.time() < end:
            a = np.random.rand(1<<22).astype('complex64')
            np.fft.fft(a)
            ops += 1
    except Exception as e:
        conn.send({'error': str(e)})
    finally:
        conn.send({'ops': ops, 'type': 'fft'})
        conn.close()

def _prime_worker(duration, conn):
    end = time.time() + duration
    found = 0
    try:
        n = 10**6
        while time.time() < end:
            for x in range(n, n+5000):
                is_prime = True
                r = int(math.sqrt(x))
                for d in range(2, r+1):
                    if x % d == 0:
                        is_prime = False
                        break
                if is_prime:
                    found += 1
            n += 5000
    except Exception as e:
        conn.send({'error': str(e)})
    finally:
        conn.send({'found': found, 'type': 'prime'})
        conn.close()

class CPUStress:
    def __init__(self):
        self.processes = []
        self._parent_conns = []

    def _start_worker(self, target, duration):
        parent, child = Pipe()
        p = mp.Process(target=target, args=(duration, child))
        p.start()
        self.processes.append(p)
        self._parent_conns.append(parent)

    def start(self, duration=60):
        cpu_count = mp.cpu_count()
        per_worker = max(1, duration // 3)
        for i in range(cpu_count):
            if i % 3 == 0:
                self._start_worker(_matrix_worker, per_worker*3)
            elif i % 3 == 1:
                self._start_worker(_fft_worker, per_worker*3)
            else:
                self._start_worker(_prime_worker, per_worker*3)

    def stop(self):
        for p in self.processes:
            p.join(timeout=1)

    def result(self):
        res = {'cpu_ops': 0, 'breakdown': {}}
        for c in self._parent_conns:
            try:
                r = c.recv()
                t = r.get('type', 'unknown')
                res['breakdown'][t] = res['breakdown'].get(t, 0) + (r.get('ops') or r.get('found') or 0)
            except Exception:
                pass
        res['processes_used'] = len(self.processes) if self.processes else mp.cpu_count()
        return res

    def current_subtest(self):
        t = int(time.time()) % 3
        return ['Matrix Multiply','FFT','Prime Search'][t]
