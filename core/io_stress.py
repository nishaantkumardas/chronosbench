"""I/O stress: write and read several large files repeatedly to saturate disk.
"""
import tempfile, os, time, threading, shuti

class IOStress:
    def __init__(self):
        self._stop = False
        self._thread = None
        self._metrics = {'write_bytes':0,'read_bytes':0,'duration_s':0}

    def _worker(self, duration):
        tempdir = tempfile.mkdtemp(prefix='chronos_io_')
        try:
            start = time.time()
            block = b'0' * (1024*1024)  # 1 MB blocks
            fname = os.path.join(tempdir, 'bigfile.bin')
            while time.time() - start < duration and not self._stop:
                with open(fname, 'wb') as f:
                    for i in range(1024):
                        f.write(block)
                        self._metrics['write_bytes'] += len(block)
                with open(fname, 'rb') as f:
                    while f.read(1024*1024):
                        self._metrics['read_bytes'] += 1024*1024
            self._metrics['duration_s'] = time.time() - start
        finally:
            try:
                shutil.rmtree(tempdir)
            except Exception:
                pass

    def start(self, duration=60):
        self._stop = False
        self._thread = threading.Thread(target=self._worker, args=(duration,), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True
        if self._thread:
            self._thread.join(timeout=2)

    def result(self):
        rb = self._metrics.get('read_bytes',0)
        wb = self._metrics.get('write_bytes',0)
        dur = max(1.0, self._metrics.get('duration_s',1.0))
        return {'read_mb_s': rb/(1024*1024)/dur, 'write_mb_s': wb/(1024*1024)/dur, 'duration_s': dur}
