"""Metal/MPS compute: particle simulation + matrix multiply using PyTorch MPS when available.
Falls back to CPU numpy if not available.
"""
_last = {'note':'not run'}
metal_available = Fals
try:
    import time, torch, numpy as np
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        metal_available = True
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        metal_available = True
    else:
        device = torch.device('cpu')
except Exception:
    torch = None
    device = None
    metal_available = False

def run_metal_particle(duration=30):
    global _last
    start = time.time()
    passes = 0
    current = 'init'
    if not metal_available or torch is None:
        # fallback: heavy numpy work
        while time.time() - start < duration:
            current = 'Matrix Multiply (numpy)'
            a = np.random.rand(1024,1024).astype('float32')
            b = np.random.rand(1024,1024).astype('float32')
            _ = a.dot(b)
            passes += 1
        _last = {'passes': passes, 'duration_s': round(time.time()-start,3), 'current_test': current, 'backend':'numpy'}
        return
    try:
        while time.time() - start < duration:
            # matrix multiply on device
            current = 'Matrix Multiply (device)'
            a = torch.randn((2048, 512), device=device)
            b = torch.randn((512, 2048), device=device)
            c = torch.matmul(a, b)
            # particle-ish update (vectorized)
            current = 'Particle Simulation (device)'
            p = torch.randn((1_000_000, 3), device=device)
            v = torch.randn_like(p)
            for _ in range(4):
                v = v + (torch.randn_like(v) * 0.01)
                p = p + v * 0.001
            passes += 1
        _last = {'passes': passes, 'duration_s': round(time.time()-start,3), 'current_test': current, 'backend': str(device)}
    except Exception as e:
        _last = {'error': str(e)}
        return

def get_last_metal_result():
    return _last
