import numpy as np

def hilbert_phase_slope(times: np.ndarray, series: np.ndarray) -> float:
    N = len(series)
    if N < 4:
        return float('nan')
    S = np.fft.fft(series)
    H = np.zeros(N)
    if N % 2 == 0:
        H[0] = 1.0
        H[1:N//2] = 2.0
        H[N//2] = 1.0
    else:
        H[0] = 1.0
        H[1:(N+1)//2] = 2.0
    a = np.fft.ifft(S * H)
    phi = np.unwrap(np.angle(a))
    m, _ = np.polyfit(times, phi, 1)
    return float(abs(m))


def project_sin_cos(z: np.ndarray, f: np.ndarray, n: int, Lz: float):
    s = np.sin(np.pi * n * z / Lz)
    c = np.cos(np.pi * n * z / Lz)
    s_norm = float(np.sum(s * s)) + 1e-300
    c_norm = float(np.sum(c * c)) + 1e-300
    p_sin = float(np.sum(f * s) / s_norm)
    p_cos = float(np.sum(f * c) / c_norm)
    return p_sin, p_cos
