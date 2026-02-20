import time
import numpy as np
import matplotlib.pyplot as plt
import serial

# =========================== USER PARAMETERS ===========================
PORT = "COM4"                 # Must match Windows Device Manager (Ports -> COMx)
BAUD = 1_000_000              # Must match Serial.begin(...) in Arduino (fast_reader.ino)

INTERVAL_US = 52              # Nominal Arduino sampling interval in microseconds (calibrate this!)
DT_S = INTERVAL_US * 1e-6     # seconds

# ADC counts (0..65535) -> voltage (0..3.3 V), as you confirmed in another chat
ADC_MAX = 65535
V_REF = 3.3                   # Volts

# Data length
N_POINTS_KEEP = 500_000       # Samples kept for plotting + autocorrelation
ACQUIRE_FACTOR = 1.2          # Acquire extra samples, then keep only the last N_POINTS_KEEP

# Serial read performance / robustness
BLOCK_SAMPLES = 2048          # Read this many samples per block (2 bytes per sample)
SERIAL_TIMEOUT_S = 2.0        # Seconds
POST_OPEN_DELAY_S = 0.0       # Optional delay after opening port

# Optional alignment check
ENABLE_ALIGNMENT_CHECK = True
ALIGN_CHECK_SAMPLES = 4000    # Used only for alignment scoring
ALIGN_REL_MARGIN = 0.03       # Require ~3% improvement to switch to offset=1

# Plot controls
PLOT_STRIDE = 10              # Plot every Nth point (downsampling for speed)
SMOOTH_WINDOW_SAMPLES = 51    # Centered moving average for PLOTTING ONLY (odd recommended; 1 disables)
SMOOTH_PAD_MODE = "edge"      # "edge" or "reflect"
Y_MARGIN_FACTOR = 1.2         # Expand y-limits away from zero

# Autocorrelation
MAX_LAG = 100_000             # Max lag in samples
SAVE_PREFIX = "colloid_3"     # Output file prefix
# ======================================================================


def auto_ylim(y, factor=1.2):
    """
    Choose y-limits based on min/max with sign-aware expansion:
      - ymax = max * factor if max > 0 else max / factor
      - ymin = min * factor if min < 0 else min / factor
    This matches your rule: "use min/max *1.2, but if min is positive use /1.2".
    """
    y = np.asarray(y, dtype=np.float64)
    y_min = float(np.min(y))
    y_max = float(np.max(y))

    if np.isclose(y_min, y_max):
        delta = abs(y_min) * 0.1
        if delta == 0:
            delta = 1e-3
        return y_min - delta, y_max + delta

    y_low = (y_min * factor) if (y_min < 0) else (y_min / factor)
    y_high = (y_max * factor) if (y_max > 0) else (y_max / factor)

    if y_low >= y_high:
        mid = 0.5 * (y_min + y_max)
        span = (y_max - y_min) * factor
        return mid - 0.5 * span, mid + 0.5 * span

    return y_low, y_high


def moving_average_centered(x, window, pad_mode="edge"):
    """
    Centered moving average (boxcar) for visualization.
    Output has the same length as input due to padding.

    IMPORTANT:
    - Use this ONLY for plotting.
    - Do NOT use the smoothed signal for autocorrelation (it biases dynamics).
    """
    x = np.asarray(x, dtype=np.float64)

    if window <= 1:
        return x.copy()
    if window % 2 == 0:
        raise ValueError("SMOOTH_WINDOW_SAMPLES should be odd (e.g. 11, 21, 51).")
    if window > x.size:
        raise ValueError("SMOOTH_WINDOW_SAMPLES is larger than the signal length.")

    pad = window // 2
    x_pad = np.pad(x, (pad, pad), mode=pad_mode)
    c = np.cumsum(np.insert(x_pad, 0, 0.0))
    y = (c[window:] - c[:-window]) / window
    return y


def read_exactly(ser, n_bytes):
    """Read exactly n_bytes from serial (looping until full or timeout)."""
    buf = bytearray()
    while len(buf) < n_bytes:
        chunk = ser.read(n_bytes - len(buf))
        if not chunk:
            raise RuntimeError(
                "Serial read timed out. Check: Arduino is streaming, COM port is correct, "
                "and no other program is holding the port."
            )
        buf.extend(chunk)
    return bytes(buf)


def read_u16_samples_fast(ser, n_samples, block_samples=2048):
    """
    Read n_samples where each sample is 2 bytes, little-endian unsigned 16-bit.
    Uses block reads + numpy.frombuffer (much faster than struct.unpack per sample).
    """
    out = np.empty(n_samples, dtype=np.uint16)
    idx = 0
    while idx < n_samples:
        n = min(block_samples, n_samples - idx)
        raw = read_exactly(ser, 2 * n)
        out[idx:idx + n] = np.frombuffer(raw, dtype="<u2")
        idx += n
    return out


def alignment_score_by_diff(u16_array):
    """
    Score an alignment by typical step size between consecutive samples.
    Real analog signals usually change smoothly; a 1-byte misalignment often creates
    pseudo-random jumps, making |diff| much larger.

    We use median(|diff|) as a robust statistic.
    """
    x = u16_array.astype(np.int32)
    d = np.diff(x)
    return float(np.median(np.abs(d)))


def read_u16_samples_aligned(ser, n_samples):
    """
    Optional alignment check:
    Reads (2*ALIGN_CHECK_SAMPLES + 1) bytes once, then compares two possible uint16 alignments:
      - offset 0: bytes [0..]
      - offset 1: bytes [1..]
    Picks the alignment with smaller typical |diff|.

    If you always press Arduino RESET right before running Python, you can set
    ENABLE_ALIGNMENT_CHECK=False for simplicity.
    """
    if not ENABLE_ALIGNMENT_CHECK or n_samples <= 1:
        return read_u16_samples_fast(ser, n_samples, block_samples=BLOCK_SAMPLES)

    n_check = min(ALIGN_CHECK_SAMPLES, max(2, n_samples))
    raw = read_exactly(ser, 2 * n_check + 1)

    a0 = np.frombuffer(raw[:-1], dtype="<u2")
    a1 = np.frombuffer(raw[1:], dtype="<u2")

    s0 = alignment_score_by_diff(a0)
    s1 = alignment_score_by_diff(a1)

    use_offset1 = (s1 < s0 * (1.0 - ALIGN_REL_MARGIN))
    chosen_offset = 1 if use_offset1 else 0
    chosen = a1 if use_offset1 else a0

    print(f"Alignment check (median|diff|): offset0={s0:.2f}, offset1={s1:.2f} -> using offset {chosen_offset}")

    out = np.empty(n_samples, dtype=np.uint16)
    n0 = min(n_samples, n_check)
    out[:n0] = chosen[:n0]

    if n_samples > n0:
        out[n0:] = read_u16_samples_fast(ser, n_samples - n0, block_samples=BLOCK_SAMPLES)

    return out


def autocorr_fft_unbiased(x, max_lag):
    """
    Autocorrelation via FFT with unbiased normalization:
        r[k] = mean_t ( (x[t]-mean) * (x[t+k]-mean) )
    Returns r[k]/r[0] for k=0..max_lag.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    N = x.size
    if N < 2:
        raise ValueError("Not enough samples for autocorrelation.")

    max_lag = int(min(max_lag, N - 1))

    nfft = 1 << int(np.ceil(np.log2(2 * N)))
    f = np.fft.rfft(x, n=nfft)
    acf = np.fft.irfft(f * np.conj(f), n=nfft)[:max_lag + 1]

    norm = np.arange(N, N - max_lag - 1, -1, dtype=np.float64)
    acf = acf / norm

    if acf[0] == 0:
        raise ValueError("Zero-variance signal: autocorrelation undefined.")
    return acf / acf[0]


def counts_to_volts(u16_counts):
    """Map 0..65535 uniformly to 0..V_REF."""
    return u16_counts.astype(np.float64) * (V_REF / ADC_MAX)


def main():
    n_acquire = int(np.ceil(N_POINTS_KEEP * ACQUIRE_FACTOR))
    t_keep_s = N_POINTS_KEEP * DT_S

    print(f"Nominal dt = {DT_S:.3e} s ({INTERVAL_US} µs)")
    print(f"Keeping {N_POINTS_KEEP} samples -> nominal duration T = {t_keep_s:.3f} s")
    print(f"Acquiring {n_acquire} samples (factor {ACQUIRE_FACTOR}) and keeping the last {N_POINTS_KEEP}")

    t0_wall = time.time()
    with serial.Serial(PORT, BAUD, timeout=SERIAL_TIMEOUT_S) as ser:
        if POST_OPEN_DELAY_S > 0:
            time.sleep(POST_OPEN_DELAY_S)

        # Clear backlog in the computer-side serial input buffer so we start "fresh"
        ser.reset_input_buffer()

        # Acquire raw ADC counts
        data_acquire_counts = read_u16_samples_aligned(ser, n_acquire)

    elapsed_s = time.time() - t0_wall
    fs_eff = n_acquire / elapsed_s if elapsed_s > 0 else float("nan")
    dt_eff = 1.0 / fs_eff if fs_eff > 0 else float("nan")
    print(f"Wall-clock acquisition time: {elapsed_s:.3f} s")
    print(f"Effective receive rate: {fs_eff:.1f} samples/s  (dt_eff ~ {dt_eff:.3e} s)")

    data_counts = data_acquire_counts[-N_POINTS_KEEP:].copy()
    data_volts = counts_to_volts(data_counts)

    t = np.arange(N_POINTS_KEEP, dtype=np.float64) * DT_S

    # ---- Plotting uses smoothed data (for display only) ----
    v_smooth = moving_average_centered(data_volts, SMOOTH_WINDOW_SAMPLES, pad_mode=SMOOTH_PAD_MODE)
    v_smooth_demean = v_smooth - np.mean(v_smooth)

    t_plot = t[::PLOT_STRIDE]
    v_plot = v_smooth[::PLOT_STRIDE]
    v_plot_demean = v_smooth_demean[::PLOT_STRIDE]

    # Plot 1: Voltage vs time (no mean subtraction)
    fig, ax = plt.subplots()
    ax.plot(t_plot, v_plot, linestyle="None", marker="o")
    ax.set_title("Serial Data (Voltage, smoothed for display)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Voltage [V]")
    ax.set_xlim(float(t_plot[0]), float(t_plot[-1]))
    ax.set_ylim(*auto_ylim(v_smooth, factor=Y_MARGIN_FACTOR))
    fig.tight_layout()
    fig.savefig(f"{SAVE_PREFIX}_data_voltage.png", dpi=200)
    plt.show()

    # Plot 2: (Voltage - mean) vs time
    fig, ax = plt.subplots()
    ax.plot(t_plot, v_plot_demean, linestyle="None", marker="o")
    ax.set_title("Serial Data (Voltage - mean, smoothed for display)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Voltage - mean [V]")
    ax.set_xlim(float(t_plot[0]), float(t_plot[-1]))
    ax.set_ylim(*auto_ylim(v_smooth_demean, factor=Y_MARGIN_FACTOR))
    fig.tight_layout()
    fig.savefig(f"{SAVE_PREFIX}_data_voltage_demean.png", dpi=200)
    plt.show()

    # ---- Autocorrelation uses RAW data (unsmoothed) ----
    acf = autocorr_fft_unbiased(data_volts, max_lag=MAX_LAG)
    tau = np.arange(acf.size, dtype=np.float64) * DT_S

    fig, ax = plt.subplots()
    ax.plot(tau, acf, linestyle="None", marker="o")
    ax.set_title("Autocorrelation of (Voltage - mean) [raw signal]")
    ax.set_xlabel("Lag time [s]")
    ax.set_ylabel("Normalized autocorrelation")
    ax.set_xlim(0.0, float(tau[-1]))
    ax.set_ylim(-1.0, 1.0)
    fig.tight_layout()
    fig.savefig(f"{SAVE_PREFIX}_acf.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()

