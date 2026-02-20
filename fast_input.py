import os
import time
import numpy as np
import matplotlib.pyplot as plt
import serial
import json
from datetime import datetime
from tqdm import tqdm

# =========================== USER PARAMETERS ===========================
PORT = "COM4"                 # Must match Windows Device Manager (Ports -> COMx)
BAUD = 1_000_000              # Must match Serial.begin(...) in Arduino (fast_reader.ino)

INTERVAL_US = 52              # Nominal Arduino sampling interval in microseconds (calibrate this!)
DT_S = INTERVAL_US * 1e-6     # seconds

# ADC counts (0..65535) -> voltage (0..3.3 V)
ADC_MAX = 65535
V_REF = 3.3                   # Volts

# Data length
N_POINTS_KEEP = 500_000       # Samples kept for plotting + autocorrelation
ACQUIRE_FACTOR = 1.2          # Acquire extra samples, then keep only the last N_POINTS_KEEP

# Serial read
SERIAL_TIMEOUT_S = 2.0        # Seconds (prevents infinite blocking if Arduino stops streaming)
BLOCK_SAMPLES = 4096          # Read this many samples per block (2 bytes per sample)

# Diagnostics
PRINT_FIRST_N = 20            # Print first N raw samples for manual protocol/alignment check

# Plot controls
PLOT_STRIDE = 10              # Plot every Nth point (downsampling for speed)
SMOOTH_WINDOW_SAMPLES = 5    # Centered moving average window for PLOTTING ONLY (odd recommended; 1 disables)
SMOOTH_PAD_MODE = "edge"      # "edge" or "reflect"
Y_MARGIN_FACTOR = 1.2         # Expand y-limits away from zero

# Autocorrelation (dot-product method; can be slow for large N and MAX_LAG)
MAX_LAG = 100_000             # Max lag in samples

# Output
SAVE_PREFIX = "colloid_3"
SAVE_RAW_NPZ = True           # Saves time_s, counts_u16, volts arrays into a .npz file (recommended)
SAVE_RAW_CSV = False          # CSV can be very large; enable only if you really need it
# ======================================================================


def get_output_dir():
    """
    Save outputs into the same folder as this .py file.
    Falls back to current working directory if __file__ is not available.
    """
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


def get_experiment_info():
    """
    Get experiment name and metadata from user input.
    """
    name = input("Enter experiment name: ").strip() or "default"
    
    enter_metadata = input("Enter metadata manually? (y/n) [n]: ").strip().lower()
    if enter_metadata == 'y':
        scatter_angle = input("Scatter angle (degrees) [90]: ").strip()
        sample_date = input("Sample date (YYYY-MM-DD) [today]: ").strip()
        sample_number = input("Sample number [1]: ").strip()
        colloid_type = input("Colloid type (brand) [unknown]: ").strip()
        diameter = input("Diameter (nm) [100]: ").strip()
        concentration = input("Concentration [1%]: ").strip()
        framerate = input("Framerate (Hz) [auto]: ").strip()
        notes = input("General notes: ").strip()
    else:
        scatter_angle = ""
        sample_date = ""
        sample_number = ""
        colloid_type = ""
        diameter = ""
        concentration = ""
        framerate = ""
        notes = ""
    
    # Set defaults
    diameter = diameter or "100"
    concentration = concentration or "1%"
    framerate = framerate or str(1/DT_S)
    notes = notes or ""
    
    capture_time = datetime.now().isoformat()
    
    metadata_raw = {
        "scatter_angle": scatter_angle,
        "sample_date": sample_date,
        "sample_number": sample_number,
        "colloid_type": colloid_type,
        "diameter": diameter,
        "concentration": concentration,
        "capture_time": capture_time,
        "framerate": framerate,
        "notes": notes
    }
    
    # Primary results metadata (placeholders)
    metadata_primary = {
        "filtering_criterion": "none",
        "range": "full",
        "rolling_average_removed": SMOOTH_WINDOW_SAMPLES,
        "notes": notes
    }
    
    return name, metadata_raw, metadata_primary


def auto_ylim(y, factor):
    """
    Choose y-limits based on min/max with sign-aware expansion:
      - ymax = max * factor if max > 0 else max / factor
      - ymin = min * factor if min < 0 else min / factor
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


def moving_average_centered(x, window, pad_mode):
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


def counts_to_volts(u16_counts):
    """Map 0..65535 uniformly to 0..V_REF."""
    return u16_counts.astype(np.float64) * (V_REF / ADC_MAX)


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


def read_u16_samples(ser, n_samples, block_samples):
    """
    Read n_samples where each sample is 2 bytes, little-endian unsigned 16-bit.
    Uses block reads + numpy.frombuffer (faster than per-sample unpack).
    """
    out = np.empty(n_samples, dtype=np.uint16)
    idx = 0
    with tqdm(total=n_samples, desc="Reading data", unit="samples") as pbar:
        while idx < n_samples:
            n = min(block_samples, n_samples - idx)
            raw = read_exactly(ser, 2 * n)
            out[idx:idx + n] = np.frombuffer(raw, dtype="<u2")
            idx += n
            pbar.update(n)
    return out


def autocorr_limited(x, max_lag):
    """
    Dot-product autocorrelation (normalized to 1 at lag=0).
    This is O(N*max_lag) and can be slow for large arrays.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    N = len(x)
    max_lag = int(min(max_lag, N - 1))

    acf = np.empty(max_lag + 1, dtype=np.float64)
    for lag in tqdm(range(max_lag + 1), desc="Computing autocorrelation"):
        acf[lag] = np.dot(x[:N - lag], x[lag:])

    if acf[0] == 0:
        raise ValueError("Zero-variance signal: autocorrelation undefined.")
    return acf / acf[0]


def main():
    base_dir = get_output_dir()
    
    # Get experiment info
    exp_name, metadata_raw, metadata_primary = get_experiment_info()
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir_name = f"{timestamp}_{exp_name}"
    
    # Create directories
    data_dir = os.path.join(base_dir, "Data")
    primary_results_dir = os.path.join(base_dir, "Primary_results")
    secondary_results_dir = os.path.join(base_dir, "Secondary_results")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(primary_results_dir, exist_ok=True)
    os.makedirs(secondary_results_dir, exist_ok=True)
    
    print(f"Experiment: {exp_dir_name}")
    print(f"Data will be saved to: {data_dir}")
    print(f"Primary results will be saved to: {primary_results_dir}")
    print(f"Secondary results will be saved to: {secondary_results_dir}")

    n_acquire = int(np.ceil(N_POINTS_KEEP * ACQUIRE_FACTOR))
    t_keep_s = N_POINTS_KEEP * DT_S
    t_acquire_s = n_acquire * DT_S

    print(f"Nominal dt = {DT_S:.3e} s ({INTERVAL_US} µs)")
    print(f"Keeping {N_POINTS_KEEP} samples -> nominal T_keep = {t_keep_s:.3f} s")
    print(f"Acquiring {n_acquire} samples (factor {ACQUIRE_FACTOR}) -> nominal T_acquire = {t_acquire_s:.3f} s")
    print(f"Plot stride = {PLOT_STRIDE}, smooth window = {SMOOTH_WINDOW_SAMPLES} samples")

    first_n = min(PRINT_FIRST_N, n_acquire)
    remaining_n = n_acquire - first_n

    t0 = time.time()
    with serial.Serial(PORT, BAUD, timeout=SERIAL_TIMEOUT_S) as ser:
        # Clear backlog in the computer-side serial input buffer so we start "fresh"
        ser.reset_input_buffer()

        # Read first N samples and print them for manual sanity check
        first_raw = read_exactly(ser, 2 * first_n)
        first_counts = np.frombuffer(first_raw, dtype="<u2")
        first_volts = counts_to_volts(first_counts)

        print(f"First {first_n} raw samples (uint16): {first_counts.tolist()}")
        print(f"First {first_n} samples (V): {[float(v) for v in first_volts]}")

        if remaining_n > 0:
            rest_counts = read_u16_samples(ser, remaining_n, block_samples=BLOCK_SAMPLES)
            data_acquire_counts = np.concatenate([first_counts, rest_counts])
        else:
            data_acquire_counts = first_counts.copy()

    elapsed_s = time.time() - t0
    n_read = data_acquire_counts.size
    fs_eff = n_read / elapsed_s if elapsed_s > 0 else float("nan")
    dt_eff = 1.0 / fs_eff if fs_eff > 0 else float("nan")
    print(f"Wall-clock acquisition time: {elapsed_s:.3f} s")
    print(f"Effective receive rate: {fs_eff:.1f} samples/s (dt_eff ~ {dt_eff:.3e} s)")

    if n_read < N_POINTS_KEEP:
        raise RuntimeError(
            f"Only received {n_read} samples, but N_POINTS_KEEP={N_POINTS_KEEP}. "
            "This usually means the Arduino is not streaming at the expected rate, "
            "or the serial link is unstable."
        )

    # Keep last N_POINTS_KEEP samples (ensures we do not include any 'startup' effects)
    data_counts = data_acquire_counts[-N_POINTS_KEEP:].astype(np.uint16, copy=False)
    data_volts = counts_to_volts(data_counts)

    # Build time axis in seconds (based on nominal dt)
    t = np.arange(N_POINTS_KEEP, dtype=np.float64) * DT_S

    # Save raw data used for analysis
    if SAVE_RAW_NPZ:
        npz_path = os.path.join(data_dir, f"{exp_dir_name}.npz")
        np.savez(npz_path, time_s=t, counts_u16=data_counts, volts=data_volts)
        print(f"Saved raw data (npz): {npz_path}")

    if SAVE_RAW_CSV:
        csv_path = os.path.join(data_dir, f"{exp_dir_name}.csv")
        arr = np.column_stack([t, data_counts.astype(np.int64), data_volts])
        np.savetxt(csv_path, arr, delimiter=",", header="time_s,counts_u16,volts", comments="")
        print(f"Saved raw data (csv): {csv_path}")

    # Save raw data metadata
    metadata_raw_path = os.path.join(data_dir, f"{exp_dir_name}.metadata")
    with open(metadata_raw_path, 'w') as f:
        json.dump(metadata_raw, f, indent=4)
    print(f"Saved raw metadata: {metadata_raw_path}")

    # ---- Plotting uses smoothed signal (display only) ----
    v_smooth = moving_average_centered(data_volts, SMOOTH_WINDOW_SAMPLES, pad_mode=SMOOTH_PAD_MODE)
    v_smooth_demean = v_smooth - np.mean(v_smooth)

    # Downsample for plotting
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

    fig_path = os.path.join(data_dir, f"{exp_dir_name}_data_voltage.png")
    fig.savefig(fig_path, dpi=200)
    print(f"Saved figure: {fig_path}")
    plt.close(fig)  # Close to avoid display issues

    # Plot 2: (Voltage - mean) vs time
    fig, ax = plt.subplots()
    ax.plot(t_plot, v_plot_demean, linestyle="None", marker="o")
    ax.set_title("Serial Data (Voltage - mean, smoothed for display)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Voltage - mean [V]")
    ax.set_xlim(float(t_plot[0]), float(t_plot[-1]))
    ax.set_ylim(*auto_ylim(v_smooth_demean, factor=Y_MARGIN_FACTOR))
    fig.tight_layout()

    fig_path = os.path.join(data_dir, f"{exp_dir_name}_data_voltage_demean.png")
    fig.savefig(fig_path, dpi=200)
    print(f"Saved figure: {fig_path}")
    plt.close(fig)

    # ---- Autocorrelation uses RAW data (unsmoothed) ----
    acf = autocorr_limited(data_volts, max_lag=MAX_LAG)
    tau = np.arange(acf.size, dtype=np.float64) * DT_S

    fig, ax = plt.subplots()
    ax.plot(tau, acf, linestyle="None", marker="o")
    ax.set_title("Autocorrelation of (Voltage - mean) [raw signal]")
    ax.set_xlabel("Lag time [s]")
    ax.set_ylabel("Normalized autocorrelation")
    ax.set_xlim(0.0, float(tau[-1]))
    ax.set_ylim(-1.0, 1.0)
    fig.tight_layout()

    fig_path = os.path.join(primary_results_dir, f"{exp_dir_name}.png")
    fig.savefig(fig_path, dpi=200)
    print(f"Saved autocorrelation figure: {fig_path}")
    plt.close(fig)

    # Save autocorrelation data as CSV
    acf_csv_path = os.path.join(primary_results_dir, f"{exp_dir_name}.csv")
    arr_acf = np.column_stack([tau, acf])
    np.savetxt(acf_csv_path, arr_acf, delimiter=",", header="lag_time_s,autocorrelation", comments="")
    print(f"Saved autocorrelation data: {acf_csv_path}")

    # Save primary results metadata
    metadata_primary_path = os.path.join(primary_results_dir, f"{exp_dir_name}.metadata")
    with open(metadata_primary_path, 'w') as f:
        json.dump(metadata_primary, f, indent=4)
    print(f"Saved primary metadata: {metadata_primary_path}")

    # Note: Tau_q calculation to be added later


if __name__ == "__main__":
    main()
