import os
import time
import numpy as np
import matplotlib.pyplot as plt
import serial
import json
from datetime import datetime
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy import stats

# =========================== USER PARAMETERS ===========================
PORT = "COM4"                 # Must match Windows Device Manager (Ports -> COMx)
BAUD = 1_000_000              # Must match Serial.begin(...) in Arduino (fast_reader.ino)

INTERVAL_US = 52              # Nominal Arduino sampling interval in microseconds (calibrate this!)
DT_S = INTERVAL_US * 1e-6     # seconds

# ADC counts (0..65535) -> voltage (0..3.3 V)
ADC_MAX = 65535
V_REF = 3.3                   # Volts

# Data length
N_POINTS_PER_REPEAT = 250_000  # Samples per acquisition
ACQUIRE_FACTOR = 1.2           # Acquire extra samples, then keep only the last N_POINTS_PER_REPEAT
N_REPEATS = 30                 # Number of acquisitions in the sweep

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
    name = input("Enter experiment name: ").strip() or "sweep"
    
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
    scatter_angle = scatter_angle or "90"
    sample_date = sample_date or datetime.now().strftime("%Y-%m-%d")
    sample_number = sample_number or "1"
    colloid_type = colloid_type or "unknown"
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


def read_u16_samples(ser, n_samples, block_samples, desc="Reading data"):
    """
    Read n_samples where each sample is 2 bytes, little-endian unsigned 16-bit.
    Uses block reads + numpy.frombuffer (faster than per-sample unpack).
    """
    out = np.empty(n_samples, dtype=np.uint16)
    idx = 0
    with tqdm(total=n_samples, desc=desc, unit="samples") as pbar:
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


def process_single_repeat(data_volts, data_counts, repeat_num, data_dir, exp_dir_name, metadata_raw):
    """
    Process and save a single repeat: raw data and plots.
    Returns the ACF data for later analysis.
    """
    # Build time axis
    t = np.arange(N_POINTS_PER_REPEAT, dtype=np.float64) * DT_S
    
    # Create repeat-specific filename
    repeat_filename = f"{exp_dir_name}_repeat{repeat_num:02d}"
    
    # Save raw data
    if SAVE_RAW_NPZ:
        npz_path = os.path.join(data_dir, f"{repeat_filename}.npz")
        np.savez(npz_path, time_s=t, counts_u16=data_counts, volts=data_volts)

    if SAVE_RAW_CSV:
        csv_path = os.path.join(data_dir, f"{repeat_filename}.csv")
        arr = np.column_stack([t, data_counts.astype(np.int64), data_volts])
        np.savetxt(csv_path, arr, delimiter=",", header="time_s,counts_u16,volts", comments="")

    # Save raw data metadata
    metadata_raw_path = os.path.join(data_dir, f"{repeat_filename}.metadata")
    with open(metadata_raw_path, 'w') as f:
        json.dump(metadata_raw, f, indent=4)

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
    ax.set_title(f"Raw Data - Repeat {repeat_num} (Voltage)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Voltage [V]")
    ax.set_xlim(float(t_plot[0]), float(t_plot[-1]))
    ax.set_ylim(*auto_ylim(v_smooth, factor=Y_MARGIN_FACTOR))
    fig.tight_layout()
    fig_path = os.path.join(data_dir, f"{repeat_filename}_voltage.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    # Plot 2: (Voltage - mean) vs time
    fig, ax = plt.subplots()
    ax.plot(t_plot, v_plot_demean, linestyle="None", marker="o")
    ax.set_title(f"Raw Data - Repeat {repeat_num} (Voltage - Mean)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Voltage - mean [V]")
    ax.set_xlim(float(t_plot[0]), float(t_plot[-1]))
    ax.set_ylim(*auto_ylim(v_smooth_demean, factor=Y_MARGIN_FACTOR))
    fig.tight_layout()
    fig_path = os.path.join(data_dir, f"{repeat_filename}_voltage_demean.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    # ---- Compute and return autocorrelation for later analysis ----
    acf = autocorr_limited(data_volts, max_lag=MAX_LAG)
    tau = np.arange(acf.size, dtype=np.float64) * DT_S
    
    return tau, acf


def exponential_model(t, amplitude, tau):
    """Exponential decay model: amplitude * exp(-t/tau)"""
    return amplitude * np.exp(-t / tau)


def fit_exponential(tau, acf, initial_guess=None):
    """
    Fit exponential decay to ACF.
    Returns (amplitude, tau, popt, pcov) or None if fitting fails.
    """
    try:
        # Use only the positive ACF region (typically good to first zero crossing or 1/e point)
        # Fit only up to where ACF is still reasonably above noise
        valid_idx = acf > 0.01  # Fit where ACF > 0.01
        if np.sum(valid_idx) < 3:  # Need at least 3 points
            valid_idx = acf > np.max(acf) * 0.05
        
        tau_fit = tau[valid_idx]
        acf_fit = acf[valid_idx]
        
        if initial_guess is None:
            # Estimate initial amplitude and tau from data
            amplitude_0 = acf_fit[0]
            # Find where acf drops to 1/e of initial value
            target = amplitude_0 / np.e
            tau_0_idx = np.argmin(np.abs(acf_fit - target))
            tau_0 = tau_fit[tau_0_idx] if tau_0_idx < len(tau_fit) else tau_fit[-1]
            initial_guess = [amplitude_0, tau_0]
        
        # Fit the exponential
        popt, pcov = curve_fit(exponential_model, tau_fit, acf_fit, p0=initial_guess, maxfev=5000)
        
        return popt[0], popt[1], popt, pcov
    except Exception as e:
        print(f"  Warning: Exponential fitting failed: {e}")
        return None


def calculate_correlation_time(amplitude, tau):
    """
    Calculate correlation time: lag time where ACF drops to 1/e of its initial value.
    If amplitude and tau are from fit, returns tau. If amplitude and tau are arrays, finds first crossing.
    """
    if amplitude is None or tau is None:
        return None
    # If amplitude and tau are arrays (raw ACF), find where ACF drops to amplitude/e
    if isinstance(amplitude, (np.ndarray, list)) and isinstance(tau, (np.ndarray, list)):
        acf = np.asarray(amplitude)
        tau_arr = np.asarray(tau)
        target = acf[0] / np.e
        idx = np.where(acf <= target)[0]
        if len(idx) == 0:
            return tau_arr[-1]
        return tau_arr[idx[0]]
    # Otherwise, for exponential fit, tau is the correlation time
    return tau


def analyze_all_acfs(all_acfs, data_dir, exp_dir_name):
    """
    Analyze all ACFs: fit exponentials, extract correlation times, and create histogram.
    """
    print(f"\n{'='*60}")
    print("ACF Analysis and Correlation Time Extraction")
    print(f"{'='*60}\n")
    
    correlation_times = []
    fits_successful = []
    
    # Fit each ACF
    for repeat_num, (tau, acf) in enumerate(all_acfs, 1):
        result = fit_exponential(tau, acf)
        if result is not None:
            amplitude, tau_corr, popt, pcov = result
            correlation_times.append(tau_corr)
            fits_successful.append(True)
            print(f"Repeat {repeat_num:2d}: tau_corr = {tau_corr:.6f} s (amplitude = {amplitude:.4f})")
            
            # Save fitted ACF data
            repeat_filename = f"{exp_dir_name}_repeat{repeat_num:02d}"
            acf_csv_path = os.path.join(data_dir, f"{repeat_filename}_acf.csv")
            arr_acf = np.column_stack([tau, acf])
            np.savetxt(acf_csv_path, arr_acf, delimiter=",", header="lag_time_s,autocorrelation", comments="")
            
            # Plot ACF with exponential fit overlay
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(tau, acf, 'o-', label='ACF', alpha=0.7, markersize=3)
            
            # Plot the exponential fit
            tau_fit_plot = tau[:int(len(tau)*0.3)]  # Plot fit over first 30% of lag range
            acf_fit = exponential_model(tau_fit_plot, amplitude, tau_corr)
            ax.plot(tau_fit_plot, acf_fit, 'r--', linewidth=2, label=f'Exponential fit (τ={tau_corr:.6f}s)')
            
            ax.axhline(y=amplitude/np.e, color='g', linestyle=':', alpha=0.5, label=f'1/e level')
            ax.axvline(x=tau_corr, color='g', linestyle=':', alpha=0.5)
            
            ax.set_title(f"Autocorrelation with Fit - Repeat {repeat_num}")
            ax.set_xlabel("Lag time [s]")
            ax.set_ylabel("Normalized autocorrelation")
            ax.set_xlim(0.0, float(tau[min(len(tau)-1, int(len(tau)*0.3))]))
            ax.set_ylim(-0.1, 1.0)
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig_path = os.path.join(data_dir, f"{repeat_filename}_acf_fit.png")
            fig.savefig(fig_path, dpi=200)
            plt.close(fig)
        else:
            fits_successful.append(False)
            print(f"Repeat {repeat_num:2d}: Fit failed")
    
    # Create histogram of correlation times
    if correlation_times:
        print(f"\n{'='*60}")
        print("Correlation Time Statistics")
        print(f"{'='*60}")
        print(f"Mean τ_corr: {np.mean(correlation_times):.6f} s")
        print(f"Std τ_corr:  {np.std(correlation_times):.6f} s")
        print(f"Min τ_corr:  {np.min(correlation_times):.6f} s")
        print(f"Max τ_corr:  {np.max(correlation_times):.6f} s")
        print(f"Successful fits: {np.sum(fits_successful)}/{len(fits_successful)}\n")
        
        # Plot histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        n_bins = max(5, len(correlation_times) // 3)
        counts, bins, patches = ax.hist(correlation_times, bins=n_bins, edgecolor='black', alpha=0.7)
        
        # Add statistics
        mean_tau = np.mean(correlation_times)
        std_tau = np.std(correlation_times)
        ax.axvline(mean_tau, color='r', linestyle='--', linewidth=2, label=f'Mean = {mean_tau:.6f} s')
        ax.axvline(mean_tau - std_tau, color='r', linestyle=':', alpha=0.7, label='±1σ')
        ax.axvline(mean_tau + std_tau, color='r', linestyle=':', alpha=0.7)
        
        ax.set_title("Distribution of Correlation Times")
        ax.set_xlabel("Correlation Time τ_corr [s]")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()
        
        hist_path = os.path.join(data_dir, f"{exp_dir_name}_correlation_time_histogram.png")
        fig.savefig(hist_path, dpi=200)
        plt.close(fig)
        
        # Save correlation times to CSV
        corr_time_csv = os.path.join(data_dir, f"{exp_dir_name}_correlation_times.csv")
        corr_time_data = np.array(correlation_times).reshape(-1, 1)
        np.savetxt(corr_time_csv, corr_time_data, delimiter=",", header="correlation_time_s", comments="")
        print(f"Saved correlation times to: {corr_time_csv}")
        print(f"Saved histogram to: {hist_path}\n")
        
        return correlation_times
    else:
        print("No successful ACF fits for histogram generation.")
        return None


def main():
    base_dir = get_output_dir()
    
    # Get experiment info (once for entire sweep)
    exp_name, metadata_raw, metadata_primary = get_experiment_info()
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir_name = f"{timestamp}_{exp_name}"
    
    # Create directories
    data_dir = os.path.join(base_dir, "Data")
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"SWEEP PROGRAM: {N_REPEATS} repeats of {N_POINTS_PER_REPEAT} samples")
    print(f"Experiment: {exp_dir_name}")
    print(f"Data directory: {data_dir}")
    print(f"{'='*60}\n")

    n_acquire = int(np.ceil(N_POINTS_PER_REPEAT * ACQUIRE_FACTOR))
    t_keep_s = N_POINTS_PER_REPEAT * DT_S
    t_acquire_s = n_acquire * DT_S

    print(f"Nominal dt = {DT_S:.3e} s ({INTERVAL_US} µs)")
    print(f"Keeping {N_POINTS_PER_REPEAT} samples per repeat -> nominal T = {t_keep_s:.3f} s")
    print(f"Acquiring {n_acquire} samples per repeat (factor {ACQUIRE_FACTOR}) -> nominal T_acquire = {t_acquire_s:.3f} s")
    print(f"Total experiment time: ~{t_acquire_s * N_REPEATS / 60:.1f} minutes\n")

    first_n = min(PRINT_FIRST_N, n_acquire)

    # Main sweep loop
    sweep_start_time = time.time()
    all_acfs = []  # Store ACF data from all repeats for post-analysis
    
    with serial.Serial(PORT, BAUD, timeout=SERIAL_TIMEOUT_S) as ser:
        for repeat_idx in tqdm(range(N_REPEATS), desc="Sweep Progress", position=0):
            repeat_num = repeat_idx + 1
            repeat_start = time.time()
            
            # Clear serial buffer
            ser.reset_input_buffer()
            
            # Read first N samples for diagnostics (only on first repeat)
            if repeat_idx == 0:
                first_raw = read_exactly(ser, 2 * first_n)
                first_counts = np.frombuffer(first_raw, dtype="<u2")
                first_volts = counts_to_volts(first_counts)
                print(f"\n[{repeat_num}/{N_REPEATS}] First {first_n} samples (uint16): {first_counts.tolist()}")
                print(f"[{repeat_num}/{N_REPEATS}] First {first_n} samples (V): {[float(v) for v in first_volts]}")
                
                remaining_n = n_acquire - first_n
                if remaining_n > 0:
                    rest_counts = read_u16_samples(ser, remaining_n, block_samples=BLOCK_SAMPLES, desc=f"Repeat {repeat_num} - Reading data")
                    data_acquire_counts = np.concatenate([first_counts, rest_counts])
                else:
                    data_acquire_counts = first_counts.copy()
            else:
                # Subsequent repeats: just read all data
                data_acquire_counts = read_u16_samples(ser, n_acquire, block_samples=BLOCK_SAMPLES, desc=f"Repeat {repeat_num} - Reading data")
            
            # Keep last N_POINTS_PER_REPEAT samples
            data_counts = data_acquire_counts[-N_POINTS_PER_REPEAT:].astype(np.uint16, copy=False)
            data_volts = counts_to_volts(data_counts)
            
            elapsed_loop = time.time() - repeat_start
            
            # Process the data for this repeat and capture ACF for later analysis
            tau, acf = process_single_repeat(data_volts, data_counts, repeat_num, data_dir, exp_dir_name, metadata_raw)
            all_acfs.append((tau, acf))
            
            print(f"[{repeat_num}/{N_REPEATS}] Complete - Time: {elapsed_loop:.1f}s\n")
    
    sweep_elapsed = time.time() - sweep_start_time
    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE: {sweep_elapsed/60:.1f} minutes")
    print(f"All {N_REPEATS} repeats saved to: {data_dir}")
    print(f"{'='*60}\n")
    
    # Perform ACF analysis on all collected data
    analyze_all_acfs(all_acfs, data_dir, exp_dir_name)


if __name__ == "__main__":
    main()
