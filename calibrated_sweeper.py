import time
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from rtlsdr import RtlSdr

DB_NAME = "rf_survey_database.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS swept_spectrum_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            band_name TEXT,
            start_mhz REAL,
            stop_mhz REAL,
            peak_power_dbfs REAL,
            avg_power_dbfs REAL,
            occupation_rate_pct REAL,
            exposure_index TEXT
        )
    """)
    conn.commit()
    conn.close()

def process_iq(samples, sample_rate, trim_fraction=0.15):
    """Computes properly normalized PSD (dBFS <= 0) and trims filter roll-off."""
    n = len(samples)
    window = np.blackman(n)
    
    # Scale complex samples to [-1.0, 1.0] standard
    norm_samples = samples / np.max(np.abs(samples) + 1e-9)
    windowed = norm_samples * window
    
    # FFT calculation with coherent gain normalization
    fft_vals = np.fft.fftshift(np.fft.fft(windowed))
    psd_linear = (np.abs(fft_vals) / np.sum(window)) ** 2
    psd_dbfs = 10.0 * np.log10(np.maximum(psd_linear, 1e-12))
    
    freq_rel = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / sample_rate))
    
    # Trim outer edges to discard IF low-pass filter attenuation
    trim = int(n * trim_fraction)
    return freq_rel[trim:-trim], psd_dbfs[trim:-trim]

def run_calibrated_sweep(start_mhz=88.0, stop_mhz=108.0, band_name="FM_Broadcast", fixed_gain_db=25.4):
    init_db()
    sdr = RtlSdr()
    
    fs = 2.4e6
    sdr.sample_rate = fs
    sdr.gain = fixed_gain_db         # Lock gain to a fixed value across every hop
    sdr.direct_sampling = 0
    
    # Effective hop step after trimming 30% total edge bandwidth
    step_mhz = (fs * 0.70) / 1e6
    center_freqs = np.arange(start_mhz + (step_mhz / 2.0), stop_mhz, step_mhz)
    
    print(f"[*] Sweeping {band_name} from {start_mhz} to {stop_mhz} MHz ({len(center_freqs)} hops)...")
    
    all_freqs = []
    all_psd = []
    
    try:
        for fc in center_freqs:
            sdr.center_freq = fc * 1e6
            time.sleep(0.015)  # LO synthesizer lock delay
            
            # Discard initial transient buffer (Flak et al.)
            _ = sdr.read_samples(4096)
            
            samples = sdr.read_samples(65536)
            f_slice, p_slice = process_iq(samples, fs, trim_fraction=0.15)
            
            all_freqs.extend((fc * 1e6 + f_slice) / 1e6)
            all_psd.extend(p_slice)
    finally:
        sdr.close()
        
    all_freqs = np.array(all_freqs)
    all_psd = np.array(all_psd)
    
    # Energy Detection with Adaptive Noise Floor (Slimeni et al.)
    noise_floor_median = float(np.median(all_psd))
    threshold_dbfs = noise_floor_median + 10.0  # 10 dB SNR detection threshold
    
    occupied_bins = all_psd > threshold_dbfs
    occupation_rate = float(np.sum(occupied_bins) / len(all_psd) * 100.0)
    
    peak_pwr = float(np.max(all_psd))
    mean_linear = np.mean(10.0 ** (all_psd / 10.0))
    avg_power = float(10.0 * np.log10(mean_linear))
    
    # Exposure categorization
    if avg_power < -50:
        exp_index = "LOW"
    elif avg_power < -30:
        exp_index = "MEDIUM"
    else:
        exp_index = "HIGH"
        
    # Store to SQLite
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO swept_spectrum_log 
        (band_name, start_mhz, stop_mhz, peak_power_dbfs, avg_power_dbfs, occupation_rate_pct, exposure_index)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (band_name, start_mhz, stop_mhz, peak_pwr, avg_power, occupation_rate, exp_index))
    conn.commit()
    conn.close()
    
    print("\n--- Calibrated Scan Results ---")
    print(f"Locked Gain           : {fixed_gain_db} dB")
    print(f"Noise Floor (Median)  : {noise_floor_median:.2f} dBFS")
    print(f"Peak Signal           : {peak_pwr:.2f} dBFS")
    print(f"Average Channel Power : {avg_power:.2f} dBFS")
    print(f"Occupation Rate (OR)  : {occupation_rate:.1f} %")
    print(f"RF Exposure Index     : {exp_index}")
    
    # Plotting
    plt.figure(figsize=(11, 4.5))
    plt.plot(all_freqs, all_psd, color='navy', lw=0.7, label='Calibrated PSD')
    plt.axhline(threshold_dbfs, color='crimson', linestyle='--', lw=1.2, 
                label=f'Energy Detection Threshold ({threshold_dbfs:.1f} dBFS)')
    plt.title(f"Calibrated Swept Spectrum ({band_name}) | Exposure: {exp_index} | OR: {occupation_rate:.1f}%")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Relative Power (dBFS)")
    plt.ylim(-90, 5)
    plt.xlim(start_mhz, stop_mhz)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Sweep GSM-900 Cellular Downlink (Tower to Mobile)
    run_calibrated_sweep(start_mhz=925.0, stop_mhz=960.0, band_name="GSM_900_Cellular", fixed_gain_db=29.7)