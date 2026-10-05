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
        CREATE TABLE IF NOT EXISTS shielding_trials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            frequency_mhz REAL,
            shield_material TEXT,
            peak_power_dbfs REAL,
            avg_power_dbfs REAL,
            noise_floor_dbfs REAL,
            occupation_rate_pct REAL
        )
    """)
    conn.commit()
    conn.close()

def capture_rf_spectrum(center_freq_mhz=98.3, sample_rate_mhz=2.048, gain_db=25.0, fft_size=16384):
    """Acquires raw baseband IQ samples and computes calibrated PSD."""
    sdr = RtlSdr()
    try:
        sdr.sample_rate = sample_rate_mhz * 1e6
        sdr.center_freq = center_freq_mhz * 1e6
        sdr.gain = gain_db
        sdr.direct_sampling = 0

        # Discard initial transient buffer to eliminate LO switching spikes (Flak et al.)
        _ = sdr.read_samples(4096)
        samples = sdr.read_samples(fft_size)
    finally:
        sdr.close()

    # Windowed FFT (Blackman window to suppress spectral leakage)
    window = np.blackman(fft_size)
    norm_samples = samples / (np.max(np.abs(samples)) + 1e-9)
    fft_vals = np.fft.fftshift(np.fft.fft(norm_samples * window))
    
    # Coherent Power Spectral Density (dBFS)
    psd_linear = (np.abs(fft_vals) / np.sum(window)) ** 2
    psd_dbfs = 10.0 * np.log10(np.maximum(psd_linear, 1e-12))
    
    freq_rel = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0 / (sample_rate_mhz * 1e6)))
    freq_axis = (center_freq_mhz * 1e6 + freq_rel) / 1e6

    # Metrics Extraction (Slimeni et al. Energy Detection)
    noise_floor = float(np.median(psd_dbfs))
    threshold = noise_floor + 10.0  # 10 dB SNR occupancy threshold
    occupied_bins = psd_dbfs > threshold
    occupation_rate = float(np.sum(occupied_bins) / fft_size * 100.0)
    
    peak_pwr = float(np.max(psd_dbfs))
    avg_pwr = float(10.0 * np.log10(np.mean(10.0 ** (psd_dbfs / 10.0))))

    return freq_axis, psd_dbfs, peak_pwr, avg_pwr, noise_floor, occupation_rate

def run_experiment(material_label="None"):
    init_db()
    fc = 98.3  # Target prominent broadcast carrier
    
    print(f"\n[*] Starting capture for material: '{material_label}' at {fc} MHz...")
    freqs, psd, peak_p, avg_p, floor_p, or_pct = capture_rf_spectrum(center_freq_mhz=fc)

    # Persist trial to SQLite
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO shielding_trials 
        (frequency_mhz, shield_material, peak_power_dbfs, avg_power_dbfs, noise_floor_dbfs, occupation_rate_pct)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (fc, material_label, peak_p, avg_p, floor_p, or_pct))
    conn.commit()
    conn.close()

    print(f"[+] Material Tested      : {material_label}")
    print(f"[+] Peak Carrier Power   : {peak_p:.2f} dBFS")
    print(f"[+] Channel Power (Avg)  : {avg_p:.2f} dBFS")
    print(f"[+] Est. Noise Floor     : {floor_p:.2f} dBFS")
    print(f"[+] Channel Occupancy    : {or_pct:.1f} %")
    return freqs, psd, peak_p

if __name__ == "__main__":
    # Change label to: "None", "Aluminium_Foil", "Steel_Can", or "Wire_Mesh"
    run_experiment(material_label="Aluminium_Foil")