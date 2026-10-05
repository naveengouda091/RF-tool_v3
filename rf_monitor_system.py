import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from rtlsdr import RtlSdr

DB_NAME = "rf_survey_database.db"

def init_database():
    """Initializes SQLite schema for RF signal mapping."""
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS rf_measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            latitude REAL,
            longitude REAL,
            band_name TEXT,
            center_freq_mhz REAL,
            peak_power_dbfs REAL,
            avg_power_dbfs REAL,
            exposure_index TEXT,
            shield_material TEXT
        )
    """)
    conn.commit()
    conn.close()

def classify_frequency_band(freq_mhz):
    """Categorizes carrier frequency into standard communication bands."""
    if 87.5 <= freq_mhz <= 108.0:
        return "FM Broadcast"
    elif 880.0 <= freq_mhz <= 960.0:
        return "GSM-900 Cellular"
    elif 1710.0 <= freq_mhz <= 1880.0:
        return "DCS / LTE Band 3"
    elif 2400.0 <= freq_mhz <= 2483.5:
        return "2.4 GHz ISM / Wi-Fi"
    else:
        return "Other / General RF"

def run_rf_scan(center_freq_mhz=98.3, sample_rate_mhz=2.4, gain_val=20.0,
                lat=15.3218, lon=74.7646, shield="None"):
    """
    Captures raw IQ baseband samples, calculates PSD via FFT,
    determines Exposure Index, and stores results to database.
    """
    init_database()
    
    print(f"[*] Initializing RTL-SDR Blog V3 at {center_freq_mhz} MHz (Gain: {gain_val} dB)...")
    sdr = RtlSdr()
    
    try:
        # Standard Quadrature Sampling Mode (direct_sampling = 0)
        sdr.direct_sampling = 0
        sdr.sample_rate = sample_rate_mhz * 1e6
        sdr.center_freq = center_freq_mhz * 1e6
        sdr.gain = gain_val

        # Discard initial transient buffer to prevent LO tuning artifacts
        _ = sdr.read_samples(4096)

        # Acquire baseband IQ data (128k samples)
        num_samples = 131072
        samples = sdr.read_samples(num_samples)
    finally:
        sdr.close()

    # Windowed Fast Fourier Transform (Hanning Window)
    window = np.hanning(num_samples)
    fft_vals = np.fft.fftshift(np.fft.fft(samples * window))
    
    # Calculate Power Spectral Density (PSD) in dBFS
    psd_linear = (np.abs(fft_vals) ** 2) / (num_samples * np.mean(window**2))
    psd_dbfs = 10.0 * np.log10(np.maximum(psd_linear, 1e-12))
    freq_axis = (np.fft.fftshift(np.fft.fftfreq(num_samples, d=1.0/(sample_rate_mhz * 1e6))) + (center_freq_mhz * 1e6)) / 1e6

    # Extract Key Parameters
    peak_power = float(np.max(psd_dbfs))
    avg_power = float(10.0 * np.log10(np.mean(psd_linear)))
    band_name = classify_frequency_band(center_freq_mhz)

    # Classify RF Exposure Index
    if avg_power < -50.0:
        exposure_index = "LOW"
    elif avg_power < -30.0:
        exposure_index = "MEDIUM"
    else:
        exposure_index = "HIGH"

    # Commit record to SQLite
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO rf_measurements 
        (latitude, longitude, band_name, center_freq_mhz, peak_power_dbfs, avg_power_dbfs, exposure_index, shield_material)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (lat, lon, band_name, center_freq_mhz, peak_power, avg_power, exposure_index, shield))
    conn.commit()
    conn.close()

    # Output Terminal Summary
    print("\n--- Measurement Summary ---")
    print(f"Detected Band    : {band_name}")
    print(f"Center Frequency : {center_freq_mhz:.2f} MHz")
    print(f"Peak Power       : {peak_power:.2f} dBFS")
    print(f"Channel Power    : {avg_power:.2f} dBFS")
    print(f"Exposure Index   : {exposure_index}")
    print(f"Shielding Tested : {shield}")
    print(f"Database Status  : Successfully recorded to '{DB_NAME}'")

    # Plot Frequency vs Relative Power Spectrum
    plt.figure(figsize=(10, 4.5))
    plt.plot(freq_axis, psd_dbfs, color="royalblue", lw=0.8, label="PSD")
    plt.axhline(peak_power, color="crimson", linestyle="--", alpha=0.7, label=f"Peak: {peak_power:.1f} dBFS")
    plt.title(f"RF Spectrum ({center_freq_mhz} MHz) | Exposure: {exposure_index} | Shield: {shield}")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Relative Power (dBFS)")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run with aluminum foil shielding
    run_rf_scan(center_freq_mhz=98.3, gain_val=20.0, shield="Aluminium_Foil")