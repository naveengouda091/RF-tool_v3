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
        CREATE TABLE IF NOT EXISTS swept_band_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            band_name TEXT,
            start_mhz REAL,
            stop_mhz REAL,
            peak_power_dbfs REAL,
            avg_power_dbfs REAL,
            noise_floor_dbfs REAL,
            occupation_rate_pct REAL
        )
    """)
    conn.commit()
    conn.close()

def process_iq_block(samples, sample_rate, trim_fraction=0.15):
    """Normalizes samples, applies Blackman window, and trims filter roll-off."""
    n = len(samples)
    window = np.blackman(n)
    norm_samples = samples / (np.max(np.abs(samples)) + 1e-9)
    fft_vals = np.fft.fftshift(np.fft.fft(norm_samples * window))
    
    psd_linear = (np.abs(fft_vals) / np.sum(window)) ** 2
    psd_dbfs = 10.0 * np.log10(np.maximum(psd_linear, 1e-12))
    freq_rel = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / sample_rate))
    
    trim = int(n * trim_fraction)
    return freq_rel[trim:-trim], psd_dbfs[trim:-trim]

def sweep_band(sdr, start_mhz, stop_mhz, sample_rate=2.4e6):
    usable_step = (sample_rate * 0.70) / 1e6
    center_freqs = np.arange(start_mhz + (usable_step / 2.0), stop_mhz, usable_step)
    
    stitched_f = []
    stitched_p = []
    
    for fc in center_freqs:
        sdr.center_freq = fc * 1e6
        time.sleep(0.012)
        _ = sdr.read_samples(4096)  # Discard LO transient DC spike
        samples = sdr.read_samples(32768)
        f_sub, p_sub = process_iq_block(samples, sample_rate)
        stitched_f.extend((fc * 1e6 + f_sub) / 1e6)
        stitched_p.extend(p_sub)
        
    return np.array(stitched_f), np.array(stitched_p)

def run_multiband_study():
    init_db()
    bands = [
        {"name": "FM Broadcast", "start": 88.0, "stop": 108.0, "gain": 20.0},
        {"name": "GSM-900 Cellular", "start": 925.0, "stop": 960.0, "gain": 29.7}
    ]
    
    results = []
    
    sdr = RtlSdr()
    sdr.sample_rate = 2.4e6
    sdr.direct_sampling = 0
    
    try:
        for b in bands:
            print(f"[*] Scanning {b['name']} ({b['start']} to {b['stop']} MHz)...")
            sdr.gain = b['gain']
            f_axis, p_axis = sweep_band(sdr, b['start'], b['stop'])
            
            # Energy Detection & Occupancy (Slimeni et al. model)
            noise_est = float(np.median(p_axis))
            threshold = noise_est + 10.0
            occupied = p_axis > threshold
            or_rate = float(np.sum(occupied) / len(p_axis) * 100.0)
            peak_p = float(np.max(p_axis))
            avg_p = float(10.0 * np.log10(np.mean(10.0 ** (p_axis / 10.0))))
            
            # Log to SQLite
            conn = sqlite3.connect(DB_NAME)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO swept_band_metrics 
                (band_name, start_mhz, stop_mhz, peak_power_dbfs, avg_power_dbfs, noise_floor_dbfs, occupation_rate_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (b['name'], b['start'], b['stop'], peak_p, avg_p, noise_est, or_rate))
            conn.commit()
            conn.close()
            
            results.append({
                "name": b['name'], "freqs": f_axis, "psd": p_axis, 
                "threshold": threshold, "peak": peak_p, "avg": avg_p,
                "noise": noise_est, "or": or_rate, "start": b['start'], "stop": b['stop']
            })
    finally:
        sdr.close()
        
    # Generate Dual-Band Figure
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    
    for i, res in enumerate(results):
        ax = axes[i]
        ax.plot(res['freqs'], res['psd'], color='navy', lw=0.6, label='Normalized PSD')
        ax.axhline(res['threshold'], color='crimson', linestyle='--', lw=1.2, 
                   label=f"Energy Detection Threshold ({res['threshold']:.1f} dBFS)")
        ax.set_title(f"{res['name']} | Peak: {res['peak']:.1f} dBFS | Noise Floor: {res['noise']:.1f} dBFS | OR: {res['or']:.1f}%", fontweight='bold')
        ax.set_ylabel("Power (dBFS)")
        ax.set_xlim(res['start'], res['stop'])
        ax.set_ylim(-85, 0)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right")
        
    axes[1].set_xlabel("Frequency (MHz)")
    plt.tight_layout()
    plt.savefig("multiband_survey_analysis.png", dpi=300)
    print("\n[+] Multiband sweep complete! Saved comparison plot as 'multiband_survey_analysis.png'")
    plt.show()

if __name__ == "__main__":
    run_multiband_study()