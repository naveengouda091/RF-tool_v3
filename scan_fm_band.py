import os
import sys

# Tell Windows and pyrtlsdr explicitly where the DLLs live
dll_folder = os.path.dirname(os.path.abspath(__file__))
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(dll_folder)
    os.add_dll_directory(r"C:\sdrsharp-x64")
os.environ['LIBRTLSDR_PATH'] = os.path.join(dll_folder, 'rtlsdr.dll')

# Now import rtlsdr and the rest
from rtlsdr import RtlSdr
import numpy as np
import matplotlib.pyplot as plt

def run_rf_snapshot(center_freq_hz=98.3e6, sample_rate=2.4e6, gain_db=20.0):
    print(f"[*] Initializing RTL-SDR at {center_freq_hz / 1e6:.1f} MHz...")
    sdr = RtlSdr()
    
    # Configure parameters matching your working SDR# settings
    sdr.sample_rate = sample_rate
    sdr.center_freq = center_freq_hz
    sdr.gain = gain_db

    # Settle local oscillator and discard initial transient buffer
    _ = sdr.read_samples(4096)

    # Read baseband IQ data (128k complex samples)
    num_samples = 131072
    samples = sdr.read_samples(num_samples)
    sdr.close()

    # Fast Fourier Transform with Hanning window
    window = np.hanning(num_samples)
    fft_complex = np.fft.fftshift(np.fft.fft(samples * window))
    
    # Power Spectral Density (dBFS)
    psd_linear = (np.abs(fft_complex) ** 2) / (num_samples * np.mean(window**2))
    psd_dbfs = 10.0 * np.log10(np.maximum(psd_linear, 1e-12))
    freq_axis = (np.fft.fftshift(np.fft.fftfreq(num_samples, d=1.0/sample_rate)) + center_freq_hz) / 1e6

    # Extract metrics
    peak_power = float(np.max(psd_dbfs))
    channel_integrated_power = float(10.0 * np.log10(np.mean(psd_linear)))

    # Compute custom RF Exposure Index
    if channel_integrated_power < -50.0:
        exposure_index = "LOW"
    elif channel_integrated_power < -30.0:
        exposure_index = "MEDIUM"
    else:
        exposure_index = "HIGH"

    return freq_axis, psd_dbfs, peak_power, channel_integrated_power, exposure_index

if __name__ == "__main__":
    freqs, psd, peak_p, total_p, exposure = run_rf_snapshot()

    print("\n--- RF Spectrum & Exposure Analysis ---")
    print(f"Target Frequency : 98.3 MHz")
    print(f"Peak Power       : {peak_p:.2f} dBFS")
    print(f"Integrated Power : {total_p:.2f} dBFS")
    print(f"Exposure Index   : {exposure}")

    # Plot frequency vs relative power
    plt.figure(figsize=(10, 4.5))
    plt.plot(freqs, psd, color="royalblue", lw=0.8, label="PSD")
    plt.axhline(peak_p, color="crimson", linestyle="--", alpha=0.7, label=f"Peak: {peak_p:.1f} dBFS")
    plt.title(f"Live Spectrum Scan (98.3 MHz) | RF Exposure Index: {exposure}")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Relative Power (dBFS)")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()