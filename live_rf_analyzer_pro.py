#!/usr/bin/env python3
"""
live_rf_analyzer_pro.py
Advanced SDR Spectrum Analyzer & Environmental RF Exposure Monitor

Features:
- Swept-FFT engine with sub-band overlap stitching
- Offset tuning & transient discarding for zero-DC artifact
- Live, Max-Hold, and Min-Hold spectral traces
- Dynamic thresholding for Channel Occupation Rate (OR)
- Total integrated channel power & Exposure Index calculation
- Shielding effectiveness attenuation mode
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons
from rtlsdr import RtlSdr

class AdvancedRFAnalyzer:
    def __init__(self, start_freq=88e6, stop_freq=108e6, sample_rate=2.4e6, 
                 fft_size=1024, overlap_ratio=0.20, db_offset=0.0):
        self.start_freq = start_freq
        self.stop_freq = stop_freq
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.overlap_ratio = overlap_ratio
        self.db_offset = db_offset  # Calibration offset to approximate dBm

        # Offset tuning parameters to avoid central DC LO leakage
        self.lo_offset = 250e3  # 250 kHz shift
        self.drop_transients = 1024  # Discard initial samples post-tune

        # Calculate sub-band step size based on overlap
        self.effective_step = self.sample_rate * (1.0 - self.overlap_ratio)
        self.center_freqs = np.arange(
            self.start_freq + self.sample_rate / 2.0,
            self.stop_freq + self.sample_rate / 2.0,
            self.effective_step
        )

        # Initialize SDR Hardware
        self.sdr = RtlSdr()
        self.sdr.sample_rate = self.sample_rate
        self.sdr.gain = 'auto'

        # Precompute global frequency axis
        self.freq_axis = None
        self.live_psd = None
        self.max_hold = None
        self.min_hold = None

        # Exposure and Shielding metrics
        self.baseline_power = None
        self.shielded_power = None
        self.total_power_db = -100.0
        self.exposure_status = "LOW"
        self.occupancy_rate = 0.0

    def capture_subband_psd(self, center_freq):
        """Captures samples with offset tuning, drops transients, and computes PSD."""
        # Tune LO offset away from center
        actual_tune_freq = center_freq + self.lo_offset
        self.sdr.center_freq = actual_tune_freq

        # Read samples and drop the settling transient buffer
        raw_samples = self.sdr.read_samples(self.drop_transients + self.fft_size * 4)
        clean_samples = raw_samples[self.drop_transients : self.drop_transients + self.fft_size * 4]

        # Shift samples back by -lo_offset in baseband
        t = np.arange(len(clean_samples)) / self.sample_rate
        shifted_samples = clean_samples * np.exp(-1j * 2 * np.pi * self.lo_offset * t)

        # Truncate to FFT window and apply Hann window
        windowed_samples = shifted_samples[:self.fft_size] * np.hanning(self.fft_size)
        fft_data = np.fft.fftshift(np.fft.fft(windowed_samples, n=self.fft_size))
        
        # PSD in dBFS + calibration offset
        psd = 10 * np.log10((np.abs(fft_data) ** 2) / (self.fft_size * self.sample_rate) + 1e-12)
        psd += self.db_offset

        sub_freqs = np.fft.fftshift(np.fft.fftfreq(self.fft_size, d=1.0/self.sample_rate)) + center_freq
        return sub_freqs, psd

    def sweep_spectrum(self):
        """Executes sequential sub-band sweep and stitches overlapping regions."""
        stitched_freqs = []
        stitched_psd = []

        # Margin to trim from both edges due to filter roll-off
        trim_bins = int((self.fft_size * self.overlap_ratio) // 2)

        for cf in self.center_freqs:
            sub_f, sub_p = self.capture_subband_psd(cf)
            if trim_bins > 0:
                sub_f = sub_f[trim_bins:-trim_bins]
                sub_p = sub_p[trim_bins:-trim_bins]

            stitched_freqs.extend(sub_f)
            stitched_psd.extend(sub_p)

        stitched_freqs = np.array(stitched_freqs)
        stitched_psd = np.array(stitched_psd)

        # Crop to the exact user-specified range
        valid_indices = np.where((stitched_freqs >= self.start_freq) & (stitched_freqs <= self.stop_freq))[0]
        self.freq_axis = stitched_freqs[valid_indices] / 1e6  # Output in MHz
        self.live_psd = stitched_psd[valid_indices]

        # Update Trace Histories
        if self.max_hold is None:
            self.max_hold = np.copy(self.live_psd)
            self.min_hold = np.copy(self.live_psd)
        else:
            self.max_hold = np.maximum(self.max_hold, self.live_psd)
            self.min_hold = np.minimum(self.min_hold, self.live_psd)

        # Calculate Analytics
        self.compute_channel_metrics()

    def compute_channel_metrics(self):
        """Calculates total channel power, exposure categorization, and occupancy."""
        # Total integrated linear power
        linear_power = np.sum(10 ** (self.live_psd / 10.0))
        self.total_power_db = 10 * np.log10(linear_power + 1e-12)

        # Dynamic Energy Detection threshold: Median noise floor + 10 dB offset
        noise_floor = np.median(self.live_psd)
        dynamic_threshold = noise_floor + 10.0
        occupied_bins = np.sum(self.live_psd > dynamic_threshold)
        self.occupancy_rate = (occupied_bins / len(self.live_psd)) * 100.0

        # RF Exposure Classification
        if self.total_power_db < -50.0:
            self.exposure_status = "LOW (Ambient / Safe)"
        elif -50.0 <= self.total_power_db <= -25.0:
            self.exposure_status = "MEDIUM (Moderate Proximity)"
        else:
            self.exposure_status = "HIGH (High Field Concentration)"

    def reset_traces(self, event=None):
        """Clears Max-Hold and Min-Hold buffers."""
        self.max_hold = None
        self.min_hold = None

    def close(self):
        self.sdr.close()


def launch_gui():
    analyzer = AdvancedRFAnalyzer(start_freq=88e6, stop_freq=108e6)

    fig, ax = plt.subplots(figsize=(11, 6))
    plt.subplots_adjust(bottom=0.25)

    analyzer.sweep_spectrum()

    line_live, = ax.plot(analyzer.freq_axis, analyzer.live_psd, color='cyan', lw=1.0, label='Live Trace')
    line_max, = ax.plot(analyzer.freq_axis, analyzer.max_hold, color='yellow', lw=1.2, linestyle='--', label='Max-Hold')
    line_min, = ax.plot(analyzer.freq_axis, analyzer.min_hold, color='magenta', lw=0.8, alpha=0.6, label='Min-Hold')

    ax.set_title("Real-Time Swept Spectrum Analyzer & RF Exposure Monitor", fontsize=12, fontweight='bold')
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Power Density (dBFS / Calibrated Scale)")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right')

    # Info text displays
    info_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                        fontsize=9, color='white')

    # GUI Buttons
    reset_ax = plt.axes([0.15, 0.08, 0.15, 0.075])
    btn_reset = Button(reset_ax, 'Reset Max/Min')
    btn_reset.on_clicked(analyzer.reset_traces)

    ref_ax = plt.axes([0.35, 0.08, 0.18, 0.075])
    btn_ref = Button(ref_ax, 'Set Shield Baseline')

    def set_baseline(event):
        analyzer.baseline_power = analyzer.total_power_db
    btn_ref.on_clicked(set_baseline)

    shield_ax = plt.axes([0.55, 0.08, 0.18, 0.075])
    btn_shield = Button(shield_ax, 'Log Shielded Power')

    def log_shield(event):
        analyzer.shielded_power = analyzer.total_power_db
    btn_shield.on_clicked(log_shield)

    # Main update loop
    plt.ion()
    plt.show()

    try:
        while plt.fignum_exists(fig.number):
            analyzer.sweep_spectrum()

            # Update trace lines
            line_live.set_ydata(analyzer.live_psd)
            line_max.set_ydata(analyzer.max_hold)
            line_min.set_ydata(analyzer.min_hold)

            ax.set_ylim(np.min(analyzer.min_hold) - 5, np.max(analyzer.max_hold) + 5)

            # Shielding attenuation string
            atten_str = "N/A"
            if analyzer.baseline_power is not None and analyzer.shielded_power is not None:
                attenuation = analyzer.baseline_power - analyzer.shielded_power
                atten_str = f"{attenuation:.2f} dB"

            # Overlay dashboard
            info_text.set_text(
                f"Total Power: {analyzer.total_power_db:.1f} dBFS | Exposure: {analyzer.exposure_status}\n"
                f"Band Occupancy Rate (OR): {analyzer.occupancy_rate:.1f}%\n"
                f"Shielding Attenuation (AdB): {atten_str}"
            )

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        analyzer.close()

if __name__ == '__main__':
    launch_gui()