import sys
import time
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from rtlsdr import RtlSdr

class EnhancedSpectrumWaterfallApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced Real-Time SDR Spectrum Analyzer (RTSA)")
        self.resize(1280, 880)

        self.sdr = None
        self.is_running = False
        self.fs = 2.048e6          # Safe, stable USB sample rate for RTL-SDR Blog V3
        self.chunk_size = 16384    # Safe hardware transfer block (multiples of 512)
        self.fft_size = 2048       # Visual FFT resolution
        self.history_depth = 220
        self.window = np.blackman(self.fft_size)
        self.window_power = np.sum(self.window)

        # Buffers
        self.max_hold_trace = np.full(self.fft_size, -120.0)
        self.waterfall_data = np.full((self.history_depth, self.fft_size), -90.0)
        self.cal_offset_db = 15.0

        self.setup_ui()

    def setup_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)

        # ---------------- CONTROL DOCK ----------------
        control_panel = QtWidgets.QVBoxLayout()
        main_layout.addLayout(control_panel, stretch=1)

        control_panel.addWidget(QtWidgets.QLabel("<h3>RF Spectrum Analyzer</h3><p style='color:gray;'>RTSA & Advanced Traces</p>"))

        control_panel.addWidget(QtWidgets.QLabel("<b>Target Band Preset:</b>"))
        self.combo_band = QtWidgets.QComboBox()
        self.combo_band.addItems([
            "FM Broadcast (98.3 MHz)",
            "GSM-900 Cellular (942.5 MHz)",
            "ISM Band (433.92 MHz)",
            "Airband ATC (125.0 MHz)"
        ])
        self.combo_band.currentIndexChanged.connect(self.change_preset)
        control_panel.addWidget(self.combo_band)

        control_panel.addWidget(QtWidgets.QLabel("<b>Center Frequency (MHz):</b>"))
        self.freq_spin = QtWidgets.QDoubleSpinBox()
        self.freq_spin.setRange(24.0, 1750.0)
        self.freq_spin.setValue(98.3)
        self.freq_spin.setDecimals(3)
        self.freq_spin.valueChanged.connect(self.update_freq)
        control_panel.addWidget(self.freq_spin)

        control_panel.addWidget(QtWidgets.QLabel("<b>Hardware RF Gain:</b>"))
        self.slider_gain = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_gain.setRange(0, 49)
        self.slider_gain.setValue(25)
        self.lbl_gain = QtWidgets.QLabel("25.0 dB")
        self.slider_gain.valueChanged.connect(lambda v: self.lbl_gain.setText(f"{v:.1f} dB"))
        self.slider_gain.valueChanged.connect(self.update_gain)
        control_panel.addWidget(self.slider_gain)
        control_panel.addWidget(self.lbl_gain)

        control_panel.addSpacing(10)
        self.btn_run = QtWidgets.QPushButton("Start Live Acquisition")
        self.btn_run.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 9px;")
        self.btn_run.clicked.connect(self.toggle_stream)
        control_panel.addWidget(self.btn_run)

        self.btn_reset_max = QtWidgets.QPushButton("Reset Max-Hold Trace")
        self.btn_reset_max.setStyleSheet("background-color: #37474f; color: white; font-weight: bold; padding: 7px;")
        self.btn_reset_max.clicked.connect(self.reset_max_hold)
        control_panel.addWidget(self.btn_reset_max)

        control_panel.addSpacing(15)
        metrics_group = QtWidgets.QGroupBox("Channel Telemetry")
        m_layout = QtWidgets.QVBoxLayout(metrics_group)
        self.hud_peak = QtWidgets.QLabel("Peak Power: -- dBFS")
        self.hud_dbm = QtWidgets.QLabel("Est. Peak: -- dBm")
        self.hud_noise = QtWidgets.QLabel("Noise Floor: -- dBFS")
        self.hud_snr = QtWidgets.QLabel("Estimated SNR: -- dB")
        self.hud_occupancy = QtWidgets.QLabel("Occupancy (OR): -- %")
        
        for lbl in [self.hud_peak, self.hud_dbm, self.hud_noise, self.hud_snr, self.hud_occupancy]:
            lbl.setStyleSheet("font-size: 11px; font-weight: bold; color: #0d47a1;")
            m_layout.addWidget(lbl)
        control_panel.addWidget(metrics_group)
        control_panel.addStretch()

        # ---------------- VIEWPORT ----------------
        plot_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(plot_layout, stretch=4)

        # Spectrum Plot
        self.spectrum_plot = pg.PlotWidget(title="Power Spectral Density (Cyan: Instantaneous | Yellow: Max-Hold)")
        self.spectrum_plot.setLabel('left', "Power", units="dBFS")
        self.spectrum_plot.setLabel('bottom', "Frequency", units="MHz")
        self.spectrum_plot.setYRange(-90, 5)
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.4)
        
        self.curve_max = self.spectrum_plot.plot(pen=pg.mkPen('#ffd600', width=1.1, style=QtCore.Qt.DotLine), name="Max-Hold")
        self.curve_psd = self.spectrum_plot.plot(pen=pg.mkPen('#00e5ff', width=1.3), name="Real-Time")
        self.thresh_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('r', style=QtCore.Qt.DashLine, width=1.4))
        self.spectrum_plot.addItem(self.thresh_line)
        plot_layout.addWidget(self.spectrum_plot, stretch=2)

        # Waterfall Spectrogram
        self.waterfall_plot = pg.PlotWidget(title="Live Spectrogram History (Waterfall)")
        self.waterfall_plot.setLabel('left', "Time Bins")
        self.waterfall_plot.setLabel('bottom', "Frequency", units="MHz")
        self.waterfall_img = pg.ImageItem()
        self.waterfall_plot.addItem(self.waterfall_img)
        self.waterfall_img.setColorMap(pg.colormap.get('inferno'))
        plot_layout.addWidget(self.waterfall_plot, stretch=2)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.process_frame)

    def reset_max_hold(self):
        self.max_hold_trace = np.full(self.fft_size, -120.0)

    def change_preset(self, idx):
        presets = [98.3, 942.5, 433.92, 125.0]
        self.freq_spin.setValue(presets[idx])

    def update_freq(self, val):
        if self.sdr:
            try:
                self.sdr.center_freq = val * 1e6
                time.sleep(0.01)
                _ = self.sdr.read_samples(4096)
            except Exception:
                pass
        self.reset_max_hold()

    def update_gain(self, val):
        if self.sdr:
            try:
                self.sdr.gain = float(val)
            except Exception:
                pass

    def toggle_stream(self):
        if not self.is_running:
            try:
                self.sdr = RtlSdr()
                self.sdr.sample_rate = self.fs
                self.sdr.center_freq = self.freq_spin.value() * 1e6
                self.sdr.gain = float(self.slider_gain.value())
                self.sdr.direct_sampling = 0

                time.sleep(0.02)
                _ = self.sdr.read_samples(8192)

                self.reset_max_hold()
                self.is_running = True
                self.btn_run.setText("Stop Acquisition")
                self.btn_run.setStyleSheet("background-color: #c62828; color: white; font-weight: bold; padding: 9px;")
                self.timer.start(40)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Hardware Error", f"Unable to open SDR:\n{e}")
        else:
            self.timer.stop()
            if self.sdr:
                try:
                    self.sdr.close()
                except Exception:
                    pass
                self.sdr = None
            self.is_running = False
            self.btn_run.setText("Start Live Acquisition")
            self.btn_run.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 9px;")

    def process_frame(self):
        if not self.sdr:
            return

        try:
            # Safe USB block read
            raw_samples = self.sdr.read_samples(self.chunk_size)
            samples = raw_samples[:self.fft_size]
        except Exception:
            # Guard against occasional USB bus glitch or frame drop
            return

        norm_samples = samples / (np.max(np.abs(samples)) + 1e-9)
        fft_vals = np.fft.fftshift(np.fft.fft(norm_samples * self.window))
        psd_linear = (np.abs(fft_vals) / self.window_power) ** 2
        psd_dbfs = 10.0 * np.log10(np.maximum(psd_linear, 1e-12))

        # Max-Hold persistence
        self.max_hold_trace = np.maximum(self.max_hold_trace, psd_dbfs)

        noise_floor = float(np.median(psd_dbfs))
        threshold = noise_floor + 10.0
        occupied_bins = psd_dbfs > threshold
        occupation_rate = float(np.sum(occupied_bins) / self.fft_size * 100.0)
        peak_power = float(np.max(psd_dbfs))
        est_dbm = peak_power - self.cal_offset_db

        fc = self.sdr.center_freq / 1e6
        f_rel = np.fft.fftshift(np.fft.fftfreq(self.fft_size, d=1.0 / self.fs)) / 1e6
        freq_axis = fc + f_rel

        # Push to plot
        self.curve_psd.setData(freq_axis, psd_dbfs)
        self.curve_max.setData(freq_axis, self.max_hold_trace)
        self.thresh_line.setValue(threshold)

        # Push to waterfall
        self.waterfall_data = np.roll(self.waterfall_data, 1, axis=0)
        self.waterfall_data[0, :] = psd_dbfs
        self.waterfall_img.setImage(self.waterfall_data.T, autoLevels=False)
        self.waterfall_img.setLevels([-85, -20])
        self.waterfall_img.setRect(QtCore.QRectF(freq_axis[0], 0, self.fs / 1e6, self.history_depth))

        # Update HUD
        self.hud_peak.setText(f"Peak Power: {peak_power:.1f} dBFS")
        self.hud_dbm.setText(f"Est. Power: {est_dbm:.1f} dBm (uncal)")
        self.hud_noise.setText(f"Noise Floor: {noise_floor:.1f} dBFS")
        self.hud_snr.setText(f"Estimated SNR: {max(0.0, peak_power - noise_floor):.1f} dB")
        self.hud_occupancy.setText(f"Occupancy (OR): {occupation_rate:.1f} %")

    def closeEvent(self, event):
        if self.is_running and self.sdr:
            try:
                self.sdr.close()
            except Exception:
                pass
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = EnhancedSpectrumWaterfallApp()
    win.show()
    sys.exit(app.exec_())