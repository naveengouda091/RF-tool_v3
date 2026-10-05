import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from rtlsdr import RtlSdr

class LiveSpectrumWaterfallApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time SDR Spectrum Analyzer & Waterfall Spectrogram")
        self.resize(1200, 850)

        # SDR & Acquisition Parameters
        self.sdr = None
        self.is_running = False
        self.fs = 2.4e6          # 2.4 MSps standard sampling rate
        self.fft_size = 2048     # Low-latency real-time FFT size
        self.history_depth = 200 # Waterfall time buffer depth
        self.window = np.blackman(self.fft_size)
        self.window_power = np.sum(self.window)

        # Waterfall 2D Ring Buffer
        self.waterfall_data = np.full((self.history_depth, self.fft_size), -90.0)

        self.setup_ui()

    def setup_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)

        # ------------------ LEFT SIDEBAR: CONTROLS & HUD ------------------
        control_panel = QtWidgets.QVBoxLayout()
        main_layout.addLayout(control_panel, stretch=1)

        header = QtWidgets.QLabel("<h3>RF Spectrum Analyzer</h3><p style='color:gray;'>Live DSP & Energy Detection</p>")
        control_panel.addWidget(header)

        # Frequency Selection Preset
        control_panel.addWidget(QtWidgets.QLabel("<b>Preset Target Band:</b>"))
        self.combo_band = QtWidgets.QComboBox()
        self.combo_band.addItems([
            "FM Broadcast (98.3 MHz)",
            "GSM-900 Cellular (942.5 MHz)",
            "ISM Band (433.92 MHz)",
            "Airband ATC (125.0 MHz)"
        ])
        self.combo_band.currentIndexChanged.connect(self.change_frequency)
        control_panel.addWidget(self.combo_band)

        # Fine-Tune Frequency Entry
        control_panel.addWidget(QtWidgets.QLabel("<b>Center Frequency (MHz):</b>"))
        self.freq_spin = QtWidgets.QDoubleSpinBox()
        self.freq_spin.setRange(24.0, 1750.0)
        self.freq_spin.setValue(98.3)
        self.freq_spin.setDecimals(3)
        self.freq_spin.valueChanged.connect(self.update_manual_freq)
        control_panel.addWidget(self.freq_spin)

        # Hardware RF Gain Slider
        control_panel.addWidget(QtWidgets.QLabel("<b>Hardware RF Gain:</b>"))
        self.slider_gain = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_gain.setRange(0, 49)
        self.slider_gain.setValue(25)
        self.lbl_gain = QtWidgets.QLabel("25.0 dB")
        self.slider_gain.valueChanged.connect(self.update_gain)
        control_panel.addWidget(self.slider_gain)
        control_panel.addWidget(self.lbl_gain)

        # Action Buttons
        control_panel.addSpacing(10)
        self.btn_run = QtWidgets.QPushButton("Start Live Acquisition")
        self.btn_run.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 10px; font-size: 13px;")
        self.btn_run.clicked.connect(self.toggle_stream)
        control_panel.addWidget(self.btn_run)

        # Real-time Metrics Card
        control_panel.addSpacing(15)
        metrics_group = QtWidgets.QGroupBox("Live Channel Telemetry")
        m_layout = QtWidgets.QVBoxLayout(metrics_group)
        self.hud_peak = QtWidgets.QLabel("Peak Power: -- dBFS")
        self.hud_noise = QtWidgets.QLabel("Noise Floor: -- dBFS")
        self.hud_snr = QtWidgets.QLabel("Estimated SNR: -- dB")
        self.hud_occupancy = QtWidgets.QLabel("Occupancy (OR): -- %")
        
        for lbl in [self.hud_peak, self.hud_noise, self.hud_snr, self.hud_occupancy]:
            lbl.setStyleSheet("font-size: 12px; font-weight: bold; color: #0d47a1;")
            m_layout.addWidget(lbl)
        control_panel.addWidget(metrics_group)
        control_panel.addStretch()

        # ------------------ RIGHT VIEWPORT: SPECTRUM + WATERFALL ------------------
        plot_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(plot_layout, stretch=4)

        # 1. Real-Time PSD Line Plot (Top)
        self.spectrum_plot = pg.PlotWidget(title="Live Power Spectral Density (Normalized PSD)")
        self.spectrum_plot.setLabel('left', "Power", units="dBFS")
        self.spectrum_plot.setLabel('bottom', "Frequency", units="MHz")
        self.spectrum_plot.setYRange(-90, 5)
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.4)
        self.curve_psd = self.spectrum_plot.plot(pen=pg.mkPen('#00e5ff', width=1.3))
        
        # Occupancy Threshold Line (Slimeni et al.)
        self.thresh_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('r', style=QtCore.Qt.DashLine, width=1.5))
        self.spectrum_plot.addItem(self.thresh_line)
        plot_layout.addWidget(self.spectrum_plot, stretch=2)

        # 2. Continuous Scrolling Spectrogram / Waterfall (Bottom)
        self.waterfall_plot = pg.PlotWidget(title="Live Spectrogram History (Waterfall)")
        self.waterfall_plot.setLabel('left', "Time Bins (History)")
        self.waterfall_plot.setLabel('bottom', "Frequency", units="MHz")
        self.waterfall_img = pg.ImageItem()
        self.waterfall_plot.addItem(self.waterfall_img)
        
        # Colormap (Inferno / High Contrast)
        colormap = pg.colormap.get('inferno')
        self.waterfall_img.setColorMap(colormap)
        plot_layout.addWidget(self.waterfall_plot, stretch=2)

        # High-Rate Acquisition Loop
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.process_next_frame)

    def change_frequency(self, idx):
        presets = [98.3, 942.5, 433.92, 125.0]
        self.freq_spin.setValue(presets[idx])

    def update_manual_freq(self, val):
        if self.sdr:
            self.sdr.center_freq = val * 1e6

    def update_gain(self, val):
        self.lbl_gain.setText(f"{val:.1f} dB")
        if self.sdr:
            self.sdr.gain = float(val)

    def toggle_stream(self):
        if not self.is_running:
            try:
                self.sdr = RtlSdr()
                self.sdr.sample_rate = self.fs
                self.sdr.center_freq = self.freq_spin.value() * 1e6
                self.sdr.gain = float(self.slider_gain.value())
                self.sdr.direct_sampling = 0

                # Discard transient settling samples (Flak et al.)
                _ = self.sdr.read_samples(4096)

                self.is_running = True
                self.btn_run.setText("Stop Acquisition")
                self.btn_run.setStyleSheet("background-color: #c62828; color: white; font-weight: bold; padding: 10px; font-size: 13px;")
                self.timer.start(30)  # ~33 FPS rendering
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Hardware Initialization Error", f"Unable to communicate with RTL-SDR:\n{e}")
        else:
            self.timer.stop()
            if self.sdr:
                self.sdr.close()
                self.sdr = None
            self.is_running = False
            self.btn_run.setText("Start Live Acquisition")
            self.btn_run.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 10px; font-size: 13px;")

    def process_next_frame(self):
        if not self.sdr:
            return

        # 1. Read Baseband IQ Samples
        samples = self.sdr.read_samples(self.fft_size)

        # 2. Windowed FFT & Normalization (Șorecău et al.)
        norm_samples = samples / (np.max(np.abs(samples)) + 1e-9)
        fft_vals = np.fft.fftshift(np.fft.fft(norm_samples * self.window))
        psd_linear = (np.abs(fft_vals) / self.window_power) ** 2
        psd_dbfs = 10.0 * np.log10(np.maximum(psd_linear, 1e-12))

        # 3. Dynamic Threshold & Occupancy (Slimeni et al.)
        noise_floor = float(np.median(psd_dbfs))
        threshold = noise_floor + 10.0
        occupied_bins = psd_dbfs > threshold
        occupation_rate = float(np.sum(occupied_bins) / self.fft_size * 100.0)
        peak_power = float(np.max(psd_dbfs))
        snr_est = max(0.0, peak_power - noise_floor)

        # 4. Frequency Axis Mapping
        fc = self.sdr.center_freq / 1e6
        f_rel = np.fft.fftshift(np.fft.fftfreq(self.fft_size, d=1.0 / self.fs)) / 1e6
        freq_axis = fc + f_rel

        # 5. Push PSD Line Data
        self.curve_psd.setData(freq_axis, psd_dbfs)
        self.thresh_line.setValue(threshold)

        # 6. Push Waterfall Image Data
        self.waterfall_data = np.roll(self.waterfall_data, 1, axis=0)
        self.waterfall_data[0, :] = psd_dbfs
        self.waterfall_img.setImage(self.waterfall_data.T, autoLevels=False)
        self.waterfall_img.setLevels([-85, -20])

        # Align Image Bounds to Frequency Scale
        df = (self.fs / 1e6) / self.fft_size
        self.waterfall_img.setRect(QtCore.QRectF(freq_axis[0], 0, self.fs / 1e6, self.history_depth))

        # 7. Update Telemetry HUD
        self.hud_peak.setText(f"Peak Power: {peak_power:.1f} dBFS")
        self.hud_noise.setText(f"Noise Floor: {noise_floor:.1f} dBFS")
        self.hud_snr.setText(f"Estimated SNR: {snr_est:.1f} dB")
        self.hud_occupancy.setText(f"Occupancy (OR): {occupation_rate:.1f} %")

    def closeEvent(self, event):
        if self.is_running and self.sdr:
            self.sdr.close()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = LiveSpectrumWaterfallApp()
    window.show()
    sys.exit(app.exec_())