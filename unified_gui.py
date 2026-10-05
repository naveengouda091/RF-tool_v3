import sys
import time
import sqlite3
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from rtlsdr import RtlSdr

DB_NAME = "rf_survey_database.db"

class RFMonitorApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SDR RF Noise Monitoring & Exposure Analyzer")
        self.resize(1100, 750)
        
        # Central SDR object
        self.sdr = None
        self.is_scanning = False
        
        # Setup GUI Layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)
        
        # Left Panel: Controls & Metrics
        control_panel = QtWidgets.QVBoxLayout()
        main_layout.addLayout(control_panel, stretch=1)
        
        # Band Selection
        control_panel.addWidget(QtWidgets.QLabel("<b>Select Target Band:</b>"))
        self.band_combo = QtWidgets.QComboBox()
        self.band_combo.addItems([
            "FM Broadcast (98.3 MHz)",
            "GSM-900 Cellular (942.5 MHz)",
            "DCS / LTE-B3 (1840.0 MHz)"
        ])
        control_panel.addWidget(self.band_combo)
        
        # Hardware Gain Slider
        control_panel.addWidget(QtWidgets.QLabel("<b>RF Tuner Gain:</b>"))
        self.gain_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gain_slider.setRange(0, 49)
        self.gain_slider.setValue(25)
        self.gain_label = QtWidgets.QLabel("25.0 dB")
        self.gain_slider.valueChanged.connect(lambda v: self.gain_label.setText(f"{v:.1f} dB"))
        control_panel.addWidget(self.gain_slider)
        control_panel.addWidget(self.gain_label)
        
        # Shielding Selection
        control_panel.addWidget(QtWidgets.QLabel("<b>Shielding Material Under Test:</b>"))
        self.shield_combo = QtWidgets.QComboBox()
        self.shield_combo.addItems(["None", "Aluminium_Foil", "Copper_Mesh", "Plastic_Enclosure"])
        control_panel.addWidget(self.shield_combo)
        
        # Location Coordinates
        control_panel.addWidget(QtWidgets.QLabel("<b>Latitude / Longitude:</b>"))
        self.lat_input = QtWidgets.QLineEdit("15.3218")
        self.lon_input = QtWidgets.QLineEdit("74.7646")
        control_panel.addWidget(self.lat_input)
        control_panel.addWidget(self.lon_input)
        
        # Action Buttons
        self.start_btn = QtWidgets.QPushButton("Start Live Acquisition")
        self.start_btn.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 6px;")
        self.start_btn.clicked.connect(self.toggle_stream)
        control_panel.addWidget(self.start_btn)
        
        self.log_btn = QtWidgets.QPushButton("Log Entry to Database")
        self.log_btn.setStyleSheet("background-color: #0277bd; color: white; font-weight: bold; padding: 6px;")
        self.log_btn.clicked.connect(self.log_current_state)
        control_panel.addWidget(self.log_btn)
        
        # Metric Display Cards
        control_panel.addSpacing(15)
        control_panel.addWidget(QtWidgets.QLabel("<b>Live Analysis:</b>"))
        self.metric_peak = QtWidgets.QLabel("Peak Power: -- dBFS")
        self.metric_avg = QtWidgets.QLabel("Channel Power: -- dBFS")
        self.metric_or = QtWidgets.QLabel("Occupation Rate: -- %")
        self.metric_index = QtWidgets.QLabel("Exposure Index: --")
        self.metric_index.setStyleSheet("font-size: 14px; font-weight: bold; color: green;")
        
        for w in [self.metric_peak, self.metric_avg, self.metric_or, self.metric_index]:
            control_panel.addWidget(w)
        control_panel.addStretch()
        
        # Right Panel: Live Spectrum & Waterfall
        plot_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(plot_layout, stretch=3)
        
        # 1. Spectrum Plot (Power vs Freq)
        self.plot_spectrum = pg.PlotWidget(title="Live Power Spectral Density (PSD)")
        self.plot_spectrum.setLabel('bottom', "Frequency", units='MHz')
        self.plot_spectrum.setLabel('left', "Power", units='dBFS')
        self.plot_spectrum.setYRange(-90, 5)
        self.plot_spectrum.showGrid(x=True, y=True, alpha=0.5)
        self.curve_psd = self.plot_spectrum.plot(pen=pg.mkPen('c', width=1.0))
        self.threshold_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('r', style=QtCore.Qt.DashLine))
        self.plot_spectrum.addItem(self.threshold_line)
        plot_layout.addWidget(self.plot_spectrum)
        
        # 2. Waterfall Heatmap
        self.plot_waterfall = pg.PlotWidget(title="Spectrogram / Waterfall History")
        self.plot_waterfall.setLabel('bottom', "Frequency", units='MHz')
        self.plot_waterfall.setLabel('left', "Time (Frames)")
        self.waterfall_img = pg.ImageItem()
        self.plot_waterfall.addItem(self.waterfall_img)
        
        colormap = pg.colormap.get('inferno')
        self.waterfall_img.setColorMap(colormap)
        plot_layout.addWidget(self.plot_waterfall)
        
        # Waterfall Buffer (100 history rows)
        self.waterfall_history = np.zeros((100, 1024))
        
        # Timer for continuous capture
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)

    def toggle_stream(self):
        if not self.is_scanning:
            try:
                self.sdr = RtlSdr()
                self.sdr.sample_rate = 2.048e6
                self.sdr.direct_sampling = 0
                self.sdr.gain = float(self.gain_slider.value())
                self.update_tuner_freq()
                
                self.is_scanning = True
                self.start_btn.setText("Stop Acquisition")
                self.start_btn.setStyleSheet("background-color: #c62828; color: white; font-weight: bold; padding: 6px;")
                self.timer.start(50)  # ~20 FPS refresh
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Hardware Error", f"Unable to open RTL-SDR:\n{e}")
        else:
            self.timer.stop()
            if self.sdr:
                self.sdr.close()
                self.sdr = None
            self.is_scanning = False
            self.start_btn.setText("Start Live Acquisition")
            self.start_btn.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 6px;")

    def update_tuner_freq(self):
        idx = self.band_combo.currentIndex()
        if idx == 0:
            self.current_center_freq = 98.3e6
        elif idx == 1:
            self.current_center_freq = 942.5e6
        else:
            self.current_center_freq = 1840.0e6
        if self.sdr:
            self.sdr.center_freq = self.current_center_freq
            self.sdr.gain = float(self.gain_slider.value())

    def update_frame(self):
        if not self.sdr:
            return
        
        # Re-check settings if changed
        self.update_tuner_freq()
        
        # Acquire samples (16k FFT points)
        n = 16384
        samples = self.sdr.read_samples(n)
        
        # Blackman windowed FFT & Normalized PSD
        window = np.blackman(n)
        norm_samples = samples / np.max(np.abs(samples) + 1e-9)
        fft_vals = np.fft.fftshift(np.fft.fft(norm_samples * window))
        psd = (np.abs(fft_vals) / np.sum(window)) ** 2
        psd_dbfs = 10.0 * np.log10(np.maximum(psd, 1e-12))
        
        # Frequency scale
        freq_rel = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / self.sdr.sample_rate))
        freq_axis = (self.current_center_freq + freq_rel) / 1e6
        
        # Metrics Calculation (Slimeni et al. Energy Detection)
        median_noise = float(np.median(psd_dbfs))
        threshold = median_noise + 12.0
        occupied_bins = psd_dbfs > threshold
        occupation_rate = float(np.sum(occupied_bins) / n * 100.0)
        peak_pwr = float(np.max(psd_dbfs))
        avg_pwr = float(10.0 * np.log10(np.mean(10.0 ** (psd_dbfs / 10.0))))
        
        # Exposure Index Mapping
        if avg_pwr < -50:
            exp_text = "LOW"
            self.metric_index.setStyleSheet("font-size: 14px; font-weight: bold; color: green;")
        elif avg_pwr < -30:
            exp_text = "MEDIUM"
            self.metric_index.setStyleSheet("font-size: 14px; font-weight: bold; color: orange;")
        else:
            exp_text = "HIGH"
            self.metric_index.setStyleSheet("font-size: 14px; font-weight: bold; color: red;")
            
        # Update Live Labels
        self.metric_peak.setText(f"Peak Power: {peak_pwr:.1f} dBFS")
        self.metric_avg.setText(f"Channel Power: {avg_pwr:.1f} dBFS")
        self.metric_or.setText(f"Occupation Rate: {occupation_rate:.1f} %")
        self.metric_index.setText(f"Exposure Index: {exp_text}")
        
        # Update Plots
        self.curve_psd.setData(freq_axis, psd_dbfs)
        self.threshold_line.setValue(threshold)
        
        # Downsample for waterfall performance (1024 bins)
        downsampled = psd_dbfs[::int(n / 1024)]
        self.waterfall_history = np.roll(self.waterfall_history, 1, axis=0)
        self.waterfall_history[0, :len(downsampled)] = downsampled
        self.waterfall_img.setImage(self.waterfall_history.T, autoLevels=False)
        self.waterfall_img.setLevels([-85, -20])

    def log_current_state(self):
        # Save current reading to SQLite
        try:
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
            lat = float(self.lat_input.text())
            lon = float(self.lon_input.text())
            band = self.band_combo.currentText().split(" ")[0]
            shield = self.shield_combo.currentText()
            peak = float(self.metric_peak.text().split(": ")[1].replace(" dBFS", ""))
            avg = float(self.metric_avg.text().split(": ")[1].replace(" dBFS", ""))
            exp_val = self.metric_index.text().split(": ")[1]
            
            cur.execute("""
                INSERT INTO rf_measurements 
                (latitude, longitude, band_name, center_freq_mhz, peak_power_dbfs, avg_power_dbfs, exposure_index, shield_material)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (lat, lon, band, self.current_center_freq / 1e6, peak, avg, exp_val, shield))
            conn.commit()
            conn.close()
            QtWidgets.QMessageBox.information(self, "Logged", f"Entry saved to {DB_NAME} successfully!")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Log Error", f"Could not save measurement:\n{e}")

    def closeEvent(self, event):
        if self.is_scanning and self.sdr:
            self.sdr.close()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = RFMonitorApp()
    window.show()
    sys.exit(app.exec_())