import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DB_NAME = "rf_survey_database.db"

def build_viva_deliverables():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("""
        SELECT id, timestamp, shield_material, peak_power_dbfs, avg_power_dbfs, noise_floor_dbfs, occupation_rate_pct 
        FROM shielding_trials 
        ORDER BY id ASC
    """, conn)
    conn.close()

    base_subset = df[df['shield_material'] == 'None']
    foil_subset = df[df['shield_material'] == 'Aluminium_Foil']

    if base_subset.empty or foil_subset.empty:
        print("[!] Missing required trial conditions in database.")
        return

    # Average baseline metrics (IDs 1 & 2)
    base_peak = base_subset['peak_power_dbfs'].mean()
    base_avg = base_subset['avg_power_dbfs'].mean()
    base_floor = base_subset['noise_floor_dbfs'].mean()
    base_or = base_subset['occupation_rate_pct'].mean()

    # Average shielded metrics (IDs 3 & 4)
    foil_peak = foil_subset['peak_power_dbfs'].mean()
    foil_avg = foil_subset['avg_power_dbfs'].mean()
    foil_floor = foil_subset['noise_floor_dbfs'].mean()
    foil_or = foil_subset['occupation_rate_pct'].mean()

    # Exact decibel and percentage power reduction calculations
    carrier_atten_db = base_peak - foil_peak
    channel_atten_db = base_avg - foil_avg
    power_ratio = 10.0 ** (-carrier_atten_db / 10.0)
    percent_rf_blocked = (1.0 - power_ratio) * 100.0

    print("\n" + "="*75)
    print("      RF NOISE MONITORING & SHIELDING ATTENUATION ANALYSIS REPORT")
    print("="*75)
    print(f"Target Frequency Band      : 98.30 MHz (FM Carrier)")
    print(f"Sampling Rate / FFT Length : 2.048 MSps / 16384 Bins")
    print(f"Detection Model            : Energy Detection (ED) Adaptive Thresholding")
    print("-"*75)
    print(f"{'Metric Parameter':<30} | {'Baseline (Unshielded)':<20} | {'Aluminium Foil Shielded':<20}")
    print("-"*75)
    print(f"{'Peak Carrier Level':<30} | {base_peak:>15.2f} dBFS | {foil_peak:>15.2f} dBFS")
    print(f"{'Total Channel Power':<30} | {base_avg:>15.2f} dBFS | {foil_avg:>15.2f} dBFS")
    print(f"{'Ambient Noise Floor':<30} | {base_floor:>15.2f} dBFS | {foil_floor:>15.2f} dBFS")
    print(f"{'Spectrum Occupancy Rate':<30} | {base_or:>15.1f} %    | {foil_or:>15.1f} %")
    print("-"*75)
    print(f"FINAL CARRIER ATTENUATION  : {carrier_atten_db:.2f} dB")
    print(f"TOTAL CHANNEL ATTENUATION  : {channel_atten_db:.2f} dB")
    print(f"EFFECTIVE RF POWER BLOCKED : {percent_rf_blocked:.2f} %")
    print("="*75 + "\n")

    # Generate Publication-Grade Comparison Figure for Thesis & Slides
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.5))

    # Subplot A: Absolute Signal Drop
    categories = ['Unshielded\n(Baseline)', 'Shielded\n(Aluminium Foil)']
    peak_levels = [base_peak, foil_peak]
    bars1 = ax1.bar(categories, peak_levels, color=['#c62828', '#2e7d32'], width=0.45, edgecolor='black', zorder=3)
    ax1.set_title("Carrier Peak Power Level", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Power Spectral Density (dBFS)", fontsize=10)
    ax1.set_ylim(-60, 0)
    ax1.grid(axis='y', linestyle=':', alpha=0.7, zorder=0)
    for b in bars1:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2., h - 4.0, f"{h:.2f} dBFS", ha='center', color='white', fontweight='bold')

    # Subplot B: Decibel Attenuation & Shielding Effectiveness
    atten_vals = [0.0, carrier_atten_db]
    bars2 = ax2.bar(categories, atten_vals, color=['#9e9e9e', '#1565c0'], width=0.45, edgecolor='black', zorder=3)
    ax2.set_title("Measured Shielding Effectiveness (SE)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Attenuation (dB) [Higher = Better]", fontsize=10)
    ax2.set_ylim(0, carrier_atten_db + 8.0)
    ax2.grid(axis='y', linestyle=':', alpha=0.7, zorder=0)
    
    # Text annotation on the shielded bar
    ax2.text(bars2[1].get_x() + bars2[1].get_width()/2., carrier_atten_db + 1.0, 
             f"+{carrier_atten_db:.2f} dB Drop\n({percent_rf_blocked:.2f}% Blocked)", 
             ha='center', fontsize=9, fontweight='bold', color='#0d47a1')

    plt.suptitle("RF Signal Attenuation & Physical Barrier Analysis (98.3 MHz)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("viva_shielding_analysis.png", dpi=300)
    print("[+] Publication figure exported successfully: 'viva_shielding_analysis.png'")
    plt.show()

if __name__ == "__main__":
    build_viva_deliverables()