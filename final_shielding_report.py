import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DB_NAME = "rf_survey_database.db"

def run_evaluation():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("""
        SELECT id, timestamp, shield_material, peak_power_dbfs, avg_power_dbfs, noise_floor_dbfs, occupation_rate_pct 
        FROM shielding_trials 
        ORDER BY id DESC
    """, conn)
    conn.close()

    print("\n--- Raw Logged Trials (Latest 8) ---")
    print(df.head(8).to_string(index=False))

    # Extract the latest actual shielded foil run and latest baseline run
    foil_rows = df[df['shield_material'] == 'Aluminium_Foil']
    base_rows = df[df['shield_material'] == 'None']

    if foil_rows.empty or base_rows.empty:
        print("[!] Missing either 'Aluminium_Foil' or 'None' records in database.")
        return

    # Use the latest capture for each
    p_base_peak = base_rows.iloc[0]['peak_power_dbfs']
    p_base_avg = base_rows.iloc[0]['avg_power_dbfs']
    p_base_floor = base_rows.iloc[0]['noise_floor_dbfs']
    base_or = base_rows.iloc[0]['occupation_rate_pct']

    p_foil_peak = foil_rows.iloc[0]['peak_power_dbfs']
    p_foil_avg = foil_rows.iloc[0]['avg_power_dbfs']
    p_foil_floor = foil_rows.iloc[0]['noise_floor_dbfs']
    foil_or = foil_rows.iloc[0]['occupation_rate_pct']

    # Attenuation Calculations
    carrier_attenuation_db = p_base_peak - p_foil_peak
    channel_attenuation_db = p_base_avg - p_foil_avg
    power_ratio = 10.0 ** (-max(0.0, carrier_attenuation_db) / 10.0)
    percent_blocked = (1.0 - power_ratio) * 100.0

    report = [
        {
            "Condition": "None (Baseline)",
            "Peak (dBFS)": f"{p_base_peak:.2f}",
            "Channel Power (dBFS)": f"{p_base_avg:.2f}",
            "Noise Floor (dBFS)": f"{p_base_floor:.2f}",
            "Attenuation (dB)": "0.00",
            "RF Power Blocked": "0.00 %",
            "Occupancy (OR)": f"{base_or:.1f} %"
        },
        {
            "Condition": "Aluminium Foil",
            "Peak (dBFS)": f"{p_foil_peak:.2f}",
            "Channel Power (dBFS)": f"{p_foil_avg:.2f}",
            "Noise Floor (dBFS)": f"{p_foil_floor:.2f}",
            "Attenuation (dB)": f"{carrier_attenuation_db:.2f}",
            "RF Power Blocked": f"{percent_blocked:.2f} %",
            "Occupancy (OR)": f"{foil_or:.1f} %"
        }
    ]

    print("\n======================= VERIFIED SHIELDING REPORT =======================")
    print(pd.DataFrame(report).to_string(index=False))
    print("=========================================================================\n")

    # Generate Bar Chart Deliverable
    labels = ['Baseline', 'Aluminium Foil']
    peaks = [p_base_peak, p_foil_peak]
    atten = [0.0, carrier_attenuation_db]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))

    # Subplot 1: Absolute Received Signal Levels
    bars1 = ax[0].bar(labels, peaks, color=['#1565c0', '#b0bec5'], edgecolor='black', width=0.5)
    ax[0].set_title("Carrier Peak Power Level", fontweight='bold')
    ax[0].set_ylabel("Power (dBFS)")
    ax[0].grid(axis='y', linestyle=':', alpha=0.7)
    for b in bars1:
        h = b.get_height()
        ax[0].text(b.get_x() + b.get_width()/2., h - 2.5, f"{h:.1f} dBFS", ha='center', color='black', fontweight='bold')

    # Subplot 2: Decibel Attenuation Blocked
    bars2 = ax[1].bar(labels, atten, color=['#78909c', '#2e7d32'], edgecolor='black', width=0.5)
    ax[1].set_title(f"Shielding Effectiveness: {carrier_attenuation_db:.1f} dB Drop", fontweight='bold')
    ax[1].set_ylabel("Attenuation (dB) [Higher = Better]")
    ax[1].set_ylim(0, max(atten) + 8)
    ax[1].grid(axis='y', linestyle=':', alpha=0.7)
    for b in bars2:
        h = b.get_height()
        if h > 0:
            ax[1].text(b.get_x() + b.get_width()/2., h + 0.6, f"{h:.1f} dB\n({percent_blocked:.1f}%)", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig("verified_shielding_report.png", dpi=300)
    print("[+] Plot saved as 'verified_shielding_report.png'")
    plt.show()

if __name__ == "__main__":
    run_evaluation()