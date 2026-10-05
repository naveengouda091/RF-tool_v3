import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DB_NAME = "rf_survey_database.db"

def generate_report():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("""
        SELECT id, timestamp, shield_material, peak_power_dbfs, avg_power_dbfs, noise_floor_dbfs, occupation_rate_pct
        FROM shielding_trials
        WHERE frequency_mhz = 98.3
        ORDER BY id DESC
    """, conn)
    conn.close()

    if df.empty or "None" not in df["shield_material"].values:
        print("[!] Insufficient data. Ensure baseline ('None') and shielded runs exist.")
        return

    # Take the average of the most recent runs for each condition (last 2 runs each)
    summary_data = []
    
    baseline_df = df[df["shield_material"] == "None"].head(2)
    base_peak = baseline_df["peak_power_dbfs"].mean()
    base_avg = baseline_df["avg_power_dbfs"].mean()
    base_floor = baseline_df["noise_floor_dbfs"].mean()
    base_or = baseline_df["occupation_rate_pct"].mean()

    summary_data.append({
        "Material": "None (Baseline)",
        "Peak (dBFS)": round(base_peak, 2),
        "Attenuation (dB)": 0.0,
        "Power Blocked (%)": "0.0 %",
        "Channel Power (dBFS)": round(base_avg, 2),
        "Noise Floor (dBFS)": round(base_floor, 2),
        "Occupancy (OR)": f"{base_or:.1f} %"
    })

    materials = [m for m in df["shield_material"].unique() if m != "None"]
    for mat in materials:
        mat_df = df[df["shield_material"] == mat].head(2)
        m_peak = mat_df["peak_power_dbfs"].mean()
        m_avg = mat_df["avg_power_dbfs"].mean()
        m_floor = mat_df["noise_floor_dbfs"].mean()
        m_or = mat_df["occupation_rate_pct"].mean()

        attenuation_db = base_peak - m_peak
        lin_ratio = 10.0 ** (-max(0.0, attenuation_db) / 10.0)
        pct_blocked = (1.0 - lin_ratio) * 100.0

        summary_data.append({
            "Material": mat,
            "Peak (dBFS)": round(m_peak, 2),
            "Attenuation (dB)": round(attenuation_db, 2),
            "Power Blocked (%)": f"{pct_blocked:.2f} %",
            "Channel Power (dBFS)": round(m_avg, 2),
            "Noise Floor (dBFS)": round(m_floor, 2),
            "Occupancy (OR)": f"{m_or:.1f} %"
        })

    report_table = pd.DataFrame(summary_data)
    print("\n======================= FINAL SHIELDING EFFECTIVENESS REPORT =======================")
    print(report_table.to_string(index=False))
    print("===================================================================================\n")

    # Generate Comparative Bar Chart for Presentation
    mats = report_table["Material"].tolist()
    attens = report_table["Attenuation (dB)"].tolist()

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(mats, attens, color=['#78909c', '#0288d1'][:len(mats)], edgecolor='black', width=0.5)
    plt.title("Measured RF Shielding Effectiveness (98.3 MHz Carrier)", fontsize=12, fontweight='bold')
    plt.ylabel("Attenuation (dB) [Higher = Better Shielding]", fontsize=10)
    plt.ylim(0, max(attens) + 8)
    plt.grid(axis='y', linestyle=':', alpha=0.7)

    for bar in bars:
        h = bar.get_height()
        if h > 0:
            plt.text(bar.get_x() + bar.get_width()/2., h + 0.6, f"{h:.2f} dB", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig("shielding_attenuation_summary.png", dpi=300)
    print("[+] Presentation chart saved as 'shielding_attenuation_summary.png'")
    plt.show()

if __name__ == "__main__":
    generate_report()