import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect("rf_survey_database.db")

print("\n" + "="*80)
print("              1. EXPERIMENTAL SHIELDING EFFECTIVENESS (VERIFIED)")
print("="*80)
df_shield = pd.read_sql_query("""
    SELECT shield_material, peak_power_dbfs, avg_power_dbfs, noise_floor_dbfs, occupation_rate_pct 
    FROM shielding_trials
""", conn)

base_sub = df_shield[df_shield['shield_material'] == 'None']
foil_sub = df_shield[df_shield['shield_material'] == 'Aluminium_Foil']

base_p = base_sub['peak_power_dbfs'].mean()
foil_p = foil_sub['peak_power_dbfs'].mean()
atten = base_p - foil_p
pct = (1.0 - 10**(-atten/10)) * 100

print(f"Baseline Unshielded Peak Power  : {base_p:.2f} dBFS")
print(f"Aluminium Foil Shielded Peak    : {foil_p:.2f} dBFS")
print(f"Measured Shielding Attenuation  : {atten:.2f} dB")
print(f"Total Incident RF Power Blocked : {pct:.2f} %")
print(f"Noise Floor Drop (Faraday)      : {base_sub['noise_floor_dbfs'].mean():.2f} dBFS -> {foil_sub['noise_floor_dbfs'].mean():.2f} dBFS")
print(f"Occupancy Rate (OR) Collapse    : {base_sub['occupation_rate_pct'].mean():.1f}% -> {foil_sub['occupation_rate_pct'].mean():.1f}%")

print("\n" + "="*80)
print("              2. MULTI-BAND SPECTRUM OCCUPANCY SURVEY")
print("="*80)
try:
    df_bands = pd.read_sql_query("SELECT band_name, start_mhz, stop_mhz, peak_power_dbfs, noise_floor_dbfs, occupation_rate_pct FROM swept_band_metrics ORDER BY id DESC LIMIT 2", conn)
    print(df_bands.to_string(index=False))
except Exception:
    print("Run 'python multiband_spectral_analysis.py' first to populate multiband records.")

conn.close()
print("="*80 + "\n")