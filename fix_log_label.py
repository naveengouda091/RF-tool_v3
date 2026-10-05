import sqlite3

conn = sqlite3.connect("rf_survey_database.db")
cur = conn.cursor()

# Update the latest entry from 'Aluminium_Foil' to 'No_Antenna_Rods'
cur.execute("""
    UPDATE rf_measurements 
    SET shield_material = 'No_Antenna_Rods' 
    WHERE id = (SELECT MAX(id) FROM rf_measurements)
""")
conn.commit()
conn.close()
print("[+] Database entry updated to 'No_Antenna_Rods'.")