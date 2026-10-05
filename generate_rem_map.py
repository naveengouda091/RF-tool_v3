import sqlite3
import urllib.request
import json
import os
import folium
from folium.plugins import HeatMap

DB_NAME = "rf_survey_database.db"
OUTPUT_HTML = "radio_environment_map.html"

def get_current_location():
    """Fetches approximate real-world coordinates via IP geolocation."""
    try:
        url = "http://ip-api.com/json/"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=4) as response:
            data = json.loads(response.read().decode())
            if data.get('status') == 'success':
                print(f"[+] Detected Location: {data.get('city')}, {data.get('regionName')} ({data.get('lat')}, {data.get('lon')})")
                return float(data['lat']), float(data['lon'])
    except Exception:
        pass
    # Fallback to college campus if offline
    return 15.3218, 74.7646

def build_accurate_rem_map():
    real_lat, real_lon = get_current_location()

    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
        SELECT band_name, center_freq_mhz, peak_power_dbfs, avg_power_dbfs, exposure_index 
        FROM rf_measurements 
        ORDER BY id DESC LIMIT 10
    """)
    records = cur.fetchall()
    conn.close()

    # Base map using standard Esri street tiles (no tokens, no watermark)
    rem_map = folium.Map(
        location=[real_lat, real_lon],
        zoom_start=17,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri Street Map | SDR RF Noise Survey"
    )

    # Add Esri Satellite Imagery toggle
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite View"
    ).add_to(rem_map)

    heat_data = []

    if records:
        for i, (band, freq, peak, avg, exposure) in enumerate(records):
            # Spread points slightly if multiple captures were taken at one desk
            offset_lat = real_lat + (i * 0.00015)
            offset_lon = real_lon + (i * 0.00015)

            norm_weight = float(max(0.1, min(1.0, (avg + 80.0) / 60.0)))
            heat_data.append([offset_lat, offset_lon, norm_weight])

            marker_color = "#2e7d32" if exposure == "LOW" else "#ef6c00" if exposure == "MEDIUM" else "#c62828"

            popup_html = f"""
            <div style="font-family: Arial; font-size: 12px; min-width: 170px;">
                <b style="color: #0d47a1; font-size: 13px;">{band}</b><br>
                <hr style="margin: 4px 0;">
                <b>Frequency:</b> {freq:.2f} MHz<br>
                <b>Peak Power:</b> {peak:.1f} dBFS<br>
                <b>Channel Pwr:</b> {avg:.1f} dBFS<br>
                <b>Exposure Index:</b> <span style="color: {marker_color}; font-weight: bold;">{exposure}</span>
            </div>
            """

            folium.CircleMarker(
                location=[offset_lat, offset_lon],
                radius=9,
                popup=folium.Popup(popup_html, max_width=250),
                color="#ffffff",
                weight=2,
                fill=True,
                fill_color=marker_color,
                fill_opacity=0.9
            ).add_to(rem_map)
    else:
        heat_data.append([real_lat, real_lon, 0.5])

    # Density heatmap layer
    HeatMap(heat_data, radius=35, blur=20, min_opacity=0.4).add_to(rem_map)
    folium.LayerControl().add_to(rem_map)

    # Project Legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 25px; left: 25px; width: 220px; height: 135px; 
                background-color: white; border: 1px solid #757575; z-index: 9999; 
                font-family: Arial; font-size: 12px; padding: 10px; 
                border-radius: 6px; box-shadow: 2px 2px 6px rgba(0,0,0,0.35);">
        <b>RF Exposure Index</b><br>
        <span style="color: #2e7d32; font-size: 15px;">&#9679;</span> Low (&lt; -50 dBFS)<br>
        <span style="color: #ef6c00; font-size: 15px;">&#9679;</span> Medium (-50 to -30 dBFS)<br>
        <span style="color: #c62828; font-size: 15px;">&#9679;</span> High (&gt; -30 dBFS)<br>
        <hr style="margin: 4px 0;">
        <small>Radio Environment Map (REM)</small>
    </div>
    '''
    rem_map.get_root().html.add_child(folium.Element(legend_html))

    full_path = os.path.abspath(OUTPUT_HTML)
    rem_map.save(full_path)
    print(f"[+] Accurate Radio Environment Map saved: {full_path}")

if __name__ == "__main__":
    build_accurate_rem_map()