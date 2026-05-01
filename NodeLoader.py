import pandas as pd
import os

# --- SETTINGS ---
# Direct link to the WRI Global Power Plant Database (v1.3.0)
WRI_CSV_URL = "https://raw.githubusercontent.com/wri/global-power-plant-database/master/output_database/global_power_plant_database.csv"
SAVE_PATH = "./data/asset_nodes.csv"

# Focus on countries along the Westerlies track
# ISO Alpha-3 codes: GBR (UK), DEU (Germany), DNK (Denmark), FRA (France), NLD (Netherlands)
EUROPE_WEST_TRACK = [ 'FRA', 'DEU', 'DNK', 'NLD', 'BEL', 'NOR', 'IRL']

def load_wind_assets():
    print("Downloading global power plant database (this may take a moment)...")
    df = pd.read_csv(WRI_CSV_URL, low_memory=False)
    
    # 1. Filter for Wind Power
    wind_df = df[df['primary_fuel'] == 'Wind'].copy()
    
    # 2. Filter for European Westerlies Track
    eu_wind = wind_df[wind_df['country'].isin(EUROPE_WEST_TRACK)].copy()
    
    # 3. Select relevant columns for your AI Model
    # name: Node ID
    # latitude/longitude: For Spatial Matrix
    # capacity_mw: To 'weight' the importance of the node
    nodes = eu_wind[['name', 'country', 'latitude', 'longitude', 'capacity_mw']]
    
    # 4. Clean data (remove assets with missing coordinates)
    nodes = nodes.dropna(subset=['latitude', 'longitude'])
    
    print(f"Success! Found {len(nodes)} wind farm nodes in the target region.")
    
    # Save for use in your ST-GCN model
    if not os.path.exists("./data"): os.makedirs("./data")
    nodes.to_csv(SAVE_PATH, index=False)
    print(f"Asset-Nodes saved to {SAVE_PATH}")
    
    return nodes

if __name__ == "__main__":
    asset_nodes = load_wind_assets()
    # Preview the top assets (likely the big offshore ones in the North Sea)
    print(asset_nodes.sort_values(by='capacity_mw', ascending=False).head(10))