import os
import pandas as pd
import requests

CSV_NAME = "Pokemon Collection - Miles (English).csv"
IMAGE_FOLDER = "card_images"

os.makedirs(IMAGE_FOLDER, exist_ok=True)
cards = pd.read_csv(CSV_NAME)

for _, row in cards.iterrows():
    card_name = str(row["Card Name"])
    unique_id = str(row["Unique ID"])
    url = row.get("Image URL")

    if pd.isna(url) or not str(url).strip():
        print(f"Skipping {unique_id} - {card_name}: no Image URL")
        continue

    safe_name = card_name.replace("/", "-").replace("\\", "-")
    safe_id = unique_id.replace("/", "-").replace("\\", "-")
    filename = f"{safe_id} - {safe_name}.jpg"
    filepath = os.path.join(IMAGE_FOLDER, filename)

    if os.path.exists(filepath):
        print(f"Already have {filepath}")
        continue

    try:
        print(f"Downloading {unique_id} - {card_name}...")
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            with open(filepath, "wb") as f:
                f.write(r.content)
        else:
            print(f"Failed {unique_id} - {card_name}: HTTP {r.status_code}")
    except Exception as e:
        print(f"Error downloading {unique_id} - {card_name}: {e}")

