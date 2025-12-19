import streamlit as st
import pandas as pd
import os
import math
from io import BytesIO
from PIL import Image
import requests
import json
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timezone
import time

# ---------- CONFIG ----------
CSV_NAME = "Pokemon Collection - Miles (English).csv"
IMAGE_FOLDER = "card_images"  # folder next to this .py file

os.makedirs(IMAGE_FOLDER, exist_ok=True)

# ---------- GOOGLE SHEETS HELPERS ----------

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

def load_service_account_info():
    """
    Try to load service account info from Streamlit secrets (for cloud),
    otherwise fall back to a local JSON file (for local testing).
    """
    # Try Streamlit secrets first (used on Streamlit Cloud)
    try:
        raw = st.secrets.get("gcp_service_account", None)
        if raw:
            return json.loads(raw)
    except Exception:
        pass

    # Local JSON file fallback
    json_path = os.path.join(os.path.dirname(__file__), "gcp_service_account.json")
    with open(json_path, "r") as f:
        return json.load(f)

def get_gsheet_client():
    try:
        service_account_info = load_service_account_info()
        credentials = Credentials.from_service_account_info(
            service_account_info,
            scopes=SCOPES,
        )
        client = gspread.authorize(credentials)
        return client
    except Exception as e:
        st.warning(f"Could not connect to Google Sheets (drafts will not be logged): {e}")
        return None

def get_draft_sheet():
    client = get_gsheet_client()
    if client is None:
        return None, None, None

    # Try to get sheet ID from secrets first, otherwise use hardcoded value
    sheet_id = None
    try:
        sheet_id = st.secrets.get("google_sheets_document_id", None)
    except Exception:
        sheet_id = None

    if sheet_id is None:
        # Fallback: hardcode your sheet ID here for local testing
        sheet_id = "1OORWkcWkwixexnHElLj0BXUSw_BL840gu2zEPMXc8j8"

    try:
        sh = client.open_by_key(sheet_id)
        drafts_ws = sh.worksheet("Drafts")
        rounds_ws = sh.worksheet("Rounds")
        return sh, drafts_ws, rounds_ws
    except Exception as e:
        st.warning(f"Could not open Google Sheet or worksheets: {e}")
        return None, None, None

def ensure_draft_row(draft_id, players):
    """
    Ensure a row exists in the Drafts sheet for this draft_id.
    If it doesn't exist, create it.
    """
    sh, drafts_ws, rounds_ws = get_draft_sheet()
    if drafts_ws is None:
        return  # Sheets not configured / unavailable

    try:
        all_values = drafts_ws.get_all_values()
        existing_ids = [row[0] for row in all_values[1:]] if len(all_values) > 1 else []
        if draft_id in existing_ids:
            return

        timestamp = datetime.now(timezone.utc).isoformat()
        p1 = players[0]["name"] if len(players) > 0 else ""
        p2 = players[1]["name"] if len(players) > 1 else ""
        p3 = players[2]["name"] if len(players) > 2 else ""
        config_json = "{}"  # placeholder for future config

        drafts_ws.append_row(
            [draft_id, timestamp, "active", p1, p2, p3, config_json]
        )
    except Exception as e:
        st.warning(f"Could not ensure draft row in Google Sheets: {e}")

def append_round_offers_to_sheet(draft_id, player_name, round_num, choices):
    """
    Log the 3 offered cards for a round into the Rounds sheet.
    """
    sh, drafts_ws, rounds_ws = get_draft_sheet()
    if rounds_ws is None:
        return  # Sheets not configured / unavailable

    try:
        timestamp = datetime.now(timezone.utc).isoformat()
        rows = []
        for idx, uid in enumerate(choices, start=1):
            rows.append([
                str(draft_id),
                str(player_name),
                int(round_num),
                int(idx),
                str(uid),
                "TRUE",   # offered
                "FALSE",  # picked (not yet)
                timestamp
            ])
        rounds_ws.append_rows(rows)
    except Exception as e:
        st.warning(f"Could not log round offers to Google Sheets: {e}")

def mark_pick_in_sheet(draft_id, player_name, round_num, picked_uid):
    """
    Mark the chosen card as picked=TRUE in the Rounds sheet for this draft.
    """
    sh, drafts_ws, rounds_ws = get_draft_sheet()
    if rounds_ws is None:
        return  # Sheets not configured / unavailable

    try:
        all_values = rounds_ws.get_all_values()
        if not all_values:
            return

        header = all_values[0]
        def col_idx(col_name):
            return header.index(col_name) + 1  # 1-based for gspread

        draft_idx = col_idx("draft_id")
        player_idx = col_idx("player_name")
        round_idx = col_idx("round")
        uid_idx = col_idx("unique_id")
        picked_idx = col_idx("picked")

        for row_num, row in enumerate(all_values[1:], start=2):
            if (
                row[draft_idx-1] == str(draft_id) and
                row[player_idx-1] == str(player_name) and
                row[round_idx-1] == str(round_num) and
                row[uid_idx-1] == str(picked_uid)
            ):
                rounds_ws.update_cell(row_num, picked_idx, "TRUE")
                break
    except Exception as e:
        st.warning(f"Could not log pick to Google Sheets: {e}")

# ---------- LOAD DATA ----------
@st.cache(show_spinner=False)
def load_cards(csv_name):
    df = pd.read_csv(csv_name)
    # Basic sanity: ensure required columns exist
    required_cols = [
        "Unique ID",
        "Card Name",
        "Card Type",
        "Card Stage",
        "Rarity",
        "Set",
        "Series",
        "Count",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing columns in CSV: {missing}")
    return df

cards = load_cards(CSV_NAME)

# ---------- GLOBAL STATE INITIALIZATION ----------
if "initialized" not in st.session_state:
    st.session_state.initialized = True

    # Global remaining Count per Unique ID
    st.session_state.global_counts = {}
    for _, row in cards.iterrows():
        uid = row["Unique ID"]
        cnt = int(row.get("Count", 0))
        st.session_state.global_counts[uid] = st.session_state.global_counts.get(uid, 0) + cnt

    # Player state: 3 players
    st.session_state.players = []
    for i in range(3):
        st.session_state.players.append(
            {
                "name": f"Player {i + 1}",
                "types": [],             # list of 3 chosen Pokémon types
                "current_round": 1,
                "seen_ids": set(),       # Unique IDs this player has seen as options
                "picked_ids": [],        # Unique IDs this player has drafted
                "picks": [],             # list of dicts with card info
                "current_choices": None  # list of 3 Unique IDs for the current round
            }
        )

# ---------- HELPER: ROUND RULES ----------
def get_round_info(round_num):
    """
    Returns a dict describing the rules for this round.
    Keys:
      - mode: "pokemon" or "trainer"
      - rarity: one of "RR", "R", "U", "C" (for pokemon mode), or None
      - trainer_stage: "Supporter", "Item", "Tool" (for trainer mode)
    """
    # Rounds 1–25: Pokémon by rarity band + types
    if round_num == 1:
        return {"mode": "pokemon", "rarity": "RR", "trainer_stage": None}
    elif 2 <= round_num <= 5:
        return {"mode": "pokemon", "rarity": "R", "trainer_stage": None}
    elif 6 <= round_num <= 13:
        return {"mode": "pokemon", "rarity": "U", "trainer_stage": None}
    elif 14 <= round_num <= 25:
        return {"mode": "pokemon", "rarity": "C", "trainer_stage": None}
    # Rounds 26–33: Trainer Supporter
    elif 26 <= round_num <= 33:
        return {"mode": "trainer", "rarity": None, "trainer_stage": "Supporter"}
    # Rounds 34–42: Trainer Item
    elif 34 <= round_num <= 42:
        return {"mode": "trainer", "rarity": None, "trainer_stage": "Item"}
    # Rounds 43–45: Trainer Tool
    elif 43 <= round_num <= 45:
        return {"mode": "trainer", "rarity": None, "trainer_stage": "Tool"}
    else:
        return None

# ---------- HELPER: IMAGE SOURCE ----------
def get_card_image_source(unique_id, card_name, image_url):
    """
    Prefer local image file:
        card_images/<Unique ID> - <Card Name>.jpg
    Fall back to the Image URL column if local file doesn't exist.
    """
    safe_name = str(card_name).replace("/", "-").replace("\\", "-").strip()
    safe_id = str(unique_id).replace("/", "-").replace("\\", "-").strip()
    filename = f"{safe_id} - {safe_name}.jpg"
    local_path = os.path.join(IMAGE_FOLDER, filename)

    if os.path.exists(local_path):
        return local_path
    elif isinstance(image_url, str) and image_url.strip():
        return image_url.strip()
    else:
        return None

# ---------- HELPER: GENERATE CHOICES FOR A ROUND ----------
def generate_round_choices(player_index):
    player = st.session_state.players[player_index]
    round_num = player["current_round"]
    rules = get_round_info(round_num)
    if rules is None:
        st.error(f"Invalid round number: {round_num}")
        return None

    seen_ids = player["seen_ids"]
    global_counts = st.session_state.global_counts

    choices = []

    if rules["mode"] == "pokemon":
        # Need player's 3 types
        if len(player["types"]) != 3:
            st.warning("This player must select exactly 3 Pokémon types before drafting.")
            return None

        rarity = rules["rarity"]
        type_slots = player["types"]  # [TypeA, TypeB, TypeC]

        used_ids_in_round = set()

        for slot_type in type_slots:
            # Pokémon definition: Card Type != Trainer and Card Stage in [Basic, Stage 1, Stage 2]
            mask = (
                (cards["Card Type"] == slot_type) &
                (cards["Rarity"] == rarity) &
                (cards["Card Type"] != "Trainer") &
                (cards["Card Stage"].isin(["Basic", "Stage 1", "Stage 2"]))
            )

            def can_use(uid):
                return (
                    uid not in seen_ids and
                    uid not in used_ids_in_round and
                    global_counts.get(uid, 0) > 0
                )

            eligible = cards[mask].copy()
            eligible = eligible[eligible["Unique ID"].map(can_use)]

            if eligible.empty:
                st.error(
                    f"No eligible cards for player {player_index + 1}, type {slot_type}, "
                    f"rarity {rarity}, round {round_num}."
                )
                return None

            chosen = eligible.sample(1).iloc[0]
            uid = chosen["Unique ID"]
            choices.append(uid)
            used_ids_in_round.add(uid)

    elif rules["mode"] == "trainer":
        stage = rules["trainer_stage"]

        mask = (
            (cards["Card Type"] == "Trainer") &
            (cards["Card Stage"] == stage)
        )

        def can_use(uid):
            return (
                uid not in seen_ids and
                global_counts.get(uid, 0) > 0
            )

        eligible = cards[mask].copy()
        eligible = eligible[eligible["Unique ID"].map(can_use)]

        if len(eligible) < 3:
            st.error(
                f"Not enough eligible Trainer cards (stage {stage}) for player "
                f"{player_index + 1}, round {round_num}."
            )
            return None

        chosen_rows = eligible.sample(3)
        choices = list(chosen_rows["Unique ID"].values)

    # Log offers to Google Sheets
    draft_id = st.session_state.get("draft_id", "LOCAL-DRAFT")
    append_round_offers_to_sheet(draft_id, player["name"], round_num, choices)

    # Apply side effects: mark seen & decrement global counts
    for uid in choices:
        player["seen_ids"].add(uid)
        st.session_state.global_counts[uid] = st.session_state.global_counts.get(uid, 0) - 1

    return choices

# ---------- SIDEBAR: DRAFT + PLAYER SETUP ----------
st.sidebar.header("Draft Setup")

# Draft ID (for logging to Sheets / future async)
if "draft_id" not in st.session_state:
    st.session_state.draft_id = f"DRAFT-{int(time.time())}"

draft_id_input = st.sidebar.text_input("Draft ID", value=st.session_state.draft_id)
draft_id = draft_id_input.strip() or st.session_state.draft_id
st.session_state.draft_id = draft_id
st.sidebar.write(f"Using draft ID: `{draft_id}`")

st.sidebar.header("Player Setup")

pokemon_types = [
    "Colorless", "Darkness", "Dragon", "Fairy", "Fighting",
    "Fire", "Grass", "Lightning", "Metal", "Psychic", "Water"
]

for i in range(3):
    p = st.session_state.players[i]

    # Stable key for player name
    name_key = f"player_{i}_name"
    p["name"] = st.sidebar.text_input(
        f"Player {i + 1} name",
        value=p["name"],
        key=name_key,
    )

    # Stable key for this player's type selection
    types_key = f"player_{i}_types"
    if types_key not in st.session_state:
        st.session_state[types_key] = p["types"] or []

    selected = st.sidebar.multiselect(
        f"{p['name']}: Choose 3 Pokémon types",
        pokemon_types,
        key=types_key,
    )

    p["types"] = selected

# Ensure this draft ID exists in the Drafts sheet (non-fatal if Sheets isn't configured)
ensure_draft_row(draft_id, st.session_state.players)

if st.sidebar.button("Reset entire draft (all players & counts)"):
    st.session_state.clear()
    st.experimental_rerun()

# ---------- MAIN UI ----------
st.title("Structured Rounds Draft")

st.markdown(
    "Each of 3 players has 45 rounds, 3 cards per round, 1 pick per round.\n\n"
    "Rounds 1–25: Pokémon by rarity band and player-specific types (slots = Type A / B / C).\n\n"
    "Rounds 26–45: Trainer rounds (Supporter, Item, Tool)."
)

# --- Catalogue view (no images) ---
st.subheader("Card Catalogue (Full List)")

catalogue_cols = [
    "Unique ID",
    "Card Name",
    "Card Type",
    "Card Stage",
    "Rarity",
    "Set",
    "Series",
    "Count",
]
existing_cols = [c for c in catalogue_cols if c in cards.columns]
st.dataframe(cards[existing_cols])

st.markdown("---")

# ---------- DRAFT SECTION ----------
st.header("Draft Controls")

player_names = [p["name"] for p in st.session_state.players]
active_index = st.selectbox(
    "Active Player",
    options=list(range(3)),
    format_func=lambda i: player_names[i]
)

player = st.session_state.players[active_index]
round_num = player["current_round"]

if round_num > 45:
    st.subheader(f"{player['name']} – Draft Complete")
else:
    st.subheader(f"{player['name']} – Round {round_num} of 45")

    round_info = get_round_info(round_num)
    if round_info is None:
        st.error(f"Invalid round number: {round_num}")
    elif round_info["mode"] == "pokemon" and len(player["types"]) != 3:
        st.warning(
            f"{player['name']} must have exactly 3 Pokémon types selected in the sidebar "
            "to continue drafting Pokémon rounds."
        )
    else:
        # Generate choices if none currently stored
        if player["current_choices"] is None:
            choices = generate_round_choices(active_index)
            player["current_choices"] = choices

        choices = player["current_choices"]

        if choices is not None:
            # Show choices
            st.write("Choose **one** of the following 3 cards:")

            choice_rows = cards[cards["Unique ID"].isin(choices)].copy()
            # Keep same order as choices
            choice_rows = choice_rows.set_index("Unique ID").loc[choices].reset_index()

            # Build labels for radio button
            labels = []
            for _, row in choice_rows.iterrows():
                uid = row["Unique ID"]
                name = row["Card Name"]
                rarity = row.get("Rarity", "")
                ctype = row.get("Card Type", "")
                stage = row.get("Card Stage", "")
                labels.append(f"[{uid}] {name} – {ctype} / {stage} / {rarity}")

            uid_order = list(choice_rows["Unique ID"].values)

            # Optional: show images first, side by side in 3 columns
            show_images = st.checkbox("Show images for these choices", value=True)

            if show_images:
                cols = st.columns(3)
                for col, (_, row) in zip(cols, choice_rows.iterrows()):
                    uid = row["Unique ID"]
                    name = row["Card Name"]
                    img_src = get_card_image_source(
                        uid,
                        name,
                        row.get("Image URL", None)
                    )
                    with col:
                        if img_src:
                            # Use width=245 so images display at roughly 245x342
                            st.image(
                                img_src,
                                width=245,
                                caption=f"[{uid}] {name}"
                            )
                        else:
                            st.write(f"[{uid}] {name}")

            # Single radio group below the images to keep "pick one" logic clean
            selected_label = st.radio(
                "Your pick:",
                options=labels,
                index=0 if labels else 0
            )

            # Find the UID corresponding to the selected label
            selected_uid = None
            for uid, label in zip(uid_order, labels):
                if label == selected_label:
                    selected_uid = uid
                    break

            if st.button("Confirm pick for this round"):
                # Record pick in local state
                picked_row = cards[cards["Unique ID"] == selected_uid].iloc[0]
                player["picked_ids"].append(selected_uid)
                player["picks"].append({
                    "Round": round_num,
                    "Unique ID": picked_row["Unique ID"],
                    "Card Name": picked_row["Card Name"],
                    "Card Type": picked_row["Card Type"],
                    "Card Stage": picked_row["Card Stage"],
                    "Rarity": picked_row["Rarity"],
                    "Set": picked_row.get("Set", ""),
                    "Series": picked_row.get("Series", "")
                })

                # Log pick to Google Sheets (non-fatal if it fails)
                draft_id = st.session_state.get("draft_id", "LOCAL-DRAFT")
                mark_pick_in_sheet(draft_id, player["name"], round_num, selected_uid)

                # Advance round and clear current choices
                player["current_round"] += 1
                player["current_choices"] = None

                st.experimental_rerun()

# ---------- DRAFT SUMMARY ----------
st.markdown("---")
st.header("Drafted Pools")

for i, p in enumerate(st.session_state.players):
    st.subheader(f"{p['name']} – Drafted Cards ({len(p['picks'])})")
    if p["picks"]:
        picks_df = pd.DataFrame(p["picks"])
        st.dataframe(picks_df)
    else:
        st.write("_No cards drafted yet._")

# ---------- EXPORT CSV ----------
st.markdown("---")
st.header("Export Draft Results (CSV)")

st.write(
    "Download a CSV of all drafted cards for all players. "
    "Each row includes player name, round, card details, and an Image Source "
    "(either local file path or Image URL)."
)

if st.button("Prepare export CSV"):
    export_rows = []
    for p in st.session_state.players:
        for pick in p["picks"]:
            uid = pick["Unique ID"]
            base = cards[cards["Unique ID"] == uid]
            if base.empty:
                img_src = None
                image_url = None
            else:
                base_row = base.iloc[0]
                image_url = base_row.get("Image URL", None)
                img_src = get_card_image_source(uid, base_row["Card Name"], image_url)

            row = {
                "Player": p["name"],
                "Round": pick["Round"],
                "Unique ID": pick["Unique ID"],
                "Card Name": pick["Card Name"],
                "Card Type": pick["Card Type"],
                "Card Stage": pick["Card Stage"],
                "Rarity": pick["Rarity"],
                "Set": pick.get("Set", ""),
                "Series": pick.get("Series", ""),
                "Image Source": img_src if img_src else image_url
            }
            export_rows.append(row)

    if not export_rows:
        st.info("No drafted cards yet to export.")
    else:
        export_df = pd.DataFrame(export_rows)
        csv_data = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download drafted cards as CSV",
            data=csv_data,
            file_name="pokemon_draft_results.csv",
            mime="text/csv"
        )

# ---------- EXPORT DRAFT IMAGE ----------
st.markdown("---")
st.header("Export Draft Image (Per Player)")

selected_player_index = st.selectbox(
    "Select player for image export",
    options=list(range(3)),
    format_func=lambda i: st.session_state.players[i]["name"]
)

selected_player = st.session_state.players[selected_player_index]

st.write(
    "This will create a single image showing all drafted cards for the selected player "
    "arranged in a grid (using card images if available)."
)

if st.button("Generate draft image for selected player"):
    if not selected_player["picks"]:
        st.info("This player has no drafted cards yet.")
    else:
        # Collect images in the order of picks
        images = []
        for pick in selected_player["picks"]:
            uid = pick["Unique ID"]
            base = cards[cards["Unique ID"] == uid]
            if base.empty:
                continue
            base_row = base.iloc[0]
            image_url = base_row.get("Image URL", None)
            src = get_card_image_source(uid, base_row["Card Name"], image_url)
            img = None
            if src:
                try:
                    if src.startswith("http://") or src.startswith("https://"):
                        resp = requests.get(src, timeout=10)
                        if resp.status_code == 200:
                            img = Image.open(BytesIO(resp.content)).convert("RGB")
                    else:
                        if os.path.exists(src):
                            img = Image.open(src).convert("RGB")
                except Exception:
                    img = None

            if img is not None:
                img = img.resize((245, 342), Image.LANCZOS)
                images.append(img)

        if not images:
            st.warning("No images could be loaded for this player's picks.")
        else:
            # Make a grid: 9 columns, enough rows for all images
            cols = 9
            card_w, card_h = 245, 342
            rows = math.ceil(len(images) / cols)

            collage_w = cols * card_w
            collage_h = rows * card_h
            collage = Image.new("RGB", (collage_w, collage_h), color=(0, 0, 0))

            for idx, img in enumerate(images):
                row_idx = idx // cols
                col_idx = idx % cols
                x = col_idx * card_w
                y = row_idx * card_h
                collage.paste(img, (x, y))

            st.image(collage, caption=f"{selected_player['name']}'s drafted cards", use_column_width=True)

            # Offer download as PNG
            buf = BytesIO()
            collage.save(buf, format="PNG")
            buf.seek(0)

            st.download_button(
                label="Download draft image as PNG",
                data=buf,
                file_name=f"{selected_player['name'].replace(' ', '_')}_draft.png",
                mime="image/png"
            )

