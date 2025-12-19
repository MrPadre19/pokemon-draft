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

# ---------- GOOGLE SHEETS HELPERS (USES st.secrets TABLE) ----------

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


def get_gsheet_client():
    """
    Build a gspread client using Streamlit secrets.
    Expects [gcp_service_account] as a table in secrets (so st.secrets['gcp_service_account'] is dict-like).
    """
    try:
        service_account_info = dict(st.secrets["gcp_service_account"])
    except Exception as e:
        if not st.session_state.get("gsheets_warning_shown", False):
            st.session_state["gsheets_warning_shown"] = True
            st.warning(
                f"Google Sheets secrets not configured; drafts will not be logged: {e}"
            )
        return None

    try:
        credentials = Credentials.from_service_account_info(
            service_account_info,
            scopes=SCOPES,
        )
        client = gspread.authorize(credentials)
        return client
    except Exception as e:
        if not st.session_state.get("gsheets_warning_shown", False):
            st.session_state["gsheets_warning_shown"] = True
            st.warning(
                f"Could not connect to Google Sheets; drafts will not be logged: {e}"
            )
        return None


def get_draft_sheet():
    client = get_gsheet_client()
    if client is None:
        return None, None, None

    try:
        sheet_id = st.secrets["google_sheets_document_id"]
        sh = client.open_by_key(sheet_id)
        drafts_ws = sh.worksheet("Drafts")
        rounds_ws = sh.worksheet("Rounds")
        return sh, drafts_ws, rounds_ws
    except Exception as e:
        if not st.session_state.get("gsheets_warning_shown", False):
            st.session_state["gsheets_warning_shown"] = True
            st.warning(
                f"Could not open Google Sheet or worksheets; drafts will not be logged: {e}"
            )
        return None, None, None


def ensure_draft_row(draft_id, players):
    """
    Ensure a row exists in the Drafts sheet for this draft_id.
    If it exists, update player names to match current UI.
    If it doesn't exist, create it.
    """
    sh, drafts_ws, rounds_ws = get_draft_sheet()
    if drafts_ws is None:
        return

    try:
        all_values = drafts_ws.get_all_values()
        if not all_values:
            return

        header = all_values[0]
        rows = all_values[1:]
        existing_ids = [row[0] for row in rows] if rows else []

        p1 = players[0]["name"] if len(players) > 0 else ""
        p2 = players[1]["name"] if len(players) > 1 else ""
        p3 = players[2]["name"] if len(players) > 2 else ""
        config_json = "{}"  # placeholder for future config

        if draft_id in existing_ids:
            # Update existing row's player names
            row_index = existing_ids.index(draft_id)  # 0-based within rows
            sheet_row_num = row_index + 2            # +2 to account for header row

            # Find column indices for p1, p2, p3 if possible
            # Expected header: [draft_id, timestamp, status, p1, p2, p3, config_json]
            try:
                p1_col = header.index("player1_name") + 1
                p2_col = header.index("player2_name") + 1
                p3_col = header.index("player3_name") + 1
                drafts_ws.update_cell(sheet_row_num, p1_col, p1)
                drafts_ws.update_cell(sheet_row_num, p2_col, p2)
                drafts_ws.update_cell(sheet_row_num, p3_col, p3)
            except ValueError:
                # If header doesn't have named columns, fall back to hard-coded positions 4,5,6
                drafts_ws.update_cell(sheet_row_num, 4, p1)
                drafts_ws.update_cell(sheet_row_num, 5, p2)
                drafts_ws.update_cell(sheet_row_num, 6, p3)

            return

        # No existing row -> append a new one
        timestamp = datetime.now(timezone.utc).isoformat()
        drafts_ws.append_row(
            [draft_id, timestamp, "active", p1, p2, p3, config_json]
        )
    except Exception as e:
        if not st.session_state.get("gsheets_warning_shown", False):
            st.session_state["gsheets_warning_shown"] = True
            st.warning(f"Could not ensure draft row in Google Sheets: {e}")


def append_round_offers_to_sheet(draft_id, player_name, round_num, choices):
    """
    Log the offered cards for a round into the Rounds sheet.
    """
    sh, drafts_ws, rounds_ws = get_draft_sheet()
    if rounds_ws is None:
        return

    try:
        timestamp = datetime.now(timezone.utc).isoformat()
        rows = []
        for idx, uid in enumerate(choices, start=1):
            rows.append(
                [
                    str(draft_id),
                    str(player_name),
                    int(round_num),
                    int(idx),
                    str(uid),
                    "TRUE",   # offered
                    "FALSE",  # picked (not yet)
                    timestamp,
                ]
            )
        rounds_ws.append_rows(rows)
    except Exception as e:
        if not st.session_state.get("gsheets_warning_shown", False):
            st.session_state["gsheets_warning_shown"] = True
            st.warning(f"Could not log round offers to Google Sheets: {e}")


def mark_pick_in_sheet(draft_id, player_name, round_num, picked_uid):
    """
    Mark the chosen card as picked=TRUE in the Rounds sheet for this draft.
    """
    sh, drafts_ws, rounds_ws = get_draft_sheet()
    if rounds_ws is None:
        return

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
                row[draft_idx - 1] == str(draft_id)
                and row[player_idx - 1] == str(player_name)
                and row[round_idx - 1] == str(round_num)
                and row[uid_idx - 1] == str(picked_uid)
            ):
                rounds_ws.update_cell(row_num, picked_idx, "TRUE")
                break
    except Exception as e:
        if not st.session_state.get("gsheets_warning_shown", False):
            st.session_state["gsheets_warning_shown"] = True
            st.warning(f"Could not log pick to Google Sheets: {e}")


# ---------- LOAD DATA ----------
@st.cache(show_spinner=False)
def load_cards(csv_name):
    df = pd.read_csv(csv_name)
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

# Map from string Unique ID to canonical value (to keep types consistent)
UNIQUE_ID_MAP = {str(u): u for u in cards["Unique ID"].unique()}

# ---------- GLOBAL STATE INITIALIZATION ----------
if "initialized" not in st.session_state:
    st.session_state.initialized = True

    # Global remaining Count per Unique ID (fresh, before any draft is applied)
    st.session_state.global_counts = {}
    for _, row in cards.iterrows():
        uid = row["Unique ID"]
        cnt = int(row.get("Count", 0))
        st.session_state.global_counts[uid] = (
            st.session_state.global_counts.get(uid, 0) + cnt
        )

    # Player state: 3 players
    st.session_state.players = []
    for i in range(3):
        st.session_state.players.append(
            {
                "name": f"Player {i + 1}",
                "types": [],
                "current_round": 1,
                "seen_ids": set(),
                "picked_ids": [],
                "picks": [],
                "current_choices": None,
            }
        )

    # Track which draft_id we've already loaded from Sheets
    st.session_state.loaded_draft_id = None


# ---------- HELPER: ROUND RULES ----------
def get_round_info(round_num):
    """
    Returns a dict describing the rules for this round.
    """
    if round_num == 1:
        return {"mode": "pokemon", "rarity": "RR", "trainer_stage": None}
    elif 2 <= round_num <= 5:
        return {"mode": "pokemon", "rarity": "R", "trainer_stage": None}
    elif 6 <= round_num <= 13:
        return {"mode": "pokemon", "rarity": "U", "trainer_stage": None}
    elif 14 <= round_num <= 25:
        return {"mode": "pokemon", "rarity": "C", "trainer_stage": None}
    elif 26 <= round_num <= 33:
        return {"mode": "trainer", "rarity": None, "trainer_stage": "Supporter"}
    elif 34 <= round_num <= 42:
        return {"mode": "trainer", "rarity": None, "trainer_stage": "Item"}
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


# ---------- HELPER: LOAD STATE FROM SHEETS FOR ASYNC DRAFT ----------
def load_draft_state_from_sheets(draft_id: str) -> bool:
    """
    If the given draft_id exists in Google Sheets, rebuild session state from it:
    - Use player names observed in Rounds (what was actually used while drafting)
    - Recompute global_counts for this draft (based on offers)
    - Rebuild each player's seen_ids, picks, and current_round
    - If a round was offered but not picked yet, re-use those offers as current_choices
    Returns True if a draft row exists (even if no rounds yet), False otherwise.
    """
    sh, drafts_ws, rounds_ws = get_draft_sheet()
    if drafts_ws is None or rounds_ws is None:
        return False

    # ----- Confirm draft exists in Drafts (by ID) -----
    draft_values = drafts_ws.get_all_values()
    if len(draft_values) < 2:
        return False  # only header, no drafts

    draft_exists = any(row and row[0] == str(draft_id) for row in draft_values[1:])
    if not draft_exists:
        return False

    # ----- Process Rounds sheet -----
    rounds_values = rounds_ws.get_all_values()
    if len(rounds_values) < 2:
        # Draft exists but has no rounds yet; keep defaults
        return True

    header = rounds_values[0]

    def col_idx(col_name: str) -> int:
        return header.index(col_name)

    try:
        draft_col = col_idx("draft_id")
        player_col = col_idx("player_name")
        round_col = col_idx("round")
        uid_col = col_idx("unique_id")
        offered_col = col_idx("offered")
        picked_col = col_idx("picked")
    except ValueError:
        # Header mismatch; don't crash the app
        return True

    relevant_rows = [
        row
        for row in rounds_values[1:]
        if len(row) > draft_col and row[draft_col] == str(draft_id)
    ]

    if not relevant_rows:
        # Draft exists but no rounds yet
        return True

    # ----- Recompute global_counts based on offers so far -----
    total_counts = {}
    for _, row in cards.iterrows():
        uid = row["Unique ID"]
        cnt = int(row.get("Count", 0))
        total_counts[uid] = total_counts.get(uid, 0) + cnt

    offers_per_uid = {}
    for row in relevant_rows:
        if len(row) <= offered_col:
            continue
        offered_val = row[offered_col].strip().upper()
        if offered_val == "TRUE":
            uid_str = row[uid_col]
            uid_val = UNIQUE_ID_MAP.get(uid_str, uid_str)
            offers_per_uid[uid_val] = offers_per_uid.get(uid_val, 0) + 1

    new_global_counts = {}
    for uid, total in total_counts.items():
        used = offers_per_uid.get(uid, 0)
        remaining = max(total - used, 0)
        new_global_counts[uid] = remaining
    st.session_state.global_counts = new_global_counts

    # ----- Build per-player info from Rounds -----
    per_player = {}
    player_order = []  # preserve first-seen order of player names

    for row in relevant_rows:
        if len(row) <= max(player_col, round_col, uid_col, offered_col, picked_col):
            continue

        p_name = row[player_col] or ""
        if p_name not in per_player:
            per_player[p_name] = {
                "seen_ids": set(),
                "offers": {},  # round -> list of uids
                "picked": [],  # list of (round, uid)
            }
            player_order.append(p_name)

        round_str = row[round_col]
        uid_str = row[uid_col]
        offered_val = row[offered_col].strip().upper()
        picked_val = row[picked_col].strip().upper()

        try:
            rnum = int(round_str)
        except ValueError:
            continue

        uid_val = UNIQUE_ID_MAP.get(uid_str, uid_str)
        info = per_player[p_name]

        if offered_val == "TRUE":
            info["seen_ids"].add(uid_val)
            info["offers"].setdefault(rnum, []).append(uid_val)

        if picked_val == "TRUE":
            info["picked"].append((rnum, uid_val))

    # Compute per-player max picked round & pending round
    for p_name, info in per_player.items():
        if info["picked"]:
            max_picked_round = max(r for (r, _) in info["picked"])
        else:
            max_picked_round = 0

        pending_round = None
        for rnum in sorted(info["offers"].keys()):
            if not any(pr == rnum for (pr, _) in info["picked"]):
                pending_round = rnum  # last offered-but-not-picked round
        info["max_picked_round"] = max_picked_round
        info["pending_round"] = pending_round

    # ----- Apply to session_state.players -----
    # Map player slots in the app to the players found in Rounds, by order
    for idx, p_state in enumerate(st.session_state.players):
        if idx < len(player_order):
            name = player_order[idx]
            info = per_player.get(
                name,
                {
                    "seen_ids": set(),
                    "offers": {},
                    "picked": [],
                    "max_picked_round": 0,
                    "pending_round": None,
                },
            )
            p_state["name"] = name
        else:
            info = {
                "seen_ids": set(),
                "offers": {},
                "picked": [],
                "max_picked_round": 0,
                "pending_round": None,
            }

        # Seen IDs
        p_state["seen_ids"] = info["seen_ids"]

        # Picked IDs and picks (sorted by round)
        sorted_picks = sorted(info["picked"], key=lambda t: t[0])
        p_state["picked_ids"] = [uid for (_, uid) in sorted_picks]

        picks_list = []
        for rnum, uid in sorted_picks:
            base = cards[cards["Unique ID"] == uid]
            if base.empty:
                continue
            base_row = base.iloc[0]
            picks_list.append(
                {
                    "Round": rnum,
                    "Unique ID": base_row["Unique ID"],
                    "Card Name": base_row["Card Name"],
                    "Card Type": base_row["Card Type"],
                    "Card Stage": base_row["Card Stage"],
                    "Rarity": base_row["Rarity"],
                    "Set": base_row.get("Set", ""),
                    "Series": base_row.get("Series", ""),
                }
            )
        p_state["picks"] = picks_list

        # Determine current_round and current_choices
        pending_round = info.get("pending_round")
        max_picked_round = info.get("max_picked_round", 0)

        if pending_round is not None and pending_round in info["offers"]:
            p_state["current_round"] = pending_round
            p_state["current_choices"] = info["offers"][pending_round]
        else:
            next_round = max_picked_round + 1 if max_picked_round > 0 else 1
            p_state["current_round"] = next_round
            p_state["current_choices"] = None

    return True


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
        if len(player["types"]) != 3:
            st.warning("This player must select exactly 3 Pokémon types before drafting.")
            return None

        rarity = rules["rarity"]
        type_slots = player["types"]
        used_ids_in_round = set()

        for slot_type in type_slots:
            mask = (
                (cards["Card Type"] == slot_type)
                & (cards["Rarity"] == rarity)
                & (cards["Card Type"] != "Trainer")
                & (cards["Card Stage"].isin(["Basic", "Stage 1", "Stage 2"]))
            )

            def can_use(uid):
                return (
                    uid not in seen_ids
                    and uid not in used_ids_in_round
                    and global_counts.get(uid, 0) > 0
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
            (cards["Card Type"] == "Trainer")
            & (cards["Card Stage"] == stage)
        )

        def can_use(uid):
            return (
                uid not in seen_ids
                and global_counts.get(uid, 0) > 0
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

    draft_id = st.session_state.get("draft_id", "LOCAL-DRAFT")
    append_round_offers_to_sheet(draft_id, player["name"], round_num, choices)

    for uid in choices:
        player["seen_ids"].add(uid)
        st.session_state.global_counts[uid] = (
            st.session_state.global_counts.get(uid, 0) - 1
        )

    return choices


# ---------- SIDEBAR: DRAFT + PLAYER SETUP ----------
st.sidebar.header("Draft Setup")

if "draft_id" not in st.session_state:
    st.session_state.draft_id = f"DRAFT-{int(time.time())}"

draft_id_input = st.sidebar.text_input("Draft ID", value=str(st.session_state.draft_id))
draft_id = draft_id_input.strip() or st.session_state.draft_id
st.session_state.draft_id = draft_id

# Load from Sheets once per draft_id (for async resume)
if (
    "loaded_draft_id" not in st.session_state
    or st.session_state.loaded_draft_id != draft_id
):
    load_draft_state_from_sheets(draft_id)
    st.session_state.loaded_draft_id = draft_id

st.sidebar.write(f"Using draft ID: `{draft_id}`")

st.sidebar.header("Player Setup")

pokemon_types = [
    "Colorless", "Darkness", "Dragon", "Fairy", "Fighting",
    "Fire", "Grass", "Lightning", "Metal", "Psychic", "Water",
]

for i in range(3):
    p = st.session_state.players[i]

    name_key = f"player_{i}_name"
    p["name"] = st.sidebar.text_input(
        f"Player {i + 1} name", value=p["name"], key=name_key
    )

    types_key = f"player_{i}_types"
    if types_key not in st.session_state:
        st.session_state[types_key] = p["types"] or []

    selected = st.sidebar.multiselect(
        f"{p['name']}: Choose 3 Pokémon types",
        pokemon_types,
        key=types_key,
    )
    p["types"] = selected

# Ensure the draft row exists / update names
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
    format_func=lambda i: player_names[i],
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
        if player["current_choices"] is None:
            choices = generate_round_choices(active_index)
            player["current_choices"] = choices

        choices = player["current_choices"]

        if choices is not None:
            st.write("Choose **one** of the following 3 cards:")

            choice_rows = cards[cards["Unique ID"].isin(choices)].copy()
            choice_rows = (
                choice_rows.set_index("Unique ID").loc[choices].reset_index()
            )

            labels = []
            for _, row in choice_rows.iterrows():
                uid = row["Unique ID"]
                name = row["Card Name"]
                rarity = row.get("Rarity", "")
                ctype = row.get("Card Type", "")
                stage = row.get("Card Stage", "")
                labels.append(f"[{uid}] {name} – {ctype} / {stage} / {rarity}")

            uid_order = list(choice_rows["Unique ID"].values)

            show_images = st.checkbox("Show images for these choices", value=True)

            if show_images:
                cols = st.columns(3)
                for col, (_, row) in zip(cols, choice_rows.iterrows()):
                    uid = row["Unique ID"]
                    name = row["Card Name"]
                    img_src = get_card_image_source(
                        uid,
                        name,
                        row.get("Image URL", None),
                    )
                    with col:
                        if img_src:
                            st.image(
                                img_src,
                                width=245,
                                caption=f"[{uid}] {name}",
                            )
                        else:
                            st.write(f"[{uid}] {name}")

            selected_label = st.radio(
                "Your pick:",
                options=labels,
                index=0 if labels else 0,
            )

            selected_uid = None
            for uid, label in zip(uid_order, labels):
                if label == selected_label:
                    selected_uid = uid
                    break

            if st.button("Confirm pick for this round"):
                picked_row = cards[cards["Unique ID"] == selected_uid].iloc[0]
                player["picked_ids"].append(selected_uid)
                player["picks"].append(
                    {
                        "Round": round_num,
                        "Unique ID": picked_row["Unique ID"],
                        "Card Name": picked_row["Card Name"],
                        "Card Type": picked_row["Card Type"],
                        "Card Stage": picked_row["Card Stage"],
                        "Rarity": picked_row["Rarity"],
                        "Set": picked_row.get("Set", ""),
                        "Series": picked_row.get("Series", ""),
                    }
                )

                draft_id = st.session_state.get("draft_id", "LOCAL-DRAFT")
                mark_pick_in_sheet(draft_id, player["name"], round_num, selected_uid)

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
                img_src = get_card_image_source(
                    uid, base_row["Card Name"], image_url
                )

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
                "Image Source": img_src if img_src else image_url,
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
            mime="text/csv",
        )

# ---------- EXPORT DRAFT IMAGE ----------
st.markdown("---")
st.header("Export Draft Image (Per Player)")

selected_player_index = st.selectbox(
    "Select player for image export",
    options=list(range(3)),
    format_func=lambda i: st.session_state.players[i]["name"],
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

            st.image(
                collage,
                caption=f"{selected_player['name']}'s drafted cards",
                use_column_width=True,
            )

            buf = BytesIO()
            collage.save(buf, format="PNG")
            buf.seek(0)

            st.download_button(
                label="Download draft image as PNG",
                data=buf,
                file_name=f"{selected_player['name'].replace(' ', '_')}_draft.png",
                mime="image/png",
            )
