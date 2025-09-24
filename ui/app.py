# ui/app.py
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# --------- Konfiguracja ----------
st.set_page_config(page_title="Netflix POC – Rekomendacje", layout="wide")
API_URL = os.getenv("API_URL", "http://api:8080")   # w docker-compose ustaw API_URL: http://api:8080
ART = Path("./artifacts")                           # wolumen z artefaktami

# --------- Dane pomocnicze ----------
@st.cache_data(show_spinner=False)
def load_maps():
    users_map = pd.read_csv(ART / "users_map.csv")
    items_map = pd.read_csv(ART / "items_map.csv")
    return users_map, items_map

users_map, items_map = load_maps()

def post_json(path, payload):
    r = requests.post(f"{API_URL}{path}", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()

def get_json(path, params=None):
    r = requests.get(f"{API_URL}{path}", params=params, timeout=10)
    r.raise_for_status()
    return r.json()

# --------- UI ----------
st.title("🎬 Netflix POC – Rekomendacje")

left, mid, right = st.columns([2, 1.2, 1.2])
with left:
    user_id = st.selectbox("Wybierz userId", options=users_map["userId"].tolist(), index=0)
with mid:
    k = st.slider("Ile rekomendacji (k)", min_value=5, max_value=30, value=10, step=1)
with right:
    mode = st.selectbox("Tryb", ["MIXED (ALS+Popularność)", "ALS only"], index=0)

alpha = 0.8
if mode.startswith("MIXED"):
    alpha = st.slider("Waga ALS (alpha)", min_value=0.0, max_value=1.0, value=0.8, step=0.05)

if st.button("Pobierz rekomendacje", type="primary"):
    st.session_state.pop("recs", None)

# --------- Pobranie wyników ----------
if "recs" not in st.session_state:
    try:
        if mode.startswith("MIXED"):
            data = get_json(f"/recommend/mixed/{user_id}", params={"k": k, "alpha": alpha})
        else:
            data = get_json(f"/recommend/user/{user_id}", params={"k": k})
        st.session_state["recs"] = data.get("items", [])
    except Exception as e:
        st.error(f"Nie udało się pobrać rekomendacji: {e}")
        st.stop()

recs = st.session_state.get("recs", [])
st.subheader("Wyniki")

if not recs:
    st.info("Brak wyników.")
else:
    ix = items_map.set_index("item_index")
    for i, rec in enumerate(recs, start=1):
        item_index = rec.get("item_index")
        movie_id = rec.get("movieId")
        title = rec.get("title") or ix.loc[item_index]["title"] if item_index in ix.index else "(brak tytułu)"

        st.markdown(f"**{i}. {title}**  \n`movieId={movie_id}` • `item_index={item_index}`")
        c1, c2, c3 = st.columns([1, 1, 6])

        with c1:
            if st.button("👍 Trafne", key=f"up_{i}"):
                try:
                    post_json("/feedback", {
                        "user_id": int(user_id),
                        "item_index": int(item_index) if item_index is not None else None,
                        "movieId": int(movie_id) if movie_id is not None else None,
                        "relevant": True,
                        "source": "ui",
                    })
                    st.success("Zapisano feedback 👍")
                except Exception as e:
                    st.error(f"Nie udało się zapisać feedbacku: {e}")

        with c2:
            if st.button("👎 Nietrafne", key=f"down_{i}"):
                try:
                    post_json("/feedback", {
                        "user_id": int(user_id),
                        "item_index": int(item_index) if item_index is not None else None,
                        "movieId": int(movie_id) if movie_id is not None else None,
                        "relevant": False,
                        "source": "ui",
                    })
                    st.warning("Zapisano feedback 👎")
                except Exception as e:
                    st.error(f"Nie udało się zapisać feedbacku: {e}")

        st.divider()
