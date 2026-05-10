# ==============================================
# pages/3_Near_52W_High_Low.py
# ==============================================
import streamlit as st
import pandas as pd
from utils import load_daily_52w, enrich_daily

st.set_page_config(page_title="Near 52-Week High/Low", layout="wide")
st.title("🏁 Near 52-Week High / Low")

# ----------------------------------------------
# Load symbols
# ----------------------------------------------
symbols = st.session_state.get("SYMBOLS", [])
if not symbols:
    st.warning("Upload symbols on Home page first.")
    st.stop()

rows = []

# ----------------------------------------------
# Process each symbol
# ----------------------------------------------
for s in symbols:
    try:
        # Load data
        d = load_daily_52w(s)

        if d.empty or len(d) < 252:
            continue

        # Enrich (indicators are fine, but NOT used for 52W price)
        d = enrich_daily(d)

        # Ensure Date column exists
        if "Date" not in d.columns:
            d = d.reset_index().rename(columns={"index": "Date"})

        # Ensure correct sorting
        d = d.sort_values("Date")

        # Latest close
        latest = d.iloc[-1]
        close_price = latest["Close"]

        # ------------------------------------------
        # ✅ TRUE 52-WEEK WINDOW (last 252 days)
        # ------------------------------------------
        d_52w = d.tail(252)

        # Get exact candles
        high_idx = d_52w["High"].idxmax()
        low_idx  = d_52w["Low"].idxmin()

        high_price = d_52w.loc[high_idx, "High"]
        low_price  = d_52w.loc[low_idx, "Low"]

        high_date = d_52w.loc[high_idx, "Date"]
        low_date  = d_52w.loc[low_idx, "Date"]

        # ------------------------------------------
        # Proximity calculation (using SAME prices)
        # ------------------------------------------
        pct_to_high = (close_price / high_price - 1.0) * 100.0
        pct_to_low  = (close_price / low_price - 1.0) * 100.0

        # ------------------------------------------
        # Store results
        # ------------------------------------------
        rows.append({
            "Symbol": s,
            "Close": round(close_price, 2),
            "52W_High": round(high_price, 2),
            "52W_High_Date": high_date.strftime("%Y-%m-%d"),
            "52W_Low": round(low_price, 2),
            "52W_Low_Date": low_date.strftime("%Y-%m-%d"),
            "% to 52W High": round(pct_to_high, 2),
            "% to 52W Low": round(pct_to_low, 2)
        })

    except Exception as e:
        st.warning(f"⚠️ Error processing {s}: {e}")
        continue

# ----------------------------------------------
# Final DataFrame
# ----------------------------------------------
df = pd.DataFrame(rows)

if df.empty:
    st.info("No symbols with valid 52-week data.")
    st.stop()

# ----------------------------------------------
# Filters (±3%)
# ----------------------------------------------
near_high = (
    df[df["% to 52W High"].between(-3, 0)]
    .sort_values("% to 52W High")
)

near_low = (
    df[df["% to 52W Low"].between(0, 3)]
    .sort_values("% to 52W Low")
)

# ----------------------------------------------
# Display tables
# ----------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🟩 Within 3% of 52W High")
    st.dataframe(near_high, use_container_width=True)

with col2:
    st.subheader("🟥 Within 3% of 52W Low")
    st.dataframe(near_low, use_container_width=True)

# ----------------------------------------------
# Download CSV
# ----------------------------------------------
st.download_button(
    "📥 Download 52W Proximity CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="near_52w.csv",
    mime="text/csv"
)
