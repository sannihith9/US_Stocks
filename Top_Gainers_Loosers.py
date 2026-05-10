from __future__ import annotations

import pandas as pd
import streamlit as st
import yfinance as yf


st.set_page_config(page_title="Top Gainers & Losers", layout="wide")
st.title("Top Gainers & Top Losers")
st.caption("Daily movers based on the latest two available Yahoo Finance closes.")

DEFAULT_SYMBOLS = "AAPL, MSFT, NVDA, META, RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS"


def clean_symbols(symbols: list[str]) -> list[str]:
    return sorted({str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()})


def parse_symbol_text(raw_symbols: str) -> list[str]:
    normalized = raw_symbols.replace("\n", ",")
    return clean_symbols(normalized.split(","))


def read_symbols_from_csv(uploaded_file) -> list[str]:
    if uploaded_file is None:
        return []

    try:
        uploaded_symbols = pd.read_csv(uploaded_file)
    except Exception as exc:  # noqa: BLE001 - Streamlit should show the user-friendly parse failure.
        st.sidebar.error(f"Could not read symbols CSV: {exc}")
        return []

    if uploaded_symbols.empty:
        return []

    symbol_column = "Symbol" if "Symbol" in uploaded_symbols.columns else uploaded_symbols.columns[0]
    return clean_symbols(uploaded_symbols[symbol_column].dropna().astype(str).tolist())


@st.cache_data(ttl=900, show_spinner=False)
def fetch_close_prices(symbols: tuple[str, ...]) -> pd.DataFrame:
    data = yf.download(
        list(symbols),
        period="5d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            return pd.DataFrame()
        close = data["Close"]
    elif "Close" in data.columns:
        close = data[["Close"]].rename(columns={"Close": symbols[0]})
    else:
        return pd.DataFrame()

    if isinstance(close, pd.Series):
        close = close.to_frame(symbols[0])

    return close.dropna(how="all")


st.sidebar.header("Watchlist")
session_symbols = clean_symbols(st.session_state.get("SYMBOLS", []))
default_symbol_text = ", ".join(session_symbols) if session_symbols else DEFAULT_SYMBOLS
raw_symbols = st.sidebar.text_area("Symbols", value=default_symbol_text, height=120)
uploaded_file = st.sidebar.file_uploader("Or upload CSV", type=["csv"])

csv_symbols = read_symbols_from_csv(uploaded_file)
symbols = csv_symbols or parse_symbol_text(raw_symbols)

if not symbols:
    st.warning("Add symbols in the sidebar or upload a CSV with a Symbol column.")
    st.stop()

with st.spinner("Fetching latest close prices..."):
    close_prices = fetch_close_prices(tuple(symbols))

if close_prices.empty or len(close_prices) < 2:
    st.error("Could not fetch enough price history to calculate daily movers.")
    st.stop()

latest_close = close_prices.ffill().iloc[-1]
previous_close = close_prices.ffill().iloc[-2]

movers = (
    pd.DataFrame(
        {
            "Previous Close": previous_close,
            "Latest Close": latest_close,
            "Change %": ((latest_close / previous_close) - 1) * 100,
        }
    )
    .replace([float("inf"), float("-inf")], pd.NA)
    .dropna(subset=["Change %"])
)

if movers.empty:
    st.warning("No valid daily change values were available for the uploaded symbols.")
    st.stop()

gainers = movers.sort_values("Change %", ascending=False).head(15)
losers = movers.sort_values("Change %", ascending=True).head(15)

st.metric("Symbols analyzed", len(movers), help="Symbols with enough close-price data.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Top Gainers")
    st.dataframe(
        gainers.style.format(
            {
                "Previous Close": "{:,.2f}",
                "Latest Close": "{:,.2f}",
                "Change %": "{:+.2f}%",
            }
        ),
        use_container_width=True,
    )

with col2:
    st.subheader("Top Losers")
    st.dataframe(
        losers.style.format(
            {
                "Previous Close": "{:,.2f}",
                "Latest Close": "{:,.2f}",
                "Change %": "{:+.2f}%",
            }
        ),
        use_container_width=True,
    )

st.download_button(
    "Download Movers CSV",
    data=movers.sort_values("Change %", ascending=False).to_csv(index_label="Symbol").encode("utf-8"),
    file_name="movers.csv",
    mime="text/csv",
)
