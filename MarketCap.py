import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Stock Fundamentals Analyzer",
    page_icon="📊",
    layout="wide"
)

# -------------------------------
# Helper Functions
# -------------------------------

def fmt_cap(v):
    if pd.isna(v):
        return "—"
    if v >= 1e12:
        return f"${v/1e12:.2f}T"
    if v >= 1e9:
        return f"${v/1e9:.2f}B"
    if v >= 1e6:
        return f"${v/1e6:.2f}M"
    return f"${v:,.0f}"


def fmt_num(v):
    if pd.isna(v):
        return "—"
    return f"{v:,.2f}"


def fmt_pct(v):
    if pd.isna(v):
        return "—"
    return f"{v:.2f}%"


# -------------------------------
# Fetch Fundamentals (SAFE)
# -------------------------------

@st.cache_data(ttl=1800)
def fetch_fundamentals(tickers):

    rows = []
    progress = st.progress(0)

    for i, ticker in enumerate(tickers):

        try:
            t = yf.Ticker(ticker)
            info = t.get_info()
        except:
            info = {}

        row = {
            "Ticker": ticker,
            "Name": info.get("longName"),
            "Sector": info.get("sector"),
            "Current Price": info.get("currentPrice"),
            "Prev Close": info.get("previousClose"),
            "Market Cap": info.get("marketCap"),
            "PE Ratio": info.get("trailingPE"),
            "Forward PE": info.get("forwardPE"),
            "EPS (TTM)": info.get("trailingEps"),
            "Revenue": info.get("totalRevenue"),
            "Profit Margin": info.get("profitMargins"),
            "ROE": info.get("returnOnEquity"),
            "Dividend Yield": info.get("dividendYield"),
            "52w High": info.get("fiftyTwoWeekHigh"),
            "52w Low": info.get("fiftyTwoWeekLow")
        }

        price = row["Current Price"]
        prev = row["Prev Close"]

        if price and prev:
            row["Change %"] = (price - prev) / prev * 100
        else:
            row["Change %"] = None

        rows.append(row)

        progress.progress((i + 1) / len(tickers))
        time.sleep(0.2)

    progress.empty()

    df = pd.DataFrame(rows)

    return df


# -------------------------------
# Header
# -------------------------------

st.title("📊 Stock Fundamentals Analyzer")

st.write(
"Upload a CSV with stock tickers and analyze fundamentals such as "
"Market Cap, PE Ratio, margins, and 52-week range."
)

# -------------------------------
# Upload CSV
# -------------------------------

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if not uploaded:
    st.info("Upload a CSV containing a column called Ticker or Symbol")
    st.stop()

raw_df = pd.read_csv(uploaded)

ticker_col = None
for c in raw_df.columns:
    if c.lower() in ["ticker","symbol","stock","scrip"]:
        ticker_col = c

if ticker_col is None:
    ticker_col = raw_df.columns[0]

tickers = raw_df[ticker_col].dropna().str.upper().unique().tolist()

st.write(f"Detected {len(tickers)} tickers")

# -------------------------------
# Fetch Button
# -------------------------------

if st.button("Analyze Stocks"):

    df = fetch_fundamentals(tickers)

    if df.empty:
        st.error("No data received from Yahoo Finance")
        st.stop()

    # Ensure columns exist
    required = ["Market Cap","PE Ratio","Change %","Dividend Yield"]

    for col in required:
        if col not in df.columns:
            df[col] = None

    # -------------------------------
    # KPI Metrics
    # -------------------------------

    valid = df[df["Market Cap"].notna()]

    # total_cap = valid["Market Cap"].sum()
    # avg_pe = df["PE Ratio"].dropna().mean()
    # gainers = (df["Change %"] > 0).sum()
    # losers = (df["Change %"] < 0).sum()

    # c1,c2,c3,c4 = st.columns(4)

    # c1.metric("Total Market Cap", fmt_cap(total_cap))
    # c2.metric("Average PE", fmt_num(avg_pe))
    # c3.metric("Gainers", gainers)
    # c4.metric("Losers", losers)

    st.divider()

    # -------------------------------
    # Data Table
    # -------------------------------

    st.subheader("Stock Data")

    st.dataframe(df, use_container_width=True)


    # -------------------------------
    # Export CSV
    # -------------------------------

    st.subheader("Export Data")

    csv = df.to_csv(index=False).encode()

    st.download_button(
        "Download CSV",
        csv,
        "stock_fundamentals.csv",
        "text/csv"
    )
