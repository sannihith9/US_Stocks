# pages/1_CPR_Breakout_Scanner.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf  # new import to fetch fundamental data
from utils import (
    load_daily, enrich_daily, detect_swings, calc_cpr, cpr_trend, latest_cpr_trend, breakout_tags
)

# =========================================
# 📐 Helper Function: Format Market Cap
#
# Market capitalization values can span from thousands to trillions of
# dollars. To make these large numbers more human‑friendly, we convert
# the raw integer into a string with an appropriate suffix.  The
# abbreviations follow common finance and resume conventions: “M” for
# millions, “B” for billions and “T” for trillions【815272332623032†L72-L90】.
# Thousands can optionally be displayed with a “K” suffix.  Numbers
# below a thousand are left unchanged.  If the value is missing or
# cannot be converted to a float, the function returns "N/A".
def format_market_cap(value: float) -> str:
    """Format a numeric market cap into a human‑readable string.

    Args:
        value: The raw market cap as a float or integer.

    Returns:
        A string representing the value in trillions (T), billions (B),
        millions (M), thousands (K), or the original number if smaller.
    """
    try:
        val = float(value)
        # guard against NaN
        if np.isnan(val):
            return "N/A"
        # trillions
        if abs(val) >= 1e12:
            return f"{val / 1e12:.2f}T"
        # billions
        if abs(val) >= 1e9:
            return f"{val / 1e9:.2f}B"
        # millions
        if abs(val) >= 1e6:
            return f"{val / 1e6:.2f}M"
        # thousands
        if abs(val) >= 1e3:
            return f"{val / 1e3:.2f}K"
        # small numbers
        return f"{val:.2f}"
    except Exception:
        return "N/A"

st.set_page_config(page_title="Complete Breakout Scanner", layout="wide")
st.title("📈Breakout Scanner")

symbols = st.session_state.get("SYMBOLS", [])
if not symbols:
    st.warning("Upload a symbols CSV on the Home page first.")
    st.stop()

with st.sidebar:
    st.header("📊 Filters")
    f_cpr = st.multiselect("CPR Trend (y vs dby)", ["Ascending","Descending","Sideways"], default=["Ascending","Sideways"])
    f_ema = st.multiselect("EMA Filter", ["Close>EMA20","Close>EMA7","Close<EMA20","Close<EMA7"], default=[])
    f_vol = st.multiselect("Volume Filter", ["Vol>Yday","Vol>5dAvg","Vol>2x5dAvg"], default=[])
    f_break = st.multiselect("Breakout Type", ["Swing","CPR_Top","CPR_Bottom","52W","PreBreakout","Confirmed"], default=[])
    f_momo = st.multiselect("Momentum Strength", ["🔥 Strong Up","⚠️ Strong Down","↔️ Mild","Neutral"], default=["🔥 Strong Up","↔️ Mild","Neutral"])
    near_52w = st.multiselect("Near 52W", ["Near 52W High","Near 52W Low"], default=[])

progress = st.progress(0.0)
rows = []

for i, sym in enumerate(symbols):
    try:
        d = load_daily(sym)  # 400d
        if len(d) < 30: 
            progress.progress((i+1)/len(symbols)); continue

        d = enrich_daily(d)
        d, sh_str, sl_str = detect_swings(d, left=2, right=2, top_n=5)

        yday, dby = d.iloc[-2], d.iloc[-3]
        y_piv, y_bc, y_tc, R1,S1,R2,S2 = calc_cpr(yday["High"], yday["Low"], yday["Close"])
        dby_piv, dby_bc, dby_tc, *_ = calc_cpr(dby["High"], dby["Low"], dby["Close"])

        today = d.iloc[-1]
        cpr_y_trend = cpr_trend(y_piv, y_bc, y_tc, dby_piv, dby_bc, dby_tc)
        latest_tr = latest_cpr_trend(today["Close"], y_bc, y_tc)

        # near 52W flags
        w_high, w_low = today["52W_High"], today["52W_Low"]
        near_flag = "—"
        if pd.notna(w_high) and today["Close"] >= 0.98 * w_high: near_flag = "Near 52W High"
        if pd.notna(w_low) and today["Close"] <= 1.02 * w_low: near_flag = "Near 52W Low"

        # pre-breakout/breakdown using last swing
        preBO = (
        pd.notna(today["Last_Swing_High"])                       # swing high exists
        and (today["Close"] < today["Last_Swing_High"])          # still below resistance (not yet broken)
        and (today["Close"] >= 0.98 * today["Last_Swing_High"])  # within −2 % of swing high
        and (today["Volume"] >= 0.6 * today["AvgVol20"])         # decent volume
        and (today.get("VWROC", 0) >= 0)                         # bullish / neutral momentum
        and (today.get("VPI", 0) >= 0)                           # confirms positive pressure
        and (cpr_y_trend in ["Ascending", "Sideways"])           # CPR alignment supports breakout
             )
        
        preBD = (
                pd.notna(today["Last_Swing_Low"])
                and (today["Close"] > today["Last_Swing_Low"])        # still above the low
                and (today["Close"] <= 1.02 * today["Last_Swing_Low"]) # within +2 % buffer
                and (today["Volume"] >= 0.6 * today["AvgVol20"])
                and (today.get("VWROC", 0) <= 0)                      # bearish / neutral momentum
                and (today.get("VPI", 0) <= 0)
                and (cpr_y_trend in ["Descending", "Sideways"])
            )

        # st.write(today["Last_Swing_Low"],"Latestswinglow",today["Close"],1.02 * today["Last_Swing_Low"])
        # momentum-confirmed swing BO/BD in last 5 swing points
        recent_swing_high = d["Swing_High"].dropna().tail(5).max() if not d["Swing_High"].dropna().empty else np.nan
        recent_swing_low  = d["Swing_Low"].dropna().tail(5).min() if not d["Swing_Low"].dropna().empty else np.nan

        swing_BO = (pd.notna(recent_swing_high) and today["Close"] >= recent_swing_high*0.999 
                    and today["Volume"] > 1.5*today["AvgVol20"] 
                    and today["VWROC"] > 0.05 and today["VPI"] > 0.2 
                    and cpr_y_trend in ["Ascending","Sideways"])
        swing_BD = (pd.notna(recent_swing_low) and today["Close"] <= recent_swing_low*1.001 
                    and today["Volume"] > 1.5*today["AvgVol20"] 
                    and today["VWROC"] < -0.05 and today["VPI"] < -0.2 
                    and cpr_y_trend in ["Descending","Sideways"])

        # conditions for filters
        cond = {
            "Close>YdayHigh": today["Close"] > yday["High"],
            "Close<YdayLow": today["Close"] < yday["Low"],
            "Close>EMA20": today["Close"] > today["EMA20"],
            "Close>EMA7": today["Close"] > today["EMA7"],
            "Close<EMA20": today["Close"] < today["EMA20"],
            "Close<EMA7": today["Close"] < today["EMA7"],
        }
        vol5 = d["Volume"].iloc[-6:-1].mean() if len(d) >= 6 else d["Volume"].mean()
        cond.update({
            "Vol>Yday": today["Volume"] > yday["Volume"],
            "Vol>5dAvg": today["Volume"] > vol5,
            "Vol>2x5dAvg": today["Volume"] > 2*vol5
        })

        # breakout tag combo
        tags = breakout_tags(today, yday["High"], yday["Low"], y_bc, y_tc)

        # score
        score = 0
        if cond["Close>YdayHigh"]: score += 1
        if cond["Vol>Yday"]: score += 1
        if cond["Close>EMA20"]: score += 1
        if cond["Close>EMA7"]: score += 1
        if cond["Vol>5dAvg"]: score += 2
        if cond["Vol>2x5dAvg"]: score += 2
        if "CPR_Top" in tags: score += 1
        if "Swing" in tags: score += 2
        if "52W" in tags: score += 2
        if swing_BO: score += 3

        # -------------------------------------------------------------------
        # 🏢 Market Capitalization
        #
        # Fetch market capitalization for the current symbol.  Market cap is
        # computed as the latest close price multiplied by the total number of
        # shares outstanding (if available).  We retrieve both the shares
        # outstanding and the reported market cap from yfinance's `Ticker.info`.
        # If shares outstanding is not available, we fall back to the reported
        # marketCap value.  Any exceptions or missing data result in a NaN.  
        market_cap = np.nan
        try:
            ticker = yf.Ticker(sym)
            info = ticker.info or {}
            shares_outstanding = info.get("sharesOutstanding")
            if shares_outstanding:
                market_cap = today["Close"] * shares_outstanding
            else:
                mcap = info.get("marketCap")
                if mcap:
                    market_cap = mcap
        except Exception:
            market_cap = np.nan

        rows.append({
            "Symbol": sym,
            "Close": round(today["Close"],2),
            "Yday_High": round(yday["High"],2),
            "Yday_Low": round(yday["Low"],2),
            "EMA20": round(today["EMA20"],2),
            "EMA7": round(today["EMA7"],2),
            "Volume": int(today["Volume"]),
            "AvgVol20": int(today["AvgVol20"]),
            "CPR_Trend": cpr_y_trend,
            "Intraday_CPR": latest_tr,
            "Momentum_Strength": today["Momentum_Strength"],
            "Momentum_Trend": today["Momentum_Trend"],
            "Strong_Resistance": sh_str,
            "Strong_Support": sl_str,
            "Latest_Swing_High": round(today.get("Last_Swing_High", np.nan),2) if pd.notna(today.get("Last_Swing_High", np.nan)) else np.nan,
            "Latest_Swing_Low": round(today.get("Last_Swing_Low", np.nan),2) if pd.notna(today.get("Last_Swing_Low", np.nan)) else np.nan,
            "Near_52W_HighLow": near_flag,
            "52W_High": round(w_high,2) if pd.notna(w_high) else np.nan,
            "52W_Low": round(w_low,2) if pd.notna(w_low) else np.nan,
            "Breakout_Type": tags,
            "PreBreakout": preBO,
            "PreBreakdown": preBD,
            "Confirmed_Breakout": swing_BO,
            "Confirmed_Breakdown": swing_BD,
            "Vol>Yday": cond["Vol>Yday"],
            "Vol>5dAvg": cond["Vol>5dAvg"],
            "Vol>2x5dAvg": cond["Vol>2x5dAvg"],
            "Close>EMA20": cond["Close>EMA20"],
            "Close>EMA7": cond["Close>EMA7"],
            "Close<EMA20": cond["Close<EMA20"],
            "Close<EMA7": cond["Close<EMA7"],
            "Score": score,
            # add market capitalization column
            "Market_Cap": round(market_cap) if pd.notna(market_cap) else np.nan
        })

    except Exception as e:
        # keep going
        pass

    progress.progress((i+1)/len(symbols))

df = pd.DataFrame(rows)
if df.empty:
    st.error("No data built. Check symbols or try again.")
    st.stop()

# Add a human‑readable market cap column
# We compute this outside the row loop so that the numerical 'Market_Cap'
# column remains available for sorting/filtering.  The formatted string
# uses the helper defined above and will show values in trillions,
# billions, millions or thousands as appropriate.
df["Market_Cap_Display"] = df["Market_Cap"].apply(format_market_cap)

# apply sidebar filters
mask = pd.Series(True, index=df.index)

if f_cpr:
    mask &= df["CPR_Trend"].isin(f_cpr)
for tag in f_break:
    if tag == "PreBreakout": mask &= df["PreBreakout"] == True
    elif tag == "Confirmed": mask &= df["Confirmed_Breakout"] == True
    else: mask &= df["Breakout_Type"].str.contains(tag, na=False)

if f_ema:
    for k in f_ema:
        mask &= df[k.replace(">",">").replace("<","<")] == True

if f_vol:
    for k in f_vol:
        mask &= df[k] == True

if f_momo:
    mask &= df["Momentum_Strength"].isin(f_momo)

if near_52w:
    mask &= df["Near_52W_HighLow"].isin(near_52w)

view = df[mask].sort_values(["Score","Close"], ascending=[False, False]).reset_index(drop=True)
# Remove the raw numeric Market_Cap column from the displayed results.  We
# still keep it in `view` for potential sorting/filtering, but the end
# user will see only the formatted Market_Cap_Display.
view_display = view.drop(columns=["Market_Cap"], errors="ignore")

st.subheader("Results")

# =========================
# 🎨 Color Styling Functions
# =========================

def highlight_breakout(row):
    styles = [''] * len(row)

    # 1️⃣ Confirmed Breakout
    if row.get("Confirmed_Breakout", False):
        styles = ['background-color: #228B22; color: white; font-weight: bold;'] * len(row)  # Dark Green

    # 2️⃣ Pre-Breakout
    elif row.get("PreBreakout", False):
        styles = ['background-color: #FFD580; color: black; font-weight: bold;'] * len(row)  # Light Orange

    # 3️⃣ Pre-Breakdown
    elif row.get("PreBreakdown", False):
        styles = ['background-color: #FFB6B6; color: black;'] * len(row)  # Light Red

    # 4️⃣ Volume > AvgVol20 and breaks EMA20 + YdayHigh
    elif (
        row["Volume"] > row["AvgVol20"]
        and row["Close"] > row["EMA20"]
        and row["Close"] > row["Yday_High"]
    ):
        styles = ['background-color: #90EE90; color: black; font-weight: bold;'] * len(row)  # Light Green

    # 5️⃣ Weak volume / below EMA
    elif row["Close"] < row["EMA20"] and row["Volume"] < row["AvgVol20"]:
        styles = ['background-color: #f4cccc; color: black;'] * len(row)  # Pale Red

    return styles


def color_cpr(val):
    if val == "Ascending":
        return "background-color: #ADD8E6;"  # light blue
    elif val == "Descending":
        return "background-color: #FFD580;"  # light orange
    else:
        return ""


# -------------------------
# 🎨 Highlight Volume Column Only
# -------------------------
def color_volume(val, avg_vol):
    try:
        if val > 2 * avg_vol:
            return "background-color: #006600; color: white; font-weight: bold;"  # Dark green
        elif val > avg_vol:
            return "background-color: #90EE90; color: black; font-weight: bold;"  # Light green
        else:
            return ""
    except:
        return ""

def apply_volume_style(df):
    """
    Apply styling to the DataFrame.

    This helper applies three levels of conditional formatting:

    1. Row‑level breakout styling via ``highlight_breakout``.
    2. Single‑cell CPR trend coloring via ``color_cpr`` on the
       ``CPR_Trend`` column (if present).
    3. Volume cell highlighting: this is handled on a per‑row basis
       rather than the earlier column‑wise approach.  We construct a
       style list for each row and assign a style only to the ``Volume``
       column.  This avoids KeyError issues when ``Volume`` or
       ``AvgVol20`` are missing and makes the logic robust across
       varying DataFrame schemas.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to style.

    Returns
    -------
    pandas.io.formats.style.Styler
        A styled DataFrame ready for display with Streamlit.
    """
    styled = df.style
    # Highlight entire row for breakout/prebreakout conditions
    styled = styled.apply(highlight_breakout, axis=1)
    # Color the CPR trend cell if present.  Use applymap on the column to
    # handle each cell individually.  We only call this when the column
    # exists to avoid KeyError on missing columns.
    if "CPR_Trend" in df.columns:
        # Use ``map`` instead of ``applymap`` because some older pandas
        # versions do not implement ``Styler.applymap``.  The ``map`` method
        # applies an elementwise function to the specified subset.
        styled = styled.map(color_cpr, subset=["CPR_Trend"])

    # Highlight the Volume cell based on the ratio of ``Volume`` to
    # ``AvgVol20``.  This is done row‑wise so we can use row values
    # directly.  For each row, we build a list of styles (one per
    # column) and set the style for the Volume column only when both
    # ``Volume`` and ``AvgVol20`` values are present.
    def highlight_volume(row):
        # Start with no styles for the row
        style_row = [''] * len(row)
        # Proceed only if the DataFrame has the necessary columns
        if "Volume" in df.columns and "AvgVol20" in df.columns:
            vol = row.get("Volume")
            avg_vol = row.get("AvgVol20")
            # Only attempt formatting if both values are not null
            if vol is not None and avg_vol is not None:
                style = color_volume(vol, avg_vol)
                if style:
                    # Find the index of the Volume column and assign the style
                    idx = list(df.columns).index("Volume")
                    style_row[idx] = style
        return style_row

    # Apply the volume highlight function row‑wise
    styled = styled.apply(highlight_volume, axis=1)
    return styled

# =========================
# 🖼️ Apply Styling
# =========================
# Apply our volume and breakout styling to the filtered and column‑pruned
# DataFrame.  This helper will highlight rows based on breakout status,
# color the CPR trend column and apply conditional formatting to the
# Volume column.  We operate on `view_display` to omit the raw
# numeric Market_Cap column from the UI.
styled_view = apply_volume_style(view_display)

st.dataframe(styled_view, width="content", height=520)

# Download button (keep same)
st.download_button(
    "📥 Download Results CSV",
    data=view.to_csv(index=False).encode("utf-8"),
    file_name="breakout_results.csv",
    mime="text/csv"
)

