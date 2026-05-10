# =====================================================
# 📈 Streamlit App
# Heikin Ashi + Swing + EMA + Volume (BUY / SELL)
# =====================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

import plotly.graph_objects as go
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score





# -----------------------------------------------------
# PAGE CONFIGsn
# -----------------------------------------------------
st.set_page_config(page_title="HA Swing Intraday Strategy", layout="wide")
st.title("📈 HA Swing Intraday Strategy")

# st.markdown("""
# **BUY**
# - HA Swing Low  
# - HA trend bullish + normal candle bullish  
# - EMA Fast > EMA Slow  
# - Volume above average  

# **SELL**
# - HA Swing High  
# - HA trend bearish + normal candle bearish  
# - EMA Fast < EMA Slow  
# - Volume above average  

# """)

# -------------------------------------------------
# 🧱 Institutional Support & Resistance Detector
# -------------------------------------------------
# -------------------------------------------------
# CLEAN Support & Resistance Detector
# -------------------------------------------------
# =====================================================
# 🔥 Strong Support & Resistance Detection
# =====================================================

def detect_clean_levels(df, lookback=200, min_touches=3, tolerance=0.003):
    recent = df.tail(lookback).copy()

    levels = []

    candidate_levels = list(recent["High"]) + list(recent["Low"])

    for level in candidate_levels:
        touches = (
            ((recent["High"] >= level * (1 - tolerance)) &
             (recent["High"] <= level * (1 + tolerance))) |
            ((recent["Low"] >= level * (1 - tolerance)) &
             (recent["Low"] <= level * (1 + tolerance)))
        ).sum()

        if touches >= min_touches:
            levels.append((level, touches))

    # Remove near duplicates
    cleaned = []
    for lvl, strength in sorted(levels, key=lambda x: -x[1]):
        if not any(abs(lvl - x[0]) < lvl * 0.005 for x in cleaned):
            cleaned.append((lvl, strength))

    return cleaned[:5]  # Keep top 6 strongest

st.caption("⚠️ This platform is for educational and research purposes only.It does not constitute investment advice or a recommendation")
st.caption("⚠️Market investments are subject to risk.")

# -----------------------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------------------
st.sidebar.header("⚙️ Filters")

INTRADAY_INTERVALS = ["2m", "5m", "15m", "1h"]
HTF_INTERVALS = ["1d", "1wk", "1mo"]

symbol = st.sidebar.text_input("Symbol (Yahoo)", value="^NSEI")
interval = st.sidebar.selectbox("Interval", ["1m","2m", "5m","15m","1h","1d","1wk","1mo"], index=1)

# -------------------------------------------------
# ⏱️ Yahoo-safe period selection
# -------------------------------------------------
if interval in ["2m", "5m","1m"]:
    period = "7d"
elif interval == "15m":
    period = "60d"
elif interval == "1h":
    period = "1y"
elif interval in ["1d", "1wk"]:
    period = "5y"
 
elif interval == "1mo":
    period = "10y"

# Yahoo safe limits
#period = "5d" if interval == "3m" else "7d"

ema_fast = st.sidebar.number_input("EMA Fast", value=20, min_value=5)
ema_slow = st.sidebar.number_input("EMA Slow", value=50, min_value=10)
rr = st.sidebar.slider("Risk : Reward", 1.0, 3.0, 1.5, 0.1)

run = st.sidebar.button("🚀 Run Strategy")


def highlight_options_signal(row):

    signal = row.get("OPTIONS_SIGNAL", "")

    if signal == "BUY_CALL":
        return ["background-color:#90EE90"] * len(row)

    elif signal == "BUY_PUT":
        return ["background-color:#FF7F7F"] * len(row)

    return [""] * len(row)


def highlight_Signal(row):

    signal = row.get("SIGNAL", "")

    if signal == "BUY":
        return ["background-color:#90EE90"] * len(row)

    elif signal == "SELL":
        return ["background-color:#FF7F7F"] * len(row)

    return [""] * len(row)

def detect_strong_levels(df, tolerance=5):

    levels = []

    for i in range(2, len(df)-2):

        high = df["High"].iloc[i]
        low = df["Low"].iloc[i]

        if high > df["High"].iloc[i-1] and high > df["High"].iloc[i+1]:
            levels.append(high)

        if low < df["Low"].iloc[i-1] and low < df["Low"].iloc[i+1]:
            levels.append(low)

    strong_levels = []

    for level in levels:

        touches = sum(abs(level-l) < tolerance for l in levels)

        if touches >= 3:
            strong_levels.append(level)

    strong_levels = sorted(set(strong_levels))

    current_price = df["Close"].iloc[-1]

    # split levels
    supports = [lvl for lvl in strong_levels if lvl < current_price]
    resistances = [lvl for lvl in strong_levels if lvl > current_price]

    # closest levels
    nearest_support = max(supports) if supports else None
    nearest_resistance = min(resistances) if resistances else None

    return nearest_support, nearest_resistance

def swing_trading_signals_v2(df):
    # -------------------------------
    # Candle & Volume Metrics
    # -------------------------------
    df['candle_size'] = df['Close'] - df['Open']
    df['body_size'] = abs(df['candle_size'])
    df['avg_body_20'] = df['body_size'].rolling(20).mean()

    df['avg_volume_20'] = df['Volume'].rolling(20).mean()

    df['bullish'] = df['Close'] > df['Open']
    df['bearish'] = df['Close'] < df['Open']

    df['strong_bullish'] = df['bullish'] & (df['body_size'] > 1.2 * df['avg_body_20'])
    df['strong_bearish'] = df['bearish'] & (df['body_size'] > 1.2 * df['avg_body_20'])

    df['vol_spike'] = df['Volume'] > 1.5 * df['avg_volume_20']

    # -------------------------------
    # Trend Filter (EMA)
    # -------------------------------
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

    df['uptrend'] = df['EMA_20'] > df['EMA_50']
    df['downtrend'] = df['EMA_20'] < df['EMA_50']

    # -------------------------------
    # Breakout / Breakdown
    # -------------------------------
    df['breakout'] = df['Close'] > df['High'].shift(1)
    df['breakdown'] = df['Close'] < df['Low'].shift(1)

    # -------------------------------
    # ATR for Dynamic Risk
    # -------------------------------
    df['tr'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    df['ATR_14'] = df['tr'].rolling(14).mean()

    # -------------------------------
    # BUY / SELL Signals
    # -------------------------------
    df['buy_signal'] = (
        df['uptrend'] &
        df['vol_spike'] &
        df['strong_bullish'] &
        df['breakout']
    )

    df['sell_signal'] = (
        df['downtrend'] &
        df['vol_spike'] &
        df['strong_bearish'] &
        df['breakdown']
    )

    # -------------------------------
    # Trade Levels
    # -------------------------------
    df['entry'] = np.where(df['buy_signal'], df['Close'],
                    np.where(df['sell_signal'], df['Close'], np.nan))

    df['stop_loss'] = np.where(
        df['buy_signal'], df['Close'] - 1.2 * df['ATR_14'],
        np.where(df['sell_signal'], df['Close'] + 1.2 * df['ATR_14'], np.nan)
    )

    df['target'] = np.where(
        df['buy_signal'], df['entry'] + 1.8 * (df['entry'] - df['stop_loss']),
        np.where(df['sell_signal'], df['entry'] - 1.8 * (df['stop_loss'] - df['entry']), np.nan)
    )

    # -------------------------------
    # Trade Setup Label
    # -------------------------------
    df['trade_setup'] = np.where(
        df['buy_signal'],
        'BUY',
        np.where(df['sell_signal'], 'SELL', '')
    )

    return df


# -----------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------
def heikin_ashi(df):
    df["HA_Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4

    ha_open = [(df["Open"].iloc[0] + df["Close"].iloc[0]) / 2]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i-1] + df["HA_Close"].iloc[i-1]) / 2)

    df["HA_Open"] = ha_open
    df["HA_High"] = df[["High", "HA_Open", "HA_Close"]].max(axis=1)
    df["HA_Low"] = df[["Low", "HA_Open", "HA_Close"]].min(axis=1)

    # Explicit bullish / bearish flags
    df["HA_Bull"] = df["HA_Close"] > df["HA_Open"]
    df["Candle_Bull"] = df["Close"] > df["Open"]

    df["HA_Color"] = np.where(df["HA_Bull"], "GREEN", "RED")
    return df


def detect_price_swings(df, lookback=2):
    """
    Detect Swing High and Swing Low using non-repainting price structure logic.

    Parameters:
    df (pd.DataFrame): DataFrame with columns ['High', 'Low']
    lookback (int): Number of candles on each side for swing confirmation

    Returns:
    pd.DataFrame: df with SWING_HIGH and SWING_LOW columns
    """

    df = df.copy()
    df["SWING_HIGH"] = False
    df["SWING_LOW"] = False

    for i in range(lookback, len(df) - lookback):
        # Swing High
        if (
            df["High"].iloc[i] > df["High"].iloc[i-1] and
            df["High"].iloc[i] > df["High"].iloc[i+1] and
            df["High"].iloc[i] > df["High"].iloc[i-lookback] and
            df["High"].iloc[i] > df["High"].iloc[i+lookback]
        ):
            df.at[df.index[i], "SWING_HIGH"] = True

        # Swing Low
        if (
            df["Low"].iloc[i] < df["Low"].iloc[i-1] and
            df["Low"].iloc[i] < df["Low"].iloc[i+1] and
            df["Low"].iloc[i] < df["Low"].iloc[i-lookback] and
            df["Low"].iloc[i] < df["Low"].iloc[i+lookback]
        ):
            df.at[df.index[i], "SWING_LOW"] = True

    return df


def detect_swings(df):
    df["HA_SWING_LOW"] = False
    df["HA_SWING_HIGH"] = False

    for i in range(2, len(df) - 2):
        # Swing Low (support)
        if (
            df["HA_Low"].iloc[i] < df["HA_Low"].iloc[i-1] and
            df["HA_Low"].iloc[i] < df["HA_Low"].iloc[i+1]
        ):
            df.at[df.index[i], "HA_SWING_LOW"] = True

        # Swing High (resistance)
        if (
            df["HA_High"].iloc[i] > df["HA_High"].iloc[i-1] and
            df["HA_High"].iloc[i] > df["HA_High"].iloc[i+1]
        ):
            df.at[df.index[i], "HA_SWING_HIGH"] = True

    df["HA_SWING_LOW_PRICE"] = np.where(df["HA_SWING_LOW"], df["HA_Low"], np.nan)
    df["HA_SWING_HIGH_PRICE"] = np.where(df["HA_SWING_HIGH"], df["HA_High"], np.nan)

    df["HA_SW_LOW"] = df["HA_SWING_LOW_PRICE"].ffill().round(2)
    df["HA_SW_HIGH"] = df["HA_SWING_HIGH_PRICE"].ffill().round(2)

    return df

# -----------------------------------------------------
# RUN STRATEGY
# -----------------------------------------------------
if run:
    df = yf.download(
        symbol,
        interval=interval,
        period=period,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        st.error("No data returned from Yahoo Finance.")
        st.stop()

    # Fix columns
    df.columns = df.columns.get_level_values(0)

    # Handle timezone correctly
    #if interval != "1d":
    if interval not in ["1d", "1wk", "1mo"]:
        # Intraday data → tz-aware
        df.index = df.index.tz_convert("Asia/Kolkata")
        df["DateTime"] = df.index
    else:
        # Daily data → tz-naive
        df.index = pd.to_datetime(df.index)
        df["DateTime"] = df.index

    if interval != "1d":
        # -----------------------------
        # 🕒 ORB WINDOW (9:15 – 9:30)
        # -----------------------------
        df["TradeDate"] = df["DateTime"].dt.date
        df["Hour"] = df["DateTime"].dt.hour
        df["Minute"] = df["DateTime"].dt.minute

        orb_window = (
            (df["Hour"] == 9) &
            (df["Minute"] >= 15) &
            (df["Minute"] < 30)
        )

        df["ORB_HIGH"] = np.nan
        df["ORB_LOW"] = np.nan

        for d in df["TradeDate"].unique():
            day_mask = df["TradeDate"] == d
            orb_mask = day_mask & orb_window

            if orb_mask.any():
                df.loc[day_mask, "ORB_HIGH"] = df.loc[orb_mask, "High"].max()
                df.loc[day_mask, "ORB_LOW"]  = df.loc[orb_mask, "Low"].min()
    else:
        # Daily → ORB not applicable
        df["TradeDate"] = df["DateTime"].dt.date
        df["ORB_HIGH"] = np.nan
        df["ORB_LOW"] = np.nan
        df["ORB_SIGNAL"] = ""

        # -------------------------------------------------


   

    # -------------------------------------------------
    # INDICATORS
    # -------------------------------------------------
    df = heikin_ashi(df)

    df[f"EMA_{ema_fast}"] = df["Close"].ewm(span=ema_fast).mean()
    df[f"EMA_{ema_slow}"] = df["Close"].ewm(span=ema_slow).mean()
    df[f"EMA_7"] = df["Close"].ewm(span=7).mean()

    tick_size = 0.05   # safe tolerance for float comparison

    df["OPEN_EQ_LOW"] = (df["Open"] - df["Low"]).abs() <= tick_size
    df["OPEN_EQ_HIGH"] = (df["High"] - df["Open"]).abs() <= tick_size


    # 🔔 OPEN TYPE (ADD HERE — EARLY)
    df["OPEN_TYPE"] = np.where(
        df["OPEN_EQ_LOW"], "OPEN=LOW",
        np.where(df["OPEN_EQ_HIGH"], "OPEN=HIGH", "")
    )

    df = detect_price_swings(df)

    df["SWING_HIGH_LEVEL"] = np.where(df["SWING_HIGH"], df["High"], np.nan)
    df["SWING_LOW_LEVEL"] = np.where(df["SWING_LOW"], df["Low"], np.nan)

    # Carry forward latest swing levels
    df["LAST_SWING_HIGH"] = df["SWING_HIGH_LEVEL"].ffill().shift(1)
    df["LAST_SWING_LOW"] = df["SWING_LOW_LEVEL"].ffill().shift(1)

   
    # Volume analysis
    df["VOL_AVG_20"] = df["Volume"].rolling(20).mean()
    df["VOL_STRONG"] = df["Volume"] > df["VOL_AVG_20"]
    df["VOL_HUGE"] = df["Volume"] > 1.2 * df["VOL_AVG_20"]
    df["RANGE"] = (df["High"] - df["Low"]).replace(0, np.nan)
    df["BODY"]  = abs(df["Close"] - df["Open"])
    # Strong decisive candle (no indecision)
    df["STRONG_BODY"] = df["BODY"] >= 0.6 * df["RANGE"]

        # 🟢 Buyer dominance → accumulation
    df["BUYER_DOMINANT"] = (
        (df["Close"] > df["Open"]) &   # bullish
        df["STRONG_BODY"] &            # strong body
        df["VOL_HUGE"]                 # strong participation
    )

    # 🔴 Seller dominance → distribution
    df["SELLER_DOMINANT"] = (
        (df["Close"] < df["Open"]) &   # bearish
        df["STRONG_BODY"] &            # strong body
        df["VOL_HUGE"]                 # strong participation
    )


        # Buyers defended price from LOW
    df["BUYER_PRICE_LEVEL"] = np.where(
        df["BUYER_DOMINANT"],
        df["Low"],
        np.nan
    )

    # Sellers defended price from HIGH
    df["SELLER_PRICE_LEVEL"] = np.where(
        df["SELLER_DOMINANT"],
        df["High"],
        np.nan
    )


    # AFTER ORB flag (CRITICAL FIX)
    if interval != "1d":
        after_orb = (
            (df["Hour"] > 9) |
            ((df["Hour"] == 9) & (df["Minute"] >= 30))
        )
    else:
        after_orb = pd.Series(True, index=df.index)  
    df["ORB_SIGNAL"] = ""

    orb_buy = (
        after_orb &
        (df["Close"] > df["ORB_HIGH"]) 
       # (df[f"EMA_{ema_fast}"] > df[f"EMA_{ema_slow}"])
    )

    orb_sell = (
        after_orb &
        (df["Close"] < df["ORB_LOW"]) #&
        #(df[f"EMA_{ema_fast}"] < df[f"EMA_{ema_slow}"])
    )

    df.loc[orb_buy, "ORB_SIGNAL"] = "ORB_BUY"
    df.loc[orb_sell, "ORB_SIGNAL"] = "ORB_SELL"


    def keep_first_signal(x):
        seen = False
        out = []
        for v in x:
            if v != "" and not seen:
                out.append(v)
                seen = True
            else:
                out.append("")
        return out

    df["ORB_SIGNAL"] = (
        df.groupby("TradeDate")["ORB_SIGNAL"]
        .transform(keep_first_signal)
    )

    df = detect_swings(df)

    # -------------------------------------------------
    # SINGLE SIGNAL PER CANDLE (PRIORITY LOGIC)
    # -------------------------------------------------
    df["SIGNAL"] = ""

    buy_cond = (
        df["HA_SWING_LOW"] &
        df["HA_Bull"] &
       # df["Candle_Bull"] &
        (df[f"EMA_{ema_fast}"] > df[f"EMA_{ema_slow}"]) &
        (df["Close"] > df[f"EMA_{ema_fast}"]) 
        #df["VOL_STRONG"]
    )

    sell_cond = (
        df["HA_SWING_HIGH"] &
        (~df["HA_Bull"]) &
       # (~df["Candle_Bull"]) &
        (df[f"EMA_{ema_fast}"] < df[f"EMA_{ema_slow}"]) &
        (df["Close"] < df[f"EMA_{ema_fast}"]) 
        #df["VOL_STRONG"]
    )

    df.loc[buy_cond, "SIGNAL"] = "BUY"
    df.loc[~buy_cond & sell_cond, "SIGNAL"] = "SELL"

    # -------------------------------------------------
    # SL & TARGET
    # -------------------------------------------------
    df["SL"] = np.where(
        df["SIGNAL"] == "BUY",
        df["HA_SW_LOW"] - 0.1,
        np.where(
            df["SIGNAL"] == "SELL",
            df["HA_SW_HIGH"] + 0.1,
            np.nan
        )
    )

    df["TARGET"] = np.where(
        df["SIGNAL"] == "BUY",
        df["Close"] + (df["Close"] - df["SL"]) * rr,
        np.where(
            df["SIGNAL"] == "SELL",
            df["Close"] - (df["SL"] - df["Close"]) * rr,
            np.nan
        )
    )


    price_bins = pd.cut(df["Close"], bins=50)

    volume_profile = df.groupby(price_bins)["Volume"].sum()

    poc_bin = volume_profile.idxmax()

    POC = poc_bin.mid

    total_volume = volume_profile.sum()

    sorted_profile = volume_profile.sort_values(ascending=False)

    cum_volume = sorted_profile.cumsum()

    value_area = sorted_profile[cum_volume <= 0.7 * total_volume]

    VAL = value_area.index.min().left
    VAH = value_area.index.max().right
    if "POC" not in df.columns:
     df["POC"] = np.nan

    df.loc[df.index[-1], "POC"] = POC
    df.loc[df.index[-1], "VAL"] = VAL
    df.loc[df.index[-1], "VAH"] = VAH

    df["BUY_SETUP"] = (
    (df["Low"] <= df["VAL"]) &
    (df["Close"] > df["VAL"]) &
    (df["VOL_STRONG"])
)
    df["SELL_SETUP"] = (
    (df["High"] >= df["VAH"]) &
    (df["Close"] < df["VAH"]) &
    (df["VOL_STRONG"])
)
    df["SIGNAL_VOLP"] = np.where(
        df["BUY_SETUP"], "BUY",
        np.where(df["SELL_SETUP"], "SELL", "HOLD")
    )

    df["OPEN_TYPE"] = np.where(
    df["OPEN_EQ_LOW"], "OPEN=LOW",
    np.where(df["OPEN_EQ_HIGH"], "OPEN=HIGH", "")
)
    df["PREV_HIGH"] = df["High"].shift(1)
    df["PREV_LOW"]  = df["Low"].shift(1)
    df["PREV_CLOSE"]  = df["Close"].shift(1)

    df["VALID_OPEN_LOW"] = (
    df["OPEN_EQ_LOW"] &
    (df["Close"] > df["PREV_HIGH"]) &                 # breakout
    (df["Volume"] >= 0.8 * df["VOL_AVG_20"]) &       # volume support
    (df[f"EMA_{ema_fast}"] > df[f"EMA_{ema_slow}"])  # trend support
)

    df["VALID_OPEN_HIGH"] = (
        df["OPEN_EQ_HIGH"] &
        (df["Close"] < df["PREV_LOW"]) &                   # breakdown
        (df["Volume"] >= 0.8 * df["VOL_AVG_20"]) &       # volume support
        (df[f"EMA_{ema_fast}"] < df[f"EMA_{ema_slow}"])  # trend support
    )

    df["OPEN_VALIDATION"] = np.where(
    df["VALID_OPEN_LOW"], "VALID OPEN=LOW (BULLISH)",
    np.where(
        df["VALID_OPEN_HIGH"], "VALID OPEN=HIGH (BEARISH)",
        ""
    )
)
  
    df["RANGE"] = (df["High"] - df["Low"]).replace(0, np.nan)
    df["BODY"] = abs(df["Close"] - df["Open"])

    df["PRICE_BREAKOUT"] = (
        (df["Close"] > df["PREV_HIGH"]) &
        (df["Close"] > df["Open"]) &
        (df["BODY"] > 0.4 * df["RANGE"])
    )

    df["PRICE_BREAKDOWN"] = (
        (df["Close"] < df["PREV_LOW"]) &
        (df["Close"] < df["Open"]) &
        (df["BODY"] > 0.4 * df["RANGE"])
    )

    # =====================================================
    # 🧠 SMART MONEY + AI MODULE
    # =====================================================

    # -------------------------
    # Smart Money Detection
    # -------------------------
    df["CANDLE_RANGE"] = df["High"] - df["Low"]

    df["UPPER_WICK"] = df["High"] - df[["Open","Close"]].max(axis=1)
    df["LOWER_WICK"] = df[["Open","Close"]].min(axis=1) - df["Low"]

    df["PRICE_CHANGE"] = df["Close"].diff()

    df["BUY_VOLUME"] = np.where(df["Close"] > df["Open"], df["Volume"], 0)
    df["SELL_VOLUME"] = np.where(df["Close"] < df["Open"], df["Volume"], 0)

    df["VOLUME_DELTA"] = df["BUY_VOLUME"] - df["SELL_VOLUME"]
    df["CUM_DELTA"] = df["VOLUME_DELTA"].cumsum()

    smart_conditions = [

        (df["PRICE_CHANGE"] < 0) &
        (df["VOLUME_DELTA"] > 0) &
        (df["LOWER_WICK"] > df["CANDLE_RANGE"] * 0.4) &
        (df["Volume"] > df["VOL_AVG_20"]),

        (df["PRICE_CHANGE"] > 0) &
        (df["VOLUME_DELTA"] < 0) &
        (df["UPPER_WICK"] > df["CANDLE_RANGE"] * 0.4) &
        (df["Volume"] > df["VOL_AVG_20"])
    ]

    smart_choices = [
        "Hidden Accumulation",
        "Hidden Distribution"
    ]

    df["SMART_SIGNAL"] = np.select(smart_conditions, smart_choices, default="")

    # -------------------------
    # Order Blocks
    # -------------------------
    df["BULLISH_OB"] = (
        (df["Close"].shift(1) < df["Open"].shift(1)) &
        (df["Close"] > df["High"].shift(1)) &
        (df["Volume"] > df["VOL_AVG_20"])
    )

    df["BEARISH_OB"] = (
        (df["Close"].shift(1) > df["Open"].shift(1)) &
        (df["Close"] < df["Low"].shift(1)) &
        (df["Volume"] > df["VOL_AVG_20"])
    )

    df["ORDER_BLOCK"] = ""

    df.loc[df["BULLISH_OB"], "ORDER_BLOCK"] = "BULLISH_OB"
    df.loc[df["BEARISH_OB"], "ORDER_BLOCK"] = "BEARISH_OB"

    # -------------------------
    # Fair Value Gap
    # -------------------------
    df["FVG_BULLISH"] = df["Low"] > df["High"].shift(2)
    df["FVG_BEARISH"] = df["High"] < df["Low"].shift(2)

    df["FVG_SIGNAL"] = ""

    df.loc[df["FVG_BULLISH"], "FVG_SIGNAL"] = "BULLISH_FVG"
    df.loc[df["FVG_BEARISH"], "FVG_SIGNAL"] = "BEARISH_FVG"

    # -------------------------
    # Liquidity Sweep
    # -------------------------
    df["BUY_LIQUIDITY"] = df["High"] > df["High"].rolling(10).max().shift(1)
    df["SELL_LIQUIDITY"] = df["Low"] < df["Low"].rolling(10).min().shift(1)

    df["LIQUIDITY_ZONE"] = ""

    df.loc[df["BUY_LIQUIDITY"], "LIQUIDITY_ZONE"] = "BUY_SIDE_LIQUIDITY"
    df.loc[df["SELL_LIQUIDITY"], "LIQUIDITY_ZONE"] = "SELL_SIDE_LIQUIDITY"

    # -------------------------
    # AI Probability Engine
    # -------------------------
    df["BUY_SCORE"] = 0
    df["SELL_SCORE"] = 0

    # EMA Trend
    df.loc[df[f"EMA_{ema_fast}"] > df[f"EMA_{ema_slow}"], "BUY_SCORE"] += 2
    df.loc[df[f"EMA_{ema_fast}"] < df[f"EMA_{ema_slow}"], "SELL_SCORE"] += 2

    # # ADX Trend
    # df.loc[(df["+DI"] > df["-DI"]) & (df["ADX"] > 20), "BUY_SCORE"] += 2
    # df.loc[(df["-DI"] > df["+DI"]) & (df["ADX"] > 20), "SELL_SCORE"] += 2

    # # Volume Expansion
    df.loc[(df["Close"] > df["Open"]) & (df["Volume"] > df["VOL_AVG_20"]), "BUY_SCORE"] += 1
    df.loc[(df["Close"] < df["Open"]) & (df["Volume"] > df["VOL_AVG_20"]), "SELL_SCORE"] += 1

    # Breakouts
    df.loc[df["PRICE_BREAKOUT"], "BUY_SCORE"] += 2
    df.loc[df["PRICE_BREAKDOWN"], "SELL_SCORE"] += 2

    # Smart Money
    df.loc[df["SMART_SIGNAL"] == "Hidden Accumulation", "BUY_SCORE"] += 3
    df.loc[df["SMART_SIGNAL"] == "Hidden Distribution", "SELL_SCORE"] += 3

    # Order Block
    df.loc[df["ORDER_BLOCK"] == "BULLISH_OB", "BUY_SCORE"] += 2
    df.loc[df["ORDER_BLOCK"] == "BEARISH_OB", "SELL_SCORE"] += 2

    # Final Probabilities
    df["TOTAL_SCORE"] = df["BUY_SCORE"] + df["SELL_SCORE"]

    df["BUY_PROB"] = (df["BUY_SCORE"] / df["TOTAL_SCORE"]) * 100
    df["SELL_PROB"] = (df["SELL_SCORE"] / df["TOTAL_SCORE"]) * 100


    # -------------------------------------------------
    # 🎯 BUY ZONE / SELL ZONE DETECTION
    # -------------------------------------------------

    # current_price = df["Close"]

    # df["BUY_ZONE"] = np.where(
    #     df["LAST_SWING_LOW"] < current_price,
    #     df["LAST_SWING_LOW"],
    #     np.nan
    # )

    # df["SELL_ZONE"] = np.where(
    #     df["LAST_SWING_HIGH"] > current_price,
    #     df["LAST_SWING_HIGH"],
    #     np.nan
    # )
    latest_price = df["Close"].iloc[-1]

   
    # Different logic for daily timeframe
    if interval in ["1d", "1wk", "1mo"]:

        lookback = 30  # last 30 candles

        df["BUY_ZONE"] = df["Low"].rolling(lookback).min()
        df["SELL_ZONE"] = df["High"].rolling(lookback).max()

    else:

        df["BUY_ZONE"] = df["LAST_SWING_LOW"]
        df["SELL_ZONE"] = df["LAST_SWING_HIGH"]


    buy_zone = df["BUY_ZONE"].iloc[-1]
    sell_zone = df["SELL_ZONE"].iloc[-1]

    # Breakout buy price
    df["BREAKOUT_BUY_PRICE"] = np.where(
        df["PRICE_BREAKOUT"],
        df["LAST_SWING_HIGH"],
        np.nan
    )

    # Breakdown sell price
    df["BREAKDOWN_SELL_PRICE"] = np.where(
        df["PRICE_BREAKDOWN"],
        df["LAST_SWING_LOW"],
        np.nan
    )

    # Institutional buy zone (order block)
    df["INSTITUTIONAL_BUY_ZONE"] = np.where(
        df["ORDER_BLOCK"] == "BULLISH_OB",
        df["Low"].shift(1),
        np.nan
    )

    # Institutional sell zone
    df["INSTITUTIONAL_SELL_ZONE"] = np.where(
        df["ORDER_BLOCK"] == "BEARISH_OB",
        df["High"].shift(1),
        np.nan
    )

    df["AI_SIGNAL"] = np.where(
        df["BUY_PROB"] > 60,
        "AI_STRONG_BUY",
        np.where(df["SELL_PROB"] > 60, "AI_STRONG_SELL", "AI_NEUTRAL")
    )



# -------------------------------
# 2️⃣ Price + Volume breakout / breakdown
# -------------------------------
    df["PV_BREAKOUT"] = (
        df["PRICE_BREAKOUT"] &
        (df["Volume"] >= 1.2 * df["VOL_AVG_20"])
    )

    df["PV_BREAKDOWN"] = (
        df["PRICE_BREAKDOWN"] &
        (df["Volume"] >= 1.2 * df["VOL_AVG_20"])
)
 

    # -------------------------------
    # 3️⃣ Price + STRONG Volume breakout / breakdown
    # -------------------------------
    df["PV_STRONG_BREAKOUT"] = (
        df["PRICE_BREAKOUT"] &
        (df["Volume"] >= 1.5 * df["VOL_AVG_20"]) &
        (df[f"EMA_{ema_fast}"] > df[f"EMA_{ema_slow}"])
    )

    df["PV_STRONG_BREAKDOWN"] = (
        df["PRICE_BREAKDOWN"] &
        (df["Volume"] >= 1.5 * df["VOL_AVG_20"]) &
        (df[f"EMA_{ema_fast}"] < df[f"EMA_{ema_slow}"])
    )

#-- Swing break out

    buffer = 0.001 * df["Close"]   # 0.1% safety buffer

    df["PRICE_SW_HIGH_BREAKOUT"] = (
        (df["Close"] > df["LAST_SWING_HIGH"] + buffer) &
        (df[f"EMA_{ema_fast}"] > df[f"EMA_{ema_slow}"]) &
        (df["Volume"] > 1.2 * df["VOL_AVG_20"]) &
        ((df["Close"] - df["Open"]) > 0.4 * (df["High"] - df["Low"]))
    )

    df["PRICE_SW_LOW_BREAKDOWN"] = (
        (df["Close"] < df["LAST_SWING_LOW"] - buffer) &
        (df[f"EMA_{ema_fast}"] < df[f"EMA_{ema_slow}"]) &
        (df["Volume"] > 1.2 * df["VOL_AVG_20"]) &
        ((df["Open"] - df["Close"]) > 0.4 * (df["High"] - df["Low"]))
    )


#BReakout Retest
    df["BREAKOUT_RETEST_BUY"] = (
        df["PRICE_BREAKOUT"] &
        (df["Low"] <= df["LAST_SWING_HIGH"]) &
        (df["Close"] > df["LAST_SWING_HIGH"])
    )

    df["BREAKOUT_RETEST_SELL"] = (
        df["PRICE_BREAKDOWN"] &
        (df["High"] >= df["LAST_SWING_LOW"]) &
        (df["Close"] < df["LAST_SWING_LOW"])
    )

    df["HA_TREND"] = np.where(
    (df["HA_Close"] > df["HA_Open"]) &
    (df["HA_Low"] == df["HA_Open"]),
    "STRONG_BULL",

    np.where(
        (df["HA_Close"] < df["HA_Open"]) &
        (df["HA_High"] == df["HA_Open"]),
        "STRONG_BEAR",
        "NEUTRAL"
    )
)
    
        # -------------------------------------------------
    # 🔥 EMA CROSS + VOLUME BREAKOUT LOGIC
    # -------------------------------------------------

    # Detect EMA Cross
    df["EMA_CROSS_UP"] = (
        (df[f"EMA_{ema_fast}"] > df[f"EMA_{ema_slow}"]) &
        (df[f"EMA_{ema_fast}"].shift(1) <= df[f"EMA_{ema_slow}"].shift(1))
    )

    df["EMA_CROSS_DOWN"] = (
        (df[f"EMA_{ema_fast}"] < df[f"EMA_{ema_slow}"]) &
        (df[f"EMA_{ema_fast}"].shift(1) >= df[f"EMA_{ema_slow}"].shift(1))
    )

    # Volume Confirmation
    df["VOL_CONFIRM"] = df["Volume"] > 1.2 * df["VOL_AVG_20"]

    # Final Signals
    df["EMA_BREAKOUT_SIGNAL"] = ""

    df.loc[
        df["EMA_CROSS_UP"] &
        (df["Close"] > df[f"EMA_{ema_fast}"]) &
        df["VOL_CONFIRM"] &
        (df["Close"] > df["Open"]),
        "EMA_BREAKOUT_SIGNAL"
    ] = "EMA_BULLISH_BREAKOUT"

    df.loc[
        df["EMA_CROSS_DOWN"] &
        (df["Close"] < df[f"EMA_{ema_fast}"]) &
        df["VOL_CONFIRM"] &
        (df["Close"] < df["Open"]),
        "EMA_BREAKOUT_SIGNAL"
    ] = "EMA_BEARISH_BREAKDOWN"

    

    # Previous Pivto
    df["P_Pivot"]=(df["PREV_HIGH"] + df["PREV_LOW"] + df["PREV_CLOSE"])/3


        # -------------------------------------------------
    # 📦 INSIDE BAR (VOLATILITY CONTRACTION)
    # -------------------------------------------------
    df["INSIDE_BAR"] = (
        (df["High"] < df["High"].shift(1)) &
        (df["Low"] > df["Low"].shift(1))
    )

    df["INSIDE_HIGH"] = df["High"].shift(1)
    df["INSIDE_LOW"]  = df["Low"].shift(1)

    df["INSIDE_BAR_BREAKOUT"] = (
        df["INSIDE_BAR"].shift(1) &
        (df["Close"] > df["INSIDE_HIGH"]) 
        #(df["Volume"] >= 1.2 * df["VOL_AVG_20"]) &
        #(df["EMA_TREND"] == "UPTREND")
    )

    df["INSIDE_BAR_BREAKDOWN"] = (
        df["INSIDE_BAR"].shift(1) &
        (df["Close"] < df["INSIDE_LOW"]) 
        # (df["Volume"] >= 1.2 * df["VOL_AVG_20"]) &
        # (df["EMA_TREND"] == "DOWNTREND")
    )


    df["BREAKOUT_TYPE"] = ""


    df.loc[
    (df["BREAKOUT_TYPE"] == "") & df["INSIDE_BAR_BREAKOUT"],
    "BREAKOUT_TYPE"
    ] = "INSIDE BAR BREAKOUT"

    df.loc[
        (df["BREAKOUT_TYPE"] == "") & df["INSIDE_BAR_BREAKDOWN"],
        "BREAKOUT_TYPE"
    ] = "INSIDE BAR BREAKDOWN"

    # -------------------------------
    # 4️⃣ Final breakout label (strong > weak)


    # -------------------------------

   

    df["MARKET_STRUCTURE"] = ""

    df["HH"] = False
    df["HL"] = False
    df["LH"] = False
    df["LL"] = False

    df.loc[df["HH"], "MARKET_STRUCTURE"] = "HH"
    df.loc[df["HL"], "MARKET_STRUCTURE"] = "HL"
    df.loc[df["LH"], "MARKET_STRUCTURE"] = "LH"
    df.loc[df["LL"], "MARKET_STRUCTURE"] = "LL"
    


    # df.loc[df["INSIDE_BAR_BREAKOUT"], "BREAKOUT_TYPE"]= "INSIDE BAR BREAKOUT"
    # df.loc[df["PRICE_BREAKOUT"], "BREAKOUT_TYPE"] = "PRICE BREAKOUT"
    # df.loc[df["PV_BREAKOUT"], "BREAKOUT_TYPE"] = "PRICE + VOLUME BREAKOUT"
    # df.loc[df["PV_STRONG_BREAKOUT"], "BREAKOUT_TYPE"] = "PRICE + STRONG[1.5%] VOLUME BREAKOUT"
    # df.loc[df["PRICE_SW_HIGH_BREAKOUT"], "BREAKOUT_TYPE"] = "PRICE + SWING_HIGH[1.5%] VOLUME BREAKOUT"
    # df.loc[df["BREAKOUT_RETEST_BUY"], "BREAKOUT_TYPE"] = "BUY-PRICE BREAKOUT + SWING_HIGH_RETEST"

    # df.loc[df["PRICE_BREAKDOWN"], "BREAKOUT_TYPE"] = "PRICE BREAKDOWN"
    # df.loc[df["PV_BREAKDOWN"], "BREAKOUT_TYPE"] = "PRICE + VOLUME BREAKDOWN"
    # df.loc[df["PV_STRONG_BREAKDOWN"], "BREAKOUT_TYPE"] = "PRICE + STRONG[1.5%] VOLUME BREAKDOWN"
    # df.loc[df["PRICE_SW_LOW_BREAKDOWN"], "BREAKOUT_TYPE"] = "PRICE + SWING_LOW[1.5%] VOLUME BREAKDOWN"
    # df.loc[df["BREAKOUT_RETEST_SELL"], "BREAKOUT_TYPE"] = "SELL-PRICE BREAKDOWN + SWING_LOW_RETEST"
    
    #df.loc[df["INSIDE_BAR_BREAKDOWN"], "BREAKOUT_TYPE"]= "INSIDE BAR BREAKDOWN"

    # -------------------------------
# FINAL BREAKOUT LABEL (PRIORITY LOGIC)
# -------------------------------

    df["BREAKOUT_TYPE"] = ""

    # 🔴 Strongest signals
    df.loc[df["PV_STRONG_BREAKOUT"], "BREAKOUT_TYPE"] = "1.PRICE + STRONG VOLUME BREAKOUT"
    df.loc[df["PV_STRONG_BREAKDOWN"], "BREAKOUT_TYPE"] = "1.PRICE + STRONG VOLUME BREAKDOWN"

    # 🟠 Medium signals
    df.loc[(df["BREAKOUT_TYPE"] == "") & df["PV_BREAKOUT"], "BREAKOUT_TYPE"] = "2.PRICE + VOLUME BREAKOUT"
    df.loc[(df["BREAKOUT_TYPE"] == "") & df["PV_BREAKDOWN"], "BREAKOUT_TYPE"] = "2.PRICE + VOLUME BREAKDOWN"

    # 🟡 Swing breakout
    df.loc[(df["BREAKOUT_TYPE"] == "") & df["PRICE_SW_HIGH_BREAKOUT"], "BREAKOUT_TYPE"] = "1.SWING HIGH BREAKOUT"
    df.loc[(df["BREAKOUT_TYPE"] == "") & df["PRICE_SW_LOW_BREAKDOWN"], "BREAKOUT_TYPE"] = "1.SWING LOW BREAKDOWN"

    # 🟢 Basic breakout
    df.loc[(df["BREAKOUT_TYPE"] == "") & df["PRICE_BREAKOUT"], "BREAKOUT_TYPE"] = "3.PRICE BREAKOUT"
    df.loc[(df["BREAKOUT_TYPE"] == "") & df["PRICE_BREAKDOWN"], "BREAKOUT_TYPE"] = "3.PRICE BREAKDOWN"

    # 🔵 Inside bar breakout
    df.loc[(df["BREAKOUT_TYPE"] == "") & df["INSIDE_BAR_BREAKOUT"], "BREAKOUT_TYPE"] = "INSIDE BAR BREAKOUT"
    df.loc[(df["BREAKOUT_TYPE"] == "") & df["INSIDE_BAR_BREAKDOWN"], "BREAKOUT_TYPE"] = "INSIDE BAR BREAKDOWN"

    # 🟣 Retest breakout
    df.loc[(df["BREAKOUT_TYPE"] == "") & df["BREAKOUT_RETEST_BUY"], "BREAKOUT_TYPE"] = "BUY-BREAKOUT RETEST BUY"
    df.loc[(df["BREAKOUT_TYPE"] == "") & df["BREAKOUT_RETEST_SELL"], "BREAKOUT_TYPE"] = "SELL-BREAKOUT RETEST SELL"

    
    level_buffer = 0.001 * df["Close"]   # ~0.1%

    df["SUPPORT_LEVEL"] = df["HA_SW_LOW"]
    df["RESISTANCE_LEVEL"] = df["HA_SW_HIGH"]

    df["SUPPORT_TOUCH"] = (
    (df["Low"] <= df["SUPPORT_LEVEL"] + level_buffer) &
    (df["Low"] >= df["SUPPORT_LEVEL"] - level_buffer)
)
    
    df["RESISTANCE_TOUCH"] = (
    (df["High"] <= df["RESISTANCE_LEVEL"] + level_buffer) &
    (df["High"] >= df["RESISTANCE_LEVEL"] - level_buffer)
)
    
    df["SUPPORT_STRENGTH"] = df["SUPPORT_TOUCH"].rolling(50).sum()
    df["RESISTANCE_STRENGTH"] = df["RESISTANCE_TOUCH"].rolling(50).sum()

    df["VALID_SUPPORT"] = np.where(
    df["SUPPORT_STRENGTH"] >= 2,
    df["SUPPORT_LEVEL"],
    np.nan
)

    df["VALID_RESISTANCE"] = np.where(
        df["RESISTANCE_STRENGTH"] >= 2,
        df["RESISTANCE_LEVEL"],
        np.nan
    )

    df["VALID_SUPPORT"] = df["VALID_SUPPORT"].ffill()
    df["VALID_RESISTANCE"] = df["VALID_RESISTANCE"].ffill()

    BUY_SUPPORT = (
    (df["Low"] < df["VALID_SUPPORT"]) &
    (df["Close"] > df["VALID_SUPPORT"])
)

    SELL_RESISTANCE = (
        (df["High"] > df["VALID_RESISTANCE"]) &
        (df["Close"] < df["VALID_RESISTANCE"])
    )


    df["SUPPORT_INFO"] = np.where(
        df["VALID_SUPPORT"].notna(),
        df["VALID_SUPPORT"].round(2).astype(str)
        + " | S="
        + df["SUPPORT_STRENGTH"].fillna(0).astype(int).astype(str),
        ""
    )

    df["RESISTANCE_INFO"] = np.where(
        df["VALID_RESISTANCE"].notna(),
        df["VALID_RESISTANCE"].round(2).astype(str)
        + " | R="
        + df["RESISTANCE_STRENGTH"].fillna(0).astype(int).astype(str),
        ""
    )

    df["Pivot"] = round((df["High"] + df["Low"] + df["Close"]) / 3,0)
    df["R1"] = round((2 * df["Pivot"]) - df["Low"], 0)
    df["S1"] = round((2 * df["Pivot"]) - df["High"], 0)
    df["R2"] = round(df["Pivot"] + (df["High"] - df["Low"]), 0)
    df["S2"] = round(df["Pivot"] - (df["High"] - df["Low"]), 0)





    def color_breakout(val):
        if isinstance(val, str):
            if "BREAKOUT" in val and "DOWN" not in val:
                return "color: green; font-weight: bold;"
            elif "BREAKDOWN" in val:
                return "color: red; font-weight: bold;"
        return ""
    # -------------------------------------------------
    # DISPLAY LAST 20 RECORDS ONLY
    # -------------------------------------------------

    # -------------------------------------------------
# 📊 PRICE + VOLUME DIRECTION LOGIC
# -------------------------------------------------
    df["PRICE_UP"] = df["Close"] > df["Close"].shift(1)
    df["PRICE_DOWN"] = df["Close"] < df["Close"].shift(1)

    df["HH_LL_SIGNAL"] = np.where(
    df["Close"] > df["High"].shift(1),
    "HH_BREAKOUT",
    np.where(
        df["Close"] < df["Low"].shift(1),
        "LL_BREAKDOWN",
        ""
    )
)

    df = swing_trading_signals_v2(df)

    
    df["Trend_"] = ""

    # FAST HH breakout (Bullish)
    df.loc[
        (df["Close"] > df["High"].shift(1)) &              # HH candle
        (df["PRICE_UP"]) &                                 # momentum
        (df[f"EMA_{ema_fast}"] > df[f"EMA_{ema_slow}"]) &  # EMA trend
        (df["Close"] > df[f"EMA_{ema_fast}"]),             # above fast EMA
        "Trend_"
    ] = "BULLISH"

    # FAST LL breakdown (Bearish)
    df.loc[
        (df["Close"] < df["Low"].shift(1)) &               # LL candle
        (df["PRICE_DOWN"]) &                               # momentum
        (df[f"EMA_{ema_fast}"] < df[f"EMA_{ema_slow}"]) &  # EMA trend
        (df["Close"] < df[f"EMA_{ema_fast}"]),             # below fast EMA
        "Trend_"
    ] = "BEARISH"





    df["VOL_EXPAND"] = df["Volume"] > df["VOL_AVG_20"]

    df["PV_BREAKOUT_SIGNAL"] = ""

    pv_breakout_cond = (
    df["PRICE_UP"] &
    df["VOL_EXPAND"] &
    (df["Close"] > df[f"EMA_{ema_fast}"]) &
    (df[f"EMA_{ema_fast}"] > df[f"EMA_{ema_slow}"])
)
    
    df.loc[
        df["Trend_"] == "BULLISH",
        "TREND"
    ] = "STRONG_BULLISH"

    df.loc[
        df["Trend_"] == "BEARISH",
        "TREND"
    ] = "STRONG_BEARISH"

    df["RANGE"] = df["High"] - df["Low"]
    df["RANGE_AVG_20"] = df["RANGE"].rolling(20).mean()

    # Volatility expansion (strength only)
    df["VOL_EXPAND"] = df["RANGE"] > 1.3 * df["RANGE_AVG_20"]

    # Direction-aware volatility
    df["VOL_EXPAND_UP"] = df["VOL_EXPAND"] & (df["Close"] > df["Open"])
    df["VOL_EXPAND_DOWN"] = df["VOL_EXPAND"] & (df["Close"] < df["Open"])

    df["VOL_SOURCE"] = "RANGE_PROXY"


        # -------------------------------------------------
    # 🔥 ADX (Trend Strength)
    # -------------------------------------------------
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    df["+DM"] = np.where((high - high.shift(1)) > (low.shift(1) - low),
                        np.maximum(high - high.shift(1), 0), 0)

    df["-DM"] = np.where((low.shift(1) - low) > (high - high.shift(1)),
                        np.maximum(low.shift(1) - low, 0), 0)

    df["TR"] = np.maximum(
        high - low,
        np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1)))
    )

    period = 14

    df["ATR_ADX"] = df["TR"].rolling(period).mean()
    df["+DI"] = 100 * (df["+DM"].rolling(period).mean() / df["ATR_ADX"])
    df["-DI"] = 100 * (df["-DM"].rolling(period).mean() / df["ATR_ADX"])

    df["DX"] = 100 * abs(df["+DI"] - df["-DI"]) / (df["+DI"] + df["-DI"])
    df["ADX"] = df["DX"].rolling(period).mean()

    # Trend strength filters
    df["ADX_STRONG"] = df["ADX"] > 20
    df["ADX_VERY_STRONG"] = df["ADX"] > 25

    df["ADX_BULL"] = (df["+DI"] > df["-DI"]) & df["ADX_STRONG"]
    df["ADX_BEAR"] = (df["-DI"] > df["+DI"]) & df["ADX_STRONG"]


    df["ADX_TREND"] = np.where(
        (df["+DI"] > df["-DI"]) & (df["ADX"] > 20),
        "BULL",
        np.where(
            (df["-DI"] > df["+DI"]) & (df["ADX"] > 20),
            "BEAR",
            "NEUTRAL"
        )
    )

    # -------------------------------------------------
# 🎯 SINGLE OPTIONS SIGNAL COLUMN
# -------------------------------------------------
    df["OPTIONS_SIGNAL"] = "NO_TRADE"

    df.loc[
        (df["Trend_"] == "BULLISH") &
        (df["VOL_EXPAND_UP"]),
        "OPTIONS_SIGNAL"
    ] = "BUY_CALL"

    df.loc[
        (df["Trend_"] == "BEARISH") &
        (df["VOL_EXPAND_DOWN"]),
        "OPTIONS_SIGNAL"
    ] = "BUY_PUT"

    df["HH_LL_SIGNAL"]=df["HH_LL_SIGNAL"]+"->" + df["Trend_"]+df["OPTIONS_SIGNAL"]




    df.loc[pv_breakout_cond, "PV_BREAKOUT_SIGNAL"] = "PV_BREAKOUT"

    pv_breakdown_cond = (
    df["PRICE_DOWN"] &
    df["VOL_EXPAND"] &
    (df["Close"] < df[f"EMA_{ema_fast}"]) &
    (df[f"EMA_{ema_fast}"] < df[f"EMA_{ema_slow}"])
)

    df.loc[pv_breakdown_cond, "PV_BREAKOUT_SIGNAL"] = "PV_BREAKDOWN"

    st.subheader(f"📊 Last 20 Intraday Candles ({interval})")

    df["BUY_VOL"] = np.where(df["Close"] > df["Open"], df["Volume"], 0)
    df["SELL_VOL"] = np.where(df["Close"] < df["Open"], df["Volume"], 0)

    # Cumulative (per day)
    df["CUM_BUY_VOL"] = df.groupby("TradeDate")["BUY_VOL"].cumsum()
    df["CUM_SELL_VOL"] = df.groupby("TradeDate")["SELL_VOL"].cumsum()

    df["NET_VOLUME_DELTA"] = df["CUM_BUY_VOL"] - df["CUM_SELL_VOL"]

    latest_delta = df["NET_VOLUME_DELTA"].iloc[-1]

    zone = (
        "🟢 BUYING ZONE" if latest_delta > 0
        else "🔴 SELLING ZONE" if latest_delta < 0
        else "⚪ NEUTRAL"
    )

    st.subheader(f"📊 Volume Zone: {zone}")
    st.write(f"📈 Net Volume Delta: {int(latest_delta)}")


     
   # -------------------------------------------------
    # 📅 Yesterday High / Low (NON-REPAINTING – FIXED)
    # -------------------------------------------------
    daily_levels = (
        df.groupby("TradeDate")
        .agg(
            DAY_HIGH=("High", "max"),
            DAY_LOW=("Low", "min")
        )
    )

    # Shift by DAY (not candle)
    daily_levels["YDAY_HIGH"] = daily_levels["DAY_HIGH"].shift(1)
    daily_levels["YDAY_LOW"]  = daily_levels["DAY_LOW"].shift(1)
    daily_levels["TODAY_HIGH"] = daily_levels["DAY_HIGH"]
    daily_levels["TODAY_LOW"]  = daily_levels["DAY_LOW"]

    # -------------------------------------------------
    # 📐 DAILY CPR CALCULATION (PREVIOUS DAY)
    # -------------------------------------------------
    daily_levels["PIVOT"] = (
        daily_levels["DAY_HIGH"] +
        daily_levels["DAY_LOW"] +
        daily_levels["DAY_HIGH"]  # close proxy safe
    ) / 3

    daily_levels["BC"] = (
        daily_levels["DAY_HIGH"] + daily_levels["DAY_LOW"]
    ) / 2

    daily_levels["TC"] = (
        (daily_levels["PIVOT"] - daily_levels["BC"]) +
        daily_levels["PIVOT"]
    )

    # CPR Width
    daily_levels["CPR_WIDTH"] = daily_levels["TC"] - daily_levels["BC"]

    # Shift → use ONLY completed day CPR
    daily_levels["YDAY_PIVOT"] = daily_levels["PIVOT"].shift(1)
    daily_levels["YDAY_BC"] = daily_levels["BC"].shift(1)
    daily_levels["YDAY_TC"] = daily_levels["TC"].shift(1)
    daily_levels["YDAY_CPR_WIDTH"] = daily_levels["CPR_WIDTH"].shift(1)

    

        # Step 2: Rolling last 5 COMPLETED days
    daily_levels["LAST_5D_HIGH"] = (
        daily_levels["DAY_HIGH"]
        .shift(1)                      # exclude current day
        .rolling(5)
        .max()
    )

    daily_levels["LAST_5D_LOW"] = (
    daily_levels["DAY_LOW"]
    .shift(1)                      # exclude current day
    .rolling(5)
    .min()
   )
    

        # Step 3: Map back to intraday candles
    df["LAST_5D_HIGH"] = df["TradeDate"].map(daily_levels["LAST_5D_HIGH"])
    df["LAST_5D_LOW"]  = df["TradeDate"].map(daily_levels["LAST_5D_LOW"])

    df["CPR_PIVOT"] = df["TradeDate"].map(daily_levels["YDAY_PIVOT"])
    df["CPR_BC"] = df["TradeDate"].map(daily_levels["YDAY_BC"])
    df["CPR_TC"] = df["TradeDate"].map(daily_levels["YDAY_TC"])
    df["CPR_WIDTH"] = df["TradeDate"].map(daily_levels["YDAY_CPR_WIDTH"])

    # Map back to intraday candles
    df["YDAY_HIGH"] = df["TradeDate"].map(daily_levels["YDAY_HIGH"])
    df["YDAY_LOW"]  = df["TradeDate"].map(daily_levels["YDAY_LOW"])

    df["TODAY_HIGH"] = df["TradeDate"].map(daily_levels["TODAY_HIGH"])
    df["TODAY_LOW"]  = df["TradeDate"].map(daily_levels["TODAY_LOW"])

    # Rolling average of CPR width (last 10 completed days)
    daily_levels["CPR_AVG_10"] = (
        daily_levels["CPR_WIDTH"]
        .shift(1)
        .rolling(10)
        .mean()
    )

    df["CPR_AVG_10"] = df["TradeDate"].map(daily_levels["CPR_AVG_10"])

    df["NARROW_CPR"] = df["CPR_WIDTH"] < df["CPR_AVG_10"]

    df["CPR_SIGNAL"] = "NO_CPR_TRADE"

    df.loc[
        (df["Close"] > df["CPR_TC"]) &
        df["NARROW_CPR"] &
        (df["ADX"] > 20) &
        (df[f"EMA_{ema_fast}"] > df[f"EMA_{ema_slow}"]),
        "CPR_SIGNAL"
    ] = "CPR_BUY"

    df.loc[
        (df["Close"] < df["CPR_BC"]) &
        df["NARROW_CPR"] &
        (df["ADX"] > 20) &
        (df[f"EMA_{ema_fast}"] < df[f"EMA_{ema_slow}"]),
        "CPR_SIGNAL"
    ] = "CPR_SELL"

    strong_levels = detect_strong_levels(df)

    support, resistance = detect_strong_levels(df)

    st.write("Strong Support:", support)
    st.write("Strong Resistance:", resistance)



    latest_date = df["TradeDate"].max()

    latest_orb = df[df["TradeDate"] == latest_date][["ORB_HIGH", "ORB_LOW"]].dropna()

    latest_row = df[df["TradeDate"] == latest_date].iloc[-1]

    yday_low  = latest_row["YDAY_LOW"]
    yday_high = latest_row["YDAY_HIGH"]

    
    today_low  = latest_row["TODAY_LOW"]
    today_high = latest_row["TODAY_HIGH"]

    
    Last_5D_High  = latest_row["LAST_5D_HIGH"]
    Last_5D_low = latest_row["LAST_5D_LOW"]

    latest = df.iloc[-1]

    st.subheader("🤖 AI Market Bias")

    col1, col2, col3 = st.columns(3)

    col1.metric("BUY Probability", f"{latest['BUY_PROB']:.1f}%")
    col2.metric("SELL Probability", f"{latest['SELL_PROB']:.1f}%")
    col3.metric("AI Signal", latest["AI_SIGNAL"])

    st.subheader("🎯 Institutional Price Zones")

    col1, col2 = st.columns(2)

    col1.metric(
        "Buy Zone (Support)",
        round(latest["BUY_ZONE"], 2) if pd.notna(latest["BUY_ZONE"]) else "N/A"
    )

    col2.metric(
        "Sell Zone (Resistance)",
        round(latest["SELL_ZONE"], 2) if pd.notna(latest["SELL_ZONE"]) else "N/A"
    )

   
   # breaoout buy
    last_breakout = df["BREAKOUT_BUY_PRICE"].dropna().iloc[-1] if not df["BREAKOUT_BUY_PRICE"].dropna().empty else None
    last_breakdown = df["BREAKDOWN_SELL_PRICE"].dropna().iloc[-1] if not df["BREAKDOWN_SELL_PRICE"].dropna().empty else None

    col1.metric(
        "Breakout Buy Price",
        round(last_breakout,2) if last_breakout else "Wait"
    )

    col2.metric(
        "Breakdown Sell Price",
        round(last_breakdown,2) if last_breakdown else "Wait"
    )



    if not latest_orb.empty:
        orb_high_val = latest_orb["ORB_HIGH"].iloc[0]
        orb_low_val  = latest_orb["ORB_LOW"].iloc[0]

        st.markdown("### 🔔 Latest ORB Levels (9:15 – 9:30 IST)")
        st.write(f"📅 **Date:** {latest_date}")
        # st.write(f"🟢 **ORB HIGH:** `{orb_high_val}`| f"🟢 **TODAY HIGH:** `{today_high}` 🟧 **YDAY HIGH:** `{round(yday_high, 2)}`| 🟧 **LAST 5day HIGH:** `{round(Last_5D_High, 2)}`")
        # st.write(
        # f"🔴 **ORB LOW:** `{orb_low_val}` | 🟧 **YDAY LOW:** `{round(yday_low, 2)}` | 🟧 **Last 5D Low:** `{round(Last_5D_low, 2)}`"
        # )
        # st.write(f"🔴 **ORB LOW:** `{orb_low_val}`")

        st.write(
        f"🟢 **ORB HIGH:** `{orb_high_val}` | "
        f"🟡 **TODAY HIGH:** `{round(today_high, 2)}` | "
        f"🟧 **YDAY HIGH:** `{round(yday_high, 2)}` | "
        f"🟦 **LAST 5D HIGH:** `{round(Last_5D_High, 2)}`"
    )
        
        st.write(
        f"🔴 **ORB LOW:** `{orb_low_val}` | "
        f"🟡 **TODAY LOW:** `{round(today_low, 2)}` | "
        f"🟧 **YDAY LOW:** `{round(yday_low, 2)}` | "
        f"🟦 **LAST 5D LOW:** `{round(Last_5D_low, 2)}`"
    )
    else:
        st.warning("⚠️ ORB levels not available for the latest trading day.")

    df[f"EMA_{ema_fast}_D"] = df[f"EMA_{ema_fast}"].round(2)    

    st.dataframe(
        df[[
            #"DateTime",
            "Open", "Close","High", "Low","BREAKOUT_TYPE","STRONG_BODY","SIGNAL","HH_LL_SIGNAL","S1","R1","S2","R2","HA_TREND","ADX_TREND","SWING_HIGH","SWING_LOW","buy_signal","BUYER_PRICE_LEVEL","SELLER_PRICE_LEVEL",
            #"OPEN_EQ_LOW", "OPEN_EQ_HIGH", "OPEN_TYPE","OPEN_VALIDATION",
            "Volume", "VOL_STRONG","POC","SIGNAL_VOLP","VAL","VAH",
            "HA_Color",
            "HA_SW_LOW", "HA_SW_HIGH","EMA_7",
             f"EMA_{ema_fast}",
             f"EMA_{ema_slow}",
            "CPR_SIGNAL", "SL", "TARGET","OPTIONS_SIGNAL"
        ]].tail(10)
    .style
    .map(color_breakout, subset=["BREAKOUT_TYPE"])
    .apply(highlight_options_signal, axis=1)
    .apply(highlight_Signal, axis=1),
    
    
    use_container_width=True
)


  




    st.success("Strategy executed successfully ✅")


    # -------------------------------------------------
    # 📊 Candlestick Chart with EMA + Breakout + Volume
    # -------------------------------------------------
    if interval != "1d":
        st.subheader("📉 Price Action: EMA + Breakout + Volume (Intraday)")
    else:
        st.subheader("📉 Price Action: EMA + Breakout + Volume (Daily Swing)")

    # Ensure chart_df is never empty
    chart_df = df.tail(150) if len(df) > 150 else df.copy()

    fig = go.Figure()

    # -------------------------
    # Candlestick
    # -------------------------
    fig.add_trace(go.Candlestick(
        x=chart_df["DateTime"],
        open=chart_df["Open"],
        high=chart_df["High"],
        low=chart_df["Low"],
        close=chart_df["Close"],
        name="Price",
        increasing=dict(
            line=dict(width=1, color="#00b050"),
            fillcolor="#00b050"
        ),
        decreasing=dict(
            line=dict(width=1, color="#ff4d4d"),
            fillcolor="#ff4d4d"
        ),
    ))

    # -------------------------
    # EMA Fast
    # -------------------------
    fig.add_trace(go.Scatter(
        x=chart_df["DateTime"],
        y=chart_df[f"EMA_{ema_fast}"],
        mode="lines",
        line=dict(width=2),
        name=f"EMA {ema_fast}"
    ))

    # -------------------------
    # EMA Slow
    # -------------------------
    fig.add_trace(go.Scatter(
        x=chart_df["DateTime"],
        y=chart_df[f"EMA_{ema_slow}"],
        mode="lines",
        line=dict(width=2),
        name=f"EMA {ema_slow}"
    ))

    # -------------------------
    # PV Breakout markers
    # -------------------------
    pv_bo_df = chart_df[chart_df["PV_BREAKOUT_SIGNAL"] == "PV_BREAKOUT"]
    pv_bd_df = chart_df[chart_df["PV_BREAKOUT_SIGNAL"] == "PV_BREAKDOWN"]

    fig.add_trace(go.Scatter(
        x=pv_bo_df["DateTime"],
        y=pv_bo_df["High"],
        mode="markers",
        marker=dict(symbol="triangle-up", size=14, color="lime"),
        name="PV Breakout"
    ))

    fig.add_trace(go.Scatter(
        x=pv_bd_df["DateTime"],
        y=pv_bd_df["Low"],
        mode="markers",
        marker=dict(symbol="triangle-down", size=14, color="red"),
        name="PV Breakdown"
    ))

    # -------------------------
    # Breakout / Breakdown markers
    # -------------------------
    breakout_df = chart_df[chart_df["BREAKOUT_TYPE"].str.contains("BREAKOUT", na=False)]
    breakdown_df = chart_df[chart_df["BREAKOUT_TYPE"].str.contains("BREAKDOWN", na=False)]

    fig.add_trace(go.Scatter(
        x=breakout_df["DateTime"],
        y=breakout_df["High"],
        mode="markers",
        marker=dict(symbol="triangle-up", size=12, color="green"),
        name="Breakout"
    ))

    fig.add_trace(go.Scatter(
        x=breakdown_df["DateTime"],
        y=breakdown_df["Low"],
        mode="markers",
        marker=dict(symbol="triangle-down", size=12, color="red"),
        name="Breakdown"
    ))

    # -------------------------
    # Volume
    # -------------------------
    fig.add_trace(go.Bar(
        x=chart_df["DateTime"],
        y=chart_df["Volume"],
        name="Volume",
        yaxis="y2",
        opacity=0.25
    ))

# -------------------------
# Strong Support / Resistance Zones
# -------------------------

    strong_levels = detect_clean_levels(chart_df)

    for lvl, strength in strong_levels:

        fig.add_shape(
            type="line",
            x0=chart_df["DateTime"].iloc[0],
            x1=chart_df["DateTime"].iloc[-1],
            y0=lvl,
            y1=lvl,
            line=dict(
                color="blue" if lvl < chart_df["Close"].iloc[-1] else "red",
                width=2
            )


     )
      
            # Add label
        fig.add_annotation(
            x=chart_df["DateTime"].iloc[-1],
            y=lvl,
            text=f"{round(lvl,2)} | S={strength}",
            showarrow=False,
            xanchor="left"
        )
        # -------------------------
    # Layout
    # -------------------------
    fig.update_layout(
        title=f"{symbol} | {interval} | Candlestick + EMA + Volume",
        height=850,
        xaxis_rangeslider_visible=False,
        yaxis=dict(title="Price", domain=[0.35, 1]),
        yaxis2=dict(title="Volume", domain=[0, 0.28], showgrid=False),
        legend=dict(orientation="h", y=1.05),
    )


# -------------------------
# EMA Breakout markers
# -------------------------

    ema_bull_df = chart_df[chart_df["EMA_BREAKOUT_SIGNAL"] == "EMA_BULLISH_BREAKOUT"]
    ema_bear_df = chart_df[chart_df["EMA_BREAKOUT_SIGNAL"] == "EMA_BEARISH_BREAKDOWN"]

    fig.add_trace(go.Scatter(
        x=ema_bull_df["DateTime"],
        y=ema_bull_df["High"],
        mode="markers",
        marker=dict(symbol="star", size=16, color="gold"),
        name="EMA Bullish Breakout"
    ))

    fig.add_trace(go.Scatter(
        x=ema_bear_df["DateTime"],
        y=ema_bear_df["Low"],
        mode="markers",
        marker=dict(symbol="star-triangle-down", size=16, color="purple"),
        name="EMA Bearish Breakdown"
    ))



    



    # -------------------------
    # SAFE rangebreaks
    # -------------------------
    if interval != "1d":
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),               # weekends
                dict(bounds=[15.5, 9.25], pattern="hour")  # NSE closed hours
            ]
        )
    else:
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"])                # daily → weekends only
            ]
        )

    # -------------------------
    # Render chart
    # -------------------------
    st.plotly_chart(fig, use_container_width=True)


    #-----------------------------------------------------
#FOOTER
    # -----------------------------------------------------
    st.markdown("---")
    st.caption("⚠️ This platform is for educational and research purposes only.It does not constitute investment advice or a recommendation")
    st.caption("⚠️Market investments are subject to risk.")
