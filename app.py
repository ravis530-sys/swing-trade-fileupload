import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
import requests

# --- CONFIG ---
LOG_FILE = "activity.log"
log_container = None  # will be set inside main()

PAGE_SIZE_DEFAULT = 20

def clear_logs():
    """Clear the activity log file."""
    open(LOG_FILE, "w").close()

def write_log(message: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"

    # Write to file
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

    # Update session logs
    if "logs" not in st.session_state:
        st.session_state["logs"] = []
    st.session_state["logs"].append(line)

    # Limit logs in memory
    if len(st.session_state["logs"]) > 500:
        st.session_state["logs"] = st.session_state["logs"][-500:]

# --- Data Loaders ---
@st.cache_data
def load_logs():
    try:
        with open(LOG_FILE, "r") as f:
            return f.read().splitlines()
    except FileNotFoundError:
        return ["No logs available."]

# --- Data Fetcher ---
def fetch_hybrid(symbol, exchange, days=180):
    """Try Yahoo Finance first, then NSE API fallback for Volume."""
    end = datetime.datetime.today()
    start = end - datetime.timedelta(days=days)

    # --- Yahoo Finance fetch ---
    try:
        if exchange == "NSE":
            ticker = f"{symbol}.NS"
        elif exchange == "BSE":
            ticker = f"{symbol}.BO"
        else:
            ticker = symbol

        df = yf.download(ticker, start=start, end=end, progress=False)
        
        # If Yahoo gave usable data with Volume, return it
        return df, "Yahoo Finance"
    except Exception:
        return pd.DataFrame(), "Yahoo Finance (failed)"

def fetch_from_nse(symbol, start, end):
    """Fetch OHLCV from NSE with proper headers & cookies."""
    url = (f"https://www.nseindia.com/api/historical/cm/equity?"
           f"symbol={symbol}&series=[%22EQ%22]"
           f"&from={start.strftime('%d-%m-%Y')}"
           f"&to={end.strftime('%d-%m-%Y')}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": "https://www.nseindia.com",
        "Host": "www.nseindia.com"
    }

    session = requests.Session()
    # 🔑 Warm up session: grab cookies
    session.get("https://www.nseindia.com", headers=headers, timeout=5)

    resp = session.get(url, headers=headers, timeout=10)
    if resp.status_code != 200:
        print("NSE error:", resp.status_code, resp.text[:200])
        return pd.DataFrame()

    data = resp.json().get("data", [])
    df = pd.DataFrame(data)
    if df.empty:
        return df

    df = df.rename(columns={
        "CH_TIMESTAMP": "Date",
        "CH_OPENING_PRICE": "Open",
        "CH_TRADE_HIGH_PRICE": "High",
        "CH_TRADE_LOW_PRICE": "Low",
        "CH_CLOSING_PRICE": "Close",
        "CH_TOT_TRADED_QTY": "Volume"
    })
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    return df

# --- Indicators (RSI, MACD, Volume filter) ---
def compute_indicators(df, symbol=None, close_col="Close"):
    # Volume handling
    if "Volume" in df.columns:
        vol = df["Volume"]
    elif symbol:
        col_name = f"Volume_{symbol}.NS"
        if col_name in df.columns:
            vol = df[col_name]
        else:
            vol = pd.Series(0, index=df.index)
    else:
        vol = pd.Series(0, index=df.index)

    df["Vol20"] = vol.rolling(20, min_periods=1).mean()
    df["VolFilter"] = df["Vol20"] > 500_000

    # RSI
    delta = df[close_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df[close_col].ewm(span=12, adjust=False).mean()
    ema26 = df[close_col].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    return df

# --- Analyze Symbol (backend only) ---
def analyze_symbol(symbol, exchange="NSE"):
    df, source = fetch_hybrid(symbol, exchange, days=180)
    if df.empty:
        return None

    #if isinstance(df.columns, pd.MultiIndex):
     #   df.columns = ["_".join([str(c) for c in col if c]) for col in df.columns]

    close_col = next((c for c in df.columns if "Close" in c or "Adj Close" in c or "close" in c), None)
    if not close_col:
        return None

    if isinstance(df[close_col], pd.DataFrame):
        df[close_col] = df[close_col].iloc[:, 0]
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")

    # Initialize indicators based on session state, defaulting to True if not set
    use_ema20 = st.session_state.get("use_ema20", True)
    use_ema50 = st.session_state.get("use_ema50", True)
    use_ema200 = st.session_state.get("use_ema200", True)
    use_rsi = st.session_state.get("use_rsi", True)
    use_macd = st.session_state.get("use_macd", True)
    use_volfilter = st.session_state.get("use_volfilter", True)
    breakout_period = st.session_state.get("breakout_period", "None")

    # Indicators
    if use_ema20:
        df['EMA20'] = df[close_col].ewm(span=20).mean()
    if use_ema50:
        df['EMA50'] = df[close_col].ewm(span=50).mean()
    if use_ema200:
        df['EMA200'] = df[close_col].ewm(span=200).mean()

    if use_rsi or use_macd or use_volfilter:
        df = compute_indicators(df, symbol=symbol, close_col=close_col)

    # Breakout Analysis
    df["Breakout"] = False
    if breakout_period == "52-Day Breakout":
        df['High_52_Week'] = df[close_col].rolling(window=52*5, min_periods=1).max() # 52 weeks * 5 trading days
        df["Breakout"] = (df[close_col].iloc[-1] >= df['High_52_Week'].iloc[-1])
    elif breakout_period == "120-Day Breakout":
        df['High_120_Day'] = df[close_col].rolling(window=120, min_periods=1).max()
        df["Breakout"] = (df[close_col].iloc[-1] >= df['High_120_Day'].iloc[-1])

    # Signals with volume filter
    buy_conditions = []
    sell_conditions = []

    if use_ema20 and use_ema200:
        buy_conditions.append(df["EMA20"] > df["EMA200"])
        sell_conditions.append(df["EMA20"] < df["EMA200"])
    if use_ema50 and use_ema200:
        buy_conditions.append(df["EMA50"] > df["EMA200"])
        sell_conditions.append(df["EMA50"] < df["EMA200"])
    if use_rsi:
        buy_conditions.append(df["RSI"] < 30)
        sell_conditions.append(df["RSI"] > 70)
    if use_macd:
        buy_conditions.append(df["MACD"] > df["Signal"])
        sell_conditions.append(df["MACD"] < df["Signal"])
    if use_volfilter:
        buy_conditions.append(df["VolFilter"])
        sell_conditions.append(df["VolFilter"])
    
    if df["Breakout"].iloc[-1] and breakout_period != "None":
        buy_conditions.append(df["Breakout"])

    # Combine conditions
    if buy_conditions:
        df["Buy"] = pd.concat(buy_conditions, axis=1).all(axis=1)
    else:
        df["Buy"] = False # No buy signals if no indicators are selected

    if sell_conditions:
        df["Sell"] = pd.concat(sell_conditions, axis=1).all(axis=1)
    else:
        df["Sell"] = False # No sell signals if no indicators are selected

    latest_price = df[close_col].iloc[-1]
    recent_window = df.tail(20)
    support = round(recent_window[close_col].min(), 2)
    resistance = round(recent_window[close_col].max(), 2)

    # logs
    log_message_parts = [f"{symbol}: "]
    if use_ema20: log_message_parts.append(f"EMA20={df['EMA20'].iloc[-1]:.2f}")
    if use_ema50: log_message_parts.append(f"EMA50={df['EMA50'].iloc[-1]:.2f}")
    if use_ema200: log_message_parts.append(f"EMA200={df['EMA200'].iloc[-1]:.2f}")
    if use_rsi: log_message_parts.append(f"RSI={df['RSI'].iloc[-1]:.2f}")
    if use_macd: log_message_parts.append(f"MACD={df['MACD'].iloc[-1]:.2f}, Signal={df['Signal'].iloc[-1]:.2f}")
    if use_volfilter: log_message_parts.append(f"VolFilter={df['VolFilter'].iloc[-1]}")
    if breakout_period != "None": log_message_parts.append(f"{breakout_period}={df['Breakout'].iloc[-1]}")
    log_message_parts.append(f"Buy={df['Buy'].iloc[-1]}, Sell={df['Sell'].iloc[-1]}")
    write_log(", ".join(log_message_parts))

    # Auto recommendation
    rec, target, sl = "HOLD", None, None
    if df["Buy"].iloc[-1]:
        rec, target, sl = "BUY", round(latest_price * 1.05, 2), support
    elif df["Sell"].iloc[-1]:
        rec, target, sl = "SELL", round(latest_price * 0.95, 2), resistance

    # Apply analysis mode
    #if analysis_mode == "Auto (Signals)":
     #   recommendation, target, stop_loss = auto_rec, auto_target, auto_sl
    #elif analysis_mode == "BUY":
     #   recommendation, target, stop_loss = "BUY", round(latest_price * 1.05, 2), support
    #elif analysis_mode == "SELL":
     #   recommendation, target, stop_loss = "SELL", round(latest_price * 0.95, 2), resistance
    #else:
     #   recommendation, target, stop_loss = "HOLD", None, None

    return {
        "Symbol": symbol,
        "Recommendation": rec,
        "Entry Price": round(latest_price, 2),
        "Target Price": target,
        "Support": support,
        "Resistance": resistance,
        "Stop Loss": sl
    }

# --- Plot Symbol (charts & stats) ---
def plot_symbol(symbol, exchange="NSE", last_n=180):
    df, source = fetch_hybrid(symbol, exchange, days=last_n)
    if df.empty:
        st.warning(f"No price data for {symbol} ({exchange})")
        return

    #if isinstance(df.columns, pd.MultiIndex):
     #   df.columns = ["_".join([str(c) for c in col if c]) for col in df.columns]

    close_col = next((c for c in df.columns if "Close" in c or "Adj Close" in c or "close" in c), None)
    if not close_col:
        st.error("No Close column found")
        return

    if isinstance(df[close_col], pd.DataFrame):
        df[close_col] = df[close_col].iloc[:, 0]
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")

    # Initialize indicators based on session state, defaulting to True if not set
    use_ema20 = st.session_state.get("use_ema20", True)
    use_ema50 = st.session_state.get("use_ema50", True)
    use_ema200 = st.session_state.get("use_ema200", True)
    use_rsi = st.session_state.get("use_rsi", True)
    use_macd = st.session_state.get("use_macd", True)
    use_volfilter = st.session_state.get("use_volfilter", True)
    breakout_period = st.session_state.get("breakout_period", "None")

    # Indicators
    if use_ema20:
        df['EMA20'] = df[close_col].ewm(span=20).mean()
    if use_ema50:
        df['EMA50'] = df[close_col].ewm(span=50).mean()
    if use_ema200:
        df['EMA200'] = df[close_col].ewm(span=200).mean()

    if use_rsi or use_macd or use_volfilter:
        df = compute_indicators(df, symbol=symbol, close_col=close_col)

    # Breakout Analysis
    df["Breakout"] = False
    if breakout_period == "52-Day Breakout":
        df['High_52_Week'] = df[close_col].rolling(window=52*5, min_periods=1).max() # 52 weeks * 5 trading days
        df["Breakout"] = (df[close_col].iloc[-1] >= df['High_52_Week'].iloc[-1])
    elif breakout_period == "120-Day Breakout":
        df['High_120_Day'] = df[close_col].rolling(window=120, min_periods=1).max()
        df["Breakout"] = (df[close_col].iloc[-1] >= df['High_120_Day'].iloc[-1])

    # Signals
    buy_conditions = []
    sell_conditions = []

    if use_ema20 and use_ema200:
        buy_conditions.append(df["EMA20"] > df["EMA200"])
        sell_conditions.append(df["EMA20"] < df["EMA200"])
    if use_ema50 and use_ema200:
        buy_conditions.append(df["EMA50"] > df["EMA200"])
        sell_conditions.append(df["EMA50"] < df["EMA200"])
    if use_rsi:
        buy_conditions.append(df["RSI"] < 30)
        sell_conditions.append(df["RSI"] > 70)
    if use_macd:
        buy_conditions.append(df["MACD"] > df["Signal"])
        sell_conditions.append(df["MACD"] < df["Signal"])
    if use_volfilter:
        buy_conditions.append(df["VolFilter"])
        sell_conditions.append(df["VolFilter"])
    
    if df["Breakout"].iloc[-1] and breakout_period != "None":
        buy_conditions.append(df["Breakout"])

    # Combine conditions
    if buy_conditions:
        df["Buy"] = pd.concat(buy_conditions, axis=1).all(axis=1)
    else:
        df["Buy"] = False # No buy signals if no indicators are selected

    if sell_conditions:
        df["Sell"] = pd.concat(sell_conditions, axis=1).all(axis=1)
    else:
        df["Sell"] = False # No sell signals if no indicators are selected

    # --- Price + EMA + Signals ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df[close_col], label="Close", color="blue")
    if use_ema20:
        ax.plot(df.index, df['EMA20'], "--", label="EMA20")
    if use_ema50:
        ax.plot(df.index, df['EMA50'], "--", label="EMA50")
    if use_ema200:
        ax.plot(df.index, df['EMA200'], "--", label="EMA200")
    
    if breakout_period == "52-Day Breakout" and 'High_52_Week' in df.columns:
        ax.plot(df.index, df['High_52_Week'], ":", label="52-Week High", color="orange")
    elif breakout_period == "120-Day Breakout" and 'High_120_Day' in df.columns:
        ax.plot(df.index, df['High_120_Day'], ":", label="120-Day High", color="purple")

    if "Buy" in df.columns:
        ax.scatter(df.index[df["Buy"]], df[close_col][df["Buy"]], marker="^", color="green", s=100, label="Buy")
    if "Sell" in df.columns:
        ax.scatter(df.index[df["Sell"]], df[close_col][df["Sell"]], marker="v", color="red", s=100, label="Sell")
    ax.legend()
    ax.set_title(f"{symbol} Price & Signals")
    st.pyplot(fig)

    # --- RSI Chart ---
    if use_rsi:
        fig_rsi, ax_rsi = plt.subplots(figsize=(10, 2))
        ax_rsi.plot(df.index, df["RSI"], color="purple")
        ax_rsi.axhline(70, color="red", linestyle="--", alpha=0.5)
        ax_rsi.axhline(30, color="green", linestyle="--", alpha=0.5)
        ax_rsi.set_title("RSI (14)")
        st.pyplot(fig_rsi)

    # --- MACD Chart ---
    if use_macd:
        fig_macd, ax_macd = plt.subplots(figsize=(10, 2))
        ax_macd.plot(df.index, df["MACD"], label="MACD", color="blue")
        ax_macd.plot(df.index, df["Signal"], label="Signal", color="orange")
        ax_macd.axhline(0, color="black", linewidth=1)
        ax_macd.legend()
        ax_macd.set_title("MACD (12,26,9)")
        st.pyplot(fig_macd)

    # --- Backtest Trades ---
    capital = 100000
    position = 0
    trades = []
    equity_curve = [capital]

    for i in range(1, len(df)):
        buy_signal = bool(df["Buy"].iloc[i])
        sell_signal = bool(df["Sell"].iloc[i])
        price = df[close_col].iloc[i]

        # Buy condition
        if buy_signal and position == 0:
            entry = price
            position = capital / entry
            capital = 0
            trades.append(("BUY", df.index[i], entry))

            # Sell condition
        elif sell_signal and position > 0:
            exit_price = price
            capital = position * exit_price
            position = 0
            trades.append(("SELL", df.index[i], exit_price))

        equity_curve.append(capital + position * price)


    if trades:
        df_trades = pd.DataFrame(trades, columns=["Type", "Date", "Price"])
        
        # Calculate Profit% for each trade
        # Assuming trades are in pairs (BUY, SELL)
        processed_trades = []
        buy_price = None
        buy_date = None
        for i, row in df_trades.iterrows():
            if row["Type"] == "BUY":
                buy_price = row["Price"]
                buy_date = row["Date"]
            elif row["Type"] == "SELL" and buy_price is not None:
                sell_price = row["Price"]
                profit_pct = ((sell_price - buy_price) / buy_price) * 100
                processed_trades.append({
                    "Buy Date": buy_date,
                    "Buy Price": buy_price,
                    "Sell Date": row["Date"],
                    "Sell Price": sell_price,
                    "Profit%": profit_pct
                })
                buy_price = None # Reset for next trade pair

        if processed_trades:
            df_trades_summary = pd.DataFrame(processed_trades)
            st.subheader("📊 Trade History")
            st.dataframe(df_trades_summary)

            # Stats
            total_trades = len(df_trades_summary)
            wins = (df_trades_summary["Profit%"] > 0).sum()
            win_rate = wins / total_trades if total_trades else 0
            avg_profit = df_trades_summary["Profit%"].mean()

            stats = {
                "Total Trades": total_trades,
                "Win Rate": f"{win_rate:.2%}",
                "Avg Profit %": round(avg_profit, 2),
                "Max Profit %": df_trades_summary["Profit%"].max(),
                "Max Loss %": df_trades_summary["Profit%"].min()
            }

            st.subheader("📈 Backtest Performance")
            st.json(stats)

            # Equity Curve
            start_capital = st.sidebar.number_input("Starting Capital (₹)", min_value=10000, value=1000000, step=50000)
            risk_fraction = st.sidebar.slider("Fraction of capital per trade", 0.1, 1.0, 1.0)
            risk_per_trade_pct = st.sidebar.slider("Risk % per trade (for ruin calc)", 0.5, 5.0, 2.0)

            portfolio = [start_capital]
            for _, trade in df_trades_summary.iterrows():
                last_capital = portfolio[-1]
                trade_capital = last_capital * risk_fraction
                profit = trade_capital * (trade["Profit%"] / 100.0)
                portfolio.append(last_capital + profit)

            equity_curve = pd.Series(portfolio[1:], index=df_trades_summary["Sell Date"])

            fig2, ax = plt.subplots(figsize=(10, 4))
            ax.plot(equity_curve.index, equity_curve.values, marker="o", color="blue")
            ax.set_title("💰 Capital-Based Equity Curve")
            ax.set_xlabel("Trade Exit Date")
            ax.set_ylabel("Portfolio Value (₹)")
            plt.xticks(rotation=45)
            st.pyplot(fig2)

            # Drawdown
            running_max = equity_curve.cummax()
            drawdowns = (equity_curve - running_max) / running_max * 100

            fig3, ax = plt.subplots(figsize=(10, 3))
            ax.fill_between(drawdowns.index, drawdowns.values, 0, color="red", alpha=0.3)
            ax.set_title("📉 Drawdown Curve")
            ax.set_ylabel("Drawdown %")
            plt.xticks(rotation=45)
            st.pyplot(fig3)

            # Risk of Ruin
            avg_win = df_trades_summary[df_trades_summary["Profit%"] > 0]["Profit%"].mean()
            avg_loss = abs(df_trades_summary[df_trades_summary["Profit%"] <= 0]["Profit%"].mean())
            win_rate = (df_trades_summary["Profit%"] > 0).mean()

            if pd.notna(avg_win) and pd.notna(avg_loss) and avg_loss > 0:
                reward_risk = avg_win / avg_loss
                capital_units = int(start_capital / (start_capital * (risk_per_trade_pct / 100)))
                risk_of_ruin = ((1 - (win_rate * reward_risk)) / (1 + (win_rate * reward_risk))) ** capital_units
                st.subheader("⚠️ Risk of Ruin Estimate")
                st.write(f"Win Rate: {win_rate:.2%}, Avg Win/Loss Ratio: {reward_risk:.2f}")
                st.write(f"Estimated Risk of Ruin: {risk_of_ruin:.6f}")
            else:
                st.info("Not enough trades to estimate Risk of Ruin.")
        else:
            st.info("No trades executed based on the selected indicators.")

    # Final Recommendation
    rec = analyze_symbol(symbol, exchange=exchange)
    if rec:
        st.subheader("📋 Trade Recommendation")
        st.table(pd.DataFrame([rec]))

def normalize_symbol_column(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common column names to 'Symbol' but avoid false matches like 'Stock Name'."""
    # Accepted aliases for stock symbol column
    valid_aliases = {
        "symbol", "ticker", "stock code", "stockcode",
        "security code", "security", "scrip"
    }
    
    mapping = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in valid_aliases:
            mapping[col] = "Symbol"
            break

    if mapping:
        df = df.rename(columns=mapping)
    return df


# --- Main ---
def main():
    # Custom CSS for dark theme and panel styling
    st.markdown(
        """
        <style>
        /* General Dark Theme */
        .stApp {
            background-color: #1a1a1a;
            color: #e0e0e0;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #e0e0e0;
        }
        p, label, .stMarkdown {
            color: white;
        }
        .stTextInput > div > div > input,
        .stSelectbox > div > div > div,
        .stFileUploader > div > div {
            background-color: #2d2d2d;
            color: #e0e0e0;
            border: 1px solid #444;
        }
        .stButton > button {
            background-color: #007bff;
            color: white;
            border-radius: 5px;
            padding: 8px 15px;
        }
        .stButton > button:hover {
            background-color: #0056b3;
        }
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #2d2d2d;
            color: #e0e0e0;
        }
        /*
        [data-testid="stMainBlockContainer"] {
            max-width: 57%;
        }*/
        /* Header and Connection Status */
        .header-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            width: 100%;
            padding-bottom: 20px;
            border-bottom: 1px solid #333;
            margin-bottom: 30px;
            text-wrap-mode: nowrap;
        }
        .platform-title {
            font-size: 2em !important;
            font-weight: bold;
            color: #e0e0e0;
            margin: 0;
        }
        .connection-status {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        .connection-status span {
            font-size: 1em;
            color: #bbb;
        }
        .connection-status .nse, .connection-status .bse {
            font-weight: bold;
        }
        .connection-status .nse {
            color: #4CAF50; /* Green for NSE */
        }
        .connection-status .bse {
            color: #FFC107; /* Amber for BSE */
        }
        .dashboard-button button {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
        }
        .dashboard-button button:hover {
            background-color: #0056b3;
        }
        [data-testid="stButton"]  {
            width: 295px;
        }
        /* Mode Selection Panels */
        .mode-container {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 40px;
            margin-bottom: 40px;
        }
        .mode-panel {
            background-color: #2d2d2d;
            border-radius: 10px;
            padding: 30px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease-in-out;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            width: 300px; /* Fixed width for uniformity */
            height: 250px; /* Fixed height for uniformity */
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        .mode-panel:hover {
            background-color: #3a3a3a;
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
        }
        .mode-panel img {
            height: 60px; /* Smaller icons */
            margin-bottom: 15px;
            filter: invert(1); /* Make icons white/light in dark mode */
        }
        .mode-panel h3 {
            color: #e0e0e0;
            font-size: 1.5em; /* Larger title */
            margin-bottom: 10px;
        }
        .mode-panel p {
            color: #bbb;
            font-size: 0.9em;
        }
        .mode-panel .sub-text {
            font-size: 0.8em;
            color: #888;
            margin-top: 10px;
        }
        /* Responsive adjustments */
        @media screen and (max-width: 768px) {
            .header-container {
                flex-direction: column;
                align-items: center;
                text-align: center;
            }
            .platform-title {
                font-size: 1.8em !important;
                margin-bottom: 10px;
            }
            .connection-status {
                flex-direction: column;
                gap: 5px;
            }
            .dashboard-button button {
                width: 100%;
                margin-top: 10px;
            }
            .mode-container {
                flex-direction: column;
                align-items: center;
                gap: 20px;
            }
            .mode-panel {
                width: 90%; /* Make panels take more width on small screens */
                height: auto; /* Allow height to adjust */
                padding: 20px;
            }
            .mode-panel h3 {
                font-size: 1.3em;
            }
            .mode-panel p {
                font-size: 0.85em;
            }
            .stFileUploader > div > div {
                width: 100% !important; /* Ensure file uploader is responsive */
            }
            [data-testid="stButton"] {
                width: 295px; /* Make sidebar buttons responsive */
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # --- Header Section ---
    st.markdown(
        """
        <div class="header-container">
            <h1 class="platform-title">Swing Trade Stock Platform</h1>
            <div class="connection-status">
                <span class="nse">🟢 NSE</span>
                <span class="bse">🟠 BSE</span>
                <div class="dashboard-button">
                    <button>Dashboard</button>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )



    # --- Mode Handling ---
    if "mode" not in st.session_state:
        st.session_state["mode"] = None

    mode = st.session_state["mode"]

    # --- Sidebar Status ---
    with st.sidebar:
        st.subheader("Technical Indicators")
        st.session_state["use_ema20"] = st.checkbox("EMA20", value=True, key="ema20_checkbox")
        st.session_state["use_ema50"] = st.checkbox("EMA50", value=True, key="ema50_checkbox")
        st.session_state["use_ema200"] = st.checkbox("EMA200", value=True, key="ema200_checkbox")
        st.session_state["use_rsi"] = st.checkbox("RSI", value=True, key="rsi_checkbox")
        st.session_state["use_macd"] = st.checkbox("MACD", value=True, key="macd_checkbox")
        st.session_state["use_volfilter"] = st.checkbox("Volume Filter", value=True, key="volfilter_checkbox")

        #st.markdown("---") # Separator

        st.subheader("Breakout Analysis")
        st.session_state["breakout_period"] = st.radio(
            "Select Breakout Period",
            ("None", "52-Day Breakout", "120-Day Breakout"),
            index=0,
            key="breakout_radio"
        )
        #st.markdown("---") # Separator

        if mode == "file":
            st.success("📂 Current Mode: File Upload")
        elif mode == "manual":
            st.success("🔍 Current Mode: Manual Input")
        else:
            st.info("ℹ️ No mode selected yet")

    # --- Mode Selection Panels ---
    col1, col2 = st.columns(2)

    with col1:
        # File Upload Panel (Visual)
        st.markdown(
            """
            <div class="mode-panel" id="file_upload_panel">
                <img src="https://img.icons8.com/ios/100/FFFFFF/upload--v1.png" alt="Upload File">
                <h3>Upload Stock List</h3>
                <p>Upload Excel/CSV file with your stock list for comprehensive analysis</p>
                <p class="sub-text">Excel/CSV | Batch Analysis</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        # Hidden Streamlit button for functionality
        st.button("Upload File Trigger", key="file_mode_trigger", on_click=lambda: st.session_state.update(mode="file"))
        st.markdown("<style>#file_mode_trigger {display: none; width: 87%;}</style>", unsafe_allow_html=True) # Hide the actual button and set width
        st.markdown(
            """
            <script>
            document.getElementById('file_upload_panel').onclick = function() {
                document.getElementById('file_mode_trigger').click();
            };
            </script>
            """,
            unsafe_allow_html=True
        )

    with col2:
        # Manual Input Panel (Visual)
        st.markdown(
            """
            <div class="mode-panel" id="manual_input_panel">
                <img src="https://img.icons8.com/ios/100/FFFFFF/edit--v1.png" alt="Manual Input">
                <h3>Manual Stock Entry</h3>
                <p>Enter specific stock symbol for detailed technical and fundamental analysis</p>
                <p class="sub-text">Single Stock | Deep Analysis</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        # Hidden Streamlit button for functionality
        st.button("Manual Input Trigger", key="manual_mode_trigger", on_click=lambda: st.session_state.update(mode="manual"))
        st.markdown("<style>#manual_mode_trigger {display: none;}</style>", unsafe_allow_html=True) # Hide the actual button
        st.markdown(
            """
            <script>
            document.getElementById('manual_input_panel').onclick = function() {
                document.getElementById('manual_mode_trigger').click();
            };
            </script>
            """,
            unsafe_allow_html=True
        )

    if mode == "file":
        uploaded_file = st.file_uploader("Upload Stock List (CSV/Excel)", type=["csv", "xlsx"])
        if uploaded_file:
            if uploaded_file.name.endswith(".csv"):
                df_uploaded = pd.read_csv(uploaded_file)
            else:
                df_uploaded = pd.read_excel(uploaded_file)

            # Try to auto-detect a symbol column
            df_uploaded = normalize_symbol_column(df_uploaded)

            if "Symbol" not in df_uploaded.columns:
                st.error("❌ The uploaded file must contain a stock symbol column (e.g., 'Symbol', 'Ticker', 'Stock Code').")
                return


            st.success(f"✅ Loaded {len(df_uploaded)} rows")

            # Pagination settings
            rows_per_page = st.sidebar.number_input("Rows per page", min_value=5, max_value=100, value=10, step=5)
            total_rows = len(df_uploaded)
            total_pages = (total_rows // rows_per_page) + int(total_rows % rows_per_page > 0)
            page = st.sidebar.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)

            start_row = (page - 1) * rows_per_page
            end_row = start_row + rows_per_page
            st.write(f"📄 Showing rows {start_row+1} to {min(end_row, total_rows)} of {total_rows}")
            st.dataframe(df_uploaded.iloc[start_row:end_row])

            # Column choice
            #column_choice = st.selectbox("Select the column containing stock symbols", df_uploaded.columns)
            if "Symbol" not in df_uploaded.columns:
                st.warning("⚠️ No standard symbol column detected. Please choose manually.")
                column_choice = st.selectbox("Select the column containing stock symbols", df_uploaded.columns)
                symbols = df_uploaded[column_choice].dropna().unique().tolist()
            else:
                column_choice = "Symbol"
                symbols = df_uploaded[column_choice].dropna().unique().tolist()

            # Analysis Mode
            analysis_options = [""] + ["Auto (Signals)", "BUY", "SELL", "HOLD"]
            analysis_mode = st.sidebar.selectbox("Select Analysis Mode", analysis_options, index=0)

            if analysis_mode == "":
                st.sidebar.warning("⚠️ Please select an analysis mode to run scans.")
                return

            # Scan all symbols
            recommendations = []
            for symbol in symbols:
                rec = analyze_symbol(symbol, exchange="NSE")
                if rec:
                    recommendations.append(rec)

            if recommendations:
                rec_df = pd.DataFrame(recommendations)
                # Apply analysis mode filter
                if analysis_mode == "BUY":
                    rec_df = rec_df[rec_df["Recommendation"] == "BUY"]
                elif analysis_mode == "SELL":
                    rec_df = rec_df[rec_df["Recommendation"] == "SELL"]
                elif analysis_mode == "HOLD":
                    rec_df = rec_df[rec_df["Recommendation"] == "HOLD"]

                st.subheader("📋 Recommendations Table")
                st.dataframe(rec_df)

                if not rec_df.empty:
                    options = [""] + rec_df["Symbol"].tolist()
                    selected_symbol = st.selectbox("Select a Signal Stock", options, index=0)
                    if selected_symbol != "":
                        plot_symbol(selected_symbol, exchange="NSE", last_n=180)
                else:
                    st.info("No actionable stocks found for this mode.")

    elif mode == "manual":
        # === Manual Search Mode (No File Uploaded) ===
        st.subheader("🔍 Manual Symbol Search")
        manual_symbol = st.text_input("Enter a stock symbol (e.g., RELIANCE, TCS, INFY):", "")

        if manual_symbol:
            write_log(f"Manual analysis started for {manual_symbol}")
            rec = analyze_symbol(manual_symbol.strip().upper(), exchange="NSE")
            if rec:
                st.subheader("📋 Manual Symbol Recommendation")
                st.table(pd.DataFrame([rec]))
                plot_symbol(manual_symbol.strip().upper(), exchange="NSE", last_n=180)
            else:
                st.warning(f"No data found for {manual_symbol}")
    else:
        st.info("👆 Please choose a mode above to continue")

    global log_container
    st.subheader("Runtime Logs")
    log_container = st.empty()
        
    # Create text area once with a key
    if "logs" not in st.session_state:
        st.session_state["logs"] = []

    if st.button("🗑️ Clear Logs"):
        clear_logs()
        st.session_state["logs"] = []
        log_container = st.empty()
        st.success("Logs cleared!")

    # Show current logs
    log_html = "<br>".join(st.session_state["logs"])
    log_container.markdown(
        f"""
        <div style="height:300px; overflow-y:auto; background-color:#111; color:#0f0; padding:10px; font-family:monospace;" id="log-box">
            {log_html}
        </div>
        <script>
            var logBox = document.getElementById('log-box');
            if (logBox) {{ logBox.scrollTop = logBox.scrollHeight; }}
        </script>
        """,
        unsafe_allow_html=True
    )
    
if __name__ == "__main__":
    main()

