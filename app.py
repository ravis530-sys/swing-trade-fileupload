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

    # Indicators
    df['EMA20'] = df[close_col].ewm(span=20).mean()
    df['EMA50'] = df[close_col].ewm(span=50).mean()
    df['EMA200'] = df[close_col].ewm(span=200).mean()

    df = compute_indicators(df, symbol=symbol, close_col=close_col)

    # Signals with volume filter
    df["Buy"] = (((df["EMA20"] > df["EMA200"]) | (df["EMA50"] > df["EMA200"])) &
                 (df["RSI"] < 30) &
                 (df["MACD"] > df["Signal"]) &
                 (df["VolFilter"]))

    df["Sell"] = (((df["EMA20"] < df["EMA200"]) | (df["EMA50"] < df["EMA200"])) &
                  (df["RSI"] > 70) &
                  (df["MACD"] < df["Signal"]) &
                 (df["VolFilter"]))

    latest_price = df[close_col].iloc[-1]
    recent_window = df.tail(20)
    support = round(recent_window[close_col].min(), 2)
    resistance = round(recent_window[close_col].max(), 2)

    #logs
        # --- Activity Logs ---
    write_log(f"{symbol}: "
              f"EMA20={df['EMA20'].iloc[-1]:.2f}, "
              f"EMA200={df['EMA200'].iloc[-1]:.2f}, "
              f"EMA50={df['EMA50'].iloc[-1]:.2f}, "
              f"RSI={df['RSI'].iloc[-1]:.2f}, "
              f"MACD={df['MACD'].iloc[-1]:.2f}, "
              f"Signal={df['Signal'].iloc[-1]:.2f}, "
              f"VolFilter={df['VolFilter'].iloc[-1]}, "
              f"Buy={df['Buy'].iloc[-1]}, "
              f"Sell={df['Sell'].iloc[-1]}")

     # Auto recommendation
    if df["Buy"].iloc[-1]:
        rec, target, sl = "BUY", round(latest_price * 1.05, 2), support
    elif df["Sell"].iloc[-1]:
        rec, target, sl = "SELL", round(latest_price * 0.95, 2), resistance
    else:
        rec, target, sl = "HOLD", None, None

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

    # Indicators
    df['EMA20'] = df[close_col].ewm(span=20).mean()
    df['EMA50'] = df[close_col].ewm(span=50).mean()
    df['EMA200'] = df[close_col].ewm(span=200).mean()

    df = compute_indicators(df, symbol=symbol, close_col=close_col)

    # Signals
    df["Buy"] = (((df["EMA20"] > df["EMA200"]) | (df["EMA50"] > df["EMA200"])) &
                 (df["RSI"] < 30) &
                 (df["MACD"] > df["Signal"]) &
                 (df["VolFilter"]))

    df["Sell"] = (((df["EMA20"] < df["EMA200"]) | (df["EMA50"] < df["EMA200"])) &
                  (df["RSI"] > 70) &
                  (df["MACD"] < df["Signal"]) &
                 (df["VolFilter"]))

    # --- Price + EMA + Signals ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df[close_col], label="Close", color="blue")
    ax.plot(df.index, df['EMA20'], "--", label="EMA20")
    ax.plot(df.index, df['EMA50'], "--", label="EMA50")
    ax.plot(df.index, df['EMA200'], "--", label="EMA200")
    ax.scatter(df.index[df["Buy"]], df[close_col][df["Buy"]], marker="^", color="green", s=100, label="Buy")
    ax.scatter(df.index[df["Sell"]], df[close_col][df["Sell"]], marker="v", color="red", s=100, label="Sell")
    ax.legend()
    ax.set_title(f"{symbol} Price & Signals")
    st.pyplot(fig)

    # --- RSI Chart ---
    fig_rsi, ax_rsi = plt.subplots(figsize=(10, 2))
    ax_rsi.plot(df.index, df["RSI"], color="purple")
    ax_rsi.axhline(70, color="red", linestyle="--", alpha=0.5)
    ax_rsi.axhline(30, color="green", linestyle="--", alpha=0.5)
    ax_rsi.set_title("RSI (14)")
    st.pyplot(fig_rsi)

    # --- MACD Chart ---
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
        df_trades = pd.DataFrame(trades)
        st.subheader("📊 Trade History")
        st.table(df_trades)

        # Stats
        total_trades = len(df_trades)
        wins = (df_trades["Profit%"] > 0).sum()
        win_rate = wins / total_trades if total_trades else 0
        avg_profit = df_trades["Profit%"].mean()

        stats = {
            "Total Trades": total_trades,
            "Win Rate": f"{win_rate:.2%}",
            "Avg Profit %": round(avg_profit, 2),
            "Max Profit %": df_trades["Profit%"].max(),
            "Max Loss %": df_trades["Profit%"].min()
        }

        st.subheader("📈 Backtest Performance")
        st.json(stats)

        # Equity Curve
        start_capital = st.sidebar.number_input("Starting Capital (₹)", min_value=10000, value=1000000, step=50000)
        risk_fraction = st.sidebar.slider("Fraction of capital per trade", 0.1, 1.0, 1.0)
        risk_per_trade_pct = st.sidebar.slider("Risk % per trade (for ruin calc)", 0.5, 5.0, 2.0)

        portfolio = [start_capital]
        for _, trade in df_trades.iterrows():
            last_capital = portfolio[-1]
            trade_capital = last_capital * risk_fraction
            profit = trade_capital * (trade["Profit%"] / 100.0)
            portfolio.append(last_capital + profit)

        equity_curve = pd.Series(portfolio[1:], index=df_trades["ExitDate"])

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
        avg_win = df_trades[df_trades["Profit%"] > 0]["Profit%"].mean()
        avg_loss = abs(df_trades[df_trades["Profit%"] <= 0]["Profit%"].mean())
        win_rate = (df_trades["Profit%"] > 0).mean()

        if pd.notna(avg_win) and pd.notna(avg_loss) and avg_loss > 0:
            reward_risk = avg_win / avg_loss
            capital_units = int(start_capital / (start_capital * (risk_per_trade_pct / 100)))
            risk_of_ruin = ((1 - (win_rate * reward_risk)) / (1 + (win_rate * reward_risk))) ** capital_units
            st.subheader("⚠️ Risk of Ruin Estimate")
            st.write(f"Win Rate: {win_rate:.2%}, Avg Win/Loss Ratio: {reward_risk:.2f}")
            st.write(f"Estimated Risk of Ruin: {risk_of_ruin:.6f}")
        else:
            st.info("Not enough trades to estimate Risk of Ruin.")

    # Final Recommendation
    rec = analyze_symbol(symbol, exchange=exchange)
    if rec:
        st.subheader("📋 Trade Recommendation")
        st.table(pd.DataFrame([rec]))

# --- Main ---
def main():
    st.title("📊 Swing Trade AI Agent")

    uploaded_file = st.file_uploader("Upload Stock List (CSV/Excel)", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df_uploaded = pd.read_csv(uploaded_file)
        else:
            df_uploaded = pd.read_excel(uploaded_file)

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
        column_choice = st.selectbox("Select the column containing stock symbols", df_uploaded.columns)
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

    global log_container
    st.subheader("Runtime Logs")
    log_container = st.empty()
        
    # Create text area once with a key
    if "logs" not in st.session_state:
        st.session_state["logs"] = []

    if st.button("🗑️ Clear Logs"):
        clear_logs()
        st.session_state["logs"] = []
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
