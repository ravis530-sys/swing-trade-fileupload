# README.md
### 📊 Swing Trade Stock Agent

An interactive **Streamlit dashboard** that analyzes Indian stocks (NSE/BSE) using **technical indicators** and provides **Buy/Sell/Hold recommendations**.  
It includes **charting, backtesting, equity curve analysis, risk of ruin, and runtime activity logs** with auto-scrolling.

---

## 🚀 Features

- 📂 Upload stock lists (CSV/Excel) with pagination support.  
- 📊 Auto, Buy, Sell, Hold analysis modes.  
- 🧮 Technical Indicators:
  - RSI (14-day, Wilder’s smoothing)
  - MACD (12, 26, 9)
  - EMA crossovers (20, 50, 200)
  - Volume filter (20-day avg > 500,000)
- 🛠️ Trade Recommendation Table (Entry, Target, Support, Resistance, Stop Loss).  
- 📈 Interactive charts:
  - Price + EMA overlays
  - RSI chart
  - MACD chart
  - Equity curve & Drawdown
- 📝 Live activity logs with auto-scroll and a dark theme.
- 🗑️ Clear Logs button.
- 💾 Logs are persisted to activity.log.

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/swing-trade-fileupload.git
cd swing-trade-fileupload
```
2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\\Scripts\\activate      # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```
## Usage

```bash
streamlit run app.py
```
## 📖 How to Use
1. Upload a Stock List (CSV/Excel).
2. Browse Table with Pagination.
3. Select Symbol Column.
4. Choose Analysis Mode.
5. Review Recommendations Table.
6. Select a Stock → view charts and backtests.
7. Monitor Runtime Logs in real time.

## 📂 Project Structure

```lua
swing-trade-aiagent/
│── app.py
│── requirements.txt
│── activity.log
│── README.md
```
## 📜 License

MIT License.
