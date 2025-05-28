# 🚀 Setup für US500 Forecast App (inkl. Cleanup & Backup)

!pip install yfinance streamlit streamlit-autorefresh scikit-learn matplotlib plotly pyngrok pandas_datareader snscrape nltk --quiet
!pip install pyngrok --index-url=https://pypi.org/simple

import os, shutil, datetime, time, socket, subprocess
from pyngrok import conf, ngrok
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime
import glob
today_str = datetime.now().strftime("%Y-%m-%d")

RESET_LOGS = False
APP_FILENAME = "app.py"
APP_PORT = 8501
NGROK_TOKEN = "2xB84xP48MVVpa7WOjuVT9OgiUI_2pDz4T89jbtkBdgvrLtV4"

# Cleanup
!pkill -f streamlit || true
!killall streamlit || true
!killall ngrok || true
!pkill ngrok || true
!pkill -f ngrok || true
!rm -rf __pycache__ */__pycache__ *.pyc
!rm -rf /root/.ngrok2 /root/.config/ngrok
!streamlit cache clear

# Jeden Tag ein neues Logfile für Forecast-Log
today_str = datetime.now().strftime("%Y-%m-%d")
forecast_log_file = f"spy_forecast_log_{today_str}.csv"
LOG_FILES = [forecast_log_file, "spy_intraday_history.csv"]

# Backup/Reset nur, wenn das File heute noch nicht existiert
for file in LOG_FILES:
    if os.path.exists(file):
        if RESET_LOGS:
            os.remove(file)
        else:
            today = datetime.now().strftime("%Y%m%d")
            backup_pattern = f"{file.replace('.csv','')}_backup_{today}_*.csv"
            if not glob.glob(backup_pattern):
                ts = datetime.now().strftime("%Y%m%d_%H%M")
                shutil.copy(file, f"{file.replace('.csv','')}_backup_{ts}.csv")

app_code = '''\
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import datetime
import yfinance as yf
import os, joblib, requests, glob, shutil
from sklearn.ensemble import RandomForestRegressor
from bs4 import BeautifulSoup
from streamlit_autorefresh import st_autorefresh
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from plotly.subplots import make_subplots
import nltk
from datetime import datetime
from sklearn.metrics import mean_absolute_error

def train_and_evaluate_regression(train, test, feature_cols, target_col="target"):
    # Entferne Zeilen mit NaN
    train = train.dropna(subset=feature_cols + [target_col])
    test = test.dropna(subset=feature_cols + [target_col])
    if len(train) == 0 or len(test) == 0:
        # Optional: print("Nicht genug Daten für Training/Test")
        return None, None, None, None, None, None, None
    X_train, y_train = train[feature_cols], train[target_col]
    X_test, y_test = test[feature_cols], test[target_col]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    # Nochmals auf NaN checken (optional)
    mask = ~np.isnan(preds)
    preds = preds[mask]
    y_test = y_test.iloc[mask]
    if len(preds) == 0:
        return model, preds, None, None, None, None, None
    mae = mean_absolute_error(y_test, preds)
    hitrate = np.mean(np.sign(y_test) == np.sign(preds))
    avg_pred = np.mean(preds)
    avg_true = np.mean(y_test)
    std_pred = np.std(preds)
    return model, preds, mae, hitrate, avg_pred, avg_true, std_pred

def time_series_train_test_split(df, test_size=0.2):
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test

nltk.download('vader_lexicon', quiet=True)

# ========================== KONSTANTEN & INTERVAL OPTIONS ==========================
today_str = datetime.now().strftime("%Y-%m-%d")
forecast_log_file = f"spy_forecast_log_{today_str}.csv"
LOG_FILES = [forecast_log_file, "spy_intraday_history.csv"]
MODEL_FILE = "forecast_model_5min.pkl"
REPAIRED_LOG_FILE = "spy_forecast_log_repaired.csv"
FINNHUB_API_KEY = "d0ldkdhr01qhb027s8fgd0ldkdhr01qhb027s8g0"
RESET_LOGS = False

interval_options = [
    ("10 Sek", pd.Timedelta(seconds=10)),
    ("1 Min", pd.Timedelta(minutes=1)),
    ("2 Min", pd.Timedelta(minutes=2)),
    ("5 Min", pd.Timedelta(minutes=5)),
    ("15 Min", pd.Timedelta(minutes=15)),
    ("30 Min", pd.Timedelta(minutes=30)),
    ("1 Std", pd.Timedelta(hours=1)),
    ("2 Std", pd.Timedelta(hours=2)),
    ("3 Std", pd.Timedelta(hours=3)),
    ("5 Std", pd.Timedelta(hours=5)),
    ("8 Std", pd.Timedelta(hours=8)),
    ("10 Std", pd.Timedelta(hours=10)),
    ("Bis Handelsende", None),
]

st.set_page_config(page_title='US500 Forecast Tool', layout='wide')

sentiment_score = None
sentiment_sources = None

# ========== Log-Backup oder Reset ===============
for file in LOG_FILES:
    if os.path.exists(file):
        if RESET_LOGS:
            os.remove(file)
        else:
            today = datetime.now().strftime("%Y%m%d")
            backup_pattern = f"{file.replace('.csv','')}_backup_{today}_*.csv"
            if not glob.glob(backup_pattern):
                ts = datetime.now().strftime("%Y%m%d_%H%M")
                shutil.copy(file, f"{file.replace('.csv','')}_backup_{ts}.csv")

# ========================== ML: Dummy-Modell & Auto-Training ==========================
def create_dummy_model():
    X_dummy = pd.DataFrame({
        "RSI": np.random.uniform(30, 70, 200),
        "MACD": np.random.normal(0, 1, 200),
        "MACD_signal": np.random.normal(0, 1, 200),
        "BB_range": np.random.uniform(1, 20, 200),
        "Sentiment": np.random.uniform(-1, 1, 200),
    })
    y_dummy = np.random.uniform(-0.02, 0.02, 200)
    model = RandomForestRegressor(n_estimators=25, random_state=42)
    model.fit(X_dummy, y_dummy)
    selected_features = list(X_dummy.columns)
    joblib.dump({"model": model, "selected_features": selected_features}, MODEL_FILE)

def load_all_logs(log_patterns=["spy_forecast_log*.csv", "spy_intraday_history*.csv"]):
    files = []
    for pat in log_patterns:
        files.extend(glob.glob(pat))
    dfs = []
    for file in files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return None
    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["Time"])
    return df_all

def ensure_target_column(df, n_steps=5):
    if "target" not in df.columns and "Price" in df.columns:
        df = df.sort_values("Time")
        df["target"] = df["Price"].shift(-n_steps) / df["Price"] - 1
    return df

def auto_train_from_all_logs(model_file=MODEL_FILE, min_rows=100, target_col="target", n_steps=5):
    df = load_all_logs()
    if df is None:
        return False
    df = ensure_target_column(df, n_steps=n_steps)
    if target_col not in df.columns or len(df) < min_rows:
        return False
    feature_cols = [col for col in df.columns if col not in ["Time", "Price", target_col]]
    X = df[feature_cols]
    y = df[target_col]
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    if len(y) < min_rows:
        return False
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump({"model": model, "selected_features": feature_cols}, model_file)
    return True

if not auto_train_from_all_logs():
    if not os.path.exists(MODEL_FILE):
        create_dummy_model()

# ========== Hilfsfunktionen ==========
def get_finnhub_quote(symbol="ES=F", api_key=FINNHUB_API_KEY):
    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={api_key}"
    r = requests.get(url, timeout=6)
    if r.status_code == 200:
        data = r.json()
        return {
            "current": data.get("c"),
            "high": data.get("h"),
            "low": data.get("l"),
            "open": data.get("o"),
            "prevclose": data.get("pc"),
            "timestamp": data.get("t")
        }
    else:
        return None

def get_zurich_now():
    return pd.Timestamp.now(tz='Europe/Zurich')

def get_price(ticker):
    try:
        yahoo_ticker = ticker.replace('^', '%5E')
        if ticker == "US500":
            yahoo_ticker = "US500.cash"
        url = f"https://finance.yahoo.com/quote/{yahoo_ticker}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        soup = BeautifulSoup(r.text, "html.parser")
        s = soup.find("fin-streamer", {"data-field": "regularMarketPrice"})
        return float(s.text.replace(",", ""))
    except Exception as e:
        st.warning(f"⚠️ Preis konnte nicht geladen werden: {e}")
        return None

analyzer = SentimentIntensityAnalyzer()
def get_sentiment_finviz():
    try:
        url = f"https://finviz.com/quote.ashx?t=ES=F"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=8)
        response.encoding = "utf-8"
        soup = BeautifulSoup(response.text, "html.parser")
        news_table = soup.find(id="news-table")
        headlines = [row.find("a").get_text(strip=True) for row in news_table.find_all("tr") if row.find("a")]
        if not headlines:
            return (0.0, [])
        scores = [analyzer.polarity_scores(h)["compound"] for h in headlines]
        return (np.mean(scores), headlines)
    except Exception:
        return (None, [])

def get_sentiment_yahoo():
    try:
        url = "https://news.search.yahoo.com/search?p=S%26P500"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=7)
        soup = BeautifulSoup(resp.text, "html.parser")
        headlines = [item.get_text(strip=True) for item in soup.find_all("h4", class_="s-title")]
        if not headlines:
            return (0.0, [])
        scores = [analyzer.polarity_scores(h)["compound"] for h in headlines]
        return (np.mean(scores), headlines)
    except Exception:
        return (None, [])

def get_sentiment_all():
    s_fin, hl_fin = get_sentiment_finviz()
    s_yahoo, hl_yahoo = get_sentiment_yahoo()
    valid = [s for s in [s_fin, s_yahoo] if s is not None]
    if valid:
        s_agg = np.mean(valid)
    else:
        s_agg = None
    sources = {
        "Finviz": (s_fin, hl_fin),
        "Yahoo": (s_yahoo, hl_yahoo),
    }
    return s_agg, sources

    st.write(sentiment_score, sentiment_sources)

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = delta.clip(upper=0).abs().rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_ema(series, window=10):
    return series.ewm(span=window, adjust=False).mean()

def compute_macd(series, fast=12, slow=26, signal=9):
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def compute_bollinger(series, window=20, std=2):
    mean = series.rolling(window).mean()
    stddev = series.rolling(window).std()
    return mean, mean + std * stddev, mean - std * stddev

def repair_csv(log_file, repaired_log_file):
    if not os.path.exists(log_file):
        return False
    repaired = False
    with open(log_file, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
    if len(lines) < 2:
        with open(repaired_log_file, 'w', encoding='utf-8') as fout:
            fout.writelines(lines)
        return False
    header = lines[0]
    n_cols = len(header.strip().split(","))
    good_lines = [header]
    for line in lines[1:]:
        if len(line.strip().split(",")) == n_cols:
            good_lines.append(line)
        else:
            repaired = True
    if repaired:
        with open(repaired_log_file, 'w', encoding='utf-8') as fout:
            fout.writelines(good_lines)
    return repaired

def compute_hit_rate(df_log, interval_td):
    if df_log is None or len(df_log) < 20 or interval_td is None:
        return np.nan
    df_log = df_log.copy()
    df_log["Time"] = pd.to_datetime(df_log["Time"], errors="coerce", utc=True).dt.tz_convert("Europe/Zurich")
    df_log = df_log.sort_values("Time")
    df_log = df_log.dropna(subset=["Forecast", "Price"])
    df_log["Forecast"] = pd.to_numeric(df_log["Forecast"], errors="coerce")
    df_log["Price"] = pd.to_numeric(df_log["Price"], errors="coerce")
    hits = []
    for i, row in df_log.iterrows():
        t0 = row["Time"]
        t1 = t0 + interval_td
        future = df_log[(df_log["Time"] > t0) & (df_log["Time"] <= t1)]
        if len(future) == 0:
            continue
        real_return = (future.iloc[-1]["Price"] / row["Price"]) - 1
        forecast = row["Forecast"]
        if pd.isna(forecast) or pd.isna(real_return):
            continue
        if (forecast > 0 and real_return > 0) or (forecast < 0 and real_return < 0):
            hits.append(1)
        else:
            hits.append(0)
    if len(hits) == 0:
        return np.nan
    return np.mean(hits)

def calc_features(row, sentiment_score, selected_features, finnhub_data, volatility_window):
    # row muss ein dict sein!
    feature_map = {
        "SMA_5": lambda r: r.get("SMA", np.nan),
        "SMA_10": lambda r: r.get("SMA", np.nan),
        "SMA_30": lambda r: r.get("SMA", np.nan),
        "EMA_10": lambda r: r.get("EMA", np.nan),
        "RSI": lambda r: r.get("RSI", 50),
        "MACD": lambda r: r.get("MACD", 0),
        "Signal": lambda r: r.get("MACD_signal", 0),
        "BB_range": lambda r: (r.get("BB_upper", 0)-r.get("BB_lower", 0)) if "BB_upper" in r and "BB_lower" in r else 0,
        f"Volatility_{volatility_window}": lambda r: r.get("Volatility", np.nan),
        "Sentiment": lambda r: sentiment_score if sentiment_score is not None else 0,
        "Finnhub_High": lambda r: finnhub_data["high"] if finnhub_data else np.nan,
        "Finnhub_Low": lambda r: finnhub_data["low"] if finnhub_data else np.nan,
        "Finnhub_Open": lambda r: finnhub_data["open"] if finnhub_data else np.nan,
        "Finnhub_PrevClose": lambda r: finnhub_data["prevclose"] if finnhub_data else np.nan,
    }
    feats = []
    if selected_features:
        for feat in selected_features:
            val = feature_map.get(feat, lambda r: np.nan)(row)
            feats.append(val)
    else:
        feats = [
            row.get("RSI", 50),
            row.get("MACD", 0),
            row.get("MACD_signal", 0),
            (row.get("BB_upper", 0)-row.get("BB_lower", 0)) if "BB_upper" in row and "BB_lower" in row else 0,
            sentiment_score if sentiment_score is not None else 0
        ]
    return feats

def robust_live_logging(
    price,
    sentiment_score,
    df,
    log_file="spy_forecast_log.csv",
    model_file="forecast_model_5min.pkl",
    finnhub_data=None,
    volatility_window=10
):
    import os
    import pandas as pd
    import joblib
    import numpy as np

    now = pd.Timestamp.now(tz="Europe/Zurich").replace(microsecond=0)
    if len(df) == 0:
        return
    last_row = df.iloc[-1].to_dict()
    feats_row = {
        "Time": now,
        "Price": price,
        "RSI": last_row.get("RSI", np.nan),
        "MACD": last_row.get("MACD", np.nan),
        "MACD_signal": last_row.get("MACD_signal", np.nan),
        "BB_upper": last_row.get("BB_upper", np.nan),
        "BB_lower": last_row.get("BB_lower", np.nan),
        "BB_range": (last_row.get("BB_upper", np.nan) - last_row.get("BB_lower", np.nan))
            if not pd.isna(last_row.get("BB_upper", np.nan)) and not pd.isna(last_row.get("BB_lower", np.nan)) else np.nan,
        "Volatility": last_row.get("Volatility", np.nan),
        "Sentiment": sentiment_score if sentiment_score is not None else np.nan,
    }
    if finnhub_data:
        feats_row["Finnhub_High"] = finnhub_data.get("high", np.nan)
        feats_row["Finnhub_Low"] = finnhub_data.get("low", np.nan)
        feats_row["Finnhub_Open"] = finnhub_data.get("open", np.nan)
        feats_row["Finnhub_PrevClose"] = finnhub_data.get("prevclose", np.nan)
    if os.path.exists(model_file):
        try:
            data = joblib.load(model_file)
            model = data["model"]
            selected_features = data["selected_features"]
            feats_for_model = []
            for feat in selected_features:
                feats_for_model.append(feats_row.get(feat, np.nan))
            pred = float(model.predict([feats_for_model])[0])
        except Exception as e:
            pred = np.nan
    else:
        pred = np.nan
    feats_row["Forecast"] = pred
    df_log_new = pd.DataFrame([feats_row])
    write_header = not os.path.exists(log_file)
    append = True
    if not write_header:
        try:
            df_existing = pd.read_csv(log_file)
            if not df_existing.empty:
                last_time_existing = pd.to_datetime(df_existing["Time"]).max()
                if pd.Timestamp(now) <= last_time_existing:
                    append = False
        except Exception:
            append = True
    if append:
        df_log_new.to_csv(log_file, mode="a", header=write_header, index=False)

# ========== TICKER-AUSWAHL (vor Sidebar laden, für globale Verwendung!) ==========
if "TICKER" not in st.session_state:
    st.session_state["TICKER"] = "ES=F"

# 2. Sentiment ERST LADEN!
sentiment_score, sentiment_sources = get_sentiment_all()

# ========================== SIDEBAR & UI ==========================
with st.sidebar:
    try:
        reload_options = {
            "5 Sekunden": 5000,
            "10 Sekunden": 10000,
            "30 Sekunden": 30000,
            "1 Minute": 60000,
            "2 Minuten": 120000
        }
        reload_keys = list(reload_options.keys())
        reload_label = st.selectbox(
            "Reload-Intervall",
            reload_keys,
            index=1,
            key="reload_interval_select"
        )
        st_autorefresh(interval=reload_options[reload_label], key="autorefresh")

        csv_export = st.checkbox("Chartdaten als CSV exportieren", value=False, key="csv_export_checkbox")
        with st.expander("ℹ️ Indikator-Erklärungen", expanded=False):
            st.markdown("""
**SMA (Simple Moving Average)**
_Glättet den Kursverlauf und hilft, Trends zu erkennen._

**EMA (Exponential Moving Average)**
_Reagiert schneller auf Kursänderungen als der SMA._

**Bollinger-Bänder**
_Zeigen Volatilität und Extrembereiche im Kursverlauf._

**RSI (Relative Strength Index)**
_Misst die relative Stärke/Schwäche und signalisiert überkauft/überverkauft._

**MACD (Moving Average Convergence Divergence)**
_Trendfolgeindikator zur Bestimmung von Trendstärke und Trendwechseln._

**Volatility (Volatilität)**
_Gibt die Schwankungsbreite des Kurses an._

**Forecast (ML-Prognose)**
_Machine-Learning-basierte Prognose der Kursrichtung für das gewählte Zeitintervall._
            """)
        with st.expander("Ticker-Auswahl", expanded=False):
            st.markdown("""
- **ES=F:** S&P 500 Future (fast 24/7, reagiert schnell auf News) [empfohlen]
- **US500:** CFD-Kontrakt auf den S&P 500 (nahezu 24/5 Handelszeit, Brokerabhängig)
- **^GSPC:** S&P 500 Index (offizieller Markt, keine Nachthandel)
- **SPY:** ETF auf den S&P 500 (börsengehandelt, US-Handelszeiten)

*Hinweis: Für durchgehende Daten empfiehlt sich ES=F oder US500.*
            """)
            TICKER = st.selectbox(
                "Ticker wählen",
                ["ES=F", "US500", "^GSPC", "SPY"],
                index=0,
                key="ticker_selectbox"
            )
        with st.expander("Indikatoren anzeigen/ausblenden", expanded=True):
            show_sma = st.checkbox("SMA", value=True, key="sma_checkbox")
            st.caption("SMA: Gleitender Durchschnitt – erkennt Trends.")
            show_ema = st.checkbox("EMA", value=True, key="ema_checkbox")
            st.caption("EMA: Reagiert schneller auf Kursänderungen.")
            show_boll = st.checkbox("Bollinger-Bänder", value=True, key="boll_checkbox")
            st.caption("Bollinger-Bänder: Zeigen Volatilität und Extremzonen.")
            show_rsi = st.checkbox("RSI", value=True, key="rsi_checkbox")
            st.caption("RSI: Relative Stärke, Extremzonen.")
            show_macd = st.checkbox("MACD", value=True, key="macd_checkbox")
            st.caption("MACD: Trendfolge, Trendwechsel.")
            show_volatility = st.checkbox("Volatilität", value=True, key="volatility_checkbox")
            st.caption("Volatilität: Schwankungsbreite.")
        with st.expander("Indikator-Einstellungen", expanded=False):
            slider_height = 90
            st.markdown(
                f"""
                <style>
                div[data-baseweb="slider"] {{
                    min-height: {slider_height}px !important;
                    margin-bottom: 18px !important;
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )
            sma_window = st.slider("SMA-Fenster", 2, 50, 20, key="sma_window_slider")
            st.caption("Kürzeres Fenster: Schnellere, aber volatilere Trend-Erkennung.")
            ema_window = st.slider("EMA-Fenster", 2, 50, 10, key="ema_window_slider")
            st.caption("Kürzeres Fenster: Empfindlicher gegenüber Kursänderungen.")
            rsi_window = st.slider("RSI-Fenster", 2, 50, 14, key="rsi_window_slider")
            st.caption("Kürzeres Fenster: Sensibler auf kurzfristige Übertreibungen.")
            macd_fast = st.slider("MACD Fast", 2, 50, 12, key="macd_fast_slider")
            st.caption("Kürzer = schnellere Trendwenden, aber mehr Rauschen.")
            macd_slow = st.slider("MACD Slow", 2, 100, 26, key="macd_slow_slider")
            st.caption("Längere Perioden = glatterer MACD.")
            macd_signal = st.slider("MACD Signal", 2, 50, 9, key="macd_signal_slider")
            st.caption("Kürzer = schnellere MACD-Signale.")
            boll_window = st.slider("Bollinger Fenster", 2, 50, 20, key="boll_window_slider")
            st.caption("Kürzer = schneller auf Volatilitätsänderungen.")
            boll_std = st.slider("Bollinger Std-Abw.", 1, 4, 2, key="boll_std_slider")
            st.caption("Größere Abweichung = breitere Bänder.")
            volatility_window = st.slider("Volatility Fenster", 2, 50, 10, key="volatility_window_slider")
            st.caption("Kürzer = empfindlicher auf kurzfristige Schwankungen.")
            st.markdown("> **Hinweis:** Je kürzer die Periode, desto schneller reagieren die Indikatoren – aber desto mehr Rauschen!")
        with st.expander("Y-Achse (Preis)", expanded=False):
            use_autoscale = st.checkbox("Y-Achse automatisch skalieren", value=True, key="autoscale_checkbox")
            user_ymin = user_ymax = None
            if 'df' in st.session_state and not st.session_state.df.empty:
                tmp_min = float(st.session_state.df["Price"].min()) - 5
                tmp_max = float(st.session_state.df["Price"].max()) + 5
            else:
                tmp_min = 0.0
                tmp_max = 10.0
            if not use_autoscale:
                user_ymin = st.number_input("Y-Achse (min)", value=tmp_min, step=1.0, format="%.2f", key="ymin_numberinput")
                user_ymax = st.number_input("Y-Achse (max)", value=tmp_max, step=1.0, format="%.2f", key="ymax_numberinput")
            st.caption("Wenn die Autoskalierung deaktiviert ist, kannst du den sichtbaren Preisbereich manuell einstellen.")
        with st.expander("Kommentare", expanded=False):
            kommentare = st.text_area("Deine Notizen / Kommentare", "", key="kommentare_textarea")
            st.markdown("Hier kannst du beliebige Kommentare, Tradingideen oder Beobachtungen zu diesem Run notieren.")
        with st.sidebar:
            with st.expander("Sentiment", expanded=False):
                if sentiment_sources is not None:
                    for name, (score, headlines) in sentiment_sources.items():
                        st.markdown(f"**{name}**: {'{:+.2f}'.format(score) if score is not None else 'n/a'}")
                        if headlines:
                            for hl in headlines[:20]:
                                st.write(f"• {hl}")
                        else:
                            st.info("Keine Daten verfügbar.")
                else:
                    st.info("Keine Sentiment-Quellen verfügbar.")

        df = st.session_state.df if "df" in st.session_state else pd.DataFrame()
        # --- Debug Info ---
        debug_show = st.checkbox("Debug-Infos anzeigen", value=False, key="debug_checkbox")
        if debug_show:
            st.markdown("---")
            st.markdown("#### Debug-Infos")
            st.write("Aktuelle Zeit (Europe/Zurich):", pd.Timestamp.now(tz="Europe/Zurich"))
            if isinstance(df, pd.DataFrame) and not df.empty:
                st.write("Letzter Zeitstempel im DataFrame:", df.index.max())
                st.write("Letzte Zeile:", df.tail(1))
                st.write("Zeitbereich im Chart:", df.index.min(), "bis", df.index.max())
                if "df_visible" in locals():
                    st.write("Zeitbereich im sichtbaren Chart:", df_visible.index.min(), "bis", df_visible.index.max())
                if 'xaxis_range' in locals() and xaxis_range is not None:
                    st.write("Zeitbereich im sichtbaren Chart:", xaxis_range[0], "bis", xaxis_range[1])
            else:
                st.info("DataFrame ist leer oder nicht initialisiert.")

        # --- Best-Practice-ML-Report Schalter ---
        show_bp_report = st.checkbox("Best-Practice-ML-Report anzeigen", value=True, key="bp_report_checkbox")
        # --- Log löschen Button ---
        if "log_reset_click" not in st.session_state:
            st.session_state.log_reset_click = False
        st.markdown("---")
        if st.button("🗑️ Alle Logs löschen (Reset für Training)", key="log_reset_button"):
            st.session_state.log_reset_click = True
        if st.session_state.log_reset_click:
            st.warning(
                "⚠️ **Achtung:** Das Löschen entfernt **alle Logdateien** für Prognose und Training. "
                "Ein erneutes Training beginnt mit komplett leeren Daten. "
                "**Fortfahren?**"
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Ja, löschen!", key="log_reset_confirm_btn"):
                    log_files = [
                        "spy_forecast_log.csv",
                        "spy_intraday_history.csv",
                        "spy_forecast_log_repaired.csv"
                    ]
                    deleted = []
                    for lf in log_files:
                        if os.path.exists(lf):
                            os.remove(lf)
                            deleted.append(lf)
                    st.success("Alle Logs wurden gelöscht: " + ", ".join(deleted) if deleted else "Keine Logs gefunden.")
                    st.session_state.log_reset_click = False
            with col2:
                if st.button("Abbrechen", key="log_reset_cancel_btn"):
                    st.session_state.log_reset_click = False

    except Exception as e:
        st.error(f"Sidebar-Fehler: {e}")

# ========================== TITEL ==========================
st.markdown(
    '<h3 style="margin-bottom:0.5em;">US500/ES=F Live-Prognose & Sentimentanalyse (inkl. Finnhub Live-Daten, Multi-Zeitintervall-Prognose, Zürich-Zeit)</h3>',
    unsafe_allow_html=True
)

# ====== INITIALISIERUNGEN FÜR SICHERHEIT ======
price = None
finnhub_data = None
sentiment_score = None
sentiment_sources = None

# ========================== FINNHUB & PREIS ==========================
finnhub_data = get_finnhub_quote(TICKER.replace("^", ""), FINNHUB_API_KEY)
if finnhub_data and finnhub_data["current"]:
    price = finnhub_data["current"]
    st.markdown(f"**Finnhub Kurs:** {price:.2f} (High: {finnhub_data['high']}, Low: {finnhub_data['low']}, Open: {finnhub_data['open']}, PrevClose: {finnhub_data['prevclose']})")
else:
    price = get_price(TICKER)
    if price is not None:
        st.markdown(f"**{TICKER} Kurs (Yahoo):** {price:.2f}", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Kein Kurs verfügbar.")

sentiment_score, sentiment_sources = get_sentiment_all()
# st.write("DEBUG Sentiment:", sentiment_score, sentiment_sources)
now = get_zurich_now()
if sentiment_score is not None:
    st.markdown(
        f"""**Aggregiertes Sentiment:** *{sentiment_score:+.2f}*
        <span title='Der Sentiment-Wert ist ein Mittelwert aus Nachrichtenstimmung (Finviz/Yahoo).
        Positiv = eher bullishe Nachrichten, Negativ = bearishe Nachrichten.
        Wertebereich: -1 bis +1.' style='vertical-align:middle; margin-left:2px; cursor:help;'>
            <svg width="16" height="16" viewBox="0 0 20 20" fill="none" style="vertical-align:-2px;">
                <circle cx="10" cy="10" r="9" fill="#1f77b4" />
                <text x="10" y="15" text-anchor="middle" font-size="13" fill="#fff" font-family="Arial" font-weight="bold">i</text>
            </svg>
        </span>""",
        unsafe_allow_html=True
    )

# ----- DF vorbereiten, LIVE-WERT ANHÄNGEN -----
now = pd.Timestamp.now(tz="Europe/Zurich").replace(second=0, microsecond=0)

def ensure_df():
    if "df" not in st.session_state or not isinstance(st.session_state.df, pd.DataFrame) or st.session_state.df.empty:
        return pd.DataFrame(columns=["Time", "Price"])
    df = st.session_state.df.copy()
    if "Time" not in df.columns:
        if df.index.name == "Time":
            df = df.reset_index()
        else:
            df = pd.DataFrame(columns=["Time", "Price"])
    return df

df = ensure_df()
if not df.empty and "Time" in df.columns:
    df["Time"] = pd.to_datetime(df["Time"], utc=True, errors="coerce")
    df = df[~df["Time"].isnull()]
    df["Time"] = df["Time"].dt.tz_convert("Europe/Zurich")
    df = df.sort_values("Time")
    df.set_index("Time", inplace=True)

if price is not None:
    if df.empty or df.index.size == 0:
        new_row = {"Price": price, "Time": now}
        df = pd.DataFrame([new_row]).set_index("Time")
    elif now > df.index.max():
        new_row = {"Price": price, "Time": now}
        df_new = pd.DataFrame([new_row]).set_index("Time")
        df = pd.concat([df, df_new])
        df = df[~df.index.duplicated(keep='last')]
    st.session_state.df = df


# ----- Prognose-Berechnung und -Ausgabe -----
prognose_text = "Seitwärts"
prognose_kurs = "N/A"
prognose_prozent = "N/A"
prognose_color = "gray"
if price is not None and os.path.exists(MODEL_FILE):
    data = joblib.load(MODEL_FILE)
    model = data["model"]
    selected_features = data["selected_features"]
    if len(df) > 0:
        feats_row = df.iloc[-1].to_dict()
        feats = calc_features(feats_row, sentiment_score, selected_features, finnhub_data, volatility_window)
        try:
            prognose_aktuell = float(model.predict([feats])[0])
            prognose_kurs = price * (1 + prognose_aktuell)
            prognose_prozent = prognose_aktuell * 100
            if prognose_aktuell > 0.02:
                prognose_text = "Steigend"
                prognose_color = "green"
            elif prognose_aktuell < -0.02:
                prognose_text = "Sinkend"
                prognose_color = "red"
            else:
                prognose_text = "Seitwärts"
                prognose_color = "gray"
        except Exception:
            prognose_kurs = "N/A"
            prognose_prozent = "N/A"
            prognose_text = "?"
            prognose_color = "gray"
prognose_kurs_display = f"{prognose_kurs:.2f}" if isinstance(prognose_kurs, (int, float)) else "N/A"
prognose_prozent_display = f"{prognose_prozent:+.2f}%" if isinstance(prognose_prozent, (int, float)) else "N/A"

st.markdown(
    f"<b>Prognose (nächste 5 Minuten):</b> "
    f"<span style='color:{prognose_color};'>{prognose_text}</span> | "
    f"Kursprognose: {prognose_kurs_display} | "
    f"Änderung: <span style='color:{prognose_color};'>{prognose_prozent_display}</span>",
    unsafe_allow_html=True
)

# ========================== HISTORY, LOG, SESSION ==========================
if "zeitintervall_idx" not in st.session_state:
    st.session_state.zeitintervall_idx = 3

if "forecast_vals" not in st.session_state:
    st.session_state.forecast_vals = []

if "history_loaded" not in st.session_state:
    try:
        hist = yf.download(tickers=TICKER, interval="1m", period="1d", progress=False)
        if not hist.empty:
            price_col = "Close" if "Close" in hist.columns else hist.columns[0]
            df_hist = hist[price_col].reset_index()
            df_hist.columns = ["Time", "Price"]
            df_hist["Time"] = pd.to_datetime(df_hist["Time"], utc=True)
            df_hist = df_hist[~df_hist["Time"].isnull()]
            df_hist["Time"] = df_hist["Time"].dt.tz_convert("Europe/Zurich")
            st.session_state.df = pd.concat([df_hist, st.session_state.df], ignore_index=True).drop_duplicates(subset="Time")
        st.session_state.history_loaded = True
    except Exception as e:
        st.warning(f"⚠️ Historische Minuten-Daten konnten nicht geladen werden: {e}")

repaired = repair_csv(LOG_FILES[0], REPAIRED_LOG_FILE)
log_to_read = REPAIRED_LOG_FILE if repaired else LOG_FILES[0]
if os.path.exists(log_to_read):
    try:
        df_log = pd.read_csv(log_to_read)
        df_log["Time"] = pd.to_datetime(df_log["Time"], utc=True, errors="coerce")
        df_log = df_log[~df_log["Time"].isnull()]
        df_log["Time"] = df_log["Time"].dt.tz_convert("Europe/Zurich")
    except Exception as e:
        st.error(f"Logdatei konnte nicht geladen werden: {e}")
        df_log = None
else:
    df_log = None

# ========================== INTERVALLAUSWAHL ==========================
st.markdown(
    '<div style="font-size:0.95em; margin-top:1em; margin-bottom:0.5em; color:#444;"><b>Zeitraum für Chart auswählen:</b></div>',
    unsafe_allow_html=True
)
btn_css = """
<style>
div.stButton > button,
button[data-baseweb="button"],
div.stButton > button *,
div.stButton > button::after,
div.stButton > button::before {
    font-size: 0.7rem !important;
    padding: 0.10rem 0.18rem !important;
    min-width: 28px !important;
    min-height: 1em !important;
    height: 1.2em !important;
    margin: 0 -2px 0 0;
    line-height: 1.1 !important;
    font-weight: 400;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
}
</style>
"""
st.markdown(btn_css, unsafe_allow_html=True)

col_buttons = st.columns(len(interval_options))
for i, (label, _) in enumerate(interval_options):
    with col_buttons[i]:
        if st.button(label, key=f"intervalbtn_{label}"):
            st.session_state.zeitintervall_idx = i

# ========================== DATENAUFBEREITUNG ==========================
df = st.session_state.df.copy()
if "Time" not in df.columns:
    if df.index.name == "Time":
        df = df.reset_index()
df["Time"] = pd.to_datetime(df["Time"], utc=True, errors="coerce")
df = df[~df["Time"].isnull()]
df["Time"] = df["Time"].dt.tz_convert("Europe/Zurich")
df = df.sort_values("Time")
df.set_index("Time", inplace=True)

# ========= Chart-Ausschnitt (x/y-Ranges) =========
delta = interval_options[st.session_state.zeitintervall_idx][1]
xaxis_range = None
yaxis_range = None

if delta is not None:
    latest_time = df.index.max()
    if latest_time is not pd.NaT and latest_time is not None:
        if latest_time.tzinfo is None:
            latest_time = latest_time.tz_localize("Europe/Zurich")
        start_time = latest_time - delta
        xaxis_range = [start_time, latest_time]
        df_visible = df[(df.index >= start_time) & (df.index <= latest_time)]
        ydata = []
        for col in ["Price", "SMA", "EMA", "BB_upper", "BB_lower"]:
            if col in df_visible:
                ydata.append(df_visible[col].dropna().values)
        if ydata:
            yvals = np.concatenate(ydata)
            yvals = yvals[np.isfinite(yvals)]
            ymin, ymax = np.nanmin(yvals), np.nanmax(yvals)
            yrange = ymax - ymin
            margin = max(0.5, yrange * 0.02)
            yaxis_range = [ymin - margin, ymax + margin]
        else:
            yaxis_range = None

# ----- Y-ACHSE: MANUELLE ÜBERSCHREIBUNG AUS SIDEBAR -----
if not use_autoscale and user_ymin is not None and user_ymax is not None:
    yaxis_range = [user_ymin, user_ymax]

# ========================== FORECAST & CHART ==========================
if len(df) > 30:
    if show_sma: df["SMA"] = df["Price"].rolling(window=sma_window).mean()
    if show_ema: df["EMA"] = compute_ema(df["Price"], window=ema_window)
    if show_rsi: df["RSI"] = compute_rsi(df["Price"], window=rsi_window)
    if show_macd: df["MACD"], df["MACD_signal"] = compute_macd(df["Price"], fast=macd_fast, slow=macd_slow, signal=macd_signal)
    if show_boll: df["SMA_Boll"], df["BB_upper"], df["BB_lower"] = compute_bollinger(df["Price"], window=boll_window, std=boll_std)
    if show_volatility: df["Volatility"] = df["Price"].rolling(volatility_window).std()
    if os.path.exists(MODEL_FILE):
        data = joblib.load(MODEL_FILE)
        model = data["model"]
        selected_features = data["selected_features"]
        df["Forecast"] = np.nan
        for idx, row in df.iterrows():
            feats = calc_features(row.to_dict(), sentiment_score, selected_features, finnhub_data, volatility_window)
            try:
                pred = float(model.predict([feats])[0])
            except Exception:
                pred = np.nan
            if not np.isnan(pred) and not np.isnan(row["Price"]):
                df.at[idx, "Forecast"] = row["Price"] * (1 + pred)
    else:
        df["Forecast"] = np.nan

    robust_live_logging(
        price=price,
        sentiment_score=sentiment_score,
        df=df,
        log_file=LOG_FILES[0],
        model_file=MODEL_FILE,
        finnhub_data=finnhub_data,
        volatility_window=volatility_window
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Price"], name="Preis", line=dict(width=2)))
    if show_sma and "SMA" in df: fig.add_trace(go.Scatter(x=df.index, y=df["SMA"], name="SMA", line=dict(width=1)))
    if show_ema and "EMA" in df: fig.add_trace(go.Scatter(x=df.index, y=df["EMA"], name="EMA", line=dict(width=1, color="#bb2222")))
    if show_boll and "BB_upper" in df and "BB_lower" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Upper", line=dict(dash="dot", width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Lower", line=dict(dash="dot", width=1)))
    if show_macd and "MACD" in df: fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", yaxis="y2", line=dict(width=1)))
    if show_rsi and "RSI" in df: fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", yaxis="y3", line=dict(width=1)))
    if show_volatility and "Volatility" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["Volatility"], name=f"Volatility ({volatility_window})", yaxis="y4", line=dict(width=1)))
    if "Forecast" in df and not df["Forecast"].isnull().all():
        fig.add_trace(go.Scatter(x=df.index, y=df["Forecast"], name="Forecast (5min)", line=dict(dash="dot", color="orange", width=2)))

    fig.update_layout(
        height=800,
        margin=dict(t=60, b=120, l=90, r=120),
        xaxis=dict(
            title="Zeit",
            title_font=dict(size=20),
            tickfont=dict(size=16),
            rangeslider=dict(visible=True, thickness=0.13, bgcolor='#e0e0e0'),
            range=xaxis_range if xaxis_range else None,
        ),
        yaxis=dict(
            title="Preis",
            title_font=dict(size=20),
            tickfont=dict(size=17),
            range=yaxis_range if yaxis_range else None,
        ),
        yaxis2=dict(
            title="MACD",
            overlaying="y",
            side="right",
            anchor="free",
            position=0.88,
            title_font=dict(size=18),
            tickfont=dict(size=15)
        ),
        yaxis3=dict(
            title="RSI",
            overlaying="y",
            side="right",
            anchor="free",
            position=0.93,
            title_font=dict(size=18),
            tickfont=dict(size=15)
        ),
        yaxis4=dict(
            title="Volatility",
            overlaying="y",
            side="right",
            anchor="free",
            position=0.98,
            title_font=dict(size=18),
            tickfont=dict(size=15)
        ),
        hovermode="x unified",
        uirevision="persistent",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.08,
            xanchor="center",
            x=0.5,
            font=dict(size=16)
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    if csv_export:
        st.download_button("Log als CSV", df.to_csv(index=True), file_name="log.csv", mime="text/csv")
else:
    st.warning("Noch nicht genügend Preisdaten für Analyse verfügbar.")

# ==== Statistik- und ML-Tabellen wie gehabt ====
import os
import numpy as np
import pandas as pd
import joblib
import datetime
from plotly.subplots import make_subplots

# --- ML-Prognose (Return!) erzeugen ---
if "Forecast" in df.columns and "Price" in df.columns:
    if (df["Forecast"] > 10).any():
        df["ML_Prognose"] = (df["Forecast"] / df["Price"]) - 1
    else:
        df["ML_Prognose"] = df["Forecast"]
else:
    df["ML_Prognose"] = np.nan

# --- Zielwert-Berechnung mit Handelsende-Logik ---
def add_target_return_for_timedelta_with_handelsende(
    df, delta, price_col="Price", target_col="target", close_time=datetime.time(22, 0)
):
    df = df.copy()
    df[target_col] = np.nan
    if df.empty:
        return df
    # Robust: times sicher zu DatetimeIndex casten
    if not isinstance(df.index, pd.DatetimeIndex):
        times = pd.to_datetime(df["Time"], errors="coerce")
    else:
        times = df.index
    times = pd.DatetimeIndex(times)
    delta = pd.to_timedelta(delta) if delta is not None else None
    for i in range(len(df)):
        t0 = times[i]
        if pd.isnull(t0):
            continue
        if delta is not None:
            t1 = t0 + delta
            if t1.time() > close_time:
                continue
            future_idx = np.where(times >= t1)[0]
            if len(future_idx) > 0:
                j = future_idx[0]
                df.iloc[i, df.columns.get_loc(target_col)] = (
                    df.iloc[j][price_col] / df.iloc[i][price_col] - 1
                )
        else:
            t_tag = t0.date()
            mask = (times.date == t_tag) & (times.time <= close_time)
            if mask.any():
                t_end = times[mask][-1]
                df.at[t0, target_col] = (
                    df.loc[t_end][price_col] / df.loc[t0][price_col] - 1
                )
    return df

# --- Intervall-Statistik-Berechnung ---
prognose_ergebnisse = []
model_dict = {}  # <-- HIER initialisieren
realtime_preds = []  # Für alle Intervalle jeweils die aktuelle (letzte) Prognose

for label, delta in interval_options:
    df_target = add_target_return_for_timedelta_with_handelsende(df, delta)
    if not df_target.empty and df_target["target"].notna().sum() > 50:
        feature_cols = [col for col in df_target.columns if col not in ["Time", "Price", "target"]]
        train, test = time_series_train_test_split(df_target)
        model, preds, mae, hitrate, avg_pred, avg_true, std_pred = train_and_evaluate_regression(
            train, test, feature_cols, target_col="target"
        )
        model_dict[label] = model

    # Statistik für das Intervall
    valid = (~df_target["target"].isna()) & (~df_target["ML_Prognose"].isna())
    if valid.any():
        avg_prognose = df_target.loc[valid, "ML_Prognose"].mean()
        wahrscheinlichkeit = (df_target.loc[valid, "ML_Prognose"] > 0).mean()
        trefferquote = (
            np.sign(df_target.loc[valid, "ML_Prognose"])
            == np.sign(df_target.loc[valid, "target"])
        ).mean()
        sample_size = valid.sum()
    else:
        avg_prognose = wahrscheinlichkeit = trefferquote = sample_size = None

    prognose_ergebnisse.append(
        {
            "Intervall": label,
            "Durchschnitt": avg_prognose,
            "Wahrscheinlichkeit": wahrscheinlichkeit,
            "Trefferquote": trefferquote,
            "Samplegröße": sample_size,
        }
    )

# --- Reload Prognose (aktuell) für alle Intervalle via Modellvorhersage ---
# Annahmen:
# - model_dict: {'10 Sek': model_10s, '1 Min': model_1min, ...} (Modelle pro Intervall)
# - build_features(df) gibt den letzten Feature-Vektor für das Modell zurück
realtime_preds = []
for label, delta in interval_options:
    model = model_dict.get(label)
    if model is not None:
        # Features für letzten Datenpunkt bauen (ggf. anpassen!)
        features = build_features(df.iloc[[-1]])
        try:
            pred = model.predict(features)[0]
            realtime_preds.append(pred)
        except Exception:
            realtime_preds.append(None)
    else:
        realtime_preds.append(None)

# --- Plotly Chart mit ALLEN aktuellen Prognosen ("Reload Prognose (aktuell)") ---
def pct_label(vals, digits=1):
    return [f"{v:.{digits}f}%" if v is not None and not np.isnan(v) else "n/a" for v in vals]

x_vals = [d["Intervall"] for d in prognose_ergebnisse]
wahrsch = [d["Wahrscheinlichkeit"]*100 if d["Wahrscheinlichkeit"] is not None else None for d in prognose_ergebnisse]
trefferquote = [d["Trefferquote"]*100 if d["Trefferquote"] is not None else None for d in prognose_ergebnisse]
durchschnitt = [d["Durchschnitt"]*100 if d["Durchschnitt"] is not None else None for d in prognose_ergebnisse]
reload_all = [v*100 if v is not None else None for v in realtime_preds]

fig_bar = make_subplots(specs=[[{"secondary_y": True}]])

# Durchschnittliche ML-Prognose-Linie
fig_bar.add_trace(
    go.Scatter(
        x=x_vals,
        y=durchschnitt,
        name="Durchschnittliche ML-Prognose",
        mode="lines+markers+text",
        line=dict(color="#0057b8", width=4),
        marker=dict(size=10, color="#0057b8"),
        text=pct_label(durchschnitt),
        textposition="top center",
        textfont=dict(size=10, color="#0057b8", family="Arial"),
        opacity=1,
    ),
    secondary_y=False,
)

# Reload Prognose (aktuell) für ALLE Intervalle als gepunktete Linie
fig_bar.add_trace(
    go.Scatter(
        x=x_vals,
        y=reload_all,
        name="Reload Prognose (aktuell, alle Intervalle)",
        mode="lines+markers+text",
        line=dict(color="#b80057", width=3, dash="dot"),
        marker=dict(size=9, color="#b80057"),
        text=pct_label(reload_all),
        textposition="bottom center",
        textfont=dict(size=10, color="#b80057", family="Arial"),
        opacity=0.95,
    ),
    secondary_y=False,
)

# Wahrscheinlichkeit als Balken
fig_bar.add_trace(
    go.Bar(
        x=x_vals,
        y=wahrsch,
        name="ML Eintrittswahrscheinlichkeit",
        marker_color="rgba(34, 139, 34, 0.38)",
        opacity=0.38,
        width=0.65,
        text=pct_label(wahrsch),
        textposition="outside",
        textfont=dict(size=10, color="rgba(34,139,34,0.9)", family="Arial"),
    ),
    secondary_y=True,
)

# Trefferquote als Linie
fig_bar.add_trace(
    go.Scatter(
        x=x_vals,
        y=trefferquote,
        name="ML Trefferquote",
        mode="lines+markers+text",
        line=dict(color="orange", width=2, dash="dash"),
        marker=dict(size=7, color="orange", opacity=0.7),
        text=pct_label(trefferquote),
        textposition="bottom center",
        textfont=dict(size=10, color="orange", family="Arial"),
        opacity=0.7
    ),
    secondary_y=True,
)

fig_bar.update_layout(
    barmode='overlay',
    title="ML-Statistiken pro Intervall",
    xaxis_title="Intervall",
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.18,
        xanchor="center",
        x=0.5,
        font=dict(size=13)
    ),
    margin=dict(b=90),
    height=500
)
fig_bar.update_yaxes(
    title_text="Prognose [%]",
    secondary_y=False,
    range=[-3, 3],
    zeroline=True,
    zerolinewidth=2,
    zerolinecolor='#888'
)
fig_bar.update_yaxes(
    title_text="Wahrscheinlichkeit / Trefferquote [%]",
    secondary_y=True,
    range=[0, 100],
    tickformat=".0f'%'",
    color="orange"
)

st.plotly_chart(fig_bar, use_container_width=True)

# --- Tabelle unter dem Chart ---
last_price = df["Price"].iloc[-1] if not df.empty and "Price" in df.columns else np.nan
df_table = pd.DataFrame([
    {
        "Intervall": d["Intervall"],
        "Kursprognose": (
            "n/a" if d["Durchschnitt"] is None or np.isnan(d["Durchschnitt"]) or np.isnan(last_price)
            else f"{last_price * (1 + d['Durchschnitt']):.2f}"
        ),
        "Prognose (%)": (
            "n/a" if d["Durchschnitt"] is None or np.isnan(d["Durchschnitt"])
            else f"{d['Durchschnitt']*100:+.2f}%"
        ),
        "ML-Eintrittswahrscheinlichkeit (%)": (
            "n/a" if d["Wahrscheinlichkeit"] is None or np.isnan(d["Wahrscheinlichkeit"])
            else f"{d['Wahrscheinlichkeit']*100:.1f}%"
        ),
        "Trefferquote (%)": (
            "n/a" if d["Trefferquote"] is None or np.isnan(d["Trefferquote"])
            else f"{d['Trefferquote']*100:.1f}%"
        ),
        "Reload Prognose (aktuell)": (
            "n/a" if realtime_preds[i] is None or np.isnan(realtime_preds[i])
            else f"{realtime_preds[i]*100:+.2f}%"
        ),
    }
    for i, d in enumerate(prognose_ergebnisse)
])

df_table = df_table.fillna("n/a")
st.dataframe(df_table, use_container_width=True)

# --- Tabelle unter dem Chart ---
last_price = df["Price"].iloc[-1] if not df.empty and "Price" in df.columns else np.nan
df_table = pd.DataFrame([
    {
        "Intervall": d["Intervall"],
        "Kursprognose": (
            "n/a" if d["Durchschnitt"] is None or np.isnan(d["Durchschnitt"]) or np.isnan(last_price)
            else f"{last_price * (1 + d['Durchschnitt']):.2f}"
        ),
        "Prognose (%)": (
            "n/a" if d["Durchschnitt"] is None or np.isnan(d["Durchschnitt"])
            else f"{d['Durchschnitt']*100:+.2f}%"
        ),
        "ML-Eintrittswahrscheinlichkeit (%)": (
            "n/a" if d["Wahrscheinlichkeit"] is None or np.isnan(d["Wahrscheinlichkeit"])
            else f"{d['Wahrscheinlichkeit']*100:.1f}%"
        ),
        "Trefferquote (%)": (
            "n/a" if d["Trefferquote"] is None or np.isnan(d["Trefferquote"])
            else f"{d['Trefferquote']*100:.1f}%"
        ),
        "Reload Prognose (aktuell)": (
            "n/a" if realtime_preds[i] is None or np.isnan(realtime_preds[i])
            else f"{realtime_preds[i]*100:+.2f}%"
        ),
    }
    for i, d in enumerate(prognose_ergebnisse)
])

df_table = df_table.fillna("n/a")
st.dataframe(df_table, use_container_width=True)

# --- Tabelle unter dem Chart, NUR EINMAL anzeigen! ---
last_price = df["Price"].iloc[-1] if not df.empty and "Price" in df.columns else np.nan
df_table = pd.DataFrame([
    {
        "Intervall": d["Intervall"],
        "Kursprognose": (
            "n/a" if d["Durchschnitt"] is None or np.isnan(d["Durchschnitt"]) or np.isnan(last_price)
            else f"{last_price * (1 + d['Durchschnitt']):.2f}"
        ),
        "Prognose (%)": (
            "n/a" if d["Durchschnitt"] is None or np.isnan(d["Durchschnitt"])
            else f"{d['Durchschnitt']*100:+.2f}%"
        ),
        "ML-Eintrittswahrscheinlichkeit (%)": (
            "n/a" if d["Wahrscheinlichkeit"] is None or np.isnan(d["Wahrscheinlichkeit"])
            else f"{d['Wahrscheinlichkeit']*100:.1f}%"
        ),
        "Trefferquote (%)": (
            "n/a" if d["Trefferquote"] is None or np.isnan(d["Trefferquote"])
            else f"{d['Trefferquote']*100:.1f}%"
        ),
    }
    for d in prognose_ergebnisse
])

df_table = df_table.fillna("n/a")
st.dataframe(df_table, use_container_width=True)

# --- Robustere Auswertung für kleine Intervalle ---
# Tipp: Stelle sicher, dass beim Erstellen von prognose_ergebnisse auch für kleine Intervalle,
#       z.B. len(df_interval) < 2, trotzdem möglichst Werte berechnet oder wenigstens 'n/a' gesetzt werden.

# --- Debug-Ausgaben nur wenn Checkbox aktiviert ---
if debug_show:
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Debug-Infos")
    st.sidebar.write("Aktuelle Zeit (Europe/Zurich):", pd.Timestamp.now(tz="Europe/Zurich"))
    if isinstance(df, pd.DataFrame) and not df.empty:
        st.sidebar.write("Letzter Zeitstempel im DataFrame:", df.index.max())
        st.sidebar.write("Letzte Zeile:", df.tail(1))
        st.sidebar.write("Zeitbereich im Chart:", df.index.min(), "bis", df.index.max())
        st.write("Zeitbereich im Chart:", df.index.min(), "bis", df.index.max())
        st.write("Zeitbereich im sichtbaren Chart:", df_visible.index.min(), "bis", df_visible.index.max())
        st.write("Aktuelle Zeit (Europe/Zurich):", pd.Timestamp.now(tz="Europe/Zurich"))
        st.write("Letzter Zeitstempel im DataFrame:", df.index.max())
        st.write("Letzte Zeile:", df.tail(1))
        if xaxis_range is not None:
            st.sidebar.write("Zeitbereich im sichtbaren Chart:", xaxis_range[0], "bis", xaxis_range[1])
    else:
        st.sidebar.info("DataFrame ist leer oder nicht initialisiert.")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, confusion_matrix, roc_auc_score
import plotly.graph_objs as go

# --- 1. Feature Engineering & Target ---
def add_features(df, rsi_window=14, ema_window=10, macd_fast=12, macd_slow=26, macd_signal=9, boll_window=20, boll_std=2, volatility_window=10, sentiment=None, finnhub_data=None):
    df = df.copy()
    df["RSI"] = compute_rsi(df["Price"], window=rsi_window)
    df["EMA"] = compute_ema(df["Price"], window=ema_window)
    df["MACD"], df["MACD_signal"] = compute_macd(df["Price"], fast=macd_fast, slow=macd_slow, signal=macd_signal)
    _, df["BB_upper"], df["BB_lower"] = compute_bollinger(df["Price"], window=boll_window, std=boll_std)
    df["Volatility"] = df["Price"].rolling(volatility_window).std()
    if sentiment is not None:
        df["Sentiment"] = sentiment
    if finnhub_data is not None:
        for key, val in finnhub_data.items():
            df[f"Finnhub_{key}"] = val
    return df

# --- Zielspalten über echtes Zeitintervall ---
def add_target_return_for_timedelta(df, delta, price_col="Price", target_col="target"):
    import pandas as pd
    import numpy as np
    df = df.copy()
    df[target_col] = np.nan

    # Robust: Index zu pd.DatetimeIndex machen, falls nötig
    if not isinstance(df.index, pd.DatetimeIndex):
        times = pd.to_datetime(df["Time"])
    else:
        times = df.index

    # Robust: delta zu pd.Timedelta machen, falls nötig
    if not isinstance(delta, pd.Timedelta):
        delta = pd.to_timedelta(delta)

    for i in range(len(df)):
        t0 = times[i]
        if not isinstance(t0, pd.Timestamp):
            t0 = pd.Timestamp(t0)
        # t0 + delta ist jetzt sicher!
        future_idx = np.where(times > t0 + delta)[0]
        if len(future_idx) > 0:
            j = future_idx[0]
            df.iloc[i, df.columns.get_loc(target_col)] = df.iloc[j][price_col] / df.iloc[i][price_col] - 1
    return df

def add_target_classification_for_timedelta(df, delta: pd.Timedelta, threshold=0.002, price_col="Price"):
    df = add_target_return_for_timedelta(df, delta, price_col=price_col, target_col="target")
    df["target_class"] = pd.cut(
        df["target"],
        [-np.inf, -threshold, threshold, np.inf],
        labels=["Down", "Neutral", "Up"]
    )
    return df

import datetime
def add_target_return_for_timedelta_with_handelsende(df, delta, price_col="Price", target_col="target", close_time=datetime.time(22, 0)):
    df = df.copy()
    df[target_col] = np.nan
    if df.empty:
       return df
    times = pd.to_datetime(df["Time"]) if not isinstance(df.index, pd.DatetimeIndex) else df.index
    delta = pd.to_timedelta(delta) if delta is not None else None
    for i in range(len(df)):
        t0 = times[i]
        if delta is not None:
            t1 = t0 + delta
            # WENN Ziel nach Handelsschluss, dann kein Label!
            if t1.time() > close_time:
                continue
            # Finde ersten Index >= t1
            future_idx = np.where(times >= t1)[0]
            if len(future_idx) > 0:
                j = future_idx[0]
                df.iloc[i, df.columns.get_loc(target_col)] = df.iloc[j][price_col] / df.iloc[i][price_col] - 1
        else:
            # Spezialfall: Bis Handelsende
            t_tag = t0.date()
            mask = (times.date == t_tag) & (times.time <= close_time)
            if mask.any():
                t_end = times[mask][-1]
                df.at[t0, target_col] = df.loc[t_end][price_col] / df.loc[t0][price_col] - 1
    return df

# --- 2. Zeitreihen-Split ---
def time_series_train_test_split(df, test_size=0.2):
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test

# --- 3. Modell-Training & Bewertung ---
def train_and_evaluate_regression(train, test, feature_cols, target_col="target"):
    X_train, y_train = train[feature_cols], train[target_col]
    X_test, y_test = test[feature_cols], test[target_col]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    hitrate = np.mean(np.sign(y_test) == np.sign(preds))
    avg_pred = np.mean(preds)
    avg_true = np.mean(y_test)
    std_pred = np.std(preds)
    return model, preds, mae, hitrate, avg_pred, avg_true, std_pred

def train_and_evaluate_classification(train, test, feature_cols, target_col="target_class"):
    from sklearn.ensemble import RandomForestClassifier
    X_train, y_train = train[feature_cols], train[target_col]
    X_test, y_test = test[feature_cols], test[target_col]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds, labels=["Up", "Neutral", "Down"])
    try:
        roc_auc = roc_auc_score(pd.get_dummies(y_test), pd.get_dummies(preds), multi_class="ovo")
    except Exception:
        roc_auc = np.nan
    accuracy = np.mean(y_test == preds)
    return model, preds, cm, accuracy, roc_auc

# --- 4. Best Practice Reporting (Streamlit, mit echten Zeitintervallen) ---
def best_practice_report(
    df,
    sentiment=None,
    finnhub_data=None,
    rsi_window=14,
    ema_window=10,
    macd_fast=12,
    macd_slow=26,
    macd_signal=9,
    boll_window=20,
    boll_std=2,
    volatility_window=10,
    regression_intervals=None,
    classification=True,
    class_threshold=0.002
):
    import streamlit as st
    from plotly.subplots import make_subplots
    import plotly.graph_objs as go
    import pandas as pd
    import numpy as np

    if regression_intervals is None:
        regression_intervals = [(label, delta) for label, delta in interval_options if delta is not None]

    all_results = []
    all_results_clf = []
    for label, delta in regression_intervals:
        if delta is None:
            continue
        # --- Feature Engineering ---
        df_feat = add_features(
            df,
            rsi_window,
            ema_window,
            macd_fast,
            macd_slow,
            macd_signal,
            boll_window,
            boll_std,
            volatility_window,
            sentiment=sentiment,
            finnhub_data=finnhub_data
        )
        # --- Regression: Ziel als echter Zeitabstand ---
        df_reg = add_target_return_for_timedelta(df_feat, delta).dropna(subset=["target"])
        if len(df_reg) < 50:
            continue
        train, test = time_series_train_test_split(df_reg)
        feature_cols = [c for c in df_reg.columns if c not in ["Time", "Price", "target", "target_class"]]
        model, preds, mae, hitrate, avg_pred, avg_true, std_pred = train_and_evaluate_regression(train, test, feature_cols)
        all_results.append({
            "Intervall": label,
            "MAE": mae,
            "Trefferquote": hitrate,
            "Prognose Mittelwert": avg_pred,
            "Real Mittelwert": avg_true,
            "Prognose Std": std_pred
        })
        # --- Klassifikation: Ziel als echter Zeitabstand ---
        if classification:
            df_clf = add_target_classification_for_timedelta(df_feat, delta, threshold=class_threshold).dropna(subset=["target_class"])
            if len(df_clf) < 50:
                continue
            trainc, testc = time_series_train_test_split(df_clf)
            modelc, predc, cm, acc, roc_auc = train_and_evaluate_classification(trainc, testc, feature_cols, target_col="target_class")
            all_results_clf.append({
                "Intervall": label,
                "Accuracy": acc,
                "ROC AUC": roc_auc,
                "ConfusionMatrix": cm
            })

    # --- Regression Reporting ---
    df_report = pd.DataFrame(all_results)
    if not df_report.empty:
        # Werte schön formatieren
        def fmt_pct(x):
            return f"{x*100:+.2f}%" if pd.notnull(x) else "n/a"
        def fmt_val(x):
            return f"{x:.2f}" if pd.notnull(x) else "n/a"

        df_report_disp = pd.DataFrame({
            "Intervall": df_report["Intervall"],
            "MAE (Fehler)": df_report["MAE"].apply(fmt_pct),
            "Trefferquote (%)": df_report["Trefferquote"].apply(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "n/a"),
            "Forecast Ø (%)": df_report["Prognose Mittelwert"].apply(fmt_pct),
            "Real Ø (%)": df_report["Real Mittelwert"].apply(fmt_pct),
            "Forecast Std (%)": df_report["Prognose Std"].apply(fmt_pct),
        })

        st.write("### [Best Practice] ML-Regression-Report (Zeitfenster-Daten)")
        st.dataframe(df_report_disp, use_container_width=True)

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Linke Y-Achse (Balken)
        fig.add_trace(
            go.Bar(
                x=df_report["Intervall"],
                y=df_report["MAE"],
                name="MAE (Fehler)",
                marker_color="#1f77b4"
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Bar(
                x=df_report["Intervall"],
                y=df_report["Prognose Mittelwert"],
                name="Forecast Ø",
                marker_color="#2ca02c"
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Bar(
                x=df_report["Intervall"],
                y=df_report["Real Mittelwert"],
                name="Real Ø",
                marker_color="#d62728"
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Bar(
                x=df_report["Intervall"],
                y=df_report["Prognose Std"],
                name="Forecast Std",
                marker_color="#9467bd"
            ),
            secondary_y=False,
        )

        # Rechte Y-Achse (Trefferquote als Linie)
        fig.add_trace(
            go.Scatter(
                x=df_report["Intervall"],
                y=df_report["Trefferquote"],
                name="Trefferquote",
                mode="lines+markers",
                line=dict(color="orange", width=4),
                marker=dict(size=10, color="orange"),
            ),
            secondary_y=True,
        )

        fig.update_layout(
            barmode='group',
            title="ML-Performance je Intervall (Daten, echtes Zeitfenster)",
            xaxis_title="Intervall",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=14)
            ),
            margin=dict(b=90),
            bargap=0.13,
            height=600
        )
        fig.update_yaxes(
            title_text="Fehler/Mittelwerte",
            secondary_y=False
        )
        fig.update_yaxes(
            title_text="Trefferquote",
            secondary_y=True,
            range=[0,1],
            tickformat=".0%",
            color="orange"
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Nicht genügend Daten für Best-Practice-ML-Regression-Statistik.")

    # --- Klassifikation Reporting ---
    if classification and all_results_clf:
        st.write("### [Best Practice] ML-Klassifikation-Report (Daten)")
        for item in all_results_clf:
            st.write(f"**{item['Intervall']}**: Accuracy: {item['Accuracy']:.2f}, ROC AUC: {item['ROC AUC']:.2f}")
            st.write("Confusion Matrix (Up/Neutral/Down):")
            st.write(item["ConfusionMatrix"])

if show_bp_report:
    intervals_for_report = [(label, delta) for (label, delta) in interval_options if delta is not None]
    best_practice_report(
        df=df,
        sentiment=sentiment_score if "sentiment_score" in locals() else None,
        finnhub_data=finnhub_data if "finnhub_data" in locals() else None,
        rsi_window=rsi_window,
        ema_window=ema_window,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
        boll_window=boll_window,
        boll_std=boll_std,
        volatility_window=volatility_window,
        regression_intervals=intervals_for_report,
        classification=True
    )
'''

# 📄 app.py schreiben
if os.path.exists(APP_FILENAME): os.remove(APP_FILENAME)
with open(APP_FILENAME, "w") as f:
    f.write(app_code)
print("✅ app.py erfolgreich erstellt.")

# Starte Streamlit + ngrok Tunnel

conf.get_default().auth_token = NGROK_TOKEN
try: ngrok.kill()
except: pass

print("🚀 Starte Streamlit...")
streamlit_proc = subprocess.Popen(["streamlit", "run", APP_FILENAME])
time.sleep(2)

def port_is_open(port=APP_PORT):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("localhost", port)) == 0

print("⏳ Warte auf Port 8501...")
for i in range(60):
    if port_is_open():
        print("✅ Streamlit läuft.")
        break
    time.sleep(2)
else:
    raise RuntimeError("❌ Fehler: Port 8501 nicht aktiv.")

print("🌍 Starte ngrok...")
public_url = ngrok.connect(APP_PORT)
print(f"🔗 App erreichbar unter:\n{public_url}")


#runtest
!ps aux | grep streamlit
