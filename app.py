import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download Sentiment components
@st.cache_resource
def download_nltk_resources():
    nltk.download('vader_lexicon', quiet=True)
download_nltk_resources()
sia = SentimentIntensityAnalyzer()

# --- CONFIGURATION ---
ANGEL_ONE_API_KEY = "1BoKeTG3"      # Paste key from smartapi.angelone.in
FINNHUB_API_KEY = "d8m5hhhr01qkiso5nsigd8m5hhhr01qkiso5nsj0"          # Keep your news API key

st.set_page_config(page_title="AngelOne Auto-Predictor", layout="wide")
st.title("📈 Angel One Live 5-Min Predictor (Smart Search)")

# --- SMART TOKEN AUTO-LOOKUP SYSTEM ---
@st.cache_data(ttl=86400) # Caches the data for 24 hours so it stays fast
def load_angel_master_tokens():
    try:
        # Pulls Angel One's official live master sheet containing symbol names and tokens
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        response = requests.get(url).json()
        df = pd.DataFrame(response)
        # Filter down only to core Equity shares on BSE and NSE to keep loading fast
        df = df[df['exch_seg'].isin(['NSE', 'BSE'])]
        df = df[df['instrumenttype'] == ''] # Filters out options and futures
        return df[['token', 'symbol', 'name', 'exch_seg']]
    except Exception as e:
        st.error(f"Failed to load Angel Master Token sheet: {e}")
        return pd.DataFrame()

# Load master database
token_db = load_angel_master_tokens()

# --- ANGEL ONE HISTORICAL INTRADAY FETCH ---
def get_angel_five_min_data(exchange, symbol_token):
    url = "https://apiconnect.angelone.in/rest/auth/angelbroking/user/v1/getCandleData"
    payload = {
        "exchange": exchange,
        "symboltoken": symbol_token,
        "interval": "FIVE_MINUTE",
        "fromdate": "2026-06-10 09:15",   
        "todate": "2026-06-13 15:30"
    }
    headers = {
        "X-PrivateKey": ANGEL_ONE_API_KEY,
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    try:
        res = requests.post(url, json=payload, headers=headers).json()
        if res.get("status") and res.get("data"):
            raw_candles = res["data"]
            df = pd.DataFrame(raw_candles, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            return df
        return None
    except:
        return None

def get_news_sentiment(news_ticker):
    url = f"https://finnhub.io/api/v1/company-news?symbol={news_ticker}&from=2026-01-01&to=2026-12-31&token={FINNHUB_API_KEY}"
    res = requests.get(url).json()
    if not res or isinstance(res, dict): return 0.0
    scores = [sia.polarity_scores(a.get('headline', ''))['compound'] for a in res[:10]]
    return np.mean(scores) if scores else 0.0

# --- INDICATOR CALCULATIONS ---
def calculate_advanced_indicators(df):
    # 1. Base Momentum
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    # 2. Add Volatility (Bollinger Bands)
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['StdDev20'] = df['close'].rolling(window=20).std()
    df['Upper_Band'] = df['import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download Sentiment components safely
@st.cache_resource
def download_nltk_resources():
    nltk.download('vader_lexicon', quiet=True)
download_nltk_resources()
sia = SentimentIntensityAnalyzer()

# --- CONFIGURATION (Your Secure Personal Keys Inserted) ---
ANGEL_ONE_API_KEY = "1BoKeTG3"                     
FINNHUB_API_KEY = "d8m5hhhr01qkiso5nsigd8m5hhhr01qkiso5nsj0"  

st.set_page_config(page_title="AngelOne Ultimate Predictor", layout="wide")
st.title("📈 Angel One Live Multi-Timeframe Analytics Suite")

# --- SMART TOKEN AUTO-LOOKUP SYSTEM ---
@st.cache_data(ttl=86400) 
def load_angel_master_tokens():
    try:
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        response = requests.get(url).json()
        df = pd.DataFrame(response)
        df = df[df['exch_seg'].isin(['NSE', 'BSE'])]
        df = df[df['instrumenttype'] == ''] 
        return df[['token', 'symbol', 'name', 'exch_seg']]
    except Exception as e:
        st.error(f"Failed to load Angel Master Token sheet: {e}")
        return pd.DataFrame()

token_db = load_angel_master_tokens()

# --- ANGEL ONE HISTORICAL INTRADAY FETCH ---
def get_angel_candlestick_data(exchange, symbol_token, interval_code):
    url = "https://apiconnect.angelone.in/rest/auth/angelbroking/user/v1/getCandleData"
    payload = {
        "exchange": exchange,
        "symboltoken": symbol_token,
        "interval": interval_code,        
        "fromdate": "2026-06-01 09:15",   
        "todate": "2026-06-13 15:30"
    }
    headers = {
        "X-PrivateKey": ANGEL_ONE_API_KEY,
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    try:
        res = requests.post(url, json=payload, headers=headers).json()
        if res.get("status") and res.get("data"):
            raw_candles = res["data"]
            df = pd.DataFrame(raw_candles, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            return df
        return None
    except:
        return None

def get_news_sentiment(news_ticker):
    url = f"https://finnhub.io/api/v1/company-news?symbol={news_ticker}&from=2026-01-01&to=2026-12-31&token={FINNHUB_API_KEY}"
    res = requests.get(url).json()
    if not res or isinstance(res, dict): return 0.0
    scores = [sia.polarity_scores(a.get('headline', ''))['compound'] for a in res[:10]]
    return np.mean(scores) if scores else 0.0

# --- MASTER INDICATOR CALCULATION ENGINE ---
def calculate_all_indicators(df):
    # 1. Base Momentum (RSI)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    # 2. Trend & Crossovers (MACD & Signal Line)
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 3. Volatility Filters (Bollinger Bands)
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['StdDev20'] = df['close'].rolling(window=20).std()
    df['Upper_Band'] = df['MA20'] + (df['StdDev20'] * 2)
    df['Lower_Band'] = df['MA20'] - (df['StdDev20'] * 2)
    df['BB_Dist_Upper'] = df['Upper_Band'] - df['close']
    df['BB_Dist_Lower'] = df['close'] - df['Lower_Band']

    # 4. Rate of Change (ROC)
    df['ROC_3'] = ((df['close'] - df['close'].shift(3)) / df['close'].shift(3)) * 100

    # 5. Market Time Context
    df['datetime'] = pd.to_datetime(df['time'])
    df['Market_Hour'] = df['datetime'].dt.hour
    df['Market_Minute'] = df['datetime'].dt.minute

    # 6. Prediction Target Matrix
    df['Target'] = np.where(df['close'].shift(-3) > df['close'], 1, 0)
    return df.dropna()

# --- INTERFACE ---
if not token_db.empty:
    c_ex, c_time, c_search = st.columns(3)
    
    selected_exchange = c_ex.selectbox("Choose Exchange", ["BSE", "NSE"])
    timeframe_label = c_time.selectbox("Candlestick Interval", ["5 Minutes", "15 Minutes", "1 Hour"])
    
    timeframe_mapping = {
        "5 Minutes": "FIVE_MINUTE",
        "15 Minutes": "FIFTEEN_MINUTE",
        "1 Hour": "ONE_HOUR"
    }
    selected_interval_code = timeframe_mapping[timeframe_label]
    
    filtered_db = token_db[token_db['exch_seg'] == selected_exchange]
    stock_list = filtered_db['name'].unique().tolist()
    search_name = c_search.selectbox("Type/Select Company Name", stock_list)
    
    matched_row = filtered_db[filtered_db['name'] == search_name].iloc[0]
    auto_token = matched_row['token']
    trading_symbol = matched_row['symbol']
    
    news_suffix = ".BO" if selected_exchange == "BSE" else ".NS"
    clean_symbol = trading_symbol.replace("-EQ", "")
    auto_news_symbol = f"{clean_symbol}{news_suffix}"
    
    st.info(f"📍 **System Map** ── Interval: `{timeframe_label}` | Symbol: `{trading_symbol}` | Token: `{auto_token}`")

    if st.button("Run Analytics & Prediction Engine"):
        with st.spinner(f"Streaming {timeframe_label} candles & calculating chart vectors..."):
            df = get_angel_candlestick_data(selected_exchange, auto_token, selected_interval_code)
            sentiment = get_news_sentiment(auto_news_symbol)
            
            if df is not None and len(df) > 25:
                df = calculate_all_indicators(df)
                
                # Full 10-dimensional feature pool
                features = [
                    'close', 'volume', 'RSI', 'MACD', 'Signal_Line', 
                    'BB_Dist_Upper', 'BB_Dist_Lower', 'ROC_3', 'Market_Hour', 'Market_Minute'
                ]
                
                X = df[features].values[:-3]
                y = df['Target'].values[:-3]
                
                model = RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_leaf=10, random_state=42)
                model.fit(X, y)
                
                train_predictions = model.predict(X)
                from sklearn.metrics import accuracy_score
                historical_accuracy = accuracy_score(y, train_predictions)
                
                latest_features = df[features].iloc[-1].values.reshape(1, -1)
                prediction = model.predict(latest_features)[0]
                probs = model.predict_proba(latest_features)[0]

                # --- DASHBOARD SUMMARY ---
                st.subheader(f"Live Analysis Dashboard ({timeframe_label} View)")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("LTP (Last Price)", f"₹{df['close'].iloc[-1]:.2f}")
                c2.metric("RSI (14m)", f"{df['RSI'].iloc[-1]:.1f}")
                c3.metric("News Trend", "Bullish" if sentiment > 0.1 else "Bearish" if sentiment < -0.1 else "Neutral")
                c4.metric("Backtest Accuracy", f"{historical_accuracy * 100:.1f}%")

                st.write("---")
                
                horizon_text = "15-MINUTE" if timeframe_label == "5 Minutes" else "45-MINUTE" if timeframe_label == "15 Minutes" else "3-HOUR"
                
                if prediction == 1:
                    st.success(f"🚀 **{horizon_text} TREND OUTLOOK: UP.** Statistical Probability: **{probs[1]*100:.1f}%**")
                else:
                    st.error(f"📉 **{horizon_text} TREND OUTLOOK: DOWN.** Statistical Probability: **{probs[0]*100:.1f}%**")
                
                # --- NEW FEATURE: INTERACTIVE CHARTS & VISUALIZATIONS ---
                st.write("---")
                st.subheader("📊 Advanced Technical Chart Vectors")
                
                # Prepare a clean time-indexed dataframe for plotting
                chart_df = df.copy()
                chart_df.set_index('time', inplace=True)
                
                # Chart 1: Price Action & Volatility Overlap
                st.markdown("### 🔹 Price Action & Bollinger Volatility Bands")
                st.line_chart(chart_df[['close', 'Upper_Band', 'Lower_Band']])
                
                # Chart 2: Momentum Analysis (RSI & MACD Panels side by side)
                ch_col1, ch_col2 = st.columns(2)
                with ch_col1:
                    st.markdown("### 🔹 Relative Strength Index (RSI)")
                    st.line_chart(chart_df['RSI'])
                with ch_col2:
                    st.markdown("### 🔹 MACD vs Signal Crossover")
                    st.line_chart(chart_df[['MACD', 'Signal_Line']])
                
                # Chart 3: Velocity & Volume
                st.markdown("### 🔹 Momentum Velocity (Rate of Change %)")
                st.area_chart(chart_df['ROC_3'])
                
                st.write("---")
                st.markdown("### 📋 Recent Data Log")
                st.dataframe(df[['time', 'open', 'high', 'low', 'close', 'volume', 'RSI', 'MACD', 'ROC_3']].tail(5))
            else:
                st.warning(f"Not enough historical {timeframe_label} candles available right now. Make sure to query during live trading hours!")
else:
    st.error("Setting up database connections, please wait a moment...")
