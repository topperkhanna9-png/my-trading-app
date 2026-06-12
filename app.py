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
def calculate_indicators(df):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['Vol_MA'] = df['volume'].rolling(window=10).mean()
    df['Volume_Spike'] = np.where(df['volume'] > (df['Vol_MA'] * 1.5), 1, 0)
    df['Target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    return df.dropna()

# --- INTERFACE ---
if not token_db.empty:
    col_ex, col_search = st.columns(2)
    
    # 1. Select Exchange
    selected_exchange = col_ex.selectbox("Choose Exchange", ["BSE", "NSE"])
    
    # Filter token database by exchange choice
    filtered_db = token_db[token_db['exch_seg'] == selected_exchange]
    stock_list = filtered_db['name'].unique().tolist()
    
    # 2. Directly type company name with an autocompleting list
    search_name = col_search.selectbox("Type/Select Company Name (e.g., RELIANCE, TATA MOTORS)", stock_list)
    
    # Auto-extract Token & Trading Symbol details instantly
    matched_row = filtered_db[filtered_db['name'] == search_name].iloc[0]
    auto_token = matched_row['token']
    trading_symbol = matched_row['symbol']
    
    # Format news ticker automatically (e.g., SBIN.BO or SBIN.NS)
    news_suffix = ".BO" if selected_exchange == "BSE" else ".NS"
    clean_symbol = trading_symbol.replace("-EQ", "")
    auto_news_symbol = f"{clean_symbol}{news_suffix}"
    
    # Visual Confirmation on App
    st.info(f"📍 **Auto-Mapped Details** ── Symbol: `{trading_symbol}` | Token ID: `{auto_token}` | News Tracker: `{auto_news_symbol}`")

    if st.button("Run Intraday Prediction Engine"):
        with st.spinner(f"Streaming live candles for {search_name}..."):
            df = get_angel_five_min_data(selected_exchange, auto_token)
            sentiment = get_news_sentiment(auto_news_symbol)
            
            if df is not None and len(df) > 15:
                df = calculate_indicators(df)
                features = ['close', 'volume', 'RSI', 'MACD', 'Signal_Line', 'Volume_Spike']
                
                X = df[features].values[:-1]
                y = df['Target'].values[:-1]
                
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                model.fit(X, y)
                
                latest_features = df[features].iloc[-1].values.reshape(1, -1)
                prediction = model.predict(latest_features)[0]
                probs = model.predict_proba(latest_features)[0]

                st.subheader(f"Live Analysis for {search_name}")
                c1, c2, c3 = st.columns(3)
                c1.metric("LTP (Last Traded Price)", f"₹{df['close'].iloc[-1]:.2f}")
                c2.metric("RSI (14m)", f"{df['RSI'].iloc[-1]:.1f}")
                c3.metric("News Trend", "Bullish" if sentiment > 0.1 else "Bearish" if sentiment < -0.1 else "Neutral")

                st.write("---")
                if prediction == 1:
                    st.success(f"🚀 **5-MINUTE PREDICTION: UP.** Probability: **{probs[1]*100:.1f}%**")
                else:
                    st.error(f"📉 **5-MINUTE PREDICTION: DOWN.** Probability: **{probs[0]*100:.1f}%**")
                    
                st.dataframe(df[['time', 'open', 'high', 'low', 'close', 'volume', 'RSI']].tail(5))
            else:
                st.warning("Insufficient trading volume data returned for this window. Make sure you test during active market hours (9:15 AM - 3:30 PM)!")
else:
    st.error("Setting up database connections, please wait a moment...")
