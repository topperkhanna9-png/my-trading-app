 import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download Sentiment Analysis components
@st.cache_resource
def download_nltk_resources():
    nltk.download('vader_lexicon', quiet=True)
download_nltk_resources()
sia = SentimentIntensityAnalyzer()

# --- CONFIGURATION (Your Secure Access Keys) ---
ANGEL_ONE_API_KEY = "1BoKeTG3"      # Paste key from smartapi.angelone.in
FINNHUB_API_KEY = "d8m5hhhr01qkiso5nsigd8m5hhhr01qkiso5nsj0"          # Keep your news API key

st.set_page_config(page_title="AngelOne 5m Predictor", layout="wide")
st.title("📈 Angel One Live 5-Min Predictor")

# --- ANGEL ONE HISTORICAL INTRADAY FETCH ---
def get_angel_five_min_data(exchange, symbol_token):
    # This reaches directly into Angel One's high-speed historical intraday routers
    url = "https://apiconnect.angelone.in/rest/auth/angelbroking/user/v1/getCandleData"
    
    # We fetch data spanning today and yesterday to capture 5-min bars
    payload = {
        "exchange": exchange,             # "BSE" or "NSE"
        "symboltoken": symbol_token,       # E.g., "500112" for SBI on BSE
        "interval": "FIVE_MINUTE",
        "fromdate": "2026-06-10 09:15",   # Programmatic window
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
            # Format: [Time, Open, High, Low, Close, Volume]
            raw_candles = res["data"]
            df = pd.DataFrame(raw_candles, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            return df
        else:
            st.error(f"Data Fetch Failed: {res.get('message', 'Unknown Error')}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {e}")
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
col_ex, col_tok, col_news = st.columns(3)
exchange = col_ex.selectbox("Exchange", ["BSE", "NSE"])
token = col_tok.text_input("Angel Token ID (e.g., 500112 for SBIN on BSE)", "500112")
news_symbol = col_news.text_input("Global News Ticker (e.g., SBIN.BO)", "SBIN.BO")

if st.button("Run Intraday Prediction Engine"):
    with st.spinner("Streaming real-time candles from Angel One..."):
        df = get_angel_five_min_data(exchange, token)
        sentiment = get_news_sentiment(news_symbol)
        
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

            st.subheader(f"Live Analysis (Token: {token} via {exchange})")
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
            st.warning("Insufficient intraday candles found for this token right now. Try checking during live market hours!")
