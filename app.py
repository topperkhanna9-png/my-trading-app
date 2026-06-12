import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Initialize NLTK for Live News Sentiment Analysis
@st.cache_resource
def download_nltk_resources():
    nltk.download('vader_lexicon', quiet=True)

download_nltk_resources()
sia = SentimentIntensityAnalyzer()

# --- CONFIGURATION (Use Free-Tier API Keys) ---
# Get free keys from twelvedata.com and finnhub.io
TWELVE_DATA_API_KEY = "8c68a32bd673433f885dc28196958f79" 
FINNHUB_API_KEY = "d8m5hhhr01qkiso5nsigd8m5hhhr01qkiso5nsj0"

st.set_page_config(page_title="Intraday 5m Predictor", layout="wide")
st.title("📈 Intraday 5-Min Candlestick Predictor")

# --- DATA FETCHING ---
def get_stock_data(ticker):
    # Fetches 5-minute interval OHLCV data
    url = f"https://api.twelvedata.com/time_series?symbol={ticker}&interval=5min&outputsize=100&apikey={TWELVE_DATA_API_KEY}"
    res = requests.get(url).json()
    if "values" not in res:
        st.error("Error fetching data. Check your API Key or Ticker.")
        return None
    df = pd.DataFrame(res['values'])
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)
    df = df.iloc[::-1].reset_index(drop=True) # Chronological order
    return df

def get_news_sentiment(ticker):
    # Fetches real-time market news for sentiment features
    url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from=2026-01-01&to=2026-12-31&token={FINNHUB_API_KEY}"
    res = requests.get(url).json()
    if not res or isinstance(res, dict):
        return 0.0 # Neutral if error or no news
    
    scores = []
    for article in res[:10]: # Analyze top 10 recent headlines
        headline = article.get('headline', '')
        scores.append(sia.polarity_scores(headline)['compound'])
    return np.mean(scores) if scores else 0.0

# --- FEATURE ENGINEERING ---
def calculate_indicators(df):
    # Intraday Strategy: RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    # Intraday Strategy: MACD (Moving Average Convergence Divergence)
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Volume Factor: Volume Relative to Moving Average
    df['Vol_MA'] = df['volume'].rolling(window=10).mean()
    df['Volume_Spike'] = np.where(df['volume'] > (df['Vol_MA'] * 1.5), 1, 0)

    # Target Variable: Next Close > Current Close (1 = Up, 0 = Down)
    df['Target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    return df.dropna()

# --- UI CONTROLS ---
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, INFY):", "AAPL").upper()

if st.button("Analyze & Predict"):
    with st.spinner("Analyzing market patterns and news sentiment..."):
        df = get_stock_data(ticker)
        sentiment = get_news_sentiment(ticker)
        
        if df is not None:
            df = calculate_indicators(df)
            
            # Split features and train a light Random Forest model
            features = ['close', 'volume', 'RSI', 'MACD', 'Signal_Line', 'Volume_Spike']
            X = df[features].values[:-1] # Drop last row since its target is unknown
            y = df['Target'].values[:-1]
            
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X, y)
            
            # Predict for the absolute latest, live 5-minute candle
            latest_features = df[features].iloc[-1].values.reshape(1, -1)
            prediction = model.predict(latest_features)[0]
            probs = model.predict_proba(latest_features)[0]

            # Display Results
            st.subheader(f"Analysis for {ticker} (Latest 5m Bar)")
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${df['close'].iloc[-1]:.2f}")
            col2.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.1f}")
            
            # Map sentiment value to text
            sent_text = "Bullish" if sentiment > 0.1 else "Bearish" if sentiment < -0.1 else "Neutral"
            col3.metric("News Sentiment Score", sent_text, delta=f"{sentiment:.2f}")

            st.write("---")
            if prediction == 1:
                st.success(f"🚀 **PREDICTION: UP.** Probability Next Candle Closes Higher: **{probs[1]*100:.1f}%**")
            else:
                st.error(f"📉 **PREDICTION: DOWN.** Probability Next Candle Closes Lower: **{probs[0]*100:.1f}%**")
                
            # Render a quick tabular view of historical data processed
            st.dataframe(df[['open', 'high', 'low', 'close', 'volume', 'RSI', 'MACD']].tail(5))
