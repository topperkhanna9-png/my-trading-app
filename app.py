import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Download Sentiment components safely
@st.cache_resource
def download_nltk_resources():
    nltk.download('vader_lexicon', quiet=True)
download_nltk_resources()
sia = SentimentIntensityAnalyzer()

# --- TERMINAL CONFIGURATION ---
FINNHUB_API_KEY = "d8m5hhhr01qkiso5nsigd8m5hhhr01qkiso5nsj0"  

st.set_page_config(page_title="Quantum AI Terminal", layout="wide", initial_sidebar_state="collapsed")

# Custom Premium Dark Theme Styling
st.markdown("""
    <style>
    .main { background-color: #0d1117; }
    div[data-testid="stMetricValue"] { font-size: 22px; font-weight: bold; color: #00f0ff; }
    div[data-testid="stMetricLabel"] { font-size: 13px; color: #8b949e; }
    .reportview-container .main .block-container{ max-width: 95%; }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ QUANTUM AI TRADING TERMINAL")
st.caption("Universal Multi-Asset Coverage Engine — Powered by Dynamic Autocomplete Streams")
st.write("---")

# --- LIVE ASSET AUTOCOMPLETE CRAWLER ---
def search_live_tickers(query):
    """Queries live databases to look up ANY stock matching user search terms"""
    if not query or len(query) < 2:
        return {
            "RELIANCE INDUSTRIES (NSE)": "RELIANCE.NS",
            "TATA MOTORS LTD (NSE)": "TATAMOTORS.NS",
            "STATE BANK OF INDIA (NSE)": "SBIN.NS",
            "HDFC BANK LTD (NSE)": "HDFCBANK.NS",
            "GROWW (BSE)": "GROWW.BO"
        }
    
    # Live Yahoo Finance Query API
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}&lang=en-US&quotesCount=15"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    try:
        res = requests.get(url, headers=headers).json()
        suggestions = {}
        for quote in res.get('quotes', []):
            symbol = quote.get('symbol', '')
            # Filter specifically for Indian stock market tickers (.NS = NSE, .BO = BSE)
            if symbol.endswith('.NS') or symbol.endswith('.BO'):
                name = quote.get('shortname', quote.get('longname', symbol))
                exch = "NSE" if symbol.endswith('.NS') else "BSE"
                display_label = f"🏢 {name} ({exch}) [{symbol.split('.')[0]}]"
                suggestions[display_label] = symbol
        
        if not suggestions:
            return {f"No custom Indian tickers found for '{query}'. Press Enter to try raw input.": f"{query.upper()}.NS"}
        return suggestions
    except:
        return {f"Searching for '{query}'...": f"{query.upper()}.NS"}

# --- SAFE CANDLESTICK CONVERTER ---
def get_candlestick_data(symbol, interval_label):
    tf_map = {"5 Minutes": "5m", "15 Minutes": "15m", "1 Hour": "1h"}
    interval = tf_map.get(interval_label, "5m")
    
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    headers = {"User-Agent": "Mozilla/5.0"}
    params = {"range": "30d", "interval": interval}
    
    try:
        res = requests.get(url, params=params, headers=headers).json()
        result = res['chart']['result'][0]
        timestamps = result['timestamp']
        indicators = result['indicators']['quote'][0]
        
        df = pd.DataFrame({
            'time': pd.to_datetime(timestamps, unit='s', utc=True).tz_convert('Asia/Kolkata').strftime('%Y-%m-%d %H:%M'),
            'open': indicators['open'],
            'high': indicators['high'],
            'low': indicators['low'],
            'close': indicators['close'],
            'volume': indicators['volume']
        })
        return df.dropna()
    except:
        return None

def get_news_sentiment(news_ticker):
    url = f"https://finnhub.io/api/v1/company-news?symbol={news_ticker}&from=2026-01-01&to=2026-12-31&token={FINNHUB_API_KEY}"
    try:
        res = requests.get(url).json()
        if not res or isinstance(res, dict): return 0.0
        scores = [sia.polarity_scores(a.get('headline', ''))['compound'] for a in res[:10]]
        return np.mean(scores) if scores else 0.0
    except:
        return 0.0

# --- MATH ALGORITHM CORE ---
def calculate_all_indicators(df):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['MA20'] = df['close'].rolling(window=20).mean()
    df['StdDev20'] = df['close'].rolling(window=20).std()
    df['Upper_Band'] = df['MA20'] + (df['StdDev20'] * 2)
    df['Lower_Band'] = df['MA20'] - (df['StdDev20'] * 2)
    df['BB_Dist_Upper'] = df['Upper_Band'] - df['close']
    df['BB_Dist_Lower'] = df['close'] - df['Lower_Band']

    df['ROC_3'] = ((df['close'] - df['close'].shift(3)) / df['close'].shift(3)) * 100

    df['datetime'] = pd.to_datetime(df['time'])
    df['Market_Hour'] = df['datetime'].dt.hour
    df['Market_Minute'] = df['datetime'].dt.minute

    df['Target'] = np.where(df['close'].shift(-3) > df['close'], 1, 0)
    return df.dropna()

# --- INTERACTIVE CONTROL DOCK ---
with st.container(border=True):
    c_input, c_dropdown, c_time = st.columns([1.5, 2, 1])
    
    # 1. Type ANY keyword here!
    search_keyword = c_input.text_input("🔍 Search Any Stock (e.g., TATA, ADANI, SUZLON, ZOMATO)", value="TATA")
    
    # Dynamic backend calculation hooks suggestions based on text above
    live_suggestions = search_live_tickers(search_keyword)
    
    # 2. Dropdown switches its contents instantly based on what you typed
    selected_asset_label = c_dropdown.selectbox("🎯 Confirm Selection", list(live_suggestions.keys()))
    ticker_symbol = live_suggestions[selected_asset_label]
    
    timeframe_label = c_time.selectbox("⏱️ Timeframe", ["5 Minutes", "15 Minutes", "1 Hour"])

st.write("")
if st.button("🚀 RUN QUANTUM PREDICTION ENGINE", use_container_width=True):
    with st.spinner("Synchronizing streaming datasets..."):
        df = get_candlestick_data(ticker_symbol, timeframe_label)
        sentiment = get_news_sentiment(ticker_symbol.split('.')[0])
        
        if df is not None and len(df) > 25:
            df = calculate_all_indicators(df)
            
            features = ['close', 'volume', 'RSI', 'MACD', 'Signal_Line', 'BB_Dist_Upper', 'BB_Dist_Lower', 'ROC_3', 'Market_Hour', 'Market_Minute']
            X = df[features].values[:-3]
            y = df['Target'].values[:-3]
            
            model = RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_leaf=10, random_state=42)
            model.fit(X, y)
            
            historical_accuracy = model.score(X, y)
            latest_features = df[features].iloc[-1].values.reshape(1, -1)
            prediction = model.predict(latest_features)[0]
            probs = model.predict_proba(latest_features)[0]

            # --- METRICS HUD ---
            with st.container(border=True):
                m1, m2, m3, m4 = st.columns(4)
                price_delta = df['close'].iloc[-1] - df['close'].iloc[-2]
                m1.metric("LTP (Last Price)", f"₹{df['close'].iloc[-1]:.2f}", f"{price_delta:+.2f}")
                
                rsi_val = df['RSI'].iloc[-1]
                rsi_status = "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral"
                m2.metric("RSI Status", f"{rsi_val:.1f}", rsi_status)
                
                sent_txt = "🟢 Bullish" if sentiment > 0.05 else "🔴 Bearish" if sentiment < -0.05 else "Neutral"
                m3.metric("News Trend", sent_txt, f"Score: {sentiment:+.2f}")
                m4.metric("Engine Reliability", f"{historical_accuracy * 100:.1f}%", "Historical Match")

            # --- TARGET PREDICTION ALERT PANEL ---
            horizon_text = "15-MIN" if timeframe_label == "5 Minutes" else "45-MIN" if timeframe_label == "15 Minutes" else "3-HOUR"
            if prediction == 1:
                st.markdown(f"""
                <div style="background-color:rgba(46, 204, 113, 0.12); padding:16px; border-radius:8px; border-left:6px solid #2ecc71; margin-bottom:15px;">
                    <span style="color:#2ecc71; font-size:18px; font-weight:bold;">🚀 TARGET METRIC OUTLOOK: UP DIRECTIONAL SHIFT</span><br>
                    <span style="color:#ffffff; font-size:13px;">Pattern recognition trees map upside probability at <b>{probs[1]*100:.1f}%</b> over the next {horizon_text} window.</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color:rgba(231, 76, 60, 0.12); padding:16px; border-radius:8px; border-left:6px solid #e74c3c; margin-bottom:15px;">
                    <span style="color:#e74c3c; font-size:18px; font-weight:bold;">📉 TARGET METRIC OUTLOOK: DOWN DIRECTIONAL SHIFT</span><br>
                    <span style="color:#ffffff; font-size:13px;">Pattern recognition trees map distribution/downside probability at <b>{probs[0]*100:.1f}%</b> over the next {horizon_text} window.</span>
                </div>
                """, unsafe_allow_html=True)

            # --- USER-FRIENDLY ZOOMABLE MULTI-CHART ---
            st.subheader("📊 Chart Workspace Terminal")
            st.caption("📱 *Mobile/Desktop Zoom Manual*: Click and drag your finger or cursor to frame any narrow window to zoom in closely on individual candles. Double tap to reset view completely.")
            
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.55, 0.22, 0.23])
            
            # Panel 1: Japanese Candlesticks with Bollinger Tracks
            fig.add_trace(go.Candlestick(
                x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                name='Candle', increasing_line_color='#2ecc71', decreasing_line_color='#e74c3c'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=df['time'], y=df['Upper_Band'], name='BB Upper', line=dict(color='rgba(52, 152, 219, 0.5)', width=1, dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['time'], y=df['MA20'], name='20 MA Basis', line=dict(color='#f39c12', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['time'], y=df['Lower_Band'], name='BB Lower', line=dict(color='rgba(52, 152, 219, 0.5)', width=1, dash='dash')), row=1, col=1)
            
            # Panel 2: RSI Oscillator Track
            fig.add_trace(go.Scatter(x=df['time'], y=df['RSI'], name='RSI', line=dict(color='#9b59b6', width=1.5)), row=2, col=1)
            fig.add_shape(type="line", x0=df['time'].iloc[0], x1=df['time'].iloc[-1], y0=70, y1=70, line=dict(color="#e74c3c", width=1, dash="dot"), row=2, col=1)
            fig.add_shape(type="line", x0=df['time'].iloc[0], x1=df['time'].iloc[-1], y0=30, y1=30, line=dict(color="#2ecc71", width=1, dash="dot"), row=2, col=1)
            
            # Panel 3: MACD Signals
            fig.add_trace(go.Scatter(x=df['time'], y=df['MACD'], name='MACD Line', line=dict(color='#1abc9c', width=1.5)), row=3, col=1)
            fig.add_trace(go.Scatter(x=df['time'], y=df['Signal_Line'], name='Signal Line', line=dict(color='#e67e22', width=1.2, dash='dot')), row=3, col=1)
            
            fig.update_layout(
                height=780, template="plotly_dark", xaxis_rangeslider_visible=False,
                hovermode="x unified", margin=dict(l=15, r=15, t=10, b=15),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True, 'responsive': True})
            
            with st.expander("📋 View Core Historical Data Spreadsheet"):
                st.dataframe(df[['time', 'open', 'high', 'low', 'close', 'volume', 'RSI', 'MACD']].tail(10), use_container_width=True)
        else:
            st.error("This specific asset doesn't have sufficient trading volume rows matching this timeframe. Try a different symbol key or adjust time interval.")

