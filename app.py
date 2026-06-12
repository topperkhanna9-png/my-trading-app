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

# --- CONFIGURATION ---
ANGEL_ONE_API_KEY = "1BoKeTG3"                     
FINNHUB_API_KEY = "d8m5hhhr01qkiso5nsigd8m5hhhr01qkiso5nsj0"  

st.set_page_config(page_title="Quantum AI Terminal", layout="wide", initial_sidebar_state="collapsed")

# Custom Premium Styling
st.markdown("""
    <style>
    .main { background-color: #0d1117; }
    div[data-testid="stMetricValue"] { font-size: 24px; font-weight: bold; }
    div[data-testid="stMetricLabel"] { font-size: 14px; color: #8b949e; }
    .reportview-container .main .block-container{ max-width: 95%; }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ QUANTUM AI TRADING TERMINAL")
st.caption("Powered by Random Forest Core Classifiers & Live Multi-Asset Autocomplete Search Engines")
st.write("---")

# --- DYNAMIC TICKER AUTOCOMPLETE SEARCH ENGINE ---
def search_live_tickers(query):
    """Fetches real-time matching assets matching user text entries dynamically"""
    if not query or len(query) < 2:
        # Default starting fallback options
        return {
            "RELIANCE INDUSTRIES": "RELIANCE.NS",
            "TATA MOTORS": "TATAMOTORS.NS",
            "STATE BANK OF INDIA": "SBIN.NS",
            "HDFC BANK": "HDFCBANK.NS",
            "INFOSYS LTD": "INFY.NS"
        }
    
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}&lang=en-US&quotesCount=10"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        res = requests.get(url, headers=headers).json()
        suggestions = {}
        for quote in res.get('quotes', []):
            symbol = quote.get('symbol', '')
            # Filter results for Indian Exchanges (NSE and BSE)
            if symbol.endswith('.NS') or symbol.endswith('.BO'):
                name = quote.get('shortname', quote.get('longname', symbol))
                display_label = f"{name} ({symbol.split('.')[1]}) [{symbol}]"
                suggestions[display_label] = symbol
        
        if not suggestions:
            return {f"No specific Indian assets found for '{query}'": f"{query.upper()}.NS"}
        return suggestions
    except:
        return {f"Searching for '{query}'...": f"{query.upper()}.NS"}

# --- UNIVERSAL FAILSAFE CANDLESTICK FETCH ---
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
    except Exception as e:
        st.error(f"⚠️ Market data synchronization offline: {str(e)}")
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

# --- INDICATOR ENGINE ---
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

# --- CONTROL CONTROL PANEL ---
with st.container(border=True):
    c_search_input, c_search_select, c_time = st.columns([1.5, 2, 1])
    
    # 1. Type keywords here to query the asset database
    user_search = c_search_input.text_input("🔍 Type Company Key (e.g., TATA, ADANI, INFY)", value="RELIANCE")
    
    # Get dynamic matching listings from live streams
    matched_options = search_live_tickers(user_search)
    
    # 2. Select corresponding item derived from results dropdown
    selected_label = c_search_select.selectbox("🎯 Select Targeted Asset", list(matched_options.keys()))
    ticker_symbol = matched_options[selected_label]
    
    timeframe_label = c_time.selectbox("⏱️ Evaluation Frame", ["5 Minutes", "15 Minutes", "1 Hour"])

st.write("")
if st.button("🚀 INITIALIZE REAL-TIME QUANT ENGINE", use_container_width=True):
    with st.spinner("Decoding candlestick signals and assembling prediction arrays..."):
        df = get_candlestick_data(ticker_symbol, timeframe_label)
        sentiment = get_news_sentiment(ticker_symbol.split('.')[0])
        
        if df is not None and len(df) > 25:
            df = calculate_all_indicators(df)
            
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

            # --- COLOR-CODED CORE METRIC DASHBOARD ---
            st.write("")
            with st.container(border=True):
                st.subheader("📊 Execution Vector Metrics")
                m1, m2, m3, m4 = st.columns(4)
                
                price_change = df['close'].iloc[-1] - df['close'].iloc[-2]
                price_color = "normal" if price_change == 0 else "inverse" if price_change < 0 else "normal"
                m1.metric("LTP (Last Traded Price)", f"₹{df['close'].iloc[-1]:.2f}", f"{price_change:+.2f}", delta_color=price_color)
                
                rsi_val = df['RSI'].iloc[-1]
                rsi_status = "⚠️ Overbought" if rsi_val > 70 else "⚠️ Oversold" if rsi_val < 30 else "Neutral Momentum"
                m2.metric("RSI Momentum", f"{rsi_val:.1f}", rsi_status)
                
                sent_txt = "🟢 Bullish Vector" if sentiment > 0.05 else "🔴 Bearish Pressure" if sentiment < -0.05 else "Neutral"
                m3.metric("News Analytics Stream", sent_txt, f"Score: {sentiment:+.2f}")
                
                m4.metric("Backtest Engine Stability", f"{historical_accuracy * 100:.1f}%", "Confidence Rating")

            # --- PREDICTIVE CALLOUT CONTAINER ---
            horizon_text = "15-MINUTE" if timeframe_label == "5 Minutes" else "45-MINUTE" if timeframe_label == "15 Minutes" else "3-HOUR"
            with st.container(border=True):
                if prediction == 1:
                    st.markdown(f"""
                    <div style="background-color:rgba(46, 204, 113, 0.15); padding:15px; border-radius:8px; border-left:6px solid #2ecc71;">
                        <span style="color:#2ecc71; font-size:20px; font-weight:bold;">🚀 AI FORECAST OUTLOOK: UP (BULLISH CONVERGENCE)</span><br>
                        <span style="color:#ffffff; font-size:14px;">The system maps an upward trajectory over the next <b>{horizon_text}</b> horizon with a directional match certainty of <b>{probs[1]*100:.1f}%</b>.</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color:rgba(231, 76, 60, 0.15); padding:15px; border-radius:8px; border-left:6px solid #e74c3c;">
                        <span style="color:#e74c3c; font-size:20px; font-weight:bold;">📉 AI FORECAST OUTLOOK: DOWN (BEARISH DISTRIBUTION)</span><br>
                        <span style="color:#ffffff; font-size:14px;">The system maps downward distributor pressure over the next <b>{horizon_text}</b> horizon with a directional match certainty of <b>{probs[0]*100:.1f}%</b>.</span>
                    </div>
                    """, unsafe_allow_html=True)

            # --- INTERACTIVE ZOOMABLE ADVANCED CHART SUITE ---
            st.write("")
            st.subheader("📈 Interactive Multi-Panel Technical Graph Suite")
            
            fig = make_subplots(
                rows=3, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.05,
                row_heights=[0.5, 0.25, 0.25]
            )
            
            # Panel 1: Candlesticks & Bollinger Bands
            fig.add_trace(go.Candlestick(
                x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                name='Candlestick', increasing_line_color='#2ecc71', decreasing_line_color='#e74c3c'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=df['time'], y=df['Upper_Band'], name='BB Upper', line=dict(color='#3498db', width=1, dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['time'], y=df['MA20'], name='20 MA Baseline', line=dict(color='#f39c12', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['time'], y=df['Lower_Band'], name='BB Lower', line=dict(color='#3498db', width=1, dash='dash')), row=1, col=1)
            
            # Panel 2: RSI
            fig.add_trace(go.Scatter(x=df['time'], y=df['RSI'], name='RSI Momentum', line=dict(color='#9b59b6', width=1.5)), row=2, col=1)
            fig.add_shape(type="line", x0=df['time'].iloc[0], x1=df['time'].iloc[-1], y0=70, y1=70, line=dict(color="red", width=1, dash="dot"), row=2, col=1)
            fig.add_shape(type="line", x0=df['time'].iloc[0], x1=df['time'].iloc[-1], y0=30, y1=30, line=dict(color="green", width=1, dash="dot"), row=2, col=1)
            
            # Panel 3: MACD Engine
            fig.add_trace(go.Scatter(x=df['time'], y=df['MACD'], name='MACD Line', line=dict(color='#1abc9c', width=1.5)), row=3, col=1)
            fig.add_trace(go.Scatter(x=df['time'], y=df['Signal_Line'], name='Signal Line', line=dict(color='#e67e22', width=1.2, dash='dot')), row=3, col=1)
            
            fig.update_layout(
                height=750,
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=30, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True, config={
                'scrollZoom': True,           
                'displayModeBar': True,        
                'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
                'responsive': True             
            })
            
            # --- ROW VOLATILITY ANALYSIS LOG ---
            with st.expander("📋 Inspect Dense Data Log (Last 10 Candles)"):
                st.dataframe(
                    df[['time', 'open', 'high', 'low', 'close', 'volume', 'RSI', 'MACD', 'ROC_3']].tail(10),
                    use_container_width=True
                )
        else:
            st.error("Data matrix contains insufficient candle metrics. Try refreshing or updating intervals.")
