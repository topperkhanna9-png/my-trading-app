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
FINNHUB_API_KEY = "d8m5hhhr01qkiso5nsigd8m5hhhr01qkiso5nsj0"  

st.set_page_config(page_title="Quantum AI Workspace", layout="wide")

# Custom Dark Theme Styles
st.markdown("""
    <style>
    .main { background-color: #0d1117; }
    div[data-testid="stMetricValue"] { font-size: 20px; font-weight: bold; color: #00f0ff; }
    div[data-testid="stMetricLabel"] { font-size: 13px; color: #8b949e; }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ QUANTUM UNIFIED RADAR & RISK WORKSPACE")
st.caption("Nifty 50 Core Screeners + Custom Multi-Exchange Autocomplete Lookups with Confidence-Scaled Risk Matrix Models")
st.write("---")

# --- CORE DATA EXTRACTION REGISTRY ---
NIFTY_50_POOL = {
    "ADANI ENTERPRISES": "ADANIENT.NS", "ADANI PORTS": "ADANIPORTS.NS", "APOLLO HOSPITALS": "APOLLOHOSP.NS",
    "ASIAN PAINTS": "ASIANPAINT.NS", "AXIS BANK": "AXISBANK.NS", "BAJAJ AUTO": "BAJAJ-AUTO.NS",
    "BAJAJ FINANCE": "BAJFINANCE.NS", "BAJAJ FINSERV": "BAJAJFINSV.NS", "BHARTI AIRTEL": "BHARTIARTL.NS",
    "BPCL": "BPCL.NS", "BRITANNIA": "BRITANNIA.NS", "CIPLA": "CIPLA.NS", "COAL INDIA": "COALINDIA.NS",
    "DIVI'S LABS": "DIVISLAB.NS", "DR. REDDY'S": "DRREDDY.NS", "EICHER MOTORS": "EICHERMOT.NS",
    "GRASIM": "GRASIM.NS", "HCL TECH": "HCLTECH.NS", "HDFC BANK": "HDFCBANK.NS", "HDFC LIFE": "HDFCLIFE.NS",
    "HERO MOTOCORP": "HEROMOTOCO.NS", "HINDALCO": "HINDALCO.NS", "HINDUSTAN UNILEVER": "HINDUNILVR.NS",
    "ICICI BANK": "ICICIBANK.NS", "ITC LTD": "ITC.NS", "INDUSIND BANK": "INDUSINDBK.NS",
    "INFOSYS": "INFY.NS", "JSW STEEL": "JSWSTEEL.NS", "KOTAK MAHINDRA BANK": "KOTAKBANK.NS",
    "L&T": "LT.NS", "LTIMINDTREE": "LTIM.NS", "M&M": "M&M.NS", "MARUTI SUZUKI": "MARUTI.NS",
    "NESTLE INDIA": "NESTLEIND.NS", "NTPC": "NTPC.NS", "ONGC": "ONGC.NS", "POWER GRID": "POWERGRID.NS",
    "RELIANCE INDUSTRIES": "RELIANCE.NS", "SBI LIFE": "SBILIFE.NS", "STATE BANK OF INDIA": "SBIN.NS",
    "SUN PHARMA": "SUNPHARMA.NS", "TATA CONSUMER": "TATACONSUM.NS", "TATA MOTORS": "TATAMOTORS.NS",
    "TATA STEEL": "TATASTEEL.NS", "TCS": "TCS.NS", "TECH MAHINDRA": "TECHM.NS", "TITAN COMPANY": "TITAN.NS",
    "ULTRATECH CEMENT": "ULTRACEMCO.NS", "WIPRO": "WIPRO.NS"
}

def search_live_tickers(query):
    if not query or len(query) < 2:
        return {"RELIANCE INDUSTRIES (NSE)": "RELIANCE.NS", "TATA MOTORS (NSE)": "TATAMOTORS.NS"}
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}&lang=en-US&quotesCount=10"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers).json()
        suggestions = {}
        for quote in res.get('quotes', []):
            symbol = quote.get('symbol', '')
            if symbol.endswith('.NS') or symbol.endswith('.BO'):
                name = quote.get('shortname', quote.get('longname', symbol))
                ex = "NSE" if symbol.endswith('.NS') else "BSE"
                suggestions[f"🏢 {name} ({ex}) [{symbol.split('.')[0]}]"] = symbol
        return suggestions if suggestions else {f"Use Entry: '{query.upper()}'": f"{query.upper()}.NS"}
    except:
        return {f"Searching '{query}'...": f"{query.upper()}.NS"}

def get_candlestick_data(symbol, interval_label):
    tf_map = {"5 Minutes": "5m", "15 Minutes": "15m", "1 Hour": "1h"}
    interval = tf_map.get(interval_label, "15m")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    headers = {"User-Agent": "Mozilla/5.0"}
    params = {"range": "15d", "interval": interval}
    try:
        res = requests.get(url, params=params, headers=headers).json()
        result = res['chart']['result'][0]
        df = pd.DataFrame({
            'time': pd.to_datetime(result['timestamp'], unit='s', utc=True).tz_convert('Asia/Kolkata').strftime('%Y-%m-%d %H:%M'),
            'open': result['indicators']['quote'][0]['open'], 'high': result['indicators']['quote'][0]['high'],
            'low': result['indicators']['quote'][0]['low'], 'close': result['indicators']['quote'][0]['close'],
            'volume': result['indicators']['quote'][0]['volume']
        })
        return df.dropna()
    except:
        return None

def calculate_all_indicators(df):
    if len(df) < 21: return None
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))

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

def get_news_sentiment(news_ticker):
    url = f"https://finnhub.io/api/v1/company-news?symbol={news_ticker}&from=2026-01-01&to=2026-12-31&token={FINNHUB_API_KEY}"
    try:
        res = requests.get(url).json()
        scores = [sia.polarity_scores(a.get('headline', ''))['compound'] for a in res[:10]]
        return np.mean(scores) if scores else 0.0
    except:
        return 0.0

# --- TAB OPERATIONS ARCHITECTURE ---
tab1, tab2 = st.tabs(["🛰️ Real-Time Index Scanner", "📈 Single Asset Workspace Terminal"])

# --- TAB 1: SCREENER ENGINE ---
with tab1:
    st.subheader("Nifty 50 High-Probability Sector Radar")
    idx_tf = st.selectbox("Scanner Frequency Profile", ["15 Minutes", "1 Hour"], key="idx_tf")
    
    if st.button("🚀 EXECUTE RADAR STREAM SCAN", use_container_width=True):
        scan_results = []
        p_bar = st.progress(0)
        
        for idx, (name, ticker) in enumerate(NIFTY_50_POOL.items()):
            p_bar.progress((idx + 1) / len(NIFTY_50_POOL))
            raw_data = get_candlestick_data(ticker, idx_tf)
            
            if raw_data is not None and len(raw_data) > 25:
                df_p = calculate_all_indicators(raw_data)
                if df_p is not None:
                    features = ['close', 'volume', 'RSI', 'MACD', 'Signal_Line', 'BB_Dist_Upper', 'BB_Dist_Lower', 'ROC_3', 'Market_Hour', 'Market_Minute']
                    X = df_p[features].values[:-3]
                    y = df_p['Target'].values[:-3]
                    
                    if len(np.unique(y)) > 1:
                        model = RandomForestClassifier(n_estimators=60, max_depth=4, random_state=42)
                        model.fit(X, y)
                        latest_v = df_p[features].iloc[-1].values.reshape(1, -1)
                        prob_up = model.predict_proba(latest_v)[0][1] * 100
                        
                        scan_results.append({
                            "Name": name, "Code": ticker, "LTP": df_p['close'].iloc[-1],
                            "RSI": df_p['RSI'].iloc[-1], "Probability": prob_up,
                            "LB": df_p['Lower_Band'].iloc[-1], "UB": df_p['Upper_Band'].iloc[-1]
                        })
                        
        if scan_results:
            res_df = pd.DataFrame(scan_results).sort_values(by="Probability", ascending=False).reset_index(drop=True)
            st.success("🎯 Index Scan Complete! Top Setups Listed Below:")
            
            for i, row in res_df.head(5).iterrows():
                entry = row["LTP"]
                prob = row["Probability"]
                
                # ADAPTIVE RISK PROFILING COMPUTATION ENGINE
                if prob >= 70:
                    rr_ratio = 3.0
                    rr_label = "1:3 (Aggressive Confluence Expansion)"
                elif prob >= 60:
                    rr_ratio = 2.0
                    rr_label = "1:2 (Optimal Structural Extension)"
                else:
                    rr_ratio = 1.5
                    rr_label = "1:1.5 (Standard Momentum Scale)"
                    
                stop_loss = row["LB"] if row["LB"] < entry else entry * 0.988
                risk = entry - stop_loss
                target = entry + (risk * rr_ratio)
                
                with st.container(border=True):
                    col1, col2, col3 = st.columns([1.5, 1.5, 3])
                    col1.markdown(f"### **#{i+1} {row['Name']}**")
                    col1.markdown(f"Ticker: `{row['Code']}`")
                    col2.metric("AI Win Probability", f"{prob:.1f}%")
                    col2.metric("LTP", f"₹{entry:.2f}")
                    
                    col3.markdown(f"🛡️ **Adaptive Matrix Risk Profile: {rr_label}**")
                    col3.write(f"🟢 **Entry Floor:** ₹{entry:.2f} | 🛑 **Stop Loss:** ₹{stop_loss:.2f} `(Risk: ₹{risk:.2f})`")
                    col3.write(f"🎯 **Target Profit Objective:** ₹{target:.2f} `(Reward Target: ₹{risk * rr_ratio:.2f})`")
        else:
            st.error("Data tracking pipeline temporarily unavailable. Rerun the radar module.")

# --- TAB 2: INDIVIDUAL WORKSPACE (PREVIOUS FEATURES RETAINED) ---
with tab2:
    st.subheader("Manual Search Analytics Terminal")
    c_in, c_drop, c_t = st.columns([1.5, 2, 1])
    
    # Text Input Autocomplete Engine
    keyword = c_in.text_input("🔍 Type Custom Stock Keyword (e.g. ADANI, SUZLON, TATA)", value="RELIANCE")
    suggestions = search_live_tickers(keyword)
    selected_lbl = c_drop.selectbox("🎯 Target Live Asset Dropdown", list(suggestions.keys()))
    asset_ticker = suggestions[selected_lbl]
    timeframe = c_t.selectbox("⏱️ Frame Frequency", ["5 Minutes", "15 Minutes", "1 Hour"], index=1)
    
    if st.button("🚀 INITIALIZE TARGET ASSET QUANT LAB", use_container_width=True):
        with st.spinner("Assembling feature matrix arrays..."):
            df = get_candlestick_data(asset_ticker, timeframe)
            sentiment = get_news_sentiment(asset_ticker.split('.')[0])
            
            if df is not None and len(df) > 25:
                df = calculate_all_indicators(df)
                if df is not None:
                    features = ['close', 'volume', 'RSI', 'MACD', 'Signal_Line', 'BB_Dist_Upper', 'BB_Dist_Lower', 'ROC_3', 'Market_Hour', 'Market_Minute']
                    X = df[features].values[:-3]
                    y = df['Target'].values[:-3]
                    
                    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
                    model.fit(X, y)
                    acc = model.score(X, y)
                    
                    latest_f = df[features].iloc[-1].values.reshape(1, -1)
                    prediction = model.predict(latest_f)[0]
                    probs = model.predict_proba(latest_f)[0]
                    prob_win = probs[1] * 100 if prediction == 1 else probs[0] * 100

                    # Adaptive Risk Assessment for single asset selection
                    raw_prob = probs[1] * 100
                    if raw_prob >= 70: rr_scale = 3.0
                    elif raw_prob >= 60: rr_scale = 2.0
                    else: rr_scale = 1.5

                    # Metric Rows Display
                    with st.container(border=True):
                        m1, m2, m3, m4 = st.columns(4)
                        p_chg = df['close'].iloc[-1] - df['close'].iloc[-2]
                        m1.metric("LTP (Last Price)", f"₹{df['close'].iloc[-1]:.2f}", f"{p_chg:+.2f}")
                        m2.metric("RSI Tracker", f"{df['RSI'].iloc[-1]:.1f}", "Overbought" if df['RSI'].iloc[-1] > 70 else "Oversold" if df['RSI'].iloc[-1] < 30 else "Neutral")
                        m3.metric("News Trend Tracker", "🟢 Bullish" if sentiment > 0.05 else "🔴 Bearish" if sentiment < -0.05 else "Neutral", f"Score: {sentiment:+.2f}")
                        m4.metric("Backtest Reliability", f"{acc*100:.1f}%")

                    # Target Callouts Container Blocks
                    horizon_t = "15-MIN" if timeframe == "5 Minutes" else "45-MIN" if timeframe == "15 Minutes" else "3-HOUR"
                    if prediction == 1:
                        st.markdown(f"""
                        <div style="background-color:rgba(46, 204, 113, 0.12); padding:16px; border-radius:8px; border-left:6px solid #2ecc71; margin-bottom:15px;">
                            <span style="color:#2ecc71; font-size:18px; font-weight:bold;">🚀 TARGET METRIC OUTLOOK: UP DIRECTIONAL SHIFT (CONVERGENCE)</span><br>
                            <span style="color:#ffffff; font-size:13px;">Pattern recognition engines map upside probability at <b>{raw_prob:.1f}%</b> over the next {horizon_t} tracking window. Target Risk-to-Reward Ratio scaled to <b>1:{rr_scale}</b>.</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background-color:rgba(231, 76, 60, 0.12); padding:16px; border-radius:8px; border-left:6px solid #e74c3c; margin-bottom:15px;">
                            <span style="color:#e74c3c; font-size:18px; font-weight:bold;">📉 TARGET METRIC OUTLOOK: DOWN DIRECTIONAL SHIFT (DISTRIBUTION)</span><br>
                            <span style="color:#ffffff; font-size:13px;">Pattern recognition engines map distribution downside probability at <b>{probs[0]*100:.1f}%</b> over the next {horizon_t} tracking window.</span>
                        </div>
                        """, unsafe_allow_html=True)

                    # Subplot layout rendering
                    st.subheader("📊 Chart Workspace Terminal")
                    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.55, 0.22, 0.23])
                    fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candle', increasing_line_color='#2ecc71', decreasing_line_color='#e74c3c'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df['time'], y=df['Upper_Band'], name='BB Upper', line=dict(color='rgba(52, 152, 219, 0.5)', width=1, dash='dash')), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df['time'], y=df['MA20'], name='20 MA Basis', line=dict(color='#f39c12', width=1.5)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df['time'], y=df['Lower_Band'], name='BB Lower', line=dict(color='rgba(52, 152, 219, 0.5)', width=1, dash='dash')), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df['time'], y=df['RSI'], name='RSI', line=dict(color='#9b59b6', width=1.5)), row=2, col=1)
                    fig.add_trace(go.Scatter(x=df['time'], y=df['MACD'], name='MACD Line', line=dict(color='#1abc9c', width=1.5)), row=3, col=1)
                    fig.add_trace(go.Scatter(x=df['time'], y=df['Signal_Line'], name='Signal Line', line=dict(color='#e67e22', width=1.2, dash='dot')), row=3, col=1)
                    
                    fig.update_layout(height=750, template="plotly_dark", xaxis_rangeslider_visible=False, hovermode="x unified", margin=dict(l=15, r=15, t=10, b=15))
                    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True, 'responsive': True})
            else:
                st.error("Insufficient liquidity depth for this token ticker. Try expanding the framing frequency.")


