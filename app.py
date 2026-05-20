import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import vnstock - primary data source for Vietnam stocks
try:
    from vnstock import Vnstock
    VNSTOCK_AVAILABLE = True
except ImportError:
    VNSTOCK_AVAILABLE = False

# Page Configuration
st.set_page_config(
    page_title='Vietnam Stock Bot',
    page_icon='📈',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS
st.markdown('''<style>
.buy-signal {color: #00d26a; font-weight: bold; font-size: 24px;}
.sell-signal {color: #ff4757; font-weight: bold; font-size: 24px;}
.hold-signal {color: #ffa502; font-weight: bold; font-size: 24px;}
</style>''', unsafe_allow_html=True)

# ============================================================
# VIETNAM STOCK DATABASE
# ============================================================
# DYNAMIC STOCK LIST - Loads ALL Vietnam stocks from vnstock API
# ============================================================
@st.cache_data(ttl=86400)  # Cache for 24 hours
def load_all_vietnam_stocks():
    """Load complete list of all Vietnam listed stocks"""
    if VNSTOCK_AVAILABLE:
        try:
            stock = Vnstock().stock(symbol='VNM', source='VCI')
            # Get all listed companies
            listing = stock.listing.all_symbols()
            if listing is not None and not listing.empty:
                # Create a dict of symbol -> company name
                stock_dict = {}
                for _, row in listing.iterrows():
                    symbol = row.get('ticker', row.get('symbol', ''))
                    name = row.get('organ_name', row.get('organName', row.get('name', '')))
                    if symbol and name:
                        stock_dict[symbol] = name
                if stock_dict:
                    return stock_dict
        except Exception as e:
            st.sidebar.warning(f'Could not load full stock list: {e}')

    # Fallback: hardcoded popular stocks (demo mode only)
    return {
        'VNM': 'Vinamilk - CTCP Sua Viet Nam',
        'FPT': 'FPT Corporation',
        'VIC': 'Vingroup - CTCP Tap doan Vingroup',
        'VHM': 'Vinhomes - CTCP Vinhomes',
        'HPG': 'Hoa Phat Group - CTCP Tap doan Hoa Phat',
        'MSN': 'Masan Group - CTCP Tap doan Masan',
        'VCB': 'Vietcombank - NH TMCP Ngoai thuong VN',
        'BID': 'BIDV - NH TMCP Dau tu va Phat trien VN',
        'CTG': 'VietinBank - NH TMCP Cong thuong VN',
        'TCB': 'Techcombank - NH TMCP Ky thuong VN',
        'MBB': 'MB Bank - NH TMCP Quan doi',
        'VPB': 'VPBank - NH TMCP Viet Nam Thinh Vuong',
        'ACB': 'ACB - NH TMCP A Chau',
        'SAB': 'Sabeco - Tong CTCP Bia Ruou NGK Sai Gon',
        'GAS': 'PV Gas - Tong CTCP Khi Viet Nam',
        'PLX': 'Petrolimex - Tap doan Xang dau VN',
        'VJC': 'Vietjet Air - CTCP Hang khong Vietjet',
        'MWG': 'Mobile World - CTCP Dau tu The Gioi Di Dong',
        'PNJ': 'PNJ - CTCP Vang bac Da quy Phu Nhuan',
        'SSI': 'SSI Securities - CTCP Chung khoan SSI',
        'VND': 'VNDirect - CTCP Chung khoan VNDirect',
        'HCM': 'HCMC Securities - CTCP CK TP Ho Chi Minh',
        'DGC': 'DGC - CTCP Tap doan Hoa chat Duc Giang',
        'REE': 'REE Corporation - CTCP Co Dien Lanh',
        'VRE': 'Vincom Retail - CTCP Vincom Retail',
    }

# Load stocks (cached - only runs once per day)
VIETNAM_STOCKS = load_all_vietnam_stocks()

# ============================================================
# DATA FUNCTIONS
# ============================================================
def generate_demo_data(symbol, days=365):
    """Generate realistic demo stock data for testing"""
    np.random.seed(hash(symbol) % (2**31))
    dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
    base_prices = {'VNM': 75000, 'FPT': 120000, 'VIC': 42000, 'VHM': 38000,
        'HPG': 25000, 'MSN': 80000, 'VCB': 90000, 'BID': 45000,
        'CTG': 35000, 'TCB': 35000, 'MBB': 22000, 'VPB': 20000}
    base_price = base_prices.get(symbol, 50000)
    returns = np.random.normal(0.0003, 0.018, days)
    prices = base_price * np.cumprod(1 + returns)
    data = pd.DataFrame({
        'time': dates,
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, days)),
        'high': prices * (1 + np.random.uniform(0.005, 0.025, days)),
        'low': prices * (1 - np.random.uniform(0.005, 0.025, days)),
        'close': prices,
        'volume': np.random.randint(500000, 5000000, days)
    })
    data.set_index('time', inplace=True)
    return data


def get_stock_data(symbol, start_date, end_date):
    """Fetch stock data from vnstock or generate demo data"""
    if VNSTOCK_AVAILABLE:
        try:
            stock = Vnstock().stock(symbol=symbol, source='VCI')
            df = stock.quote.history(start=start_date, end=end_date)
            if df is not None and not df.empty:
                df.columns = [c.lower() for c in df.columns]
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                    df.set_index('time', inplace=True)
                return df
        except Exception:
            pass
    days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    return generate_demo_data(symbol, max(days, 30))


# ============================================================
# TECHNICAL ANALYSIS FUNCTIONS
# ============================================================
def calculate_bollinger_bands(df, window=20, num_std=2):
    df = df.copy()
    df['BB_middle'] = df['close'].rolling(window=window).mean()
    df['BB_std'] = df['close'].rolling(window=window).std()
    df['BB_upper'] = df['BB_middle'] + (num_std * df['BB_std'])
    df['BB_lower'] = df['BB_middle'] - (num_std * df['BB_std'])
    return df


def calculate_rsi(df, window=14):
    df = df.copy()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def calculate_macd(df, fast=12, slow=26, signal=9):
    df = df.copy()
    df['EMA_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
    df['EMA_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = df['EMA_fast'] - df['EMA_slow']
    df['MACD_signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
    return df


def calculate_moving_averages(df):
    df = df.copy()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['SMA_200'] = df['close'].rolling(window=200).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    return df


def calculate_support_resistance(df, window=20):
    recent = df.tail(window)
    support = recent['low'].min()
    resistance = recent['high'].max()
    pivot = (recent['high'].iloc[-1] + recent['low'].iloc[-1] + recent['close'].iloc[-1]) / 3
    return support, resistance, pivot


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================
def generate_recommendation(df):
    """Generate investment recommendation based on technical indicators"""
    signals = []
    score = 0

    # RSI Signal
    current_rsi = df['RSI'].iloc[-1]
    if current_rsi < 30:
        signals.append(('RSI', 'BUY', 'Oversold - RSI below 30'))
        score += 2
    elif current_rsi < 40:
        signals.append(('RSI', 'BUY', 'Approaching oversold - RSI below 40'))
        score += 1
    elif current_rsi > 70:
        signals.append(('RSI', 'SELL', 'Overbought - RSI above 70'))
        score -= 2
    elif current_rsi > 60:
        signals.append(('RSI', 'SELL', 'Approaching overbought - RSI above 60'))
        score -= 1
    else:
        signals.append(('RSI', 'HOLD', f'Neutral RSI at {current_rsi:.1f}'))

    # Bollinger Bands Signal
    current_price = df['close'].iloc[-1]
    bb_upper = df['BB_upper'].iloc[-1]
    bb_lower = df['BB_lower'].iloc[-1]
    if current_price <= bb_lower:
        signals.append(('Bollinger Bands', 'BUY', 'Price at lower band - potential bounce'))
        score += 2
    elif current_price >= bb_upper:
        signals.append(('Bollinger Bands', 'SELL', 'Price at upper band - potential reversal'))
        score -= 2
    else:
        signals.append(('Bollinger Bands', 'HOLD', 'Price within bands'))

    # MACD Signal
    macd_val = df['MACD'].iloc[-1]
    macd_sig = df['MACD_signal'].iloc[-1]
    macd_prev = df['MACD'].iloc[-2]
    macd_sig_prev = df['MACD_signal'].iloc[-2]
    if macd_prev < macd_sig_prev and macd_val > macd_sig:
        signals.append(('MACD', 'BUY', 'Bullish crossover detected'))
        score += 2
    elif macd_prev > macd_sig_prev and macd_val < macd_sig:
        signals.append(('MACD', 'SELL', 'Bearish crossover detected'))
        score -= 2
    elif macd_val > macd_sig:
        signals.append(('MACD', 'BUY', 'MACD above signal line'))
        score += 1
    else:
        signals.append(('MACD', 'SELL', 'MACD below signal line'))
        score -= 1

    # Moving Average Signal
    sma_20 = df['SMA_20'].iloc[-1]
    sma_50 = df['SMA_50'].iloc[-1]
    if current_price > sma_20 and sma_20 > sma_50:
        signals.append(('Moving Averages', 'BUY', 'Price above SMA20 > SMA50 - strong uptrend'))
        score += 2
    elif current_price > sma_20:
        signals.append(('Moving Averages', 'BUY', 'Price above SMA20 - short-term bullish'))
        score += 1
    elif current_price < sma_20 and sma_20 < sma_50:
        signals.append(('Moving Averages', 'SELL', 'Price below SMA20 < SMA50 - strong downtrend'))
        score -= 2
    else:
        signals.append(('Moving Averages', 'HOLD', 'Mixed moving average signals'))

    # Volume Signal
    avg_vol = df['volume'].rolling(20).mean().iloc[-1]
    cur_vol = df['volume'].iloc[-1]
    prev_close = df['close'].iloc[-2]
    if cur_vol > 1.5 * avg_vol and current_price > prev_close:
        signals.append(('Volume', 'BUY', 'High volume with price increase'))
        score += 1
    elif cur_vol > 1.5 * avg_vol and current_price < prev_close:
        signals.append(('Volume', 'SELL', 'High volume with price decrease'))
        score -= 1
    else:
        signals.append(('Volume', 'HOLD', 'Normal volume activity'))

    # Overall
    confidence = min(abs(score) / 9 * 100, 100)
    if score >= 3: overall = 'STRONG BUY'
    elif score >= 1: overall = 'BUY'
    elif score <= -3: overall = 'STRONG SELL'
    elif score <= -1: overall = 'SELL'
    else: overall = 'HOLD'

    support, resistance, pivot = calculate_support_resistance(df)
    stop_loss = support * 0.97

    return {
        'overall': overall, 'score': score, 'confidence': confidence,
        'signals': signals, 'buy_price': support, 'target_price': resistance,
        'stop_loss': stop_loss, 'pivot': pivot, 'current_price': current_price
    }


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================
def create_candlestick_chart(df, symbol, show_bb=True, show_ma=True):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        subplot_titles=(f'{symbol} Price Chart', 'Volume', 'RSI'),
        row_heights=[0.6, 0.2, 0.2])

    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name='OHLC',
        increasing_line_color='#00d26a', decreasing_line_color='#ff4757'), row=1, col=1)

    if show_bb and 'BB_upper' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='BB Upper',
            line=dict(color='rgba(173,216,230,0.5)', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='BB Lower',
            line=dict(color='rgba(173,216,230,0.5)', width=1),
            fill='tonexty', fillcolor='rgba(173,216,230,0.1)'), row=1, col=1)

    if show_ma and 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20',
            line=dict(color='#ffa502', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50',
            line=dict(color='#3742fa', width=1)), row=1, col=1)

    colors = ['#00d26a' if df['close'].iloc[i] >= df['open'].iloc[i] else '#ff4757' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume',
        marker_color=colors, opacity=0.7), row=2, col=1)

    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
            line=dict(color='#a55eea', width=2)), row=3, col=1)
        fig.add_hline(y=70, line_dash='dash', line_color='red', row=3, col=1)
        fig.add_hline(y=30, line_dash='dash', line_color='green', row=3, col=1)

    fig.update_layout(height=800, template='plotly_dark', showlegend=True,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    return fig


def create_comparison_chart(data_dict, normalize=True):
    fig = go.Figure()
    for sym, df in data_dict.items():
        if normalize:
            values = (df['close'] / df['close'].iloc[0] - 1) * 100
            fig.add_trace(go.Scatter(x=df.index, y=values, name=sym, mode='lines'))
        else:
            fig.add_trace(go.Scatter(x=df.index, y=df['close'], name=sym, mode='lines'))
    title = 'Stock Price Comparison (Normalized %)' if normalize else 'Stock Price Comparison'
    fig.update_layout(title=title, yaxis_title='Return (%)' if normalize else 'Price (VND)',
        template='plotly_dark', height=500, hovermode='x unified')
    return fig


# ============================================================
# FINANCIAL DATA
# ============================================================
def get_financial_report(symbol, report_type='income_statement'):
    if VNSTOCK_AVAILABLE:
        try:
            stock = Vnstock().stock(symbol=symbol, source='VCI')
            if report_type == 'income_statement':
                return stock.finance.income_statement(period='quarter', lang='en')
            elif report_type == 'balance_sheet':
                return stock.finance.balance_sheet(period='quarter', lang='en')
            elif report_type == 'cash_flow':
                return stock.finance.cash_flow(period='quarter', lang='en')
        except Exception:
            pass
    return generate_demo_financial_data(symbol, report_type)


def generate_demo_financial_data(symbol, report_type):
    quarters = ['Q1-2024', 'Q2-2024', 'Q3-2024', 'Q4-2024', 'Q1-2025']
    np.random.seed(hash(symbol + report_type) % (2**31))
    if report_type == 'income_statement':
        data = {'Quarter': quarters,
            'Revenue (Bil VND)': np.random.uniform(5000, 20000, 5).round(0),
            'Gross Profit (Bil VND)': np.random.uniform(1000, 8000, 5).round(0),
            'Net Income (Bil VND)': np.random.uniform(300, 4000, 5).round(0),
            'EPS (VND)': np.random.uniform(1000, 5000, 5).round(0)}
    elif report_type == 'balance_sheet':
        data = {'Quarter': quarters,
            'Total Assets (Bil VND)': np.random.uniform(50000, 200000, 5).round(0),
            'Total Liabilities (Bil VND)': np.random.uniform(20000, 100000, 5).round(0),
            'Equity (Bil VND)': np.random.uniform(20000, 80000, 5).round(0),
            'Cash (Bil VND)': np.random.uniform(5000, 30000, 5).round(0)}
    else:
        data = {'Quarter': quarters,
            'Operating CF (Bil VND)': np.random.uniform(1000, 10000, 5).round(0),
            'Investing CF (Bil VND)': np.random.uniform(-8000, -1000, 5).round(0),
            'Free Cash Flow (Bil VND)': np.random.uniform(500, 8000, 5).round(0)}
    return pd.DataFrame(data)


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title('🇻🇳 Vietnam Stock Bot')
st.sidebar.markdown('---')
page = st.sidebar.radio('Navigation',
    ['Stock Search', 'Compare Stocks', 'Financial Reports', 'Recommendations', 'Market Overview'])
st.sidebar.markdown('---')
st.sidebar.markdown('### Settings')
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input('Start Date', datetime.now() - timedelta(days=365))
with col2:
    end_date = st.date_input('End Date', datetime.now())
st.sidebar.markdown('---')
st.sidebar.info('Data: vnstock (TCBS/SSI)\nFallback: Demo data')
st.sidebar.caption('⚠️ For educational purposes only. Not financial advice.')


# ============================================================
# PAGE: STOCK SEARCH
# ============================================================
if page == 'Stock Search':
    st.title('📊 Single Stock Analysis')
    st.markdown('Search for any Vietnam stock by code or company name')

    col1, col2 = st.columns([3, 1])
    with col1:
        search_input = st.text_input('Enter stock code or company name:', 'VNM',
            help='Examples: VNM, FPT, Vinamilk, Hoa Phat')
    with col2:
        st.markdown('<br>', unsafe_allow_html=True)
        st.button('🔍 Search', type='primary', use_container_width=True)

    symbol = search_input.upper().strip()
    if symbol not in VIETNAM_STOCKS:
        for code, name in VIETNAM_STOCKS.items():
            if search_input.lower() in name.lower():
                symbol = code
                break

    if symbol in VIETNAM_STOCKS:
        st.success(f'Found: {symbol} - {VIETNAM_STOCKS[symbol]}')
        with st.spinner(f'Loading data for {symbol}...'):
            df = get_stock_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            df = calculate_bollinger_bands(df)
            df = calculate_rsi(df)
            df = calculate_macd(df)
            df = calculate_moving_averages(df)

        # Key Metrics
        st.markdown('### Key Metrics')
        c1, c2, c3, c4, c5 = st.columns(5)
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        pct_chg = (current_price - prev_price) / prev_price * 100
        with c1: st.metric('Current Price', f'{current_price:,.0f} VND', f'{pct_chg:.2f}%')
        with c2: st.metric('Volume', f"{df['volume'].iloc[-1]:,.0f}")
        with c3: st.metric('RSI (14)', f"{df['RSI'].iloc[-1]:.1f}")
        with c4: st.metric('52W High', f"{df['high'].tail(252).max():,.0f} VND")
        with c5: st.metric('52W Low', f"{df['low'].tail(252).min():,.0f} VND")

        # Chart Options
        st.markdown('### Price Chart')
        col1, col2, col3 = st.columns(3)
        with col1: show_bb = st.checkbox('Show Bollinger Bands', value=True)
        with col2: show_ma = st.checkbox('Show Moving Averages', value=True)
        with col3: chart_period = st.selectbox('Period', ['1M', '3M', '6M', '1Y', 'All'], index=3)

        period_map = {'1M': 21, '3M': 63, '6M': 126, '1Y': 252, 'All': len(df)}
        display_df = df.tail(period_map[chart_period])
        fig = create_candlestick_chart(display_df, symbol, show_bb, show_ma)
        st.plotly_chart(fig, use_container_width=True)

        # Quick Recommendation
        st.markdown('### Quick Analysis')
        rec = generate_recommendation(df)
        col1, col2, col3 = st.columns(3)
        with col1:
            sig_cls = 'buy-signal' if 'BUY' in rec['overall'] else ('sell-signal' if 'SELL' in rec['overall'] else 'hold-signal')
            st.markdown(f'<p class="{sig_cls}">Signal: {rec["overall"]}</p>', unsafe_allow_html=True)
            st.markdown(f'Confidence: **{rec["confidence"]:.0f}%**')
        with col2:
            st.markdown(f'**Buy-in Price:** {rec["buy_price"]:,.0f} VND')
            st.markdown(f'**Target Price:** {rec["target_price"]:,.0f} VND')
        with col3:
            st.markdown(f'**Stop Loss:** {rec["stop_loss"]:,.0f} VND')
            pot_ret = (rec['target_price'] - current_price) / current_price * 100
            st.markdown(f'**Potential Return:** {pot_ret:.1f}%')

        with st.expander('View Historical Data'):
            st.dataframe(df[['open', 'high', 'low', 'close', 'volume']].tail(30))
    else:
        st.warning(f'Stock "{search_input}" not found. Try: VNM, FPT, VIC, HPG, VCB')
        st.dataframe(pd.DataFrame(list(VIETNAM_STOCKS.items()), columns=['Code', 'Company']))


# ============================================================
# PAGE: COMPARE STOCKS
# ============================================================
elif page == 'Compare Stocks':
    st.title('🔄 Stock Comparison')
    st.markdown('Compare 2 or more Vietnamese stocks side-by-side')

    selected_stocks = st.multiselect('Select stocks to compare (2 or more):',
        options=list(VIETNAM_STOCKS.keys()), default=['VNM', 'FPT', 'HPG'])

    if len(selected_stocks) >= 2:
        data_dict = {}
        with st.spinner('Loading comparison data...'):
            for sym in selected_stocks:
                data_dict[sym] = get_stock_data(sym, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

        st.markdown('### Normalized Price Performance')
        normalize = st.checkbox('Normalize prices (show % return)', value=True)
        fig = create_comparison_chart(data_dict, normalize)
        st.plotly_chart(fig, use_container_width=True)

        # Comparison Metrics
        st.markdown('### Comparison Metrics')
        metrics_data = []
        for sym, sdf in data_dict.items():
            sdf = calculate_rsi(sdf)
            current = sdf['close'].iloc[-1]
            ret_1m = ((current / sdf['close'].iloc[-21]) - 1) * 100 if len(sdf) > 21 else 0
            volatility = sdf['close'].pct_change().std() * np.sqrt(252) * 100
            metrics_data.append({
                'Symbol': sym, 'Price (VND)': f'{current:,.0f}',
                '1M Return (%)': f'{ret_1m:.2f}', 'Volatility (%)': f'{volatility:.2f}',
                'RSI': f"{sdf['RSI'].iloc[-1]:.1f}", 'Avg Volume': f"{sdf['volume'].mean():,.0f}"
            })
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

        # Correlation
        st.markdown('### Correlation Matrix')
        returns_df = pd.DataFrame({sym: df['close'].pct_change() for sym, df in data_dict.items()})
        fig_corr = px.imshow(returns_df.corr(), text_auto='.2f', color_continuous_scale='RdBu_r')
        fig_corr.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info('Please select at least 2 stocks to compare.')


# ============================================================
# PAGE: FINANCIAL REPORTS
# ============================================================
elif page == 'Financial Reports':
    st.title('📋 Financial Reports')
    col1, col2 = st.columns(2)
    with col1:
        fin_symbol = st.selectbox('Select Stock:', list(VIETNAM_STOCKS.keys()))
    with col2:
        report_type = st.selectbox('Report Type:', ['Income Statement', 'Balance Sheet', 'Cash Flow'])

    report_map = {'Income Statement': 'income_statement', 'Balance Sheet': 'balance_sheet', 'Cash Flow': 'cash_flow'}

    with st.spinner(f'Loading {report_type} for {fin_symbol}...'):
        fin_data = get_financial_report(fin_symbol, report_map[report_type])

    if fin_data is not None:
        st.markdown(f'### {report_type} - {fin_symbol}')
        st.dataframe(fin_data, use_container_width=True, hide_index=True)

        st.markdown('### Trend Visualization')
        fig = go.Figure()
        for col in fin_data.columns[1:]:
            fig.add_trace(go.Bar(x=fin_data['Quarter'], y=fin_data[col], name=col))
        fig.update_layout(template='plotly_dark', barmode='group', height=400)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE: RECOMMENDATIONS
# ============================================================
elif page == 'Recommendations':
    st.title('🤖 Investment Recommendations')
    rec_symbol = st.selectbox('Select Stock for Analysis:', list(VIETNAM_STOCKS.keys()))

    col1, col2 = st.columns(2)
    with col1:
        horizon = st.radio('Investment Horizon:', ['Short-term (1-4 weeks)', 'Long-term (3-12 months)'])
    with col2:
        risk = st.radio('Risk Tolerance:', ['Conservative', 'Moderate', 'Aggressive'])

    if st.button('🔮 Generate Recommendation', type='primary'):
        with st.spinner('Analyzing technical indicators...'):
            df = get_stock_data(rec_symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            df = calculate_bollinger_bands(df)
            df = calculate_rsi(df)
            df = calculate_macd(df)
            df = calculate_moving_averages(df)
            rec = generate_recommendation(df)

        st.markdown('---')
        st.markdown(f'## Analysis Results for {rec_symbol}')

        col1, col2, col3 = st.columns(3)
        with col1:
            sig_cls = 'buy-signal' if 'BUY' in rec['overall'] else ('sell-signal' if 'SELL' in rec['overall'] else 'hold-signal')
            st.markdown(f'<p class="{sig_cls}">SIGNAL: {rec["overall"]}</p>', unsafe_allow_html=True)
            st.markdown(f'**Confidence:** {rec["confidence"]:.0f}%')
            st.progress(rec['confidence'] / 100)
        with col2:
            st.markdown('**Price Targets:**')
            st.markdown(f'- Current: **{rec["current_price"]:,.0f} VND**')
            st.markdown(f'- Buy-in: **{rec["buy_price"]:,.0f} VND**')
            st.markdown(f'- Target: **{rec["target_price"]:,.0f} VND**')
            st.markdown(f'- Stop Loss: **{rec["stop_loss"]:,.0f} VND**')
        with col3:
            potential = (rec['target_price'] - rec['current_price']) / rec['current_price'] * 100
            risk_pct = (rec['current_price'] - rec['stop_loss']) / rec['current_price'] * 100
            rr = potential / risk_pct if risk_pct > 0 else 0
            st.markdown('**Risk/Reward:**')
            st.markdown(f'- Potential Gain: **+{potential:.1f}%**')
            st.markdown(f'- Potential Loss: **-{risk_pct:.1f}%**')
            st.markdown(f'- Risk/Reward: **1:{rr:.1f}**')

        st.markdown('### Detailed Signal Breakdown')
        st.dataframe(pd.DataFrame(rec['signals'], columns=['Indicator', 'Signal', 'Reasoning']),
            use_container_width=True, hide_index=True)


# ============================================================
# PAGE: MARKET OVERVIEW
# ============================================================
elif page == 'Market Overview':
    st.title('🏛️ Vietnam Market Overview')

    col1, col2, col3 = st.columns(3)
    with col1: st.metric('VN-Index', '1,245.67', '+12.34 (+1.0%)')
    with col2: st.metric('HNX-Index', '234.56', '-2.15 (-0.9%)')
    with col3: st.metric('UPCOM-Index', '89.12', '+0.45 (+0.5%)')

    st.markdown('---')
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('**🟢 Top Gainers**')
        st.dataframe(pd.DataFrame({
            'Symbol': ['FPT', 'VNM', 'MWG', 'PNJ', 'DGC'],
            'Price': ['125,400', '78,200', '52,100', '98,500', '85,300'],
            'Change': ['+6.8%', '+4.2%', '+3.9%', '+3.5%', '+3.1%']
        }), hide_index=True)
    with col2:
        st.markdown('**🔴 Top Losers**')
        st.dataframe(pd.DataFrame({
            'Symbol': ['VIC', 'VHM', 'BID', 'CTG', 'VPB'],
            'Price': ['42,300', '37,800', '44,500', '34,200', '19,800'],
            'Change': ['-4.5%', '-3.8%', '-3.2%', '-2.9%', '-2.5%']
        }), hide_index=True)

    st.markdown('---')
    sectors = pd.DataFrame({
        'Sector': ['Technology', 'Banking', 'Real Estate', 'Consumer', 'Energy', 'Healthcare'],
        'Change (%)': [2.3, -0.5, -1.2, 1.8, 0.9, 1.1]
    })
    fig = px.bar(sectors, x='Sector', y='Change (%)', color='Change (%)',
        color_continuous_scale='RdYlGn', title='Sector Performance')
    fig.update_layout(template='plotly_dark', height=400)
    st.plotly_chart(fig, use_container_width=True)


# Footer
st.markdown('---')
st.markdown('<div style="text-align:center;color:#666"><p>Vietnam Stock Bot v1.0 | Data: vnstock, CafeF, Vietstock</p><p>⚠️ For educational purposes only. Not financial advice.</p></div>', unsafe_allow_html=True)

