import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ==========================================
# 1. 配置与样式设计 (Luxury & Minimalist)
# ==========================================
st.set_page_config(
    page_title="QUANT | PROJECTION",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS：极简深色轻奢风
st.markdown("""
    <style>
        /* 全局背景与字体 */
        .stApp {
            background-color: #0E1117; /* 深炭灰 */
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        }
        
        /* 侧边栏样式 */
        [data-testid="stSidebar"] {
            background-color: #161B22;
            border-right: 1px solid #262730;
        }
        
        /* 标题样式 */
        h1, h2, h3 {
            color: #E0E0E0;
            font-weight: 300 !important; /* 极细字体 */
            letter-spacing: 1.5px;
        }
        
        /* 指标卡片自定义样式 */
        div.metric-container {
            background-color: #1E232D;
            border: 1px solid #303642;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            transition: transform 0.2s;
            text-align: center;
        }
        div.metric-container:hover {
            transform: translateY(-2px);
            border-color: #D4AF37; /* 香槟金悬停边框 */
        }
        .metric-label {
            color: #8B929E;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 5px;
        }
        .metric-value {
            color: #F0F2F6;
            font-size: 28px;
            font-weight: 400;
        }
        .metric-delta {
            font-size: 12px;
            font-weight: 500;
        }
        .positive { color: #D4AF37; } /* 香槟金代表盈利 */
        .negative { color: #FF6B6B; }
        
        /* 移除 Streamlit 默认元素 */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* 按钮样式 */
        .stButton button {
            background-color: transparent;
            border: 1px solid #D4AF37;
            color: #D4AF37;
            border-radius: 5px;
            transition: all 0.3s;
        }
        .stButton button:hover {
            background-color: #D4AF37;
            color: #0E1117;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 数据模拟引擎 (Mock Data)
# ==========================================
def generate_data(years=3, volatility=0.2, expected_return=0.15):
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=years*252, freq='B')
    
    # 模拟策略收益
    returns = np.random.normal(expected_return/252, volatility/np.sqrt(252), len(dates))
    price = 100 * np.cumprod(1 + returns)
    
    # 模拟基准收益
    bench_returns = np.random.normal(0.08/252, 0.15/np.sqrt(252), len(dates))
    bench_price = 100 * np.cumprod(1 + bench_returns)
    
    df = pd.DataFrame({
        'Date': dates,
        'Strategy': price,
        'Benchmark': bench_price,
        'Returns': returns
    }).set_index('Date')
    
    # 计算回撤
    roll_max = df['Strategy'].cummax()
    df['Drawdown'] = (df['Strategy'] - roll_max) / roll_max
    
    return df

def monte_carlo_simulation(start_price, days=252, sims=1000, mu=0.15, sigma=0.2):
    dt = 1/252
    simulation_df = pd.DataFrame()
    
    # 几何布朗运动
    paths = np.zeros((days, sims))
    paths[0] = start_price
    
    for t in range(1, days):
        rand = np.random.standard_normal(sims)
        paths[t] = paths[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rand)
        
    return paths

# ==========================================
# 3. 页面布局逻辑
# ==========================================

# --- Sidebar: 策略控制台 ---
st.sidebar.title("STRATEGY CONTROL")
st.sidebar.markdown("---")
selected_strategy = st.sidebar.selectbox("Select Strategy", ["Alpha Neutral", "Trend Following", "Global Macro"])
risk_level = st.sidebar.slider("Risk Exposure (Volatility)", 0.05, 0.40, 0.15, 0.01)
leverage = st.sidebar.slider("Leverage", 1.0, 3.0, 1.0, 0.1)
st.sidebar.markdown("---")
st.sidebar.caption("Created for High Net Worth Clients")

# 生成数据
df = generate_data(volatility=risk_level * leverage, expected_return=0.10 * leverage)

# --- Header: 核心指标 ---
st.title("QUANTITATIVE PERFORMANCE")
st.markdown(f"**Strategy:** {selected_strategy} | **Leverage:** {leverage}x")

col1, col2, col3, col4 = st.columns(4)

# 计算指标
total_ret = (df['Strategy'].iloc[-1] / df['Strategy'].iloc[0]) - 1
sharpe = (df['Returns'].mean() * 252) / (df['Returns'].std() * np.sqrt(252))
max_dd = df['Drawdown'].min()
vol = df['Returns'].std() * np.sqrt(252)

# 渲染自定义卡片函数
def render_metric(label, value, delta=None, is_percent=True):
    fmt_val = f"{value:.2%}" if is_percent and label != "Sharpe Ratio" else f"{value:.2f}"
    if label == "Sharpe Ratio": fmt_val = f"{value:.2f}"
    
    delta_html = ""
    if delta:
        color_class = "positive" if delta > 0 else "negative"
        sign = "+" if delta > 0 else ""
        delta_html = f'<div class="metric-delta {color_class}">{sign}{delta:.2%} vs Bench</div>'
    
    return f"""
    <div class="metric-container">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{fmt_val}</div>
        {delta_html}
    </div>
    """

with col1: st.markdown(render_metric("Total Return", total_ret, total_ret - 0.2), unsafe_allow_html=True)
with col2: st.markdown(render_metric("Sharpe Ratio", sharpe, is_percent=False), unsafe_allow_html=True)
with col3: st.markdown(render_metric("Max Drawdown", max_dd), unsafe_allow_html=True)
with col4: st.markdown(render_metric("Volatility", vol), unsafe_allow_html=True)

st.markdown("###") # Spacer

# ==========================================
# 4. 图表绘制 (Plotly)
# ==========================================

# 颜色配置 (Luxury Palette)
COLOR_STRATEGY = "#D4AF37" # 香槟金
COLOR_BENCHMARK = "#4A4E57" # 雅灰
COLOR_BG = "#0E1117"
COLOR_GRID = "#262730"

# Tab 分页
tab1, tab2 = st.tabs(["HISTORICAL BACKTEST", "FUTURE PROJECTION"])

with tab1:
    # --- 历史净值图 ---
    fig_hist = go.Figure()
    
    # Benchmark
    fig_hist.add_trace(go.Scatter(
        x=df.index, y=df['Benchmark'],
        mode='lines',
        name='Benchmark',
        line=dict(color=COLOR_BENCHMARK, width=1, dash='dash'),
        hovertemplate='Bench: %{y:.2f}<extra></extra>'
    ))
    
    # Strategy
    fig_hist.add_trace(go.Scatter(
        x=df.index, y=df['Strategy'],
        mode='lines',
        name='Strategy',
        line=dict(color=COLOR_STRATEGY, width=2.5),
        fill='tozeroy', # 填充背景
        fillcolor='rgba(212, 175, 55, 0.05)', # 极淡的金色填充
        hovertemplate='NAV: %{y:.2f}<extra></extra>'
    ))

    fig_hist.update_layout(
        title="",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=450,
        hovermode="x unified",
        xaxis=dict(showgrid=False, showline=True, linecolor=COLOR_GRID),
        yaxis=dict(showgrid=True, gridcolor=COLOR_GRID, zeroline=False, title="Net Asset Value"),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # --- 回撤图 ---
    with st.expander("Analyze Drawdown Risk"):
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=df.index, y=df['Drawdown'],
            mode='lines',
            line=dict(color='#FF6B6B', width=1),
            fill='tozeroy',
            fillcolor='rgba(255, 107, 107, 0.1)',
            name='Drawdown'
        ))
        fig_dd.update_layout(
            template="plotly_dark",
            height=200,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(tickformat=".1%", showgrid=True, gridcolor=COLOR_GRID),
            xaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig_dd, use_container_width=True)

with tab2:
    # --- 未来预测 (Monte Carlo) ---
    st.markdown("##### 12-Month Expected Performance Range (95% Confidence)")
    
    # 生成模拟数据
    last_price = df['Strategy'].iloc[-1]
    sim_days = 252
    paths = monte_carlo_simulation(last_price, days=sim_days, sims=500, mu=0.10 * leverage, sigma=vol)
    
    # 计算分位数
    median_path = np.median(paths, axis=1)
    p95 = np.percentile(paths, 95, axis=1)
    p05 = np.percentile(paths, 5, axis=1)
    
    future_dates = [df.index[-1] + timedelta(days=x) for x in range(sim_days)]
    
    fig_proj = go.Figure()
    
    # 95% 置信区间 (扇形区域)
    fig_proj.add_trace(go.Scatter(
        x=future_dates + future_dates[::-1],
        y=np.concatenate([p95, p05[::-1]]),
        fill='toself',
        fillcolor='rgba(212, 175, 55, 0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name='95% Confidence'
    ))
    
    # 中位数路径
    fig_proj.add_trace(go.Scatter(
        x=future_dates, y=median_path,
        mode='lines',
        line=dict(color=COLOR_STRATEGY, width=2, dash='dot'),
        name='Expected (Median)'
    ))
    
    # 连接历史线
    fig_proj.add_trace(go.Scatter(
        x=df.index[-60:], y=df['Strategy'].iloc[-60:],
        mode='lines',
        line=dict(color='#8B929E', width=2),
        name='History (Last 60d)'
    ))

    fig_proj.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500,
        xaxis=dict(showgrid=False, linecolor=COLOR_GRID),
        yaxis=dict(showgrid=True, gridcolor=COLOR_GRID, title="Projected Value"),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    st.plotly_chart(fig_proj, use_container_width=True)
    
    st.info("Disclaimer: Future projections are based on Monte Carlo simulations and do not guarantee future results.")