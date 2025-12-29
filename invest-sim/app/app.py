import streamlit as st  # pyright: ignore[reportMissingImports]
import pandas as pd  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
import plotly.graph_objects as go  # pyright: ignore[reportMissingImports]
from datetime import datetime, timedelta
from statistics import NormalDist
from typing import Optional

# 引入后端桥接 (保持原有引用)
from bridge import InvestSimBridge
from invest_sim.backend.input_modeling.fitting import fit_normal
from invest_sim.option_simulator import (
    OptionLeg,
    OptionMarginSimulator,
    bs_delta,
    bs_gamma,
    bs_price,
    bs_vega,
)

# ==========================================
# 1. 核心配置 & 视觉系统 (Visual Identity)
# ==========================================
st.set_page_config(
    page_title="QUANT | TERMINAL",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 调色板：黑金流光 (Professional Dark Mode)
COLORS = {
    "bg": "#0E1117",
    "card_bg": "#161B22",
    "border": "#30363D",
    "text_main": "#E6EDF3",
    "text_sub": "#8B949E",
    "gold": "#D29922",       # 更加沉稳的金色
    "gold_dim": "rgba(210, 153, 34, 0.15)",
    "red": "#F85149",
    "green": "#3FB950",
    "blue": "#58A6FF",
    "grid": "#21262D"
}

# Session State 初始化
if "bootstrap_returns" not in st.session_state:
    st.session_state["bootstrap_returns"] = None
if "fitted_normal_params" not in st.session_state:
    st.session_state["fitted_normal_params"] = None
if "input_model_choice" not in st.session_state:
    st.session_state["input_model_choice"] = "Normal"
if "show_welcome" not in st.session_state:
    st.session_state["show_welcome"] = True
if "user_has_run_backtest" not in st.session_state:
    st.session_state["user_has_run_backtest"] = False
if "user_has_run_projection" not in st.session_state:
    st.session_state["user_has_run_projection"] = False
if "show_settings_dialog" not in st.session_state:
    st.session_state["show_settings_dialog"] = False
if "show_input_modeling_dialog" not in st.session_state:
    st.session_state["show_input_modeling_dialog"] = False
if "backtest_history" not in st.session_state:
    st.session_state["backtest_history"] = []
if "strategy_comparison" not in st.session_state:
    st.session_state["strategy_comparison"] = []
if "transaction_cost_rate" not in st.session_state:
    st.session_state["transaction_cost_rate"] = 0.001  # 默认0.1%交易成本
if "slippage_rate" not in st.session_state:
    st.session_state["slippage_rate"] = 0.0005  # 默认0.05%滑点

# 注入极简轻奢 CSS (Bloomberg Terminal Style)
st.markdown(f"""
    <style>
        /* 引入字体 */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

        /* 全局重置 */
        .stApp {{
            background-color: {COLORS['bg']};
            font-family: 'Inter', sans-serif;
            color: {COLORS['text_main']};
        }}
        
        /* 紧凑布局 */
        .block-container {{
            padding-top: 2rem !important;
            padding-bottom: 3rem !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
        }}
        
        /* 侧边栏 */
        [data-testid="stSidebar"] {{
            background-color: #010409;
            border-right: 1px solid {COLORS['border']};
        }}
        
        /* 标题排版 */
        h1, h2, h3 {{
            font-family: 'Inter', sans-serif;
            font-weight: 400 !important;
            letter-spacing: 1px !important;
            text-transform: uppercase;
            color: {COLORS['text_main']};
        }}
        h4, h5, h6 {{
            color: {COLORS['text_sub']};
            font-weight: 500;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 1rem;
        }}
        
        /* 输入框美化 */
        .stTextInput > div > div, .stNumberInput > div > div, .stSelectbox > div > div {{
            background-color: #0D1117;
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
            color: {COLORS['text_main']};
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
        }}
        .stTextInput > div > div:focus-within, .stNumberInput > div > div:focus-within {{
            border-color: {COLORS['gold']};
            box-shadow: none;
        }}

        /* 按钮美化 */
        .stButton button {{
            background: transparent;
            border: 1px solid {COLORS['border']};
            color: {COLORS['gold']};
            font-family: 'JetBrains Mono', monospace;
            font-weight: 500;
            text-transform: uppercase;
            border-radius: 4px;
            transition: all 0.2s;
        }}
        .stButton button:hover {{
            border-color: {COLORS['gold']};
            background: {COLORS['gold_dim']};
            color: {COLORS['gold']};
        }}
        .stButton button:active {{
            background: {COLORS['gold']};
            color: #000;
        }}

        /* Metric 卡片 */
        div[data-testid="metric-container"] {{
            background-color: {COLORS['card_bg']};
            border: 1px solid {COLORS['border']};
            padding: 10px 15px;
            border-radius: 6px;
        }}
        div[data-testid="metric-container"] label {{
            font-size: 0.7rem;
            letter-spacing: 1px;
            color: {COLORS['text_sub']};
        }}
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.5rem;
            color: {COLORS['text_main']};
        }}
        
        /* Tabs 样式 */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 24px;
            border-bottom: 1px solid {COLORS['border']};
        }}
        .stTabs [data-baseweb="tab"] {{
            height: 40px;
            white-space: pre-wrap;
            border-radius: 4px 4px 0 0;
            color: {COLORS['text_sub']};
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
        }}
        .stTabs [aria-selected="true"] {{
            color: {COLORS['gold']} !important;
            border-bottom-color: {COLORS['gold']} !important;
            background-color: transparent;
        }}
        
        /* Expander */
        .streamlit-expanderHeader {{
            background-color: {COLORS['card_bg']};
            color: {COLORS['text_main']};
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
        }}
        
        /* 去除页脚 */
        footer {{visibility: hidden;}}
        #MainMenu {{visibility: hidden;}}
        
        /* 自定义分割线 */
        hr {{
            border: 0;
            border-top: 1px solid {COLORS['border']};
            margin: 1.5rem 0;
        }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 高级绘图函数 (Plotly Refined)
# ==========================================

# ==========================================
# 风险指标计算辅助函数
# ==========================================

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """计算 Sortino 比率（只考虑下行波动率）"""
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return 0.0
    
    downside_std = downside_returns.std() * np.sqrt(periods_per_year)
    if downside_std == 0:
        return 0.0
    
    annualized_return = returns.mean() * periods_per_year
    return (annualized_return - risk_free_rate) / downside_std

def calculate_calmar_ratio(annualized_return: float, max_drawdown: float) -> float:
    """计算 Calmar 比率（年化收益 / 最大回撤）"""
    if max_drawdown == 0:
        return 0.0
    return annualized_return / abs(max_drawdown)

def calculate_max_drawdown_duration(portfolio_values: pd.Series) -> int:
    """计算最大回撤持续时间（天数）"""
    cumulative_peaks = portfolio_values.expanding().max()
    drawdowns = (portfolio_values - cumulative_peaks) / cumulative_peaks
    
    max_dd = drawdowns.min()
    max_dd_date = drawdowns.idxmin()
    
    # 找到回撤开始日期（峰值日期）
    peak_date = portfolio_values[:max_dd_date].idxmax()
    
    # 计算持续时间
    duration = (max_dd_date - peak_date).days
    return max(0, duration)

def get_chart_layout(height=400):
    return dict(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=height,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(
            showgrid=True, 
            gridcolor=COLORS['grid'], 
            gridwidth=1,
            linecolor=COLORS['border'], 
            tickfont=dict(family='JetBrains Mono', color=COLORS['text_sub'], size=10)
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor=COLORS['grid'], 
            gridwidth=1,
            zerolinecolor=COLORS['border'],
            tickfont=dict(family='JetBrains Mono', color=COLORS['text_sub'], size=10)
        ),
        legend=dict(
            orientation="h", 
            y=1.02, x=1, 
            xanchor="right", 
            font=dict(family="Inter", size=10, color=COLORS['text_sub']),
            bgcolor='rgba(0,0,0,0)'
        ),
        hovermode="x unified"
    )

def plot_monte_carlo_fan(dates, paths, median_path):
    dates_arr = np.asarray(dates)
    p95 = np.percentile(paths, 95, axis=1)
    p05 = np.percentile(paths, 5, axis=1)
    p75 = np.percentile(paths, 75, axis=1)
    p25 = np.percentile(paths, 25, axis=1)

    fig = go.Figure()
    
    # 90% Confidence Interval
    fig.add_trace(go.Scatter(
        x=np.concatenate([dates_arr, dates_arr[::-1]]),
        y=np.concatenate([p95, p05[::-1]]),
        fill='toself', fillcolor='rgba(210, 153, 34, 0.05)',
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))

    # 50% Confidence Interval
    fig.add_trace(go.Scatter(
        x=np.concatenate([dates_arr, dates_arr[::-1]]),
        y=np.concatenate([p75, p25[::-1]]),
        fill='toself', fillcolor='rgba(210, 153, 34, 0.15)',
        line=dict(width=0), name='50% Conf. Interval'
    ))

    # Median
    fig.add_trace(go.Scatter(
        x=dates_arr, y=median_path, mode='lines',
        line=dict(color=COLORS['gold'], width=2),
        name='Median'
    ))

    fig.update_layout(**get_chart_layout(450))
    fig.update_layout(title="Projected Wealth Cone")
    return fig

def plot_nav_curve(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Portfolio'],
        mode='lines', name='Strategy',
        line=dict(color=COLORS['gold'], width=2),
        fill='tozeroy', fillcolor='rgba(210, 153, 34, 0.05)'
    ))
    fig.update_layout(**get_chart_layout(400))
    fig.update_layout(title="Net Asset Value")
    return fig

def render_hud_card(label, value, sub_value=None, sub_color=COLORS['text_sub']):
    """渲染 HTML 风格的 HUD 卡片 (Deprecated in favor of st.metric for this version but kept for compatibility)"""
    st.metric(label, value, sub_value)

def describe_input_model(model: Optional[dict]) -> str:
    if not model:
        return "输入模型：默认 normal 分布。"
    params = model.get("params", {})
    params_text = ", ".join(f"{k}={v}" for k, v in params.items()) or "N/A"
    return f"Model: {model.get('dist_name', 'normal')} ({params_text})"

def generate_backtest_report_markdown(
    strategy_name: str,
    initial_capital: float,
    leverage: float,
    risk_free_rate: float,
    metrics: dict,
    sortino: float,
    calmar: float,
    max_dd_duration: int,
    portfolio_returns: Optional[np.ndarray],
    input_model_info: Optional[dict] = None,
    conclusion_data: Optional[dict] = None
) -> str:
    """生成完整的回测报告Markdown文档"""
    from datetime import datetime
    
    report_time = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
    
    # 计算综合评分（与UI中相同的逻辑）
    score = 0
    if metrics['total_return'] > 0.2:
        ret_score = 30
    elif metrics['total_return'] > 0.1:
        ret_score = 20
    elif metrics['total_return'] > 0:
        ret_score = 10
    else:
        ret_score = 0
    score += ret_score
    
    sharpe_score = min(30, max(0, int(metrics['sharpe'] * 10)))
    score += sharpe_score
    
    if metrics['max_dd'] > -0.1:
        risk_score = 20
    elif metrics['max_dd'] > -0.2:
        risk_score = 15
    elif metrics['max_dd'] > -0.3:
        risk_score = 10
    else:
        risk_score = 5
    score += risk_score
    
    vol_score = max(0, 20 - int(metrics['volatility'] * 100))
    score += vol_score
    
    if score >= 80:
        overall_rating = "优秀 ⭐⭐⭐⭐⭐"
        recommendation = "强烈推荐"
    elif score >= 65:
        overall_rating = "良好 ⭐⭐⭐⭐"
        recommendation = "推荐"
    elif score >= 50:
        overall_rating = "一般 ⭐⭐⭐"
        recommendation = "可考虑"
    elif score >= 35:
        overall_rating = "较差 ⭐⭐"
        recommendation = "需改进"
    else:
        overall_rating = "差 ⭐"
        recommendation = "不推荐"
    
    # 生成报告内容
    report = f"""# 投资组合回测分析报告

**生成时间**: {report_time}  
**报告类型**: 历史回测分析

---

## 一、输入建模信息

"""
    
    # 输入建模信息
    if input_model_info:
        dist_name = input_model_info.get("dist_name", "Normal")
        params = input_model_info.get("params", {})
        report += f"""
**选择的分布模型**: {dist_name}

**分布参数**:
"""
        for key, value in params.items():
            if isinstance(value, float):
                report += f"- {key}: {value:.6f}\n"
            else:
                report += f"- {key}: {value}\n"
    else:
        input_model_choice = st.session_state.get("input_model_choice", "Normal")
        report += f"""
**选择的分布模型**: {input_model_choice}

**说明**: 本次回测使用历史数据，未进行输入建模分析。如需进行输入建模，请在"输入建模"功能中分析数据分布特征。
"""
    
    report += f"""

---

## 二、回测配置

**策略算法**: {strategy_name}  
**初始资本**: {initial_capital:,.2f} 元  
**杠杆比率**: {leverage:.2f}x  
**无风险利率**: {risk_free_rate:.2%}  

---

## 三、回测结果

### 3.1 核心绩效指标

| 指标 | 数值 |
|------|------|
| 总收益率 | {metrics['total_return']:.2%} |
| 年化收益率 | {metrics.get('annualized_return', 0):.2%} |
| Sharpe比率 | {metrics['sharpe']:.2f} |
| Sortino比率 | {sortino:.2f} |
| Calmar比率 | {calmar:.2f} |
| 最大回撤 | {metrics['max_dd']:.2%} |
| 最大回撤持续时间 | {max_dd_duration} 天 |
| 波动率 | {metrics['volatility']:.2%} |

### 3.2 风险指标

"""
    
    if portfolio_returns is not None and len(portfolio_returns) > 0:
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        report += f"""
| 指标 | 数值 |
|------|------|
| VaR (95%) | {var_95:.2%} |
| CVaR (95%) | {cvar_95:.2%} |
"""
    else:
        report += "风险指标数据不可用。\n"
    
    report += f"""

### 3.3 综合评估

**综合评分**: {score}/100  
**总体评价**: {overall_rating}  
**建议**: {recommendation}

---

## 四、策略优势分析

"""
    
    # 策略优势
    advantages = []
    if metrics['sharpe'] > 1.5:
        advantages.append("**风险调整后收益优秀** - Sharpe比率超过1.5，说明策略在控制风险的同时获得了良好收益")
    elif metrics['sharpe'] > 1.0:
        advantages.append("**风险调整后收益良好** - Sharpe比率超过1.0，策略表现优于市场平均水平")
    
    if metrics['max_dd'] > -0.15:
        advantages.append("**回撤控制良好** - 最大回撤小于15%，风险控制能力较强")
    
    if sortino > 1.5:
        advantages.append("**下行风险控制优秀** - Sortino比率较高，说明策略在下跌时表现更好")
    
    if calmar > 1.0:
        advantages.append("**收益回撤比优秀** - Calmar比率超过1.0，说明收益能力远强于最大损失")
    
    if metrics['volatility'] < 0.15:
        advantages.append("**波动率较低** - 组合波动性较小，适合稳健型投资者")
    
    if not advantages:
        advantages.append("策略表现中规中矩，无明显突出优势")
    
    for adv in advantages:
        report += f"- {adv}\n"
    
    report += "\n### 4.2 需要关注的风险点\n\n"
    
    # 风险关注点
    concerns = []
    if metrics['total_return'] < 0:
        concerns.append("**出现亏损** - 总收益率为负，需要重新评估策略或市场环境")
    elif metrics['total_return'] < 0.05:
        concerns.append("**收益偏低** - 总收益率低于5%，可能不如无风险资产")
    
    if metrics['sharpe'] < 0.5:
        concerns.append("**风险调整收益较差** - Sharpe比率低于0.5，风险收益比不理想")
    
    if metrics['max_dd'] < -0.3:
        concerns.append("**回撤较大** - 最大回撤超过30%，风险较高，需要评估承受能力")
    
    if metrics['volatility'] > 0.25:
        concerns.append("**波动率较高** - 组合波动性较大，可能不适合风险厌恶型投资者")
    
    if sortino < 0.5:
        concerns.append("**下行风险控制不足** - Sortino比率较低，下跌时损失可能较大")
    
    if not concerns:
        concerns.append("策略表现良好，无明显风险点")
    
    for concern in concerns:
        report += f"- {concern}\n"
    
    report += f"""

---

## 五、策略适用性评估

### 5.1 适合的投资者类型

"""
    
    investor_types = []
    if metrics['volatility'] < 0.12 and metrics['max_dd'] > -0.15:
        investor_types.append("✅ **风险厌恶型** - 低波动、低回撤")
    
    if metrics['sharpe'] > 1.0 and metrics['total_return'] > 0.1:
        investor_types.append("✅ **平衡型** - 收益风险平衡")
    
    if metrics['total_return'] > 0.15 and metrics['sharpe'] > 1.2:
        investor_types.append("✅ **成长型** - 追求较高收益")
    
    if not investor_types:
        investor_types.append("⚠️ 需要根据个人风险偏好谨慎评估")
    
    for it in investor_types:
        report += f"{it}\n"
    
    report += "\n### 5.2 市场环境适应性\n\n"
    
    market_conditions = []
    if metrics['sharpe'] > 1.0:
        market_conditions.append("✅ **趋势市场** - 表现良好")
    
    if sortino > metrics['sharpe']:
        market_conditions.append("✅ **震荡市场** - 下行风险控制好")
    
    if metrics['volatility'] < 0.15:
        market_conditions.append("✅ **波动市场** - 稳定性好")
    
    if not market_conditions:
        market_conditions.append("⚠️ 需要结合具体市场环境分析")
    
    for mc in market_conditions:
        report += f"{mc}\n"
    
    report += "\n### 5.3 优化建议\n\n"
    
    optimizations = []
    if metrics['sharpe'] < 1.0:
        optimizations.append("💡 考虑调整策略参数以提高风险调整收益")
    
    if metrics['max_dd'] < -0.2:
        optimizations.append("💡 增加风险控制措施，降低最大回撤")
    
    if metrics['volatility'] > 0.2:
        optimizations.append("💡 考虑增加低波动资产以降低组合波动")
    
    if calmar < 0.5:
        optimizations.append("💡 优化收益回撤比，提高策略效率")
    
    if not optimizations:
        optimizations.append("✅ 策略表现良好，可继续使用")
    
    for opt in optimizations:
        report += f"{opt}\n"
    
    report += f"""

---

## 六、最终结论与决策建议

### 6.1 策略表现总结

本次回测显示，**{strategy_name}**策略在测试期间取得了{'良好' if score >= 65 else '一般' if score >= 50 else '较差'}的表现。

**核心发现：**
- 总收益率为 **{metrics['total_return']:.2%}**，{'表现优秀' if metrics['total_return'] > 0.15 else '表现良好' if metrics['total_return'] > 0.05 else '表现一般' if metrics['total_return'] > 0 else '出现亏损'}
- 风险调整后收益（Sharpe比率）为 **{metrics['sharpe']:.2f}**，{'优于市场平均水平' if metrics['sharpe'] > 1.0 else '低于市场平均水平'}
- 最大回撤为 **{metrics['max_dd']:.2%}**，{'风险控制良好' if metrics['max_dd'] > -0.15 else '风险控制一般' if metrics['max_dd'] > -0.25 else '风险较高'}
- 组合波动率为 **{metrics['volatility']:.2%}**，{'波动性较低' if metrics['volatility'] < 0.15 else '波动性中等' if metrics['volatility'] < 0.25 else '波动性较高'}

### 6.2 决策建议

{'✅ 该策略表现优秀，建议继续使用或适当增加配置' if score >= 80 else '✅ 该策略表现良好，可以继续使用' if score >= 65 else '⚠️ 该策略表现一般，建议优化参数或考虑其他策略' if score >= 50 else '❌ 该策略表现较差，建议重新评估或更换策略'}

### 6.3 风险提示

- ⚠️ **历史表现不代表未来收益** - 回测结果基于历史数据，实际投资可能面临不同市场环境
- ⚠️ **市场环境变化** - 策略在不同市场环境下表现可能差异较大
- ⚠️ **风险承受能力** - 建议结合个人风险承受能力做出最终决策
- ⚠️ **分散投资** - 建议不要将所有资金投入单一策略

---

## 七、附录

### 7.1 图表说明

本报告包含以下可视化分析（详见系统界面）：

1. **净值曲线（NAV Curve）** - 展示投资组合价值随时间的变化
2. **回撤分析（Drawdown Analysis）** - 展示组合从峰值下降的幅度
3. **收益分布（Returns Distribution）** - 展示收益率的统计分布特征
4. **资产权重（Asset Weights）** - 展示各资产在组合中的配置变化
5. **滚动分析（Rolling Analysis）** - 展示关键指标的滚动窗口分析

### 7.2 指标说明

- **总收益率**: 整个回测期间的总收益百分比
- **年化收益率**: 将总收益率年化后的数值
- **Sharpe比率**: 风险调整后收益指标，数值越高越好
- **Sortino比率**: 只考虑下行风险的风险调整收益指标
- **Calmar比率**: 年化收益与最大回撤的比值
- **最大回撤**: 从峰值到谷底的最大跌幅
- **波动率**: 收益率的标准差，衡量风险
- **VaR (95%)**: 95%置信度下的风险价值
- **CVaR (95%)**: 95%置信度下的条件风险价值

---

**报告生成**: Invest-Sim 投资组合模拟系统  
**版本**: 1.0  
**免责声明**: 本报告仅供参考，不构成投资建议。投资有风险，决策需谨慎。
"""
    
    return report

# ==========================================
# 3. Derivatives Lab (UI 重构版)
# ==========================================

def render_derivatives_lab() -> None:
    """
    Modernized Derivatives Lab UI
    Layout: Split View (Control Deck | Analysis Dashboard)
    """

def render_derivatives_lab() -> None:
    """
    Modernized Derivatives Lab UI
    Layout: Split View (Control Deck | Analysis Dashboard)
    """
    
    # --- HEADER: Market Ticker ---
    # 使用 Container 模拟顶部状态栏
    with st.container():
        h1, h2, h3, h4 = st.columns([1.5, 1, 1, 1])
        with h1:
            st.markdown("### ❖ DERIVATIVES LAB <span style='font-size:12px; color:#8B949E; border:1px solid #30363D; padding:2px 6px; border-radius:4px;'>PRO</span>", unsafe_allow_html=True)
        with h2:
            spot_price = st.number_input("SPOT PRICE", value=100.0, step=0.5, format="%.2f")
        with h3:
            implied_vol = st.number_input("IMPLIED VOL (σ)", value=0.20, step=0.01, format="%.2f")
        with h4:
            days_to_maturity = st.number_input("DAYS TO EXP", value=30, step=1)
    
    st.markdown("---")

    # --- MAIN SPLIT LAYOUT ---
    col_controls, col_dashboard = st.columns([1, 2.2], gap="large")

    # =========================================================
    # LEFT PANEL: CONTROL DECK
    # =========================================================
    with col_controls:
        # 1. Strategy Configuration
        st.markdown("##### 🛠 STRATEGY CONFIG")
        with st.container():
            strategy_name = st.selectbox(
                "Strategy Template",
                [
                    "Single Leg", "Vertical Spread (Bull Call)", "Vertical Spread (Bear Put)",
                    "Straddle", "Strangle", "Butterfly (Call)", "Iron Condor", "Custom (Manual Legs)"
                ]
            )
            
            # Dynamic Params
            spread_width = strangle_distance = wing_width = ic_width = ic_width2 = None
            
            # Base Params
            c_leg1, c_leg2 = st.columns(2)
            with c_leg1: strike_price = st.number_input("Anchor Strike", value=100.0, step=1.0)
            with c_leg2: contract_size = st.number_input("Size", value=100, step=1)

            # Strategy Specific Inputs
            if strategy_name in ["Vertical Spread (Bull Call)", "Vertical Spread (Bear Put)"]:
                spread_width = st.number_input("Spread Width", value=5.0)
            elif strategy_name == "Strangle":
                strangle_distance = st.number_input("Strangle Dist", value=5.0)
            elif strategy_name == "Butterfly (Call)":
                wing_width = st.number_input("Wing Width", value=5.0)
            elif strategy_name == "Iron Condor":
                ic_c1, ic_c2 = st.columns(2)
                with ic_c1: ic_width = st.number_input("Short Width", value=5.0)
                with ic_c2: ic_width2 = st.number_input("Long Width", value=10.0)
            elif strategy_name == "Single Leg":
                c_opt1, c_opt2 = st.columns(2)
                with c_opt1: option_type = st.selectbox("Type", ["Call", "Put"])
                with c_opt2: position_side = st.selectbox("Side", ["Long", "Short"])
            else:
                # Custom defaults
                option_type = "Call"
                position_side = "Long"

            # --- Logic: Build Strategy Legs ---
            def build_strategy_legs():
                legs = []
                if strategy_name == "Single Leg":
                    legs = [OptionLeg(option_type, position_side, strike_price, contract_size)]
                elif strategy_name == "Vertical Spread (Bull Call)" and spread_width:
                    legs = [
                        OptionLeg("call", "long", strike_price, contract_size),
                        OptionLeg("call", "short", strike_price + spread_width, contract_size),
                    ]
                elif strategy_name == "Vertical Spread (Bear Put)" and spread_width:
                    legs = [
                        OptionLeg("put", "long", strike_price, contract_size),
                        OptionLeg("put", "short", strike_price - spread_width, contract_size),
                    ]
                elif strategy_name == "Straddle":
                    legs = [
                        OptionLeg("call", "long", strike_price, contract_size),
                        OptionLeg("put", "long", strike_price, contract_size),
                    ]
                elif strategy_name == "Strangle" and strangle_distance:
                    legs = [
                        OptionLeg("call", "long", strike_price + strangle_distance, contract_size),
                        OptionLeg("put", "long", strike_price - strangle_distance, contract_size),
                    ]
                elif strategy_name == "Butterfly (Call)" and wing_width:
                    legs = [
                        OptionLeg("call", "long", strike_price - wing_width, contract_size),
                        OptionLeg("call", "short", strike_price, 2 * contract_size),
                        OptionLeg("call", "long", strike_price + wing_width, contract_size),
                    ]
                elif strategy_name == "Iron Condor" and ic_width and ic_width2:
                    legs = [
                        OptionLeg("call", "short", strike_price + ic_width, contract_size),
                        OptionLeg("call", "long", strike_price + ic_width2, contract_size),
                        OptionLeg("put", "short", strike_price - ic_width, contract_size),
                        OptionLeg("put", "long", strike_price - ic_width2, contract_size),
                    ]
                else:
                    # Fallback / Custom
                    legs = [OptionLeg("call", "long", strike_price, contract_size)]
                return legs
            
            strategy_legs = build_strategy_legs()
            # For pricing compatibility if Single Leg
            if strategy_name != "Single Leg":
                # Dummy values for single-leg functions to avoid errors, 
                # though multi-leg usually aggregates.
                option_type_calc = "Call" 
                position_side_calc = "Long"
            else:
                option_type_calc = option_type
                position_side_calc = position_side

        st.markdown("---")

        # 2. Advanced Environment Configs
        with st.expander("⚙️ RISK & MARGIN PARAMETERS", expanded=False):
            st.caption("Environment")
            risk_free_rate = st.number_input("Risk Free Rate (r)", value=0.02, step=0.005, format="%.3f")
            
            st.caption("Margin Rules")
            m1, m2 = st.columns(2)
            with m1: initial_margin = st.number_input("Init Margin", value=0.2)
            with m2: maint_margin = st.number_input("Maint Margin", value=0.1)
            
            scan_risk = st.number_input("Scan Risk Factor", value=0.20)
            min_margin = st.number_input("Min Margin Factor", value=0.10)
            
            st.caption("Delta Hedging")
            enable_hedge = st.checkbox("Active Hedging", value=False)
            if enable_hedge:
                h1, h2 = st.columns(2)
                with h1: hedge_freq = st.number_input("Freq (Days)", 1, value=1)
                with h2: hedge_thr = st.number_input("Delta Thr", 0.0, value=0.0)
            else:
                hedge_freq, hedge_thr = 1, 0.0

            st.caption("Volatility Model")
            dynamic_vol = st.checkbox("Dynamic Vol (Crash)", value=False)
            vol_sens = st.number_input("Vol Sensitivity (k)", 0.0, value=5.0) if dynamic_vol else 0.0
        
        with st.expander("🎲 SIMULATION ASSUMPTIONS", expanded=False):
            st.caption("Random Walk Parameters")
            sim_mu = st.number_input("Drift (Daily)", value=0.0005, format="%.6f")
            sim_sigma = st.number_input("Vol (Daily)", value=0.02, format="%.4f")
            ref_equity = st.number_input("Ref. Equity", value=100000.0, step=10000.0)

    # =========================================================
    # RIGHT PANEL: ANALYSIS DASHBOARD
    # =========================================================
    with col_dashboard:
        
        # --- SECTION 1: LIVE GREEKS & PAYOFF (Interactive) ---
        T_years = days_to_maturity / 365.0
        
        # Calculate Greeks "On the Fly" for Anchor Leg (for display purposes)
        # Note: True multi-leg Greeks are complex sums, here we show Anchor or indicative
        bs_p = bs_d = bs_g = bs_v = 0.0
        
        # Try to calculate BS for the "Anchor" strike/type
        # If multi-leg, we use the first leg or the anchor input
        calc_type = option_type_calc if strategy_name == "Single Leg" else "call" # Default to Call for generic view
        
        try:
            bs_p = float(np.squeeze(bs_price(spot_price, strike_price, T_years, risk_free_rate, implied_vol, calc_type)))
            bs_d = float(np.squeeze(bs_delta(spot_price, strike_price, T_years, risk_free_rate, implied_vol, calc_type)))
            bs_g = float(np.squeeze(bs_gamma(spot_price, strike_price, T_years, risk_free_rate, implied_vol)))
            bs_v = float(np.squeeze(bs_vega(spot_price, strike_price, T_years, risk_free_rate, implied_vol)))
        except:
            pass
        
        # Display Greeks
        st.markdown(f"##### ⚡ LIVE METRICS (Anchor: {calc_type.title()} @ {strike_price})")
        g1, g2, g3, g4 = st.columns(4)
        with g1: st.metric("BS Price", f"${bs_p:.2f}")
        with g2: st.metric("Delta", f"{bs_d:.3f}", delta_color="off")
        with g3: st.metric("Gamma", f"{bs_g:.4f}", delta_color="off")
        with g4: st.metric("Vega", f"{bs_v:.2f}", delta_color="off")

        # Payoff Chart (Always visible)
        s_grid = np.linspace(0.5 * spot_price, 1.5 * spot_price, 200)
        payoff = np.zeros_like(s_grid)
        for leg in strategy_legs:
            intrinsic = np.maximum(s_grid - leg.strike, 0) if leg.option_type == "call" else np.maximum(leg.strike - s_grid, 0)
            payoff += leg.multiplier * intrinsic * leg.contract_size
        
        fig_payoff = go.Figure()
        fig_payoff.add_trace(go.Scatter(
            x=s_grid, y=payoff, mode="lines", 
            line=dict(color=COLORS['gold'], width=2), 
            fill='tozeroy', fillcolor='rgba(210, 153, 34, 0.1)',
            name="Payoff"
        ))
        fig_payoff.add_vline(x=spot_price, line=dict(color=COLORS['text_sub'], dash="dash"), annotation_text="Spot")
        fig_payoff.add_hline(y=0, line=dict(color=COLORS['border']))
        fig_payoff.update_layout(
            title="Strategy Payoff at Maturity",
            **get_chart_layout(300)
        )
        st.plotly_chart(fig_payoff, use_container_width=True)

        # --- SECTION 2: SIMULATION ENGINE (Tabs) ---
        st.markdown("##### 🔬 SIMULATION LAB")
        
        tab_static, tab_path, tab_mc = st.tabs(["📊 MARGIN ANALYSIS", "📈 PATH SIMULATOR", "🎲 MONTE CARLO"])
        
        # --- TAB 1: STATIC MARGIN ---
        with tab_static:
            st.caption("Analyze Short Option Margin Requirements vs Underlying Price.")
            
            if st.button("Compute Margin Curve", key="btn_static", use_container_width=True):
                # Logic copied from original
                if position_side_calc != "Short" and strategy_name == "Single Leg":
                    st.warning("Switch side to 'Short' to see relevant margin data.")
                else:
                    s_grid_m = np.linspace(0.5 * strike_price, 1.5 * strike_price, 100)
                    # Simplified margin scan logic for the Anchor Leg (Short)
                    # For complex strategies, this needs full portfolio margin logic (backend dependent)
                    # Here we approximate using the single leg logic for demonstration or the first leg
                    
                    # Compute Price Curve
                    price_curve = bs_price(s_grid_m, strike_price, T_years, risk_free_rate, implied_vol, "call" if "Call" in strategy_name else "put")
                    otm = np.maximum(strike_price - s_grid_m, 0) if "Call" in strategy_name else np.maximum(s_grid_m - strike_price, 0)
                    
                    scan_part = price_curve + scan_risk * s_grid_m - otm
                    min_part = price_curve + min_margin * s_grid_m
                    margin_per_unit = np.maximum(np.maximum(scan_part, min_part), 0.0)
                    margin_per_contract = margin_per_unit * contract_size
                    
                    fig_margin = go.Figure()
                    fig_margin.add_trace(go.Scatter(x=s_grid_m, y=margin_per_contract, mode="lines", name="Margin Req", line=dict(color=COLORS['red'])))
                    fig_margin.add_hline(y=ref_equity, line=dict(color=COLORS['text_sub'], dash="dash"), annotation_text="Equity")
                    fig_margin.update_layout(title="Margin Req vs Spot", **get_chart_layout(300))
                    st.plotly_chart(fig_margin, use_container_width=True)

        # --- TAB 2: SINGLE PATH ---
        with tab_path:
            p_col1, p_col2 = st.columns(2)
            with p_col1: sim_days = st.number_input("Duration (Days)", 10, 365, 60, key="path_days")
            with p_col2: 
                st.markdown(f"<div style='padding-top:28px'></div>", unsafe_allow_html=True)
                run_path = st.button("▶ Run Path Simulation", key="btn_path", use_container_width=True)
            
            if run_path:
                simulator = OptionMarginSimulator(
                    option_type_calc, position_side_calc, strike_price, contract_size, spot_price,
                    implied_vol, risk_free_rate, days_to_maturity, scan_risk, min_margin, maint_margin, 
                    sim_mu, sim_sigma, ref_equity,
                    enable_hedge=enable_hedge, hedge_frequency=hedge_freq, hedge_threshold=hedge_thr,
                    dynamic_vol=dynamic_vol, vol_sensitivity=vol_sens, legs=strategy_legs
                )
                res = simulator.run_single_path(sim_days)
                
                # Plotting
                c1, c2 = st.columns(2)
                with c1:
                    fig_spot = go.Figure()
                    fig_spot.add_trace(go.Scatter(y=res['spot_path'], name='Spot', line=dict(color=COLORS['gold'])))
                    fig_spot.update_layout(title="Spot Price Path", **get_chart_layout(250))
                    st.plotly_chart(fig_spot, use_container_width=True)
                with c2:
                    fig_eq = go.Figure()
                    fig_eq.add_trace(go.Scatter(y=res['equity_path'], name='Equity', line=dict(color=COLORS['green'])))
                    fig_eq.add_trace(go.Scatter(y=res['margin_path'], name='Margin', line=dict(color=COLORS['red'])))
                    if res['liquidation_day']:
                        fig_eq.add_vline(x=res['liquidation_day'], line=dict(color='white', dash='dot'))
                    fig_eq.update_layout(title="Equity vs Margin", **get_chart_layout(250))
                    st.plotly_chart(fig_eq, use_container_width=True)

        # --- TAB 3: MONTE CARLO ---
        with tab_mc:
            mc_c1, mc_c2 = st.columns(2)
            with mc_c1: mc_paths = st.number_input("Paths", 100, 5000, 500)
            with mc_c2: 
                st.markdown(f"<div style='padding-top:28px'></div>", unsafe_allow_html=True)
                run_mc = st.button("▶ Run Monte Carlo", key="btn_mc", type="primary", use_container_width=True)
            
            if run_mc:
                with st.spinner("Simulating Scenarios..."):
                    simulator = OptionMarginSimulator(
                        option_type_calc, position_side_calc, strike_price, contract_size, spot_price,
                        implied_vol, risk_free_rate, days_to_maturity, scan_risk, min_margin, maint_margin,
                        sim_mu, sim_sigma, ref_equity,
                        enable_hedge=enable_hedge, hedge_frequency=hedge_freq, hedge_threshold=hedge_thr,
                        dynamic_vol=dynamic_vol, vol_sensitivity=vol_sens, legs=strategy_legs
                    )
                    mc_days_input = sim_days # Reuse from prev tab or add new input
                    mc_res = simulator.run_monte_carlo(mc_paths, mc_days_input)
                    
                    # Metrics
                    breaches = (mc_res['liquidation_days'] < mc_days_input).mean()
                    final_eq = mc_res['equity_paths'][:, -1]
                    
                    m1, m2, m3 = st.columns(3)
                    with m1: st.metric("Margin Call Prob", f"{breaches:.1%}")
                    with m2: st.metric("Median Equity", f"${np.median(final_eq):,.0f}")
                    with m3: st.metric("CVaR (5%)", f"${np.percentile(final_eq, 5):,.0f}", delta_color="inverse")
                    
                    # Fan Chart
                    st.plotly_chart(
                        plot_monte_carlo_fan(
                            np.arange(mc_days_input+1), 
                            mc_res['equity_paths'], 
                            np.median(mc_res['equity_paths'], axis=0)
                        ), 
                        use_container_width=True
                    )
                    
                    # Worst Paths
                    st.markdown("###### Worst Case Scenarios")
                    worst_indices = np.argsort(final_eq)[:3]
                    fig_worst = go.Figure()
                    for idx in worst_indices:
                        fig_worst.add_trace(go.Scatter(y=mc_res['equity_paths'][idx], mode='lines', line=dict(width=1), name=f"Path {idx}"))
                    fig_worst.add_trace(go.Scatter(y=mc_res['equity_paths'].mean(axis=0), mode='lines', line=dict(color=COLORS['gold'], width=2), name="Avg"))
                    fig_worst.update_layout(title="Worst Equity Paths", **get_chart_layout(250))
                    st.plotly_chart(fig_worst, use_container_width=True)

# ==========================================
# 4. 侧边栏控制台 (Control Panel)
# ==========================================
st.sidebar.markdown("## INVEST SIM <span style='font-size:10px; opacity:0.5'>PRO</span>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# 模式选择
mode = st.sidebar.radio(
    "SYSTEM MODE",
    [
        "BACKTEST (Historical)",
        "PROJECTION (Monte Carlo)",
        "DERIVATIVES LAB (Options / Margin)",
    ],
    label_visibility="collapsed",
)

if mode != "DERIVATIVES LAB (Options / Margin)":
    # 初始化设置值
    if "settings_strategy" not in st.session_state:
        st.session_state["settings_strategy"] = "Equal Weight"
    if "settings_initial_capital" not in st.session_state:
        st.session_state["settings_initial_capital"] = 100000
    if "settings_leverage" not in st.session_state:
        st.session_state["settings_leverage"] = 1.0
    if "settings_risk_free" not in st.session_state:
        st.session_state["settings_risk_free"] = 0.02
    
    # 设置按钮和当前配置摘要
    st.sidebar.markdown("### ⚙️ CONFIGURATION")
    
    # 当前配置摘要卡片
    st.sidebar.markdown(f"""
    <div class="settings-summary">
        <div class="settings-summary-item">
            <span class="settings-summary-label">策略</span>
            <span class="settings-summary-value">{st.session_state["settings_strategy"]}</span>
        </div>
        <div class="settings-summary-item">
            <span class="settings-summary-label">初始资金</span>
            <span class="settings-summary-value">${st.session_state["settings_initial_capital"]:,.0f}</span>
        </div>
        <div class="settings-summary-item">
            <span class="settings-summary-label">杠杆</span>
            <span class="settings-summary-value">{st.session_state["settings_leverage"]}x</span>
        </div>
        <div class="settings-summary-item">
            <span class="settings-summary-label">无风险利率</span>
            <span class="settings-summary-value">{st.session_state["settings_risk_free"]:.1%}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 打开设置对话框按钮
    if st.sidebar.button("⚙️ 打开设置", use_container_width=True, type="primary"):
        st.session_state["show_settings_dialog"] = True
        st.rerun()
    
    # 打开输入建模对话框按钮
    if st.sidebar.button("📊 输入建模", use_container_width=True, type="secondary"):
        st.session_state["show_input_modeling_dialog"] = True
        st.rerun()
    
    # 策略说明
    strategy_descriptions = {
        "Fixed Weights": "保持固定权重分配，定期再平衡",
        "Target Risk": "根据目标波动率动态调整权重",
        "Adaptive Rebalance": "仅在权重偏离阈值时再平衡",
        "Equal Weight": "所有资产等权重分配（1/N策略）",
        "Risk Parity": "风险平价，各资产风险贡献相等",
        "Minimum Variance": "最小方差组合，优化波动率",
        "Momentum": "动量策略，增持表现好的资产",
        "Mean Reversion": "均值回归，反向调整偏离资产",
    }
    
    available_strategies = InvestSimBridge.get_available_strategies()
    
    # 使用session state中的值
    strategy_name_global = st.session_state["settings_strategy"]
    initial_capital = st.session_state["settings_initial_capital"]
    leverage = st.session_state["settings_leverage"]
    risk_free = st.session_state["settings_risk_free"]
    
    # 设置对话框（使用条件渲染替代st.dialog）
    if st.session_state.get("show_settings_dialog", False):
        # 使用容器和条件渲染实现弹窗效果
        st.markdown("---")
        st.markdown("### ⚙️ 投资配置设置")
        st.markdown("---")
        
        # 策略配置
        st.markdown("#### 📊 策略配置")
        strategy_name_global = st.selectbox(
            "Algorithm（策略算法）", 
            available_strategies,
            index=available_strategies.index(strategy_name_global) if strategy_name_global in available_strategies else 0,
            help="选择投资策略算法"
        )
        
        # 显示策略说明
        if strategy_name_global in strategy_descriptions:
            strategy_colors = {
                "Fixed Weights": "#58A6FF",
                "Target Risk": "#D29922",
                "Adaptive Rebalance": "#3FB950",
                "Equal Weight": "#58A6FF",
                "Risk Parity": "#D29922",
                "Minimum Variance": "#F85149",
                "Momentum": "#A371F7",
                "Mean Reversion": "#79C0FF",
            }
            color = strategy_colors.get(strategy_name_global, "#8B949E")
            st.info(f"💡 **{strategy_name_global}**: {strategy_descriptions[strategy_name_global]}")
        
        # 策略详细说明
        with st.expander("📚 策略详细说明", expanded=False):
            strategy_details = {
                "Fixed Weights": """
                ### 📌 固定权重策略
            
            **工作原理：**
            - 始终保持预设的目标权重分配
            - 定期再平衡，无论市场如何变化
            - 例如：60%股票 + 30%债券 + 10%现金，始终保持这个比例
            
            **适用场景：**
            - ✅ 长期投资者，相信资产配置的重要性
            - ✅ 希望策略简单可预测
            - ✅ 不追求市场择时
            
            **优点：**
            - 简单易懂，执行方便
            - 可预测性强
            - 交易成本相对较低
            
            **缺点：**
            - 不随市场变化调整
            - 可能错过市场机会
            - 风险控制能力有限
            """,
                "Target Risk": """
                ### 🎯 目标风险策略
            
            **工作原理：**
            - 根据市场波动动态调整资产权重
            - 保持组合整体风险（波动率）在目标水平
            - 市场波动大时降低风险资产，波动小时增加风险资产
            
            **适用场景：**
            - ✅ 风险敏感型投资者
            - ✅ 希望风险水平可控
            - ✅ 需要自动风险调整
            
            **优点：**
            - 风险可控，波动率稳定
            - 自动适应市场变化
            - 适合风险厌恶者
            
            **缺点：**
            - 可能降低收益潜力
            - 需要频繁调整
            - 参数设置影响大
            """,
                "Adaptive Rebalance": """
                ### 🔄 自适应再平衡策略
            
            **工作原理：**
            - 只在权重偏离目标超过阈值时才再平衡
            - 允许权重在一定范围内自然波动
            - 减少不必要的交易和成本
            
            **适用场景：**
            - ✅ 希望降低交易成本的投资者
            - ✅ 允许权重适度偏离
            - ✅ 长期持有策略
            
            **优点：**
            - 交易成本低
            - 允许权重自然波动（可能带来收益）
            - 减少过度交易
            
            **缺点：**
            - 权重可能长期偏离目标
            - 风险控制不如固定权重严格
            - 需要设置合适的阈值
            """,
                "Equal Weight": """
                ### ⚖️ 等权重策略（1/N策略）
            
            **工作原理：**
            - 所有资产分配相同权重（1/N，N为资产数量）
            - 例如：3个资产各占33.33%
            - 定期再平衡保持等权重
            
            **适用场景：**
            - ✅ 不确定如何分配权重的投资者
            - ✅ 追求简单有效的分散化
            - ✅ 不想做复杂的权重优化
            
            **优点：**
            - 极其简单，无需预测
            - 分散化效果好
            - 学术研究显示表现不错
            
            **缺点：**
            - 忽略资产特性差异
            - 可能不是最优配置
            - 对资产数量敏感
            """,
                "Risk Parity": """
                ### ⚡ 风险平价策略
            
            **工作原理：**
            - 根据资产波动率分配权重
            - 波动率低的资产权重更高，波动率高的权重更低
            - 使各资产的风险贡献相等
            
            **适用场景：**
            - ✅ 追求风险均衡的投资者
            - ✅ 希望真正分散风险
            - ✅ 不只看收益，更看风险
            
            **优点：**
            - 风险分散效果好
            - 波动率低的资产权重更高（如债券）
            - 风险贡献均衡
            
            **缺点：**
            - 可能降低收益潜力
            - 需要准确估计波动率
            - 计算相对复杂
            """,
                "Minimum Variance": """
                ### 📉 最小方差策略
            
            **工作原理：**
            - 基于资产间的协方差矩阵优化
            - 最小化组合整体波动率
            - 使用数学优化方法求解最优权重
            
            **适用场景：**
            - ✅ 风险厌恶型投资者
            - ✅ 追求最低波动率
            - ✅ 愿意牺牲部分收益换取稳定
            
            **优点：**
            - 波动率最低，风险最小
            - 基于数学优化，理论最优
            - 考虑资产相关性
            
            **缺点：**
            - 收益可能较低
            - 需要准确的协方差矩阵
            - 对数据质量要求高
            """,
                "Momentum": """
                ### 🚀 动量策略
            
            **工作原理：**
            - 增持近期表现好的资产（上涨趋势）
            - 减持近期表现差的资产（下跌趋势）
            - 相信"趋势会延续"的假设
            
            **适用场景：**
            - ✅ 相信趋势延续的投资者
            - ✅ 愿意跟随市场趋势
            - ✅ 追求超额收益
            
            **优点：**
            - 可能捕捉到趋势，获得超额收益
            - 顺应市场力量
            - 在趋势市场中表现好
            
            **缺点：**
            - 在震荡市场中可能表现差
            - 可能追涨杀跌
            - 需要设置合适的回看期
            """,
                "Mean Reversion": """
                ### 🔁 均值回归策略
            
            **工作原理：**
            - 当资产偏离目标权重时反向调整
            - 相信价格会回归均值
            - 低买高卖，反向操作
            
            **适用场景：**
            - ✅ 相信均值回归的投资者
            - ✅ 愿意逆势操作
            - ✅ 追求低买高卖
            
            **优点：**
            - 可能降低波动
            - 低买高卖，成本优势
            - 在震荡市场表现好
            
            **缺点：**
            - 在趋势市场中可能表现差
            - 需要设置合适的回归速度
            - 可能过早买入/卖出
                """,
            }
            if strategy_name_global in strategy_details:
                st.markdown(strategy_details[strategy_name_global])
            
            # 策略选择指南
            st.markdown("---")
            st.markdown("### 💡 策略选择指南")
            st.markdown("""
            **根据投资目标选择：**
            - 🎯 **追求稳定**：Fixed Weights, Equal Weight, Minimum Variance
            - 📈 **追求收益**：Momentum, Target Risk
            - ⚖️ **平衡收益风险**：Risk Parity, Adaptive Rebalance
            - 🔄 **降低波动**：Mean Reversion, Minimum Variance
            
            **根据市场环境：**
            - 📊 **趋势市场**：Momentum
            - 🔁 **震荡市场**：Mean Reversion
            - ⚡ **不确定**：Equal Weight, Risk Parity
            """)
        
        # 策略动态参数
        st.markdown("---")
        st.markdown("#### ⚙️ 策略参数")
        
        # 初始化策略参数到session state
        if strategy_name_global == "Target Risk":
            if "settings_target_vol" not in st.session_state:
                st.session_state["settings_target_vol"] = 0.15
            target_vol = st.slider("Target Volatility（目标波动率）", 0.05, 0.4, st.session_state["settings_target_vol"], 0.01,
                                  help="目标年化波动率")
            st.session_state["settings_target_vol"] = target_vol
        elif strategy_name_global == "Adaptive Rebalance":
            if "settings_threshold" not in st.session_state:
                st.session_state["settings_threshold"] = 0.05
            threshold = st.slider("Rebalance Threshold（再平衡阈值）", 0.01, 0.1, st.session_state["settings_threshold"], 0.01,
                                  help="权重偏离阈值，超过此值触发再平衡")
            st.session_state["settings_threshold"] = threshold
        elif strategy_name_global == "Momentum":
            if "settings_momentum_lookback" not in st.session_state:
                st.session_state["settings_momentum_lookback"] = 20
            if "settings_momentum_factor" not in st.session_state:
                st.session_state["settings_momentum_factor"] = 0.5
            momentum_lookback = st.slider("Lookback Periods（回看期数）", 5, 60, st.session_state["settings_momentum_lookback"], 5,
                                         help="动量计算的回看期数")
            momentum_factor = st.slider("Momentum Factor（动量因子）", 0.1, 1.0, st.session_state["settings_momentum_factor"], 0.1,
                                       help="动量调整强度")
            st.session_state["settings_momentum_lookback"] = momentum_lookback
            st.session_state["settings_momentum_factor"] = momentum_factor
        elif strategy_name_global == "Mean Reversion":
            if "settings_reversion_speed" not in st.session_state:
                st.session_state["settings_reversion_speed"] = 0.3
            reversion_speed = st.slider("Reversion Speed（回归速度）", 0.1, 1.0, st.session_state["settings_reversion_speed"], 0.1,
                                       help="均值回归速度，值越大回归越快")
            st.session_state["settings_reversion_speed"] = reversion_speed
        
        # 投资组合设置
        st.markdown("---")
        st.markdown("#### 💼 投资组合设置")
        
        # Initial Capital
        with st.expander("💰 初始资金说明", expanded=False):
            st.markdown("""
            **初始资金（Initial Capital）** 是投资组合的起始金额。
            
            **作用：**
            - 决定投资组合的起始规模
            - 影响最终收益的绝对值
            - 用于计算收益率和风险指标
            
            **设置建议：**
            - 💡 **新手**：$10,000 - $100,000（用于测试和学习）
            - 💡 **实际投资**：根据你的实际投资金额设置
            - 💡 **回测验证**：可以使用任意金额，收益率不受影响
            
            **注意事项：**
            - ⚠️ 金额过小（< $1,000）可能影响计算精度
            - ⚠️ 金额过大可能导致数值溢出
            - ✅ 收益率和风险指标与初始金额无关
            """)
        
        initial_capital = st.number_input("Initial Capital（初始资金）", value=st.session_state["settings_initial_capital"], 
                                        min_value=1000, max_value=100000000, step=10000,
                                        help="投资组合的起始金额")
        
        # Leverage Ratio
        with st.expander("⚖️ 杠杆比率说明", expanded=False):
            st.markdown("""
            **杠杆比率（Leverage Ratio）** 表示投资组合的杠杆倍数。
            
            **含义：**
            - **1.0x**：无杠杆，使用自有资金投资
            - **>1.0x**：使用杠杆，放大收益和风险
              - 1.5x = 使用50%的借款
              - 2.0x = 使用100%的借款（1:1杠杆）
              - 3.0x = 使用200%的借款（2:1杠杆）
            - **<1.0x**：保守投资，只使用部分资金
            
            **杠杆的影响：**
            - ✅ **收益放大**：盈利时收益成倍增加
            - ⚠️ **风险放大**：亏损时损失也成倍增加
            - ⚠️ **波动放大**：组合波动率成倍增加
            
            **使用建议：**
            - 💡 **新手**：建议使用 1.0x（无杠杆）
            - 💡 **稳健型**：0.5x - 1.0x
            - 💡 **激进型**：1.5x - 2.0x（需谨慎）
            - ⚠️ **高风险**：>2.0x 风险极高，可能导致爆仓
            
            **风险提示：**
            - ⚠️ 杠杆会放大所有风险指标
            - ⚠️ 高杠杆可能导致快速亏损
            - ⚠️ 需要足够的风险承受能力
            """)
        
        leverage = st.slider("Leverage Ratio（杠杆比率）", 0.5, 3.0, st.session_state["settings_leverage"], 0.1,
                             help="杠杆倍数，1.0表示无杠杆")
        
        if leverage > 2.0:
            st.warning("⚠️ 高杠杆增加风险，请谨慎使用")
        elif leverage > 1.5:
            st.info("💡 当前杠杆较高，请注意风险控制")
        
        # Risk Free Rate
        with st.expander("📈 无风险利率说明", expanded=False):
            st.markdown("""
            **无风险利率（Risk Free Rate）** 是用于计算风险调整收益的基准利率。
            
            **作用：**
            - 计算 **Sharpe比率**：衡量超额收益（超过无风险利率的部分）
            - 计算 **Sortino比率**：下行风险调整收益
            - 评估策略的 **风险调整后表现**
            
            **常用参考值：**
            - 🇺🇸 **美国**：2% - 3%（10年期国债收益率）
            - 🇨🇳 **中国**：2.5% - 3.5%（10年期国债收益率）
            - 🇪🇺 **欧洲**：1% - 2%（德国10年期国债）
            - 🇯🇵 **日本**：0% - 0.5%（接近零利率）
            
            **设置建议：**
            - 💡 **默认值**：2% - 3%（适合大多数情况）
            - 💡 **精确计算**：使用当前市场的10年期国债收益率
            - 💡 **历史回测**：使用回测期间的平均无风险利率
            
            **如何影响结果：**
            - ✅ **Sharpe比率**：无风险利率越高，Sharpe比率越低
            - ✅ **策略评估**：如果策略收益低于无风险利率，Sharpe比率为负
            - ✅ **风险溢价**：策略收益 - 无风险利率 = 风险溢价
            
            **实际应用：**
            - 📊 用于评估策略是否值得承担风险
            - 📊 对比不同策略的风险调整后表现
            - 📊 判断策略是否优于无风险投资
            """)
        
        risk_free = st.number_input("Risk Free Rate（无风险利率）", 0.0, 0.1, st.session_state["settings_risk_free"], 0.005,
                                   help="用于计算Sharpe比率的无风险利率，通常为2-3%")
        
        # 参数合理性检查
        if initial_capital < 1000:
            st.warning("⚠️ 初始资金过小可能影响回测准确性")
        elif initial_capital > 10000000:
            st.info("💡 初始资金较大，注意数值精度")
        
        # 保存和取消按钮
        st.markdown("---")
        col_save, col_cancel = st.columns(2)
        with col_save:
            if st.button("✅ 保存设置", use_container_width=True, type="primary"):
                st.session_state["settings_strategy"] = strategy_name_global
                st.session_state["settings_initial_capital"] = initial_capital
                st.session_state["settings_leverage"] = leverage
                st.session_state["settings_risk_free"] = risk_free
                # 保存策略参数
                if strategy_name_global == "Target Risk":
                    st.session_state["settings_target_vol"] = target_vol
                elif strategy_name_global == "Adaptive Rebalance":
                    st.session_state["settings_threshold"] = threshold
                elif strategy_name_global == "Momentum":
                    st.session_state["settings_momentum_lookback"] = momentum_lookback
                    st.session_state["settings_momentum_factor"] = momentum_factor
                elif strategy_name_global == "Mean Reversion":
                    st.session_state["settings_reversion_speed"] = reversion_speed
                st.session_state["show_settings_dialog"] = False
                st.rerun()
        with col_cancel:
            if st.button("❌ 取消", use_container_width=True):
                st.session_state["show_settings_dialog"] = False
                st.rerun()
    
    # 输入建模对话框（独立界面）
    if st.session_state.get("show_input_modeling_dialog", False):
        st.markdown("---")
        st.markdown("### 📊 输入建模（Input Modeling）")
        st.markdown("---")
        
        st.info("""
        **输入建模**是基于历史标的物价格数据，分析其收益率分布特征，为未来价格模拟提供建模基础。
        
        **核心作用：**
        - 📊 **分析历史数据**：从标的物价格数据中提取收益率，分析其统计特征（均值、标准差、偏度、峰度）
        - 📈 **拟合分布模型**：自动拟合多种分布模型（Normal、Student-t、Lognormal、Gamma、Beta、Weibull、Gumbel、Laplace、Cauchy、Bootstrap）
        - 📉 **评估拟合效果**：计算拟合优度指标（KS检验、Anderson-Darling检验、AIC、BIC、对数似然）
        - 🏆 **推荐最佳模型**：自动推荐拟合效果最好的分布模型
        - 🚀 **用于未来模拟**：保存选择的分布模型，供PROJECTION模式模拟未来价格走向使用
        
        **工作流程：**
        1. 上传历史标的物价格数据（CSV格式）或使用回测结果
        2. 系统自动计算收益率并分析分布特征
        3. 拟合多种分布模型并评估拟合效果
        4. 选择最佳分布模型并保存参数
        5. 在PROJECTION模式中使用该模型模拟未来价格，评估策略表现
        """)
        
        # 获取可用数据（从上传的文件或回测结果）
        available_returns = None
        data_source = None
        
        # 优先使用上传的文件数据
        if "uploaded_file_data" in st.session_state and st.session_state["uploaded_file_data"] is not None:
            try:
                market_data = InvestSimBridge.load_market_data(st.session_state["uploaded_file_data"])
                returns = market_data.pct_change().dropna()
                available_returns = returns.values.flatten()
                available_returns = available_returns[~np.isnan(available_returns)]
                data_source = "上传文件"
            except:
                pass
        
        # 如果没有上传文件，使用回测结果
        if available_returns is None and "bootstrap_returns" in st.session_state and st.session_state["bootstrap_returns"] is not None:
            available_returns = st.session_state["bootstrap_returns"]
            data_source = "回测结果"
        
        if available_returns is not None and len(available_returns) > 0:
            st.success(f"✅ 检测到数据：{len(available_returns):,} 个收益率样本（来源：{data_source}）")
            
            # 数据基本统计
            mean_ret = np.mean(available_returns)
            std_ret = np.std(available_returns)
            skew_ret = float(pd.Series(available_returns).skew())
            kurt_ret = float(pd.Series(available_returns).kurtosis())
            
            st.markdown("#### 📈 数据特征分析")
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            with col_stat1:
                st.metric("均值", f"{mean_ret:.6f}", f"{mean_ret*252:.2%} 年化")
            with col_stat2:
                st.metric("标准差", f"{std_ret:.6f}", f"{std_ret*np.sqrt(252):.2%} 年化")
            with col_stat3:
                st.metric("偏度", f"{skew_ret:.2f}", "偏度>0右偏，<0左偏")
            with col_stat4:
                st.metric("峰度", f"{kurt_ret:.2f}", "峰度>3厚尾，<3薄尾")
            
            # 分布拟合和评估
            st.markdown("#### 📊 分布拟合分析")
            
            # 定义所有可用的分布模型
            distribution_names = ["Normal", "Student-t", "Lognormal", "Gamma", "Beta", "Weibull", "Gumbel", "Laplace", "Cauchy", "Bootstrap"]
            
            # 存储所有分布的拟合结果
            fit_results = {}
            
            # 尝试拟合所有分布
            try:
                from scipy import stats as scipy_stats
                scipy_available = True
            except:
                scipy_available = False
                st.warning("⚠️ scipy未安装，部分分布拟合功能不可用")
            
            # 1. Normal分布
            try:
                normal_mean = mean_ret
                normal_vol = std_ret
                normal_params = {"mean": normal_mean, "vol": normal_vol}
                
                # 计算拟合优度指标
                if scipy_available:
                    ks_stat, ks_pvalue = scipy_stats.kstest(available_returns, lambda x: scipy_stats.norm.cdf(x, normal_mean, normal_vol))
                    log_likelihood = np.sum(scipy_stats.norm.logpdf(available_returns, normal_mean, normal_vol))
                    n_params = 2
                    aic = 2 * n_params - 2 * log_likelihood
                    bic = n_params * np.log(len(available_returns)) - 2 * log_likelihood
                    
                    # Anderson-Darling检验（需要标准化）
                    standardized = (available_returns - normal_mean) / normal_vol
                    ad_result = scipy_stats.anderson(standardized, dist='norm')
                    ad_stat = ad_result.statistic
                else:
                    ks_stat, ks_pvalue, log_likelihood, aic, bic, ad_stat = None, None, None, None, None, None
                
                fit_results["Normal"] = {
                    "params": normal_params,
                    "ks_stat": ks_stat,
                    "ks_pvalue": ks_pvalue,
                    "log_likelihood": log_likelihood,
                    "aic": aic,
                    "bic": bic,
                    "ad_stat": ad_stat,
                    "success": True
                }
            except Exception as e:
                fit_results["Normal"] = {"success": False, "error": str(e)}
            
            # 2. Student-t分布
            try:
                if scipy_available:
                    df_fitted, loc_fitted, scale_fitted = scipy_stats.t.fit(available_returns)
                    t_params = {"df": float(df_fitted), "mean": float(loc_fitted), "scale": float(scale_fitted)}
                    
                    ks_stat, ks_pvalue = scipy_stats.kstest(available_returns, lambda x: scipy_stats.t.cdf(x, df_fitted, loc=loc_fitted, scale=scale_fitted))
                    log_likelihood = np.sum(scipy_stats.t.logpdf(available_returns, df_fitted, loc=loc_fitted, scale=scale_fitted))
                    n_params = 3
                    aic = 2 * n_params - 2 * log_likelihood
                    bic = n_params * np.log(len(available_returns)) - 2 * log_likelihood
                    ad_stat = None  # Student-t没有标准的AD检验
                    
                    fit_results["Student-t"] = {
                        "params": t_params,
                        "ks_stat": ks_stat,
                        "ks_pvalue": ks_pvalue,
                        "log_likelihood": log_likelihood,
                        "aic": aic,
                        "bic": bic,
                        "ad_stat": ad_stat,
                        "success": True
                    }
                else:
                    fit_results["Student-t"] = {"success": False, "error": "scipy不可用"}
            except Exception as e:
                fit_results["Student-t"] = {"success": False, "error": str(e)}
            
            # 3. Lognormal分布（需要数据为正）
            try:
                if scipy_available and np.all(available_returns > -1):  # 收益率需要 > -100%
                    shifted_returns = available_returns + 1  # 平移使数据为正
                    s_fitted, loc_fitted, scale_fitted = scipy_stats.lognorm.fit(shifted_returns)
                    lognormal_params = {"s": float(s_fitted), "loc": float(loc_fitted), "scale": float(scale_fitted), "shift": 1.0}
                    
                    ks_stat, ks_pvalue = scipy_stats.kstest(shifted_returns, lambda x: scipy_stats.lognorm.cdf(x, s_fitted, loc=loc_fitted, scale=scale_fitted))
                    log_likelihood = np.sum(scipy_stats.lognorm.logpdf(shifted_returns, s_fitted, loc=loc_fitted, scale=scale_fitted))
                    n_params = 3
                    aic = 2 * n_params - 2 * log_likelihood
                    bic = n_params * np.log(len(available_returns)) - 2 * log_likelihood
                    ad_stat = None
                    
                    fit_results["Lognormal"] = {
                        "params": lognormal_params,
                        "ks_stat": ks_stat,
                        "ks_pvalue": ks_pvalue,
                        "log_likelihood": log_likelihood,
                        "aic": aic,
                        "bic": bic,
                        "ad_stat": ad_stat,
                        "success": True
                    }
                else:
                    fit_results["Lognormal"] = {"success": False, "error": "数据不满足lognormal要求或scipy不可用"}
            except Exception as e:
                fit_results["Lognormal"] = {"success": False, "error": str(e)}
            
            # 4. Gamma分布（需要数据为正）
            try:
                if scipy_available and np.all(available_returns > -1):
                    shifted_returns = available_returns + 1
                    a_fitted, loc_fitted, scale_fitted = scipy_stats.gamma.fit(shifted_returns)
                    gamma_params = {"a": float(a_fitted), "loc": float(loc_fitted), "scale": float(scale_fitted), "shift": 1.0}
                    
                    ks_stat, ks_pvalue = scipy_stats.kstest(shifted_returns, lambda x: scipy_stats.gamma.cdf(x, a_fitted, loc=loc_fitted, scale=scale_fitted))
                    log_likelihood = np.sum(scipy_stats.gamma.logpdf(shifted_returns, a_fitted, loc=loc_fitted, scale=scale_fitted))
                    n_params = 3
                    aic = 2 * n_params - 2 * log_likelihood
                    bic = n_params * np.log(len(available_returns)) - 2 * log_likelihood
                    ad_stat = None
                    
                    fit_results["Gamma"] = {
                        "params": gamma_params,
                        "ks_stat": ks_stat,
                        "ks_pvalue": ks_pvalue,
                        "log_likelihood": log_likelihood,
                        "aic": aic,
                        "bic": bic,
                        "ad_stat": ad_stat,
                        "success": True
                    }
                else:
                    fit_results["Gamma"] = {"success": False, "error": "数据不满足gamma要求或scipy不可用"}
            except Exception as e:
                fit_results["Gamma"] = {"success": False, "error": str(e)}
            
            # 5. Beta分布（需要数据在[0,1]范围内）
            try:
                if scipy_available:
                    # 将数据标准化到[0,1]
                    min_val, max_val = available_returns.min(), available_returns.max()
                    if max_val > min_val:
                        normalized = (available_returns - min_val) / (max_val - min_val)
                        a_fitted, b_fitted, loc_fitted, scale_fitted = scipy_stats.beta.fit(normalized)
                        beta_params = {"a": float(a_fitted), "b": float(b_fitted), "loc": float(loc_fitted), "scale": float(scale_fitted), "min": float(min_val), "max": float(max_val)}
                        
                        ks_stat, ks_pvalue = scipy_stats.kstest(normalized, lambda x: scipy_stats.beta.cdf(x, a_fitted, b_fitted, loc=loc_fitted, scale=scale_fitted))
                        log_likelihood = np.sum(scipy_stats.beta.logpdf(normalized, a_fitted, b_fitted, loc=loc_fitted, scale=scale_fitted))
                        n_params = 4
                        aic = 2 * n_params - 2 * log_likelihood
                        bic = n_params * np.log(len(available_returns)) - 2 * log_likelihood
                        ad_stat = None
                        
                        fit_results["Beta"] = {
                            "params": beta_params,
                            "ks_stat": ks_stat,
                            "ks_pvalue": ks_pvalue,
                            "log_likelihood": log_likelihood,
                            "aic": aic,
                            "bic": bic,
                            "ad_stat": ad_stat,
                            "success": True
                        }
                    else:
                        fit_results["Beta"] = {"success": False, "error": "数据范围无效"}
                else:
                    fit_results["Beta"] = {"success": False, "error": "scipy不可用"}
            except Exception as e:
                fit_results["Beta"] = {"success": False, "error": str(e)}
            
            # 6. Weibull分布（需要数据为正）
            try:
                if scipy_available and np.all(available_returns > -1):
                    shifted_returns = available_returns + 1
                    c_fitted, loc_fitted, scale_fitted = scipy_stats.weibull_min.fit(shifted_returns)
                    weibull_params = {"c": float(c_fitted), "loc": float(loc_fitted), "scale": float(scale_fitted), "shift": 1.0}
                    
                    ks_stat, ks_pvalue = scipy_stats.kstest(shifted_returns, lambda x: scipy_stats.weibull_min.cdf(x, c_fitted, loc=loc_fitted, scale=scale_fitted))
                    log_likelihood = np.sum(scipy_stats.weibull_min.logpdf(shifted_returns, c_fitted, loc=loc_fitted, scale=scale_fitted))
                    n_params = 3
                    aic = 2 * n_params - 2 * log_likelihood
                    bic = n_params * np.log(len(available_returns)) - 2 * log_likelihood
                    ad_stat = None
                    
                    fit_results["Weibull"] = {
                        "params": weibull_params,
                        "ks_stat": ks_stat,
                        "ks_pvalue": ks_pvalue,
                        "log_likelihood": log_likelihood,
                        "aic": aic,
                        "bic": bic,
                        "ad_stat": ad_stat,
                        "success": True
                    }
                else:
                    fit_results["Weibull"] = {"success": False, "error": "数据不满足weibull要求或scipy不可用"}
            except Exception as e:
                fit_results["Weibull"] = {"success": False, "error": str(e)}
            
            # 7. Gumbel分布
            try:
                if scipy_available:
                    loc_fitted, scale_fitted = scipy_stats.gumbel_l.fit(available_returns)
                    gumbel_params = {"loc": float(loc_fitted), "scale": float(scale_fitted)}
                    
                    ks_stat, ks_pvalue = scipy_stats.kstest(available_returns, lambda x: scipy_stats.gumbel_l.cdf(x, loc=loc_fitted, scale=scale_fitted))
                    log_likelihood = np.sum(scipy_stats.gumbel_l.logpdf(available_returns, loc=loc_fitted, scale=scale_fitted))
                    n_params = 2
                    aic = 2 * n_params - 2 * log_likelihood
                    bic = n_params * np.log(len(available_returns)) - 2 * log_likelihood
                    ad_stat = None
                    
                    fit_results["Gumbel"] = {
                        "params": gumbel_params,
                        "ks_stat": ks_stat,
                        "ks_pvalue": ks_pvalue,
                        "log_likelihood": log_likelihood,
                        "aic": aic,
                        "bic": bic,
                        "ad_stat": ad_stat,
                        "success": True
                    }
                else:
                    fit_results["Gumbel"] = {"success": False, "error": "scipy不可用"}
            except Exception as e:
                fit_results["Gumbel"] = {"success": False, "error": str(e)}
            
            # 8. Laplace分布
            try:
                if scipy_available:
                    loc_fitted, scale_fitted = scipy_stats.laplace.fit(available_returns)
                    laplace_params = {"loc": float(loc_fitted), "scale": float(scale_fitted)}
                    
                    ks_stat, ks_pvalue = scipy_stats.kstest(available_returns, lambda x: scipy_stats.laplace.cdf(x, loc=loc_fitted, scale=scale_fitted))
                    log_likelihood = np.sum(scipy_stats.laplace.logpdf(available_returns, loc=loc_fitted, scale=scale_fitted))
                    n_params = 2
                    aic = 2 * n_params - 2 * log_likelihood
                    bic = n_params * np.log(len(available_returns)) - 2 * log_likelihood
                    ad_stat = None
                    
                    fit_results["Laplace"] = {
                        "params": laplace_params,
                        "ks_stat": ks_stat,
                        "ks_pvalue": ks_pvalue,
                        "log_likelihood": log_likelihood,
                        "aic": aic,
                        "bic": bic,
                        "ad_stat": ad_stat,
                        "success": True
                    }
                else:
                    fit_results["Laplace"] = {"success": False, "error": "scipy不可用"}
            except Exception as e:
                fit_results["Laplace"] = {"success": False, "error": str(e)}
            
            # 9. Cauchy分布
            try:
                if scipy_available:
                    loc_fitted, scale_fitted = scipy_stats.cauchy.fit(available_returns)
                    cauchy_params = {"loc": float(loc_fitted), "scale": float(scale_fitted)}
                    
                    ks_stat, ks_pvalue = scipy_stats.kstest(available_returns, lambda x: scipy_stats.cauchy.cdf(x, loc=loc_fitted, scale=scale_fitted))
                    log_likelihood = np.sum(scipy_stats.cauchy.logpdf(available_returns, loc=loc_fitted, scale=scale_fitted))
                    n_params = 2
                    aic = 2 * n_params - 2 * log_likelihood
                    bic = n_params * np.log(len(available_returns)) - 2 * log_likelihood
                    ad_stat = None
                    
                    fit_results["Cauchy"] = {
                        "params": cauchy_params,
                        "ks_stat": ks_stat,
                        "ks_pvalue": ks_pvalue,
                        "log_likelihood": log_likelihood,
                        "aic": aic,
                        "bic": bic,
                        "ad_stat": ad_stat,
                        "success": True
                    }
                else:
                    fit_results["Cauchy"] = {"success": False, "error": "scipy不可用"}
            except Exception as e:
                fit_results["Cauchy"] = {"success": False, "error": str(e)}
            
            # 10. Bootstrap（经验分布，直接使用数据）
            try:
                # Bootstrap是经验分布，直接使用历史数据
                # 计算经验分布函数（ECDF）
                sorted_returns = np.sort(available_returns)
                n_samples = len(available_returns)
                
                # KS统计量：经验分布与自身的KS统计量应该为0（完美拟合）
                # 但我们可以计算经验分布与标准正态分布的KS统计量作为参考
                if scipy_available:
                    # 计算经验分布与标准正态分布的KS统计量（作为参考）
                    # 注意：这不是真正的拟合，只是作为比较
                    empirical_mean = np.mean(available_returns)
                    empirical_std = np.std(available_returns)
                    ks_stat_ref, ks_pvalue_ref = scipy_stats.kstest(
                        available_returns, 
                        lambda x: scipy_stats.norm.cdf(x, empirical_mean, empirical_std)
                    )
                    
                    # 对于Bootstrap，经验分布与自身的KS统计量应该为0
                    # 但我们可以使用经验分布的概率密度来计算对数似然
                    # 使用核密度估计（KDE）来计算对数似然
                    from scipy.stats import gaussian_kde
                    try:
                        kde = gaussian_kde(available_returns)
                        log_likelihood = np.sum(kde.logpdf(available_returns))
                    except:
                        # 如果KDE失败，使用经验分布的概率密度估计
                        # 对于经验分布，每个观测值的概率密度为 1/(n * bandwidth)
                        # 这里使用一个简化的估计
                        bandwidth = np.std(available_returns) * (4 / (3 * n_samples)) ** (1/5)  # Silverman's rule
                        log_likelihood = -n_samples * np.log(n_samples * bandwidth) - 0.5 * np.sum((available_returns - empirical_mean) ** 2) / (2 * bandwidth ** 2)
                    
                    # AIC和BIC：对于Bootstrap，参数数量可以认为是数据点数（或使用一个较小的值）
                    # 但通常Bootstrap的参数数量被认为是0（无参数模型）或数据点数
                    # 这里我们使用一个折中方案：参数数量 = log(n)（表示数据复杂度）
                    n_params_bootstrap = np.log(n_samples) if n_samples > 1 else 1
                    aic = 2 * n_params_bootstrap - 2 * log_likelihood
                    bic = n_params_bootstrap * np.log(n_samples) - 2 * log_likelihood
                    
                    # Bootstrap的KS统计量设为0（完美拟合自身）
                    ks_stat = 0.0
                    ks_pvalue = 1.0  # 完美拟合，p值为1
                    ad_stat = None  # Anderson-Darling不适用于经验分布
                else:
                    ks_stat, ks_pvalue, log_likelihood, aic, bic, ad_stat = None, None, None, None, None, None
                
                fit_results["Bootstrap"] = {
                    "params": {
                        "samples": n_samples,
                        "mean": float(empirical_mean),
                        "std": float(empirical_std),
                        "min": float(np.min(available_returns)),
                        "max": float(np.max(available_returns))
                    },
                    "ks_stat": ks_stat,
                    "ks_pvalue": ks_pvalue,
                    "log_likelihood": log_likelihood,
                    "aic": aic,
                    "bic": bic,
                    "ad_stat": ad_stat,
                    "success": True
                }
            except Exception as e:
                fit_results["Bootstrap"] = {"success": False, "error": str(e)}
            
            # 计算综合评分（基于多个指标）
            scores = {}
            for dist_name, result in fit_results.items():
                if result.get("success", False) and dist_name != "Bootstrap":
                    score = 0
                    # KS p值越高越好（最大1分）
                    if result.get("ks_pvalue") is not None:
                        score += result["ks_pvalue"] * 0.3
                    # AIC越低越好（归一化后最大1分）
                    if result.get("aic") is not None:
                        aic_values = [r.get("aic") for r in fit_results.values() if r.get("success") and r.get("aic") is not None]
                        if len(aic_values) > 0:
                            min_aic, max_aic = min(aic_values), max(aic_values)
                            if max_aic > min_aic:
                                score += (1 - (result["aic"] - min_aic) / (max_aic - min_aic)) * 0.3
                            else:
                                score += 0.3
                    # BIC越低越好（归一化后最大1分）
                    if result.get("bic") is not None:
                        bic_values = [r.get("bic") for r in fit_results.values() if r.get("success") and r.get("bic") is not None]
                        if len(bic_values) > 0:
                            min_bic, max_bic = min(bic_values), max(bic_values)
                            if max_bic > min_bic:
                                score += (1 - (result["bic"] - min_bic) / (max_bic - min_bic)) * 0.2
                            else:
                                score += 0.2
                    # 对数似然越高越好（归一化后最大0.2分）
                    if result.get("log_likelihood") is not None:
                        ll_values = [r.get("log_likelihood") for r in fit_results.values() if r.get("success") and r.get("log_likelihood") is not None]
                        if len(ll_values) > 0:
                            min_ll, max_ll = min(ll_values), max(ll_values)
                            if max_ll > min_ll:
                                score += ((result["log_likelihood"] - min_ll) / (max_ll - min_ll)) * 0.2
                            else:
                                score += 0.2
                    scores[dist_name] = score
                elif dist_name == "Bootstrap":
                    # Bootstrap的特殊评分（基于数据量）
                    scores[dist_name] = min(1.0, len(available_returns) / 1000) * 0.5  # 数据量越多越好
            
            # 找出最佳拟合分布
            if scores:
                best_dist = max(scores, key=scores.get)
                best_score = scores[best_dist]
            else:
                best_dist = "Normal"
                best_score = 0
            
            # 显示拟合结果汇总表
            st.markdown("#### 📊 拟合结果汇总")
            
            # 创建结果表格
            summary_data = []
            for dist_name in distribution_names:
                result = fit_results.get(dist_name, {})
                if result.get("success", False):
                    row = {
                        "分布": dist_name,
                        "拟合状态": "✅ 成功",
                        "KS统计量": f"{result.get('ks_stat', 'N/A'):.6f}" if result.get('ks_stat') is not None else "N/A",
                        "KS p值": f"{result.get('ks_pvalue', 'N/A'):.6f}" if result.get('ks_pvalue') is not None else "N/A",
                        "AIC": f"{result.get('aic', 'N/A'):.2f}" if result.get('aic') is not None else "N/A",
                        "BIC": f"{result.get('bic', 'N/A'):.2f}" if result.get('bic') is not None else "N/A",
                        "对数似然": f"{result.get('log_likelihood', 'N/A'):.2f}" if result.get('log_likelihood') is not None else "N/A",
                        "综合评分": f"{scores.get(dist_name, 0):.4f}" if dist_name in scores else "N/A"
                    }
                    summary_data.append(row)
                else:
                    row = {
                        "分布": dist_name,
                        "拟合状态": f"❌ 失败 ({result.get('error', '未知错误')})",
                        "KS统计量": "N/A",
                        "KS p值": "N/A",
                        "AIC": "N/A",
                        "BIC": "N/A",
                        "对数似然": "N/A",
                        "综合评分": "N/A"
                    }
                    summary_data.append(row)
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # 显示最佳拟合分布
            st.success(f"🏆 **最佳拟合分布**：**{best_dist}** (综合评分: {best_score:.4f})")
            st.caption("💡 综合评分综合考虑了KS检验p值、AIC、BIC和对数似然值。评分越高，拟合效果越好。")
            
            # 分布切换和可视化
            st.markdown("#### 🔄 分布切换与可视化")
            
            # 获取成功拟合的分布列表
            successful_dists = [d for d in distribution_names if fit_results.get(d, {}).get("success", False)]
            
            if len(successful_dists) > 0:
                selected_dist = st.selectbox(
                    "选择要查看的分布模型",
                    successful_dists,
                    index=successful_dists.index(best_dist) if best_dist in successful_dists else 0,
                    help="切换查看不同分布的拟合效果和参数"
                )
                
                # 显示选中分布的详细信息
                result = fit_results[selected_dist]
                params = result.get("params", {})
                
                st.markdown(f"##### 📈 {selected_dist} 分布详情")
                
                # 参数显示
                col_param1, col_param2 = st.columns(2)
                with col_param1:
                    st.markdown("**拟合参数：**")
                    for key, value in params.items():
                        if isinstance(value, float):
                            st.text(f"  • {key}: {value:.6f}")
                        else:
                            st.text(f"  • {key}: {value}")
                
                with col_param2:
                    st.markdown("**拟合优度指标：**")
                    if result.get("ks_stat") is not None:
                        st.text(f"  • KS统计量: {result['ks_stat']:.6f}")
                    if result.get("ks_pvalue") is not None:
                        p_color = "🟢" if result['ks_pvalue'] > 0.05 else "🟡" if result['ks_pvalue'] > 0.01 else "🔴"
                        st.text(f"  • KS p值: {result['ks_pvalue']:.6f} {p_color}")
                    if result.get("aic") is not None:
                        st.text(f"  • AIC: {result['aic']:.2f}")
                    if result.get("bic") is not None:
                        st.text(f"  • BIC: {result['bic']:.2f}")
                    if result.get("log_likelihood") is not None:
                        st.text(f"  • 对数似然: {result['log_likelihood']:.2f}")
                    if selected_dist in scores:
                        st.text(f"  • 综合评分: {scores[selected_dist]:.4f}")
                
                # 可视化
                x = np.linspace(available_returns.min(), available_returns.max(), 200)
                
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=available_returns,
                    name="实际数据",
                    opacity=0.5,
                    nbinsx=50,
                    marker_color=COLORS["blue"]
                ))
                
                # 根据选中的分布绘制拟合曲线
                if selected_dist == "Normal":
                    normal_y = (1 / (params["vol"] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - params["mean"]) / params["vol"]) ** 2)
                    fig_dist.add_trace(go.Scatter(
                        x=x,
                        y=normal_y * len(available_returns) * (x[1] - x[0]),
                        name=f"{selected_dist}拟合",
                        line=dict(color=COLORS["gold"], width=2)
                    ))
                elif selected_dist == "Student-t" and scipy_available:
                    try:
                        from scipy import stats as scipy_stats
                        t_y = scipy_stats.t.pdf(x, params["df"], loc=params["mean"], scale=params["scale"])
                        fig_dist.add_trace(go.Scatter(
                            x=x,
                            y=t_y * len(available_returns) * (x[1] - x[0]),
                            name=f"{selected_dist}拟合",
                            line=dict(color=COLORS["green"], width=2)
                        ))
                    except:
                        pass
                elif selected_dist == "Lognormal" and scipy_available:
                    try:
                        from scipy import stats as scipy_stats
                        shifted_x = x + params.get("shift", 1.0)
                        lognormal_y = scipy_stats.lognorm.pdf(shifted_x, params["s"], loc=params["loc"], scale=params["scale"])
                        fig_dist.add_trace(go.Scatter(
                            x=x,
                            y=lognormal_y * len(available_returns) * (x[1] - x[0]),
                            name=f"{selected_dist}拟合",
                            line=dict(color=COLORS["green"], width=2)
                        ))
                    except:
                        pass
                elif selected_dist == "Gamma" and scipy_available:
                    try:
                        from scipy import stats as scipy_stats
                        shifted_x = x + params.get("shift", 1.0)
                        gamma_y = scipy_stats.gamma.pdf(shifted_x, params["a"], loc=params["loc"], scale=params["scale"])
                        fig_dist.add_trace(go.Scatter(
                            x=x,
                            y=gamma_y * len(available_returns) * (x[1] - x[0]),
                            name=f"{selected_dist}拟合",
                            line=dict(color=COLORS["green"], width=2)
                        ))
                    except:
                        pass
                elif selected_dist == "Beta" and scipy_available:
                    try:
                        from scipy import stats as scipy_stats
                        normalized_x = (x - params["min"]) / (params["max"] - params["min"])
                        beta_y = scipy_stats.beta.pdf(normalized_x, params["a"], params["b"], loc=params["loc"], scale=params["scale"])
                        # 转换回原始尺度
                        beta_y = beta_y / (params["max"] - params["min"])
                        fig_dist.add_trace(go.Scatter(
                            x=x,
                            y=beta_y * len(available_returns) * (x[1] - x[0]),
                            name=f"{selected_dist}拟合",
                            line=dict(color=COLORS["green"], width=2)
                        ))
                    except:
                        pass
                elif selected_dist == "Weibull" and scipy_available:
                    try:
                        from scipy import stats as scipy_stats
                        shifted_x = x + params.get("shift", 1.0)
                        weibull_y = scipy_stats.weibull_min.pdf(shifted_x, params["c"], loc=params["loc"], scale=params["scale"])
                        fig_dist.add_trace(go.Scatter(
                            x=x,
                            y=weibull_y * len(available_returns) * (x[1] - x[0]),
                            name=f"{selected_dist}拟合",
                            line=dict(color=COLORS["green"], width=2)
                        ))
                    except:
                        pass
                elif selected_dist == "Gumbel" and scipy_available:
                    try:
                        from scipy import stats as scipy_stats
                        gumbel_y = scipy_stats.gumbel_l.pdf(x, loc=params["loc"], scale=params["scale"])
                        fig_dist.add_trace(go.Scatter(
                            x=x,
                            y=gumbel_y * len(available_returns) * (x[1] - x[0]),
                            name=f"{selected_dist}拟合",
                            line=dict(color=COLORS["green"], width=2)
                        ))
                    except:
                        pass
                elif selected_dist == "Laplace" and scipy_available:
                    try:
                        from scipy import stats as scipy_stats
                        laplace_y = scipy_stats.laplace.pdf(x, loc=params["loc"], scale=params["scale"])
                        fig_dist.add_trace(go.Scatter(
                            x=x,
                            y=laplace_y * len(available_returns) * (x[1] - x[0]),
                            name=f"{selected_dist}拟合",
                            line=dict(color=COLORS["green"], width=2)
                        ))
                    except:
                        pass
                elif selected_dist == "Cauchy" and scipy_available:
                    try:
                        from scipy import stats as scipy_stats
                        cauchy_y = scipy_stats.cauchy.pdf(x, loc=params["loc"], scale=params["scale"])
                        fig_dist.add_trace(go.Scatter(
                            x=x,
                            y=cauchy_y * len(available_returns) * (x[1] - x[0]),
                            name=f"{selected_dist}拟合",
                            line=dict(color=COLORS["green"], width=2)
                        ))
                    except:
                        pass
                elif selected_dist == "Bootstrap":
                    # Bootstrap不需要绘制拟合曲线，只显示直方图
                    pass
                
                fig_dist.update_layout(
                    title=f"{selected_dist} 分布拟合效果",
                    xaxis_title="收益率",
                    yaxis_title="频数",
                    template="plotly_dark",
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # 选择分布模型用于PROJECTION
                if "input_model_choice" not in st.session_state:
                    st.session_state["input_model_choice"] = best_dist
                
                input_model_type = st.selectbox(
                    "选择分布模型（用于PROJECTION模拟）",
                    successful_dists,
                    index=successful_dists.index(st.session_state["input_model_choice"]) if st.session_state["input_model_choice"] in successful_dists else successful_dists.index(best_dist) if best_dist in successful_dists else 0,
                    help="根据拟合效果选择最适合的分布模型"
                )
                
                # 保存拟合参数
                selected_result = fit_results[input_model_type]
                if input_model_type == "Normal":
                    st.session_state["fitted_normal_params"] = selected_result["params"]
                    st.caption(f"✅ Normal参数已保存：均值={selected_result['params']['mean']:.6f}, 波动率={selected_result['params']['vol']:.6f}")
                elif input_model_type == "Student-t":
                    st.session_state["fitted_student_t_params"] = selected_result["params"]
                    st.caption(f"✅ Student-t参数已保存：自由度={selected_result['params']['df']:.2f}, 均值={selected_result['params']['mean']:.6f}, 尺度={selected_result['params']['scale']:.6f}")
                elif input_model_type == "Bootstrap":
                    st.session_state["bootstrap_returns"] = available_returns
                    st.caption(f"✅ Bootstrap：已保存 {len(available_returns):,} 个历史收益率样本")
                else:
                    # 保存其他分布的参数（如果将来需要支持）
                    st.session_state[f"fitted_{input_model_type.lower().replace('-', '_')}_params"] = selected_result["params"]
                    st.caption(f"✅ {input_model_type}参数已保存")
            else:
                st.error("❌ 所有分布拟合均失败，请检查数据")
                input_model_type = "Normal"  # 默认值
            
        else:
            st.warning("⚠️ 未检测到数据。请先上传数据文件或运行回测")
            st.caption("💡 上传数据后，系统会自动分析数据特征并推荐最适合的分布模型")
            
            # 如果没有数据，提供默认选择
            if "input_model_choice" not in st.session_state:
                st.session_state["input_model_choice"] = "Normal"
            
            input_model_type = st.selectbox(
                "选择分布模型（用于PROJECTION模拟）",
                ["Normal", "Student-t", "Bootstrap"],
                index=["Normal", "Student-t", "Bootstrap"].index(st.session_state["input_model_choice"]) if st.session_state["input_model_choice"] in ["Normal", "Student-t", "Bootstrap"] else 0,
                help="未检测到数据时，将使用默认参数"
            )
        
        # 保存和取消按钮
        st.markdown("---")
        col_save, col_cancel = st.columns(2)
        with col_save:
            if st.button("✅ 保存配置", use_container_width=True, type="primary", key="save_input_modeling"):
                # 保存选择的分布类型
                st.session_state["input_model_choice"] = input_model_type
                
                # 确保保存当前选择的分布的参数
                # 如果有数据且进行了拟合
                if available_returns is not None and len(available_returns) > 0:
                    # 检查是否有拟合结果（在作用域内）
                    try:
                        if 'fit_results' in locals() and input_model_type in fit_results:
                            selected_result = fit_results.get(input_model_type, {})
                            if selected_result.get("success", False):
                                if input_model_type == "Normal":
                                    st.session_state["fitted_normal_params"] = selected_result["params"]
                                elif input_model_type == "Student-t":
                                    st.session_state["fitted_student_t_params"] = selected_result["params"]
                                elif input_model_type == "Bootstrap":
                                    st.session_state["bootstrap_returns"] = available_returns
                                else:
                                    # 保存其他分布的参数
                                    st.session_state[f"fitted_{input_model_type.lower().replace('-', '_')}_params"] = selected_result["params"]
                        else:
                            # 如果没有拟合结果，但之前已经保存过参数（在selectbox切换时保存的），则保留
                            # 如果之前没有保存，则从当前数据计算并保存
                            if input_model_type == "Normal" and "fitted_normal_params" not in st.session_state:
                                st.session_state["fitted_normal_params"] = {"mean": float(np.mean(available_returns)), "vol": float(np.std(available_returns))}
                            elif input_model_type == "Student-t" and "fitted_student_t_params" not in st.session_state:
                                st.session_state["fitted_student_t_params"] = {"mean": 0.0, "df": 5.0, "scale": float(np.std(available_returns))}
                            elif input_model_type == "Bootstrap" and "bootstrap_returns" not in st.session_state:
                                st.session_state["bootstrap_returns"] = available_returns
                    except:
                        # 如果出错，至少保存基本参数
                        if input_model_type == "Normal":
                            if "fitted_normal_params" not in st.session_state:
                                st.session_state["fitted_normal_params"] = {"mean": float(np.mean(available_returns)), "vol": float(np.std(available_returns))}
                        elif input_model_type == "Bootstrap":
                            if "bootstrap_returns" not in st.session_state:
                                st.session_state["bootstrap_returns"] = available_returns
                
                st.session_state["show_input_modeling_dialog"] = False
                st.success(f"✅ 输入建模配置已保存！已选择 {input_model_type} 分布。")
                st.rerun()
        with col_cancel:
            if st.button("❌ 取消", use_container_width=True, key="cancel_input_modeling"):
                st.session_state["show_input_modeling_dialog"] = False
                st.rerun()
    
    # 策略动态参数（从session state读取）
    strategy_params = {}
    if strategy_name_global == "Target Risk":
        strategy_params["target_vol"] = st.session_state.get("settings_target_vol", 0.15)
    elif strategy_name_global == "Adaptive Rebalance":
        strategy_params["threshold"] = st.session_state.get("settings_threshold", 0.05)
    elif strategy_name_global == "Momentum":
        strategy_params["momentum_lookback"] = st.session_state.get("settings_momentum_lookback", 20)
        strategy_params["momentum_factor"] = st.session_state.get("settings_momentum_factor", 0.5)
    elif strategy_name_global == "Mean Reversion":
        strategy_params["reversion_speed"] = st.session_state.get("settings_reversion_speed", 0.3)

st.sidebar.markdown("---")

# ==========================================
# 新功能区域
# ==========================================
if mode != "DERIVATIVES LAB (Options / Margin)":
    # 策略对比功能
    with st.sidebar.expander("🔀 策略对比", expanded=False):
        st.markdown("**同时对比多个策略的表现**")
        
        if st.button("➕ 添加当前策略到对比", use_container_width=True):
            if 'bt_result' in st.session_state:
                comparison_entry = {
                    "strategy": strategy_name_global,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "metrics": {
                        "total_return": st.session_state.get("bt_metrics", {}).get("total_return", 0),
                        "sharpe": st.session_state.get("bt_metrics", {}).get("sharpe", 0),
                        "max_drawdown": st.session_state.get("bt_metrics", {}).get("max_drawdown", 0),
                        "volatility": st.session_state.get("bt_metrics", {}).get("volatility", 0),
                    },
                    "params": {
                        "initial_capital": initial_capital,
                        "leverage": leverage,
                        "risk_free": risk_free,
                    }
                }
                st.session_state["strategy_comparison"].append(comparison_entry)
                st.success(f"✅ 已添加 {strategy_name_global} 到对比列表")
                st.rerun()
        
        if len(st.session_state["strategy_comparison"]) > 0:
            st.markdown("**对比列表：**")
            for i, entry in enumerate(st.session_state["strategy_comparison"]):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.caption(f"{i+1}. {entry['strategy']}")
                with col2:
                    if st.button("🗑️", key=f"remove_{i}", help="删除"):
                        st.session_state["strategy_comparison"].pop(i)
                        st.rerun()
            
            if st.button("📊 查看对比结果", use_container_width=True, type="primary"):
                st.session_state["show_comparison"] = True
                st.rerun()
            
            if st.button("🗑️ 清空对比列表", use_container_width=True):
                st.session_state["strategy_comparison"] = []
                st.rerun()
        else:
            st.info("💡 运行回测后，点击「添加当前策略到对比」来开始对比")
    
    # 回测历史记录
    with st.sidebar.expander("📚 回测历史", expanded=False):
        st.markdown("**查看历史回测结果**")
        
        if len(st.session_state["backtest_history"]) > 0:
            st.markdown(f"**共 {len(st.session_state['backtest_history'])} 条记录**")
            for i, record in enumerate(reversed(st.session_state["backtest_history"][-10:])):  # 只显示最近10条
                with st.expander(f"📅 {record['timestamp']} - {record['strategy']}", expanded=False):
                    st.markdown(f"**策略：** {record['strategy']}")
                    st.markdown(f"**总收益：** {record['metrics'].get('total_return', 0):.2%}")
                    st.markdown(f"**Sharpe比率：** {record['metrics'].get('sharpe', 0):.2f}")
                    if st.button("📊 查看详情", key=f"view_history_{i}"):
                        st.session_state["load_history_index"] = len(st.session_state["backtest_history"]) - 1 - i
                        st.rerun()
        else:
            st.info("💡 运行回测后，结果会自动保存到历史记录")
    
    # 交易成本设置
    with st.sidebar.expander("💰 交易成本设置", expanded=False):
        st.markdown("**配置实际交易成本**")
        
        transaction_cost = st.number_input(
            "交易费用率 (%)", 
            min_value=0.0, 
            max_value=1.0, 
            value=st.session_state["transaction_cost_rate"] * 100,
            step=0.01,
            help="每次交易的费用率，例如0.1%输入0.1"
        )
        st.session_state["transaction_cost_rate"] = transaction_cost / 100
        
        slippage = st.number_input(
            "滑点率 (%)", 
            min_value=0.0, 
            max_value=1.0, 
            value=st.session_state["slippage_rate"] * 100,
            step=0.01,
            help="交易滑点率，例如0.05%输入0.05"
        )
        st.session_state["slippage_rate"] = slippage / 100
        
        st.caption(f"💡 总成本：{(transaction_cost + slippage):.2f}%")
        
        if st.button("💾 保存成本设置", use_container_width=True):
            st.success("✅ 交易成本设置已保存")

st.sidebar.markdown("---")
st.sidebar.caption(f"System Status: ONLINE\nBackend: v2.4.0 (Bridge)")

# 帮助说明
with st.sidebar.expander("ℹ️ HELP & GUIDE", expanded=False):
    st.markdown("""
    **📊 BACKTEST MODE（回测模式）:**
    - **目的**：分析历史数据，得到标的物价格的输入建模（Input Model）并选择策略
    - 上传CSV文件（包含日期列和资产价格）
    - 选择策略并配置参数
    - **自动进行输入建模**：系统会从标的物价格数据中提取收益率分布特征
    - 查看策略的历史表现指标
    - 通过6个详细图表分析回测结果
    
    **🔮 PROJECTION MODE（预测模式）:**
    - **目的**：使用回测中得到的Input Model模拟未来价格走向，评估策略在未来表现
    - **自动使用回测结果**：使用回测中选择的策略和Input Model
    - 配置预测时间期限和模拟次数
    - 查看未来收益的概率分布
    - 获得策略在未来市场环境下的表现评估
    
    **💡 TIPS:**
    - Use synthetic data if no file uploaded
    - Adjust rebalance frequency for different strategies
    - Export results to Excel for further analysis
    
    **📚 策略选择建议：**
    - 新手：Equal Weight 或 Fixed Weights
    - 风险厌恶：Minimum Variance 或 Risk Parity
    - 追求收益：Momentum 或 Target Risk
    - 降低成本：Adaptive Rebalance
    """)
    
    # 策略快速对比
    with st.expander("📊 策略快速对比", expanded=False):
        st.markdown("""
        | 策略 | 复杂度 | 风险控制 | 收益潜力 | 交易成本 |
        |------|--------|----------|----------|----------|
        | Fixed Weights | ⭐ 低 | ⭐⭐ 中 | ⭐⭐ 中 | ⭐⭐ 中 |
        | Target Risk | ⭐⭐ 中 | ⭐⭐⭐ 高 | ⭐⭐ 中 | ⭐⭐ 中 |
        | Adaptive Rebalance | ⭐ 低 | ⭐⭐ 中 | ⭐⭐ 中 | ⭐⭐⭐ 低 |
        | Equal Weight | ⭐ 低 | ⭐⭐ 中 | ⭐⭐ 中 | ⭐⭐ 中 |
        | Risk Parity | ⭐⭐ 中 | ⭐⭐⭐ 高 | ⭐⭐ 中 | ⭐⭐ 中 |
        | Minimum Variance | ⭐⭐⭐ 高 | ⭐⭐⭐ 高 | ⭐ 低 | ⭐⭐ 中 |
        | Momentum | ⭐⭐ 中 | ⭐ 低 | ⭐⭐⭐ 高 | ⭐⭐ 中 |
        | Mean Reversion | ⭐⭐ 中 | ⭐⭐ 中 | ⭐⭐ 中 | ⭐⭐ 中 |
        """)

# ==========================================
# 5. 主界面逻辑 (Main View)
# ==========================================

# 页面标题
if mode != "DERIVATIVES LAB (Options / Margin)":
    st.title(mode.split(" ")[0])
    
    # 策略信息卡片
    col_title1, col_title2, col_title3 = st.columns([2, 1, 1])
    with col_title1:
        st.markdown(f"**Strategy:** <span style='color:{COLORS['gold']}'>{strategy_name_global}</span>", unsafe_allow_html=True)
        if strategy_name_global in strategy_descriptions:
            st.caption(f"💡 {strategy_descriptions[strategy_name_global]}")
    with col_title2:
        st.markdown(f"**Leverage:** <span style='color:{COLORS['text_main']}'>{leverage}x</span>", unsafe_allow_html=True)
    with col_title3:
        st.markdown(f"**Capital:** <span style='color:{COLORS['text_main']}'>${initial_capital:,.0f}</span>", unsafe_allow_html=True)
    
    # 策略快速说明展开区域
    with st.expander("📖 当前策略说明", expanded=False):
        strategy_quick_guide = {
            "Fixed Weights": "始终保持预设权重，定期再平衡。适合长期投资者，简单可预测。",
            "Target Risk": "动态调整权重以保持目标波动率。适合风险敏感型投资者。",
            "Adaptive Rebalance": "仅在权重偏离阈值时再平衡。适合希望降低交易成本的投资者。",
            "Equal Weight": "所有资产等权重分配（1/N策略）。适合不确定如何分配权重的投资者。",
            "Risk Parity": "根据波动率分配权重，使风险贡献相等。适合追求风险均衡的投资者。",
            "Minimum Variance": "优化协方差矩阵，最小化波动率。适合风险厌恶型投资者。",
            "Momentum": "增持表现好的资产，减持表现差的资产。适合相信趋势延续的投资者。",
            "Mean Reversion": "当资产偏离目标时反向调整。适合相信均值回归的投资者。",
        }
        if strategy_name_global in strategy_quick_guide:
            st.info(f"**{strategy_name_global}**: {strategy_quick_guide[strategy_name_global]}")
            st.markdown("💡 更多详细信息请查看左侧边栏的「📚 策略详细说明」")
    
    st.markdown("###") # Spacer

# ------------------------------------------
# SCENARIO A: 历史回测 (Backtest)
# ------------------------------------------
if mode == "BACKTEST (Historical)":
    
    # 首次使用引导
    if st.session_state.get("show_welcome", True) and not st.session_state.get("user_has_run_backtest", False):
        welcome_col1, welcome_col2 = st.columns([3, 1])
        with welcome_col1:
            st.info("""
            👋 **欢迎使用投资组合回测系统！**
            
            **快速开始指南：**
            1️⃣ **选择策略** - 在左侧边栏选择投资策略算法
            2️⃣ **配置参数** - 设置初始资金、杠杆等参数
            3️⃣ **上传数据** - 上传CSV文件或使用示例数据
            4️⃣ **运行回测** - 点击"EXECUTE BACKTEST"按钮
            5️⃣ **查看结果** - 在6个标签页中查看详细分析
            
            💡 **提示**：首次使用建议选择"Equal Weight"策略和示例数据快速体验
            """)
        with welcome_col2:
            if st.button("✅ 我知道了", use_container_width=True):
                st.session_state["show_welcome"] = False
                st.rerun()
    
    # 操作步骤指引
    st.markdown("### 📋 操作步骤")
    step_col1, step_col2, step_col3, step_col4, step_col5 = st.columns(5)
    
    # 智能判断当前步骤（根据实际配置状态）
    # 步骤1：选择策略
    # 步骤2：配置参数
    # 步骤3：准备数据（自动完成，因为可以使用示例数据）
    # 步骤4：运行回测
    # 步骤5：查看结果
    
    if 'bt_result' in st.session_state:
        current_step = 5  # 有结果，显示步骤5
    elif st.session_state.get("user_has_run_backtest", False):
        current_step = 4  # 正在运行回测
    elif initial_capital > 0 and strategy_name_global:
        current_step = 3  # 参数已配置，准备运行
    elif strategy_name_global:
        current_step = 2  # 已选择策略，需要配置参数
    else:
        current_step = 1  # 初始状态，需要选择策略
    
    step_style_active = "background-color: rgba(210, 153, 34, 0.2); border: 2px solid #D29922; padding: 10px; border-radius: 8px; text-align: center;"
    step_style_done = "background-color: rgba(63, 185, 80, 0.1); border: 2px solid #3FB950; padding: 10px; border-radius: 8px; text-align: center;"
    step_style_pending = "background-color: rgba(139, 148, 158, 0.1); border: 2px solid #8B949E; padding: 10px; border-radius: 8px; text-align: center; opacity: 0.6;"
    
    # 步骤状态判断
    step1_done = strategy_name_global and strategy_name_global in InvestSimBridge.get_available_strategies()
    step2_done = initial_capital > 0
    step3_done = True  # 总是可以使用示例数据
    step4_done = 'bt_result' in st.session_state or st.session_state.get("user_has_run_backtest", False)
    step5_done = 'bt_result' in st.session_state
    
    with step_col1:
        if step1_done:
            style = step_style_done if current_step > 1 else step_style_active
            icon = "✅" if current_step > 1 else "🔄"
        else:
            style = step_style_active
            icon = "📍"
        st.markdown(f'<div style="{style}"><strong>{icon} 步骤 1</strong><br>选择策略</div>', unsafe_allow_html=True)
    
    with step_col2:
        if step2_done:
            style = step_style_done if current_step > 2 else (step_style_active if current_step == 2 else step_style_done)
            icon = "✅" if current_step > 2 else ("🔄" if current_step == 2 else "✅")
        else:
            style = step_style_pending
            icon = "⏳"
        st.markdown(f'<div style="{style}"><strong>{icon} 步骤 2</strong><br>配置参数</div>', unsafe_allow_html=True)
    
    with step_col3:
        if step3_done:
            style = step_style_done if current_step > 3 else (step_style_active if current_step == 3 else step_style_done)
            icon = "✅" if current_step > 3 else ("🔄" if current_step == 3 else "✅")
        else:
            style = step_style_pending
            icon = "⏳"
        st.markdown(f'<div style="{style}"><strong>{icon} 步骤 3</strong><br>准备数据</div>', unsafe_allow_html=True)
    
    with step_col4:
        if step4_done:
            style = step_style_done if current_step > 4 else (step_style_active if current_step == 4 else step_style_done)
            icon = "✅" if current_step > 4 else ("🔄" if current_step == 4 else "✅")
        else:
            style = step_style_pending
            icon = "⏳"
        st.markdown(f'<div style="{style}"><strong>{icon} 步骤 4</strong><br>运行回测</div>', unsafe_allow_html=True)
    
    with step_col5:
        if step5_done:
            style = step_style_done
            icon = "✅"
        else:
            style = step_style_pending
            icon = "⏳"
        st.markdown(f'<div style="{style}"><strong>{icon} 步骤 5</strong><br>查看结果</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 文件上传区域
    with st.expander("DATA SOURCE SETTINGS", expanded=True):
        st.markdown("""
        <div style='background-color: rgba(210, 153, 34, 0.1); padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 3px solid #D29922;'>
        <small><strong>📋 Data Format:</strong> CSV file with date column (first column) and asset price columns.<br>
        <strong>Example:</strong> date, SPY, AGG, GLD<br>
        <strong>Note:</strong> If no file uploaded, synthetic data will be used for demonstration.</small>
        </div>
        """, unsafe_allow_html=True)
        
        col_file, col_reb = st.columns([2, 1])
        with col_file:
            uploaded_file = st.file_uploader("Upload Market Data (CSV)", type=['csv'], label_visibility="collapsed")
            if uploaded_file:
                # 保存上传的文件到session state，供输入建模使用
                st.session_state["uploaded_file_data"] = uploaded_file
            elif "uploaded_file_data" not in st.session_state:
                st.session_state["uploaded_file_data"] = None
            if not uploaded_file:
                st.caption("💡 Using synthetic demonstration data stream.")
                st.caption("📝 **提示**：首次使用建议先用示例数据体验，熟悉后再上传自己的数据")
        with col_reb:
            reb_freq = st.number_input("Rebalance Days", 1, 252, 21,
                                      help="Number of trading days between rebalancing. Lower = more frequent rebalancing.")
            
        # 操作检查清单
        st.markdown("#### ✅ 配置检查清单")
        checklist_items = []
        checklist_status = []
        
        if strategy_name_global:
            checklist_items.append("✅ 策略已选择")
            checklist_status.append(True)
        else:
            checklist_items.append("❌ 请选择策略")
            checklist_status.append(False)
        
        if initial_capital > 0:
            checklist_items.append("✅ 初始资金已设置")
            checklist_status.append(True)
        else:
            checklist_items.append("❌ 请设置初始资金")
            checklist_status.append(False)
        
        if uploaded_file is not None or True:  # 总是可以使用示例数据
            checklist_items.append("✅ 数据已准备（可使用示例数据）")
            checklist_status.append(True)
        
        # 显示检查清单
        for item in checklist_items:
            st.markdown(f"- {item}")
        
        # 状态提示
        if all(checklist_status):
            st.success("🎉 **所有配置已完成，可以运行回测！**")
        else:
            missing_count = len([x for x in checklist_status if not x])
            st.warning(f"⚠️ 还有 {missing_count} 项配置需要完成")
        
        run_bt = st.button("🚀 EXECUTE BACKTEST", type="primary", use_container_width=True)

    if run_bt:
        st.session_state["user_has_run_backtest"] = True
        st.session_state["show_welcome"] = False
        with st.spinner("PROCESSING HISTORICAL DATA..."):
            market_data = InvestSimBridge.load_market_data(uploaded_file)
            params = {
                "strategy": strategy_name_global,
                "leverage": leverage,
                "risk_free": risk_free,
                "capital": initial_capital,
                "rebalance_frequency": reb_freq,
                **strategy_params
            }
            bt_res = InvestSimBridge.run_backtest(params, market_data)
            st.session_state['bt_result'] = bt_res
            
            # 获取完整结果以访问权重历史
            from invest_sim.backtester import Backtester
            config = InvestSimBridge._build_backtest_config(params, market_data)
            backtester = Backtester(config)
            full_result = backtester.run(market_data)
            st.session_state['bt_full_result'] = full_result
            
            # 【关键改进】从标的物价格数据中提取收益率，用于输入建模
            # 这是回测的核心目的之一：得到过去一段时间标的物价格的input model
            asset_returns = market_data.pct_change().dropna()
            # 将所有资产的收益率展平，用于输入建模
            asset_returns_flat = asset_returns.values.flatten()
            asset_returns_flat = asset_returns_flat[~np.isnan(asset_returns_flat)]
            st.session_state['bootstrap_returns'] = asset_returns_flat
            st.session_state['backtest_market_data'] = market_data  # 保存原始价格数据
            
            # 保存回测中选择的策略，供预测使用
            st.session_state['backtest_strategy'] = strategy_name_global
            st.session_state['backtest_strategy_params'] = strategy_params.copy()
            st.session_state['backtest_params'] = {
                "leverage": leverage,
                "risk_free": risk_free,
                "capital": initial_capital,
                "rebalance_frequency": reb_freq,
            }
            
            # 自动进行输入建模（从标的物价格数据）
            with st.spinner("🔬 自动进行输入建模分析..."):
                try:
                    # 只拟合支持的三种分布：Normal, Student-t, Bootstrap
                    fit_results = {}
                    
                    # 1. Normal分布
                    try:
                        from scipy import stats as scipy_stats
                        scipy_available = True
                    except:
                        scipy_available = False
                    
                    normal_mean = float(np.mean(asset_returns_flat))
                    normal_vol = float(np.std(asset_returns_flat))
                    normal_params = {"mean": normal_mean, "vol": normal_vol}
                    
                    if scipy_available:
                        ks_stat, ks_pvalue = scipy_stats.kstest(asset_returns_flat, lambda x: scipy_stats.norm.cdf(x, normal_mean, normal_vol))
                        log_likelihood = np.sum(scipy_stats.norm.logpdf(asset_returns_flat, normal_mean, normal_vol))
                        n_params = 2
                        aic = 2 * n_params - 2 * log_likelihood
                        bic = n_params * np.log(len(asset_returns_flat)) - 2 * log_likelihood
                    else:
                        ks_stat, ks_pvalue, log_likelihood, aic, bic = None, None, None, None, None
                    
                    fit_results["Normal"] = {
                        "params": normal_params,
                        "ks_stat": ks_stat,
                        "ks_pvalue": ks_pvalue,
                        "log_likelihood": log_likelihood,
                        "aic": aic,
                        "bic": bic,
                        "success": True
                    }
                    
                    # 2. Student-t分布
                    if scipy_available:
                        try:
                            df, loc, scale = scipy_stats.t.fit(asset_returns_flat)
                            student_t_params = {"df": float(df), "mean": float(loc), "scale": float(scale)}
                            
                            # 计算拟合优度
                            ks_stat, ks_pvalue = scipy_stats.kstest(asset_returns_flat, lambda x: scipy_stats.t.cdf(x, df, loc, scale))
                            log_likelihood = np.sum(scipy_stats.t.logpdf(asset_returns_flat, df, loc, scale))
                            n_params = 3
                            aic = 2 * n_params - 2 * log_likelihood
                            bic = n_params * np.log(len(asset_returns_flat)) - 2 * log_likelihood
                            
                            fit_results["Student-t"] = {
                                "params": student_t_params,
                                "ks_stat": ks_stat,
                                "ks_pvalue": ks_pvalue,
                                "log_likelihood": log_likelihood,
                                "aic": aic,
                                "bic": bic,
                                "success": True
                            }
                        except Exception as e:
                            fit_results["Student-t"] = {"success": False, "error": str(e)}
                    else:
                        fit_results["Student-t"] = {"success": False, "error": "scipy未安装"}
                    
                    # 3. Bootstrap分布
                    fit_results["Bootstrap"] = {
                        "params": {"historical_returns": asset_returns_flat.tolist()},
                        "ks_stat": 0.0,
                        "ks_pvalue": 1.0,
                        "log_likelihood": None,  # Bootstrap没有解析式
                        "aic": None,
                        "bic": None,
                        "success": True
                    }
                    
                    # 找到最佳分布（只从支持的三种中选择）
                    best_dist = None
                    best_score = -np.inf
                    for dist_name in ["Normal", "Student-t", "Bootstrap"]:
                        result = fit_results.get(dist_name, {})
                        if result.get("success", False):
                            # 使用综合评分
                            score = 0
                            if "ks_pvalue" in result and result["ks_pvalue"] is not None and not np.isnan(result["ks_pvalue"]):
                                score += result["ks_pvalue"] * 2  # p值越高越好
                            if "aic" in result and result["aic"] is not None and not np.isnan(result["aic"]):
                                score -= result["aic"] / 1000  # AIC越低越好
                            if dist_name == "Bootstrap":
                                score += 0.5  # Bootstrap有额外加分（保留完整历史特征）
                            if score > best_score:
                                best_score = score
                                best_dist = dist_name
                    
                    if best_dist:
                        st.session_state["input_model_choice"] = best_dist
                        selected_result = fit_results[best_dist]
                        if best_dist == "Normal":
                            st.session_state["fitted_normal_params"] = selected_result["params"]
                        elif best_dist == "Student-t":
                            st.session_state["fitted_student_t_params"] = selected_result["params"]
                        elif best_dist == "Bootstrap":
                            st.session_state["bootstrap_returns"] = asset_returns_flat
                        
                        st.success(f"✅ **输入建模完成**：基于标的物价格数据，推荐使用 **{best_dist}** 分布模型（将用于未来价格预测）")
                    else:
                        # 默认使用Normal
                        st.session_state["input_model_choice"] = "Normal"
                        st.session_state["fitted_normal_params"] = normal_params
                        st.warning("⚠️ 无法确定最佳分布，使用Normal分布作为默认")
                except Exception as e:
                    import traceback
                    st.warning(f"⚠️ 自动输入建模失败：{str(e)}，将在预测时使用默认参数")
                    st.caption(f"错误详情：{traceback.format_exc()}")
            
            # 自动保存到历史记录
            history_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "strategy": strategy_name_global,
                "metrics": {
                    "total_return": bt_res.metrics.get("total_return", 0),
                    "sharpe": bt_res.metrics.get("sharpe", 0),
                    "max_drawdown": bt_res.metrics.get("max_drawdown", 0),
                    "volatility": bt_res.metrics.get("volatility", 0),
                },
                "params": {
                    "initial_capital": initial_capital,
                    "leverage": leverage,
                    "risk_free": risk_free,
                    "rebalance_frequency": reb_freq,
                },
                "result": bt_res  # 保存完整结果对象
            }
            st.session_state["backtest_history"].append(history_entry)
            
            # 保存指标到session state用于风险预警
            st.session_state["bt_metrics"] = bt_res.metrics

    # 策略对比显示
    if st.session_state.get("show_comparison", False) and len(st.session_state["strategy_comparison"]) > 0:
        st.markdown("---")
        st.markdown("### 🔀 策略对比分析")
        
        comparison_data = st.session_state["strategy_comparison"]
        
        # 创建对比表格
        comparison_df = pd.DataFrame({
            "策略": [entry["strategy"] for entry in comparison_data],
            "总收益": [f"{entry['metrics'].get('total_return', 0):.2%}" for entry in comparison_data],
            "Sharpe比率": [f"{entry['metrics'].get('sharpe', 0):.2f}" for entry in comparison_data],
            "最大回撤": [f"{entry['metrics'].get('max_drawdown', 0):.2%}" for entry in comparison_data],
            "波动率": [f"{entry['metrics'].get('volatility', 0):.2%}" for entry in comparison_data],
        })
        
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # 对比图表
        fig_comparison = go.Figure()
        
        strategies = [entry["strategy"] for entry in comparison_data]
        returns = [entry["metrics"].get("total_return", 0) * 100 for entry in comparison_data]
        sharpe = [entry["metrics"].get("sharpe", 0) for entry in comparison_data]
        
        fig_comparison.add_trace(go.Bar(
            x=strategies,
            y=returns,
            name="总收益 (%)",
            marker_color=COLORS["gold"]
        ))
        
        fig_comparison.update_layout(
            title="策略收益对比",
            template="plotly_dark",
            height=400,
            xaxis_title="策略",
            yaxis_title="总收益 (%)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        if st.button("❌ 关闭对比", use_container_width=True):
            st.session_state["show_comparison"] = False
            st.rerun()
        
        st.markdown("---")

    if 'bt_result' in st.session_state:
        # 成功提示
        st.success("✅ 回测完成！下方显示详细结果。你可以切换到不同标签页查看各种分析。")
        
        res = st.session_state['bt_result']
        metrics = res.metrics
        
        # 风险预警系统
        risk_warnings = []
        if abs(metrics.get("max_drawdown", 0)) > 0.3:  # 最大回撤超过30%
            risk_warnings.append("⚠️ **高风险**：最大回撤超过30%，建议降低杠杆或调整策略")
        if metrics.get("volatility", 0) > 0.4:  # 波动率超过40%
            risk_warnings.append("⚠️ **高波动**：年化波动率超过40%，组合风险较高")
        if metrics.get("sharpe", 0) < 0:  # Sharpe比率为负
            risk_warnings.append("⚠️ **负Sharpe比率**：策略表现低于无风险利率，建议重新评估")
        if metrics.get("max_drawdown", 0) < -0.5:  # 最大回撤超过50%
            risk_warnings.append("🚨 **极高风险**：最大回撤超过50%，存在爆仓风险！")
        
        if risk_warnings:
            st.warning("### ⚠️ 风险预警\n\n" + "\n\n".join(risk_warnings))
        
        # 计算额外风险指标
        portfolio_returns = None
        if 'Returns' in res.df.columns:
            portfolio_returns = res.df['Returns'].dropna()
        elif 'Portfolio' in res.df.columns:
            portfolio_returns = res.df['Portfolio'].pct_change().dropna()
        
        sortino = 0.0
        calmar = 0.0
        max_dd_duration = 0
        
        if portfolio_returns is not None and len(portfolio_returns) > 0:
            sortino = calculate_sortino_ratio(portfolio_returns, risk_free, 252)
            calmar = calculate_calmar_ratio(metrics.get('annualized_return', 0), metrics['max_dd'])
            if 'Portfolio' in res.df.columns:
                max_dd_duration = calculate_max_drawdown_duration(res.df['Portfolio'])
        
        # 结果查看引导
        st.success("✅ **回测完成！** 下方显示详细结果。你可以：")
        guide_result_col1, guide_result_col2, guide_result_col3 = st.columns(3)
        with guide_result_col1:
            st.markdown("""
            📊 **查看指标**
            - 6个核心指标卡片
            - 点击指标查看详细说明
            """)
        with guide_result_col2:
            st.markdown("""
            📈 **分析图表**
            - 切换6个标签页
            - 每个标签页有详细说明
            """)
        with guide_result_col3:
            st.markdown("""
            💾 **导出数据**
            - Excel完整报告
            - CSV原始数据
            """)
        st.markdown("---")
        
        # 扩展的指标显示
        st.markdown("### Performance Metrics")
        
        # 指标说明展开区域
        with st.expander("📖 Metric Definitions", expanded=False):
            st.markdown("""
            **Total Return**: Cumulative return over the entire backtest period  
            **Sharpe Ratio**: Risk-adjusted return (higher is better, typically >1 is good)  
            **Sortino Ratio**: Downside risk-adjusted return (only penalizes negative volatility)  
            **Calmar Ratio**: Annual return divided by maximum drawdown (higher is better)  
            **Max Drawdown**: Largest peak-to-trough decline (lower is better)  
            **Volatility**: Annualized standard deviation of returns (measures risk)
            """)
        
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1: 
            st.metric("Total Return", f"{metrics['total_return']:.2%}", f"CAGR: {metrics.get('annualized_return', 0):.2%}",
                     help="Total return over the backtest period. Delta shows annualized return.")
        with c2: 
            st.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}",
                     help="Measures excess return per unit of risk. >1 is good, >2 is excellent.")
        with c3: 
            st.metric("Sortino Ratio", f"{sortino:.2f}",
                     help="Similar to Sharpe but only considers downside volatility. Better for asymmetric returns.")
        with c4: 
            st.metric("Calmar Ratio", f"{calmar:.2f}",
                     help="Annual return / Max drawdown. Higher values indicate better risk-adjusted performance.")
        with c5: 
            st.metric("Max Drawdown", f"{metrics['max_dd']:.2%}", delta_color="inverse", delta=f"{max_dd_duration}d",
                     help="Largest peak-to-trough decline. Delta shows duration in days.")
        with c6: 
            st.metric("Volatility", f"{metrics['volatility']:.2%}",
                     help="Annualized standard deviation of returns. Measures portfolio risk.")

        # 多标签页图表展示
        chart_tabs = st.tabs(["📈 NAV Curve", "📊 Drawdown", "📉 Returns Distribution", "📊 Asset Weights", "📈 Rolling Analysis", "💾 Export"])
        
        with chart_tabs[0]:
            # 详细说明
            with st.expander("📖 什么是净值曲线（NAV Curve）？", expanded=False):
                st.markdown("""
                **净值（Net Asset Value, NAV）** 是投资组合的总价值，反映你的投资表现。
                
                **这个图表展示什么？**
                - 📈 **主图**：显示投资组合价值随时间的变化曲线
                - 📉 **侧边小图**：显示回撤情况（从峰值下降的幅度）
                
                **如何解读？**
                - **上升趋势**：组合价值增长，投资表现良好
                - **下降趋势**：组合价值减少，可能处于市场下跌期
                - **波动幅度**：曲线越平滑，风险越小；波动越大，风险越高
                
                **关键观察点：**
                - ✅ **最终价值 vs 初始价值**：判断总体盈亏
                - ✅ **增长趋势**：是否持续向上
                - ✅ **波动特征**：是否频繁大幅波动
                - ✅ **回撤幅度**：侧边图显示最大回撤
                
                **实际应用：**
                - 评估策略的长期表现
                - 识别最佳和最差表现时期
                - 对比不同策略的效果
                """)
            
            st.caption("💡 **NAV Curve**: Portfolio net asset value over time. Side panel shows drawdown visualization.")
        col_main, col_side = st.columns([3, 1])
        with col_main:
            st.plotly_chart(plot_nav_curve(res.df), use_container_width=True)
        with col_side:
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=res.df.index, y=res.df['Drawdown'],
                fill='tozeroy', line=dict(color=COLORS['red'], width=1),
                fillcolor='rgba(248, 81, 73, 0.1)'
            ))
            fig_dd.update_layout(**get_chart_layout(200))
            fig_dd.update_layout(title="Drawdown", yaxis=dict(showgrid=False, tickformat=".0%"))
            st.plotly_chart(fig_dd, use_container_width=True)
        
        with chart_tabs[1]:
            # 详细说明
            with st.expander("📖 什么是回撤分析（Drawdown Analysis）？", expanded=False):
                st.markdown("""
                **回撤（Drawdown）** 是指投资组合价值从历史最高点下降的幅度，是衡量风险的重要指标。
                
                **回撤如何计算？**
                - 找到每个时间点的历史最高净值（峰值）
                - 计算当前净值相对于峰值的下降百分比
                - 公式：回撤 = (当前净值 - 历史峰值) / 历史峰值
                
                **这个图表展示什么？**
                - 📉 **红色填充区域**：显示回撤的深度和持续时间
                - 📊 **回撤值**：负值表示下降，0%表示在历史高点
                - ⏱️ **持续时间**：回撤持续的天数
                
                **如何解读？**
                - **最大回撤**：整个回测期间的最大跌幅（越小越好）
                - **回撤持续时间**：从峰值到恢复的时间（越短越好）
                - **平均回撤**：所有回撤期的平均值
                - **>5%回撤次数**：严重回撤发生的频率
                
                **为什么重要？**
                - ⚠️ **风险控制**：了解最坏情况下的损失
                - 📊 **心理承受**：评估能否承受最大回撤
                - 🔄 **恢复能力**：观察组合从回撤中恢复的速度
                - 📈 **策略优化**：通过回撤数据改进策略
                
                **实际例子：**
                - 如果最大回撤是 -20%，意味着在最坏情况下，你的投资可能损失20%
                - 如果回撤持续100天，意味着需要100天才能恢复到之前的峰值
                """)
            
            st.caption("💡 **Drawdown Analysis**: Visualizes portfolio drawdowns (declines from peak). Monitor periods when portfolio value drops below previous highs.")
            # 详细回撤分析
            fig_dd_detailed = go.Figure()
            fig_dd_detailed.add_trace(go.Scatter(
                x=res.df.index, y=res.df['Drawdown'] * 100,
                fill='tozeroy', line=dict(color=COLORS['red'], width=2),
                fillcolor='rgba(248, 81, 73, 0.15)',
                name='Drawdown'
            ))
            fig_dd_detailed.update_layout(**get_chart_layout(400))
            fig_dd_detailed.update_layout(
                title="Drawdown Analysis",
                yaxis=dict(showgrid=True, tickformat=".1f", title="Drawdown (%)")
            )
            st.plotly_chart(fig_dd_detailed, use_container_width=True)
            
            # 回撤统计
            col_dd1, col_dd2, col_dd3, col_dd4 = st.columns(4)
            with col_dd1: st.metric("Max Drawdown", f"{metrics['max_dd']:.2%}")
            with col_dd2: st.metric("Duration", f"{max_dd_duration} days")
            with col_dd3:
                avg_dd = res.df['Drawdown'][res.df['Drawdown'] < 0].mean() if len(res.df['Drawdown'][res.df['Drawdown'] < 0]) > 0 else 0
                st.metric("Avg Drawdown", f"{avg_dd:.2%}")
            with col_dd4:
                dd_count = (res.df['Drawdown'] < -0.05).sum()
                st.metric(">5% Drawdowns", f"{dd_count}")
        
        with chart_tabs[2]:
            # 详细说明
            with st.expander("📖 什么是收益率分布（Returns Distribution）？", expanded=False):
                st.markdown("""
                **收益率分布** 显示投资组合每日收益率的统计特征，帮助理解收益的分布规律和风险特征。
                
                **这个图表展示什么？**
                - 📊 **直方图（金色）**：显示不同收益率区间的出现频率
                - 📈 **正态分布拟合线（蓝色虚线）**：理论上的正态分布曲线
                - 📉 **对比分析**：实际分布 vs 理论分布
                
                **关键统计指标：**
                - **平均日收益（Mean）**：所有日收益率的平均值
                - **标准差（Std Dev）**：收益率的波动程度，越大风险越高
                - **偏度（Skewness）**：
                  - 接近0：分布对称
                  - >0：右偏，有更多正收益（好）
                  - <0：左偏，有更多负收益（风险）
                - **峰度（Kurtosis）**：
                  - 接近3：接近正态分布
                  - >3：尖峰，极端收益更多（高风险）
                  - <3：平峰，收益更分散
                
                **如何解读？**
                - **理想分布**：接近正态分布，偏度接近0，峰度接近3
                - **右偏分布**：更多正收益，但可能有极端负收益
                - **左偏分布**：更多负收益，风险较高
                - **尖峰分布**：极端收益（大涨大跌）较多
                
                **实际应用：**
                - ✅ 评估收益的稳定性
                - ✅ 识别异常收益模式
                - ✅ 预测未来收益概率
                - ✅ 优化风险管理策略
                
                **风险提示：**
                - 如果分布严重左偏或峰度很高，说明策略可能存在极端风险
                - 正态分布拟合可以帮助识别实际分布与理论的偏差
                """)
            
            st.caption("💡 **Returns Distribution**: Histogram of daily returns with normal distribution fit. Check skewness (asymmetry) and kurtosis (tail risk).")
            # 收益率分布
            if portfolio_returns is not None and len(portfolio_returns) > 0:
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=portfolio_returns * 100,
                    nbinsx=50,
                    name='Returns Distribution',
                    marker_color=COLORS['gold'],
                    opacity=0.7
                ))
                
                # 添加正态分布拟合
                mean_ret = portfolio_returns.mean() * 100
                std_ret = portfolio_returns.std() * 100
                x_norm = np.linspace(portfolio_returns.min() * 100, portfolio_returns.max() * 100, 100)
                y_norm = np.exp(-0.5 * ((x_norm - mean_ret) / std_ret) ** 2) / (std_ret * np.sqrt(2 * np.pi))
                y_norm = y_norm * len(portfolio_returns) * (x_norm[1] - x_norm[0])
                
                fig_dist.add_trace(go.Scatter(
                    x=x_norm,
                    y=y_norm,
                    mode='lines',
                    name='Normal Fit',
                    line=dict(color=COLORS['blue'], width=2, dash='dash')
                ))
                
                fig_dist.update_layout(**get_chart_layout(400))
                fig_dist.update_layout(
                    title="Daily Returns Distribution",
                    xaxis=dict(title="Return (%)"),
                    yaxis=dict(title="Frequency")
                )
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # 统计信息
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                with col_stat1: st.metric("Mean Daily Return", f"{portfolio_returns.mean():.4%}")
                with col_stat2: st.metric("Std Dev", f"{portfolio_returns.std():.4%}")
                with col_stat3: st.metric("Skewness", f"{portfolio_returns.skew():.2f}")
                with col_stat4: st.metric("Kurtosis", f"{portfolio_returns.kurtosis():.2f}")
            else:
                st.info("Returns data not available for distribution analysis.")
        
        with chart_tabs[3]:
            # 详细说明
            with st.expander("📖 什么是资产权重分析？", expanded=False):
                st.markdown("""
                **资产权重（Asset Weights）** 表示你的投资组合中每个资产占总投资的比例。
                
                **这个图表展示什么？**
                - 📊 **堆叠面积图**：显示每个资产在组合中的权重如何随时间变化
                - 📈 **Y轴（0-100%）**：表示权重百分比，总和始终为100%
                - 📅 **X轴**：时间轴，显示回测期间
                
                **如何解读？**
                - **固定权重策略**：各资产权重应该保持相对稳定，线条平直
                - **目标风险策略**：权重会根据市场波动自动调整，线条会有波动
                - **自适应再平衡**：权重只在偏离目标时调整，会有阶梯状变化
                
                **为什么重要？**
                - ✅ 检查策略是否按预期执行
                - ✅ 监控再平衡频率是否合理
                - ✅ 发现权重异常波动
                - ✅ 评估策略的稳定性
                """)
            
            st.caption("💡 **Asset Weights**: Shows how portfolio allocation changes over time. Stacked area chart displays weight distribution across assets. Monitor rebalancing frequency and weight stability.")
            # 资产权重热力图
            if 'bt_full_result' in st.session_state:
                full_result = st.session_state['bt_full_result']
                weights_df = full_result.weights_history
                
                # 权重堆叠面积图
                fig_weights = go.Figure()
                for asset in full_result.asset_names:
                    fig_weights.add_trace(go.Scatter(
                        x=weights_df.index,
                        y=weights_df[asset] * 100,
                        mode='lines',
                        name=asset,
                        stackgroup='one',
                        hovertemplate=f'{asset}: %{{y:.1f}}%<extra></extra>'
                    ))
                
                fig_weights.update_layout(**get_chart_layout(400))
                fig_weights.update_layout(
                    title="Asset Allocation Over Time",
                    yaxis=dict(title="Weight (%)", range=[0, 100]),
                    xaxis=dict(title="Date")
                )
                st.plotly_chart(fig_weights, use_container_width=True)
                
                # 平均权重和权重统计
                col_w1, col_w2, col_w3, col_w4 = st.columns(4)
                with col_w1:
                    st.markdown("**Average Weights**")
                    for asset in full_result.asset_names:
                        avg_weight = weights_df[asset].mean()
                        st.metric(asset, f"{avg_weight:.1%}")
                
                with col_w2:
                    st.markdown("**Weight Range**")
                    for asset in full_result.asset_names:
                        weight_range = f"{weights_df[asset].min():.1%} - {weights_df[asset].max():.1%}"
                        st.caption(f"{asset}: {weight_range}")
                
                with col_w3:
                    st.markdown("**Weight Std Dev**")
                    for asset in full_result.asset_names:
                        weight_std = weights_df[asset].std()
                        st.metric(asset, f"{weight_std:.2%}")
                
                with col_w4:
                    st.markdown("**Rebalancing Frequency**")
                    rebal_count = (weights_df.diff().abs().sum(axis=1) > 0.01).sum()
                    st.metric("Rebalances", f"{rebal_count}")
                    st.caption(f"Out of {len(weights_df)} periods")
                    st.caption("💡 Counts periods where weights changed >1%")
            else:
                st.info("Full result data not available. Please re-run backtest.")
        
        with chart_tabs[4]:
            # 详细说明
            with st.expander("📖 什么是滚动分析（Rolling Analysis）？", expanded=False):
                st.markdown("""
                **滚动分析** 使用一个固定大小的"时间窗口"来计算指标，窗口随时间向前移动，展示指标的变化趋势。
                
                **滚动窗口是什么？**
                - 假设窗口大小是60天
                - 第1-60天：计算这60天的指标
                - 第2-61天：窗口向前移动1天，重新计算
                - 第3-62天：继续移动...
                - 这样可以得到每个时间点的"最近N天"的指标值
                
                **这个图表展示什么？**
                - 📈 **滚动Sharpe比率**：风险调整后收益的变化趋势
                - 📊 **滚动波动率**：风险水平的变化
                - 📉 **滚动年化收益**：收益能力的变化
                - ⚠️ **VaR/CVaR**：风险价值指标
                
                **如何调整窗口大小？**
                - **小窗口（30-60天）**：反映短期趋势，更敏感，波动大
                - **中等窗口（60-120天）**：平衡短期和长期，推荐使用
                - **大窗口（180-252天）**：反映长期趋势，更平滑，但滞后
                
                **关键指标解释：**
                - **滚动Sharpe比率**：
                  - >1：风险调整后表现良好
                  - <0：表现不佳，甚至不如无风险资产
                  - 趋势上升：策略表现改善
                - **滚动波动率**：
                  - 上升：风险增加
                  - 下降：风险降低
                  - 稳定：风险可控
                - **VaR (95%)**：在95%置信度下，预期最大损失
                - **CVaR (95%)**：当损失超过VaR时，平均损失是多少
                
                **实际应用：**
                - ✅ 识别策略表现的周期性变化
                - ✅ 发现风险水平的波动
                - ✅ 评估策略在不同市场环境下的表现
                - ✅ 优化再平衡时机
                
                **相关性分析：**
                - 如果显示相关性矩阵，可以查看资产之间的关联程度
                - 相关性接近+1：资产同向运动（分散化效果差）
                - 相关性接近-1：资产反向运动（分散化效果好）
                - 相关性接近0：资产独立运动（理想状态）
                """)
            
            st.caption("💡 **Rolling Analysis**: Time-varying metrics using a rolling window. Adjust window size to see short-term vs long-term trends. Includes VaR/CVaR risk measures.")
            # 滚动窗口分析
            if portfolio_returns is not None and len(portfolio_returns) > 0:
                window_size = st.slider("Rolling Window (days)", 30, 252, 60, 10,
                                       help="Number of days to include in rolling calculations. Smaller windows show more recent trends.")
                
                # 计算滚动指标
                rolling_returns = portfolio_returns.rolling(window=window_size)
                rolling_sharpe = (rolling_returns.mean() * 252) / (rolling_returns.std() * np.sqrt(252))
                rolling_vol = rolling_returns.std() * np.sqrt(252)
                rolling_mean = rolling_returns.mean() * 252
                
                # 滚动Sharpe比率
                fig_rolling_sharpe = go.Figure()
                fig_rolling_sharpe.add_trace(go.Scatter(
                    x=res.df.index[window_size-1:],
                    y=rolling_sharpe[window_size-1:],
                    mode='lines',
                    name='Rolling Sharpe',
                    line=dict(color=COLORS['gold'], width=2)
                ))
                fig_rolling_sharpe.add_hline(y=0, line_dash="dash", line_color=COLORS['text_sub'], opacity=0.5)
                fig_rolling_sharpe.update_layout(**get_chart_layout(300))
                fig_rolling_sharpe.update_layout(
                    title=f"Rolling Sharpe Ratio ({window_size}-day window)",
                    yaxis=dict(title="Sharpe Ratio")
                )
                st.plotly_chart(fig_rolling_sharpe, use_container_width=True)
                
                # 滚动波动率
                fig_rolling_vol = go.Figure()
                fig_rolling_vol.add_trace(go.Scatter(
                    x=res.df.index[window_size-1:],
                    y=rolling_vol[window_size-1:] * 100,
                    mode='lines',
                    name='Rolling Volatility',
                    line=dict(color=COLORS['red'], width=2),
                    fill='tozeroy',
                    fillcolor='rgba(248, 81, 73, 0.1)'
                ))
                fig_rolling_vol.update_layout(**get_chart_layout(300))
                fig_rolling_vol.update_layout(
                    title=f"Rolling Volatility ({window_size}-day window)",
                    yaxis=dict(title="Volatility (%)")
                )
                st.plotly_chart(fig_rolling_vol, use_container_width=True)
                
                # 滚动年化收益
                fig_rolling_ret = go.Figure()
                fig_rolling_ret.add_trace(go.Scatter(
                    x=res.df.index[window_size-1:],
                    y=rolling_mean[window_size-1:] * 100,
                    mode='lines',
                    name='Rolling Annualized Return',
                    line=dict(color=COLORS['green'], width=2)
                ))
                fig_rolling_ret.add_hline(y=0, line_dash="dash", line_color=COLORS['text_sub'], opacity=0.5)
                fig_rolling_ret.update_layout(**get_chart_layout(300))
                fig_rolling_ret.update_layout(
                    title=f"Rolling Annualized Return ({window_size}-day window)",
                    yaxis=dict(title="Return (%)")
                )
                st.plotly_chart(fig_rolling_ret, use_container_width=True)
                
                # 滚动指标统计
                col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                with col_r1:
                    st.metric("Avg Rolling Sharpe", f"{rolling_sharpe[window_size-1:].mean():.2f}")
                with col_r2:
                    st.metric("Avg Rolling Vol", f"{rolling_vol[window_size-1:].mean():.2%}")
                with col_r3:
                    st.metric("Avg Rolling Return", f"{rolling_mean[window_size-1:].mean():.2%}")
                with col_r4:
                    # VaR和CVaR
                    var_95 = np.percentile(portfolio_returns, 5)
                    cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
                    st.metric("VaR (95%)", f"{var_95:.2%}", 
                             help="Value at Risk: Worst expected loss at 95% confidence")
                    st.caption(f"CVaR: {cvar_95:.2%}")
                    st.caption("💡 CVaR = average loss when VaR is exceeded")
                
                # 相关性分析（如果有多个资产）
                if 'bt_full_result' in st.session_state:
                    full_result = st.session_state['bt_full_result']
                    if len(full_result.asset_names) > 1 and 'bt_result' in st.session_state:
                        st.markdown("---")
                        st.markdown("##### Asset Correlation Analysis")
                        
                        # 获取市场数据计算相关性
                        try:
                            market_data = InvestSimBridge.load_market_data(uploaded_file if 'uploaded_file' in locals() else None)
                            asset_returns = market_data.pct_change().dropna()
                            
                            if len(asset_returns.columns) > 1:
                                corr_matrix = asset_returns.corr()
                                
                                # 相关性热力图
                                import plotly.graph_objects as go
                                fig_corr = go.Figure(data=go.Heatmap(
                                    z=corr_matrix.values,
                                    x=corr_matrix.columns,
                                    y=corr_matrix.columns,
                                    colorscale='RdBu',
                                    zmid=0,
                                    text=corr_matrix.round(2).values,
                                    texttemplate='%{text}',
                                    textfont={"size":10},
                                    colorbar=dict(title="Correlation")
                                ))
                                fig_corr.update_layout(**get_chart_layout(400))
                                fig_corr.update_layout(title="Asset Return Correlation Matrix")
                                st.plotly_chart(fig_corr, use_container_width=True)
                                st.caption("💡 **Correlation**: Values close to +1 indicate assets move together, -1 indicates opposite movements. Lower correlation = better diversification.")
                        except:
                            pass
            else:
                st.info("Returns data not available for rolling analysis.")
        
        with chart_tabs[5]:
            # 导出功能
            st.markdown("### Export Backtest Results")
            
            # 详细说明
            with st.expander("📖 如何使用导出功能？", expanded=False):
                st.markdown("""
                **导出功能** 允许你将回测结果保存到本地文件，方便进一步分析和报告。
                
                **📊 Excel 导出（推荐）**
                
                Excel文件包含多个工作表，数据更完整：
                
                1. **NAV Data（净值数据表）**
                   - Date：日期
                   - Portfolio Value：组合净值
                   - Drawdown：回撤值
                   - 用途：绘制净值曲线、计算自定义指标
                
                2. **Weights History（权重历史表）**
                   - Date：日期
                   - 各资产列：每个资产在不同时间的权重
                   - 用途：分析资产配置变化、验证再平衡效果
                
                3. **Metrics（指标汇总表）**
                   - Metric：指标名称
                   - Value：指标数值
                   - 包含：总收益率、年化收益、Sharpe、Sortino、Calmar、最大回撤、波动率、VaR、CVaR等
                   - 用途：快速查看所有关键指标、制作报告
                
                **📄 CSV 导出（简单格式）**
                
                CSV文件格式简单，易于导入其他工具：
                - 包含：日期、组合净值、回撤
                - 格式：逗号分隔，可用Excel、Python、R等打开
                - 用途：快速数据交换、简单分析
                
                **使用建议：**
                - ✅ **制作报告**：使用Excel，包含完整数据
                - ✅ **进一步分析**：使用Excel，可以处理多个工作表
                - ✅ **数据共享**：使用CSV，兼容性好
                - ✅ **程序处理**：使用CSV，易于读取
                
                **文件命名：**
                - 自动包含时间戳，避免覆盖
                - 格式：`backtest_report_YYYYMMDD_HHMMSS.xlsx`
                - 格式：`backtest_data_YYYYMMDD_HHMMSS.csv`
                
                **注意事项：**
                - Excel导出需要安装openpyxl库
                - 如果导出失败，请检查是否有足够权限
                - 大文件可能需要较长时间生成
                """)
            
            st.caption("💡 **Export Options**: Download comprehensive backtest results in Excel (with multiple sheets including NAV data, weights history, and metrics) or CSV format for further analysis.")
            
            st.markdown("""
            <div style='background-color: rgba(88, 166, 255, 0.1); padding: 15px; border-radius: 5px; margin: 15px 0; border-left: 3px solid #58A6FF;'>
            <strong>📊 Excel Export Includes:</strong><br>
            • <strong>NAV Data</strong>: Portfolio values and drawdowns over time<br>
            • <strong>Weights History</strong>: Asset allocation changes (if available)<br>
            • <strong>Metrics</strong>: All performance indicators including VaR/CVaR<br><br>
            <strong>📄 CSV Export:</strong> Simple format with date, portfolio value, and drawdown
            </div>
            """, unsafe_allow_html=True)
            
            import io
            from datetime import datetime
            
            col_exp1, col_exp2, col_exp3 = st.columns(3)
            
            with col_exp1:
                # Excel 导出
                try:
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        # 净值数据
                        nav_df = pd.DataFrame({
                            'Date': res.df.index,
                            'Portfolio Value': res.df['Portfolio'] if 'Portfolio' in res.df.columns else res.df.iloc[:, 0],
                            'Drawdown': res.df['Drawdown']
                        })
                        nav_df.to_excel(writer, sheet_name='NAV Data', index=False)
                        
                        # 权重历史（如果有）
                        if 'bt_full_result' in st.session_state:
                            full_result = st.session_state['bt_full_result']
                            weights_export = full_result.weights_history.copy()
                            weights_export.index.name = 'Date'
                            weights_export.to_excel(writer, sheet_name='Weights History')
                        
                        # 指标汇总
                        metrics_df = pd.DataFrame({
                            'Metric': ['Total Return', 'Annualized Return', 'Sharpe Ratio', 'Sortino Ratio', 
                                      'Calmar Ratio', 'Max Drawdown', 'Volatility', 'Max DD Duration (days)',
                                      'VaR (95%)', 'CVaR (95%)'],
                            'Value': [
                                metrics['total_return'],
                                metrics.get('annualized_return', 0),
                                metrics['sharpe'],
                                sortino,
                                calmar,
                                metrics['max_dd'],
                                metrics['volatility'],
                                max_dd_duration,
                                np.percentile(portfolio_returns, 5) if portfolio_returns is not None and len(portfolio_returns) > 0 else 0,
                                portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean() if portfolio_returns is not None and len(portfolio_returns) > 0 else 0
                            ]
                        })
                        metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
                    
                    st.download_button(
                        label="📊 Download Excel Report",
                        data=output.getvalue(),
                        file_name=f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openpyxl-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Excel export failed: {str(e)}")
                    st.info("Please install openpyxl: pip install openpyxl")
            
            with col_exp2:
                # CSV 导出
                csv_data = pd.DataFrame({
                    'Date': res.df.index,
                    'Portfolio Value': res.df['Portfolio'] if 'Portfolio' in res.df.columns else res.df.iloc[:, 0],
                    'Drawdown': res.df['Drawdown']
                })
                csv_str = csv_data.to_csv(index=False)
                st.download_button(
                    label="📄 Download CSV Data",
                    data=csv_str,
                    file_name=f"backtest_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col_exp3:
                # 文档报告导出
                try:
                    # 获取输入建模信息
                    input_model_info = None
                    input_model_choice = st.session_state.get("input_model_choice", "Normal")
                    if input_model_choice == "Normal" and "fitted_normal_params" in st.session_state:
                        input_model_info = {
                            "dist_name": "Normal",
                            "params": st.session_state["fitted_normal_params"]
                        }
                    elif input_model_choice == "Student-t" and "fitted_student_t_params" in st.session_state:
                        input_model_info = {
                            "dist_name": "Student-t",
                            "params": st.session_state["fitted_student_t_params"]
                        }
                    elif input_model_choice == "Bootstrap" and "bootstrap_returns" in st.session_state:
                        input_model_info = {
                            "dist_name": "Bootstrap",
                            "params": {"samples": len(st.session_state["bootstrap_returns"])}
                        }
                    
                    # 生成报告
                    report_markdown = generate_backtest_report_markdown(
                        strategy_name=strategy_name_global,
                        initial_capital=st.session_state.get("settings_initial_capital", 1000000),
                        leverage=st.session_state.get("settings_leverage", 1.0),
                        risk_free_rate=st.session_state.get("settings_risk_free_rate", st.session_state.get("settings_risk_free", 0.03)),
                        metrics=metrics,
                        sortino=sortino,
                        calmar=calmar,
                        max_dd_duration=max_dd_duration,
                        portfolio_returns=portfolio_returns.values if portfolio_returns is not None and hasattr(portfolio_returns, 'values') else (portfolio_returns if isinstance(portfolio_returns, np.ndarray) else None),
                        input_model_info=input_model_info
                    )
                    
                    st.download_button(
                        label="📝 Download Full Report (Markdown)",
                        data=report_markdown.encode('utf-8'),
                        file_name=f"backtest_full_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        use_container_width=True,
                        help="包含从输入建模到决策建议的完整分析报告"
                    )
                except Exception as e:
                    st.error(f"Report generation failed: {str(e)}")
                    st.caption("💡 如果遇到问题，请确保已运行回测并查看错误信息")
        
        # ==========================================
        # 回测结论与决策建议（放在图表之后）
        # ==========================================
        st.markdown("---")
        st.markdown("### 🎯 回测结论与决策建议")
        
        # 生成综合评估
        conclusion_col1, conclusion_col2 = st.columns([2, 1])
        
        with conclusion_col1:
            # 综合评分（0-100）
            score = 0
            score_details = []
            
            # 收益评分（30分）
            if metrics['total_return'] > 0.2:
                ret_score = 30
                ret_comment = "优秀"
            elif metrics['total_return'] > 0.1:
                ret_score = 20
                ret_comment = "良好"
            elif metrics['total_return'] > 0:
                ret_score = 10
                ret_comment = "一般"
            else:
                ret_score = 0
                ret_comment = "亏损"
            score += ret_score
            score_details.append(f"收益表现：{ret_comment} (+{ret_score}分)")
            
            # 风险调整收益评分（30分）
            sharpe_score = min(30, max(0, int(metrics['sharpe'] * 10)))
            if metrics['sharpe'] > 1.5:
                sharpe_comment = "优秀"
            elif metrics['sharpe'] > 1.0:
                sharpe_comment = "良好"
            elif metrics['sharpe'] > 0.5:
                sharpe_comment = "一般"
            else:
                sharpe_comment = "较差"
            score += sharpe_score
            score_details.append(f"风险调整收益：{sharpe_comment} (+{sharpe_score}分)")
            
            # 风险控制评分（20分）
            if metrics['max_dd'] > -0.1:
                risk_score = 20
                risk_comment = "优秀"
            elif metrics['max_dd'] > -0.2:
                risk_score = 15
                risk_comment = "良好"
            elif metrics['max_dd'] > -0.3:
                risk_score = 10
                risk_comment = "一般"
            else:
                risk_score = 5
                risk_comment = "较差"
            score += risk_score
            score_details.append(f"风险控制：{risk_comment} (+{risk_score}分)")
            
            # 稳定性评分（20分）
            vol_score = max(0, 20 - int(metrics['volatility'] * 100))
            if metrics['volatility'] < 0.1:
                vol_comment = "非常稳定"
            elif metrics['volatility'] < 0.15:
                vol_comment = "较稳定"
            elif metrics['volatility'] < 0.2:
                vol_comment = "中等波动"
            else:
                vol_comment = "高波动"
            score += vol_score
            score_details.append(f"波动性：{vol_comment} (+{vol_score}分)")
            
            # 总体评价
            if score >= 80:
                overall_rating = "优秀 ⭐⭐⭐⭐⭐"
                rating_color = "#3FB950"
                recommendation = "强烈推荐"
            elif score >= 65:
                overall_rating = "良好 ⭐⭐⭐⭐"
                rating_color = "#58A6FF"
                recommendation = "推荐"
            elif score >= 50:
                overall_rating = "一般 ⭐⭐⭐"
                rating_color = "#D29922"
                recommendation = "可考虑"
            elif score >= 35:
                overall_rating = "较差 ⭐⭐"
                rating_color = "#F85149"
                recommendation = "需改进"
            else:
                overall_rating = "差 ⭐"
                rating_color = "#F85149"
                recommendation = "不推荐"
            
            # 显示评分卡片
            st.markdown(f"""
            <div style='background-color: rgba({int(rating_color[1:3], 16)}, {int(rating_color[3:5], 16)}, {int(rating_color[5:7], 16)}, 0.1); 
                        padding: 20px; border-radius: 10px; border-left: 4px solid {rating_color}; margin-bottom: 20px;'>
            <h3 style='color: {rating_color}; margin-top: 0;'>综合评分：{score}/100</h3>
            <h4 style='color: {rating_color};'>总体评价：{overall_rating}</h4>
            <p style='font-size: 16px;'><strong>建议：{recommendation}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # 评分详情
            with st.expander("📊 评分详情", expanded=False):
                for detail in score_details:
                    st.markdown(f"- {detail}")
                st.markdown(f"**总分：{score}/100**")
        
        with conclusion_col2:
            # 关键指标卡片
            st.markdown("#### 关键指标")
            st.metric("总收益率", f"{metrics['total_return']:.2%}")
            st.metric("Sharpe比率", f"{metrics['sharpe']:.2f}")
            st.metric("最大回撤", f"{metrics['max_dd']:.2%}")
            st.metric("波动率", f"{metrics['volatility']:.2%}")
        
        # 决策建议
        st.markdown("---")
        st.markdown("#### 💡 决策建议")
        
        advice_col1, advice_col2 = st.columns(2)
        
        with advice_col1:
            st.markdown("##### ✅ 策略优势")
            advantages = []
            
            if metrics['sharpe'] > 1.5:
                advantages.append("**风险调整后收益优秀** - Sharpe比率超过1.5，说明策略在控制风险的同时获得了良好收益")
            elif metrics['sharpe'] > 1.0:
                advantages.append("**风险调整后收益良好** - Sharpe比率超过1.0，策略表现优于市场平均水平")
            
            if metrics['max_dd'] > -0.15:
                advantages.append("**回撤控制良好** - 最大回撤小于15%，风险控制能力较强")
            
            if sortino > 1.5:
                advantages.append("**下行风险控制优秀** - Sortino比率较高，说明策略在下跌时表现更好")
            
            if calmar > 1.0:
                advantages.append("**收益回撤比优秀** - Calmar比率超过1.0，说明收益能力远强于最大损失")
            
            if metrics['volatility'] < 0.15:
                advantages.append("**波动率较低** - 组合波动性较小，适合稳健型投资者")
            
            if not advantages:
                advantages.append("策略表现中规中矩，无明显突出优势")
            
            for adv in advantages:
                st.markdown(f"- {adv}")
        
        with advice_col2:
            st.markdown("##### ⚠️ 需要关注")
            concerns = []
            
            if metrics['total_return'] < 0:
                concerns.append("**出现亏损** - 总收益率为负，需要重新评估策略或市场环境")
            elif metrics['total_return'] < 0.05:
                concerns.append("**收益偏低** - 总收益率低于5%，可能不如无风险资产")
            
            if metrics['sharpe'] < 0.5:
                concerns.append("**风险调整收益较差** - Sharpe比率低于0.5，风险收益比不理想")
            
            if metrics['max_dd'] < -0.3:
                concerns.append("**回撤较大** - 最大回撤超过30%，风险较高，需要评估承受能力")
            
            if metrics['volatility'] > 0.25:
                concerns.append("**波动率较高** - 组合波动性较大，可能不适合风险厌恶型投资者")
            
            if sortino < 0.5:
                concerns.append("**下行风险控制不足** - Sortino比率较低，下跌时损失可能较大")
            
            if not concerns:
                concerns.append("策略表现良好，无明显风险点")
            
            for concern in concerns:
                st.markdown(f"- {concern}")
        
        # 策略适用性评估
        st.markdown("---")
        st.markdown("#### 🎯 策略适用性评估")
        
        suitability_col1, suitability_col2, suitability_col3 = st.columns(3)
        
        with suitability_col1:
            st.markdown("##### 📊 适合的投资者类型")
            investor_types = []
            
            if metrics['volatility'] < 0.12 and metrics['max_dd'] > -0.15:
                investor_types.append("✅ **风险厌恶型** - 低波动、低回撤")
            
            if metrics['sharpe'] > 1.0 and metrics['total_return'] > 0.1:
                investor_types.append("✅ **平衡型** - 收益风险平衡")
            
            if metrics['total_return'] > 0.15 and metrics['sharpe'] > 1.2:
                investor_types.append("✅ **成长型** - 追求较高收益")
            
            if not investor_types:
                investor_types.append("⚠️ 需要根据个人风险偏好谨慎评估")
            
            for it in investor_types:
                st.markdown(it)
        
        with suitability_col2:
            st.markdown("##### 📈 市场环境适应性")
            market_conditions = []
            
            if metrics['sharpe'] > 1.0:
                market_conditions.append("✅ **趋势市场** - 表现良好")
            
            if sortino > metrics['sharpe']:
                market_conditions.append("✅ **震荡市场** - 下行风险控制好")
            
            if metrics['volatility'] < 0.15:
                market_conditions.append("✅ **波动市场** - 稳定性好")
            
            if not market_conditions:
                market_conditions.append("⚠️ 需要结合具体市场环境分析")
            
            for mc in market_conditions:
                st.markdown(mc)
        
        with suitability_col3:
            st.markdown("##### 🔄 优化建议")
            optimizations = []
            
            if metrics['sharpe'] < 1.0:
                optimizations.append("💡 考虑调整策略参数以提高风险调整收益")
            
            if metrics['max_dd'] < -0.2:
                optimizations.append("💡 增加风险控制措施，降低最大回撤")
            
            if metrics['volatility'] > 0.2:
                optimizations.append("💡 考虑增加低波动资产以降低组合波动")
            
            if calmar < 0.5:
                optimizations.append("💡 优化收益回撤比，提高策略效率")
            
            if not optimizations:
                optimizations.append("✅ 策略表现良好，可继续使用")
            
            for opt in optimizations:
                st.markdown(opt)
        
        # 最终结论
        st.markdown("---")
        st.markdown("#### 📝 最终结论")
        
        conclusion_text = f"""
        **策略表现总结：**
        
        本次回测显示，{strategy_name_global}策略在测试期间取得了{'良好' if score >= 65 else '一般' if score >= 50 else '较差'}的表现。
        
        **核心发现：**
        - 总收益率为 **{metrics['total_return']:.2%}**，{'表现优秀' if metrics['total_return'] > 0.15 else '表现良好' if metrics['total_return'] > 0.05 else '表现一般' if metrics['total_return'] > 0 else '出现亏损'}
        - 风险调整后收益（Sharpe比率）为 **{metrics['sharpe']:.2f}**，{'优于市场平均水平' if metrics['sharpe'] > 1.0 else '低于市场平均水平'}
        - 最大回撤为 **{metrics['max_dd']:.2%}**，{'风险控制良好' if metrics['max_dd'] > -0.15 else '风险控制一般' if metrics['max_dd'] > -0.25 else '风险较高'}
        - 组合波动率为 **{metrics['volatility']:.2%}**，{'波动性较低' if metrics['volatility'] < 0.15 else '波动性中等' if metrics['volatility'] < 0.25 else '波动性较高'}
        
        **决策建议：**
        {'✅ 该策略表现优秀，建议继续使用或适当增加配置' if score >= 80 else '✅ 该策略表现良好，可以继续使用' if score >= 65 else '⚠️ 该策略表现一般，建议优化参数或考虑其他策略' if score >= 50 else '❌ 该策略表现较差，建议重新评估或更换策略'}
        
        **风险提示：**
        - 历史表现不代表未来收益
        - 回测结果基于历史数据，实际投资可能面临不同市场环境
        - 建议结合个人风险承受能力做出最终决策
        """
        
        st.info(conclusion_text)

# ------------------------------------------
# SCENARIO B: 蒙特卡洛模拟 (Projection)
# ------------------------------------------
elif mode == "PROJECTION (Monte Carlo)":
    
    # 首次使用引导
    if st.session_state.get("show_welcome", True) and not st.session_state.get("user_has_run_projection", False):
        welcome_col1, welcome_col2 = st.columns([3, 1])
        with welcome_col1:
            st.info("""
            👋 **欢迎使用蒙特卡洛预测系统！**
            
            **📋 完整工作流程：**
            
            **第一步：回测（BACKTEST）**
            1️⃣ 上传历史标的物价格数据（CSV格式）
            2️⃣ 选择策略并配置参数
            3️⃣ 运行回测，系统自动进行输入建模（Input Modeling）
            → **目的**：从历史数据中提取标的物价格的分布特征，得到Input Model
            
            **第二步：预测（PROJECTION）**
            4️⃣ 系统自动使用回测中选择的策略和Input Model
            5️⃣ 配置预测时间期限和模拟次数
            6️⃣ 运行模拟，查看策略在未来价格走向下的表现
            → **目的**：使用Input Model模拟未来价格，评估策略表现
            
            💡 **提示**：建议先完成回测，获得Input Model后再进行预测，这样预测结果更准确
            """)
        with welcome_col2:
            if st.button("✅ 我知道了", use_container_width=True, key="welcome_projection"):
                st.session_state["show_welcome"] = False
                st.rerun()
    
    # 操作步骤指引
    st.markdown("### 📋 操作步骤")
    step_col1, step_col2, step_col3, step_col4, step_col5 = st.columns(5)
    
    # 智能判断当前步骤
    if 'mc_result' in st.session_state:
        current_step = 5  # 有结果，显示步骤5
    elif st.session_state.get("user_has_run_projection", False):
        current_step = 4  # 正在运行模拟
    elif initial_capital > 0 and strategy_name_global:
        current_step = 3  # 参数已配置，准备运行
    elif strategy_name_global:
        current_step = 2  # 已选择策略，需要配置参数
    else:
        current_step = 1  # 初始状态，需要选择策略
    
    step_style_active = "background-color: rgba(210, 153, 34, 0.2); border: 2px solid #D29922; padding: 10px; border-radius: 8px; text-align: center;"
    step_style_done = "background-color: rgba(63, 185, 80, 0.1); border: 2px solid #3FB950; padding: 10px; border-radius: 8px; text-align: center;"
    step_style_pending = "background-color: rgba(139, 148, 158, 0.1); border: 2px solid #8B949E; padding: 10px; border-radius: 8px; text-align: center; opacity: 0.6;"
    
    # 步骤状态判断
    step1_done = strategy_name_global and strategy_name_global in InvestSimBridge.get_available_strategies()
    step2_done = initial_capital > 0
    step3_done = True  # 参数配置总是可以完成
    step4_done = 'mc_result' in st.session_state or st.session_state.get("user_has_run_projection", False)
    step5_done = 'mc_result' in st.session_state
    
    with step_col1:
        if step1_done:
            style = step_style_done if current_step > 1 else step_style_active
            icon = "✅" if current_step > 1 else "🔄"
        else:
            style = step_style_active
            icon = "📍"
        st.markdown(f'<div style="{style}"><strong>{icon} 步骤 1</strong><br>选择策略</div>', unsafe_allow_html=True)
    
    with step_col2:
        if step2_done:
            style = step_style_done if current_step > 2 else (step_style_active if current_step == 2 else step_style_done)
            icon = "✅" if current_step > 2 else ("🔄" if current_step == 2 else "✅")
        else:
            style = step_style_pending
            icon = "⏳"
        st.markdown(f'<div style="{style}"><strong>{icon} 步骤 2</strong><br>配置参数</div>', unsafe_allow_html=True)
    
    with step_col3:
        if step3_done:
            style = step_style_done if current_step > 3 else (step_style_active if current_step == 3 else step_style_done)
            icon = "✅" if current_step > 3 else ("🔄" if current_step == 3 else "✅")
        else:
            style = step_style_pending
            icon = "⏳"
        st.markdown(f'<div style="{style}"><strong>{icon} 步骤 3</strong><br>设置模拟</div>', unsafe_allow_html=True)
    
    with step_col4:
        if step4_done:
            style = step_style_done if current_step > 4 else (step_style_active if current_step == 4 else step_style_done)
            icon = "✅" if current_step > 4 else ("🔄" if current_step == 4 else "✅")
        else:
            style = step_style_pending
            icon = "⏳"
        st.markdown(f'<div style="{style}"><strong>{icon} 步骤 4</strong><br>运行模拟</div>', unsafe_allow_html=True)
    
    with step_col5:
        if step5_done:
            style = step_style_done
            icon = "✅"
        else:
            style = step_style_pending
            icon = "⏳"
        st.markdown(f'<div style="{style}"><strong>{icon} 步骤 5</strong><br>查看结果</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 数据源设置
    with st.expander("DATA SOURCE SETTINGS", expanded=True):
        st.markdown("""
        <div style='background-color: rgba(210, 153, 34, 0.1); padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 3px solid #D29922;'>
        <small><strong>📋 Data Format:</strong> CSV file with date column (first column) and asset price columns.<br>
        <strong>Example:</strong> date, SPY, AGG, GLD<br>
        <strong>Note:</strong> Upload historical data to fit return distribution (especially for Bootstrap mode). If no file uploaded, default parameters will be used.</small>
        </div>
        """, unsafe_allow_html=True)
        
        col_file, col_data_info = st.columns([2, 1])
        with col_file:
            uploaded_file_projection = st.file_uploader("Upload Historical Data (CSV)", type=['csv'], 
                                                         key="projection_upload", label_visibility="collapsed")
            if not uploaded_file_projection:
                st.caption("💡 Using default parameters for return distribution.")
                st.caption("📝 **提示**：上传历史数据可以更准确地拟合收益分布，特别是使用Bootstrap模式时")
            else:
                st.success("✅ 数据已上传，将用于拟合收益分布")
                # 预览数据
                try:
                    preview_df = pd.read_csv(uploaded_file_projection, index_col=0, parse_dates=True, nrows=5)
                    st.caption(f"📊 数据预览：{len(preview_df.columns)} 个资产，前5行数据")
                except:
                    st.warning("⚠️ 数据格式可能不正确，请检查CSV格式")
        
        with col_data_info:
            if uploaded_file_projection:
                try:
                    data_info = pd.read_csv(uploaded_file_projection, index_col=0, parse_dates=True)
                    st.markdown("**数据信息**")
                    st.caption(f"资产数量: {len(data_info.columns)}")
                    st.caption(f"数据点: {len(data_info)}")
                    st.caption(f"日期范围: {data_info.index[0].date()} 至 {data_info.index[-1].date()}")
                except:
                    pass
    
    # 模拟参数设置
    with st.expander("SIMULATION PARAMETERS", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1: sim_years = st.number_input("Horizon (Years)", 1, 50, 10,
                                            help="预测时间期限（年）")
        with c2: num_trials = st.number_input("Monte Carlo Trials", 100, 5000, 1000,
                                             help="蒙特卡洛模拟次数，越多越准确但计算时间越长")
        with c3: annual_cont = st.number_input("Annual Contribution", 0, 1000000, 0,
                                               help="每年追加投资金额")
        # 检查是否有输入建模结果
        has_input_modeling = False
        input_model_choice_from_modeling = st.session_state.get("input_model_choice", None)
        input_modeling_info = None
        
        if input_model_choice_from_modeling:
            # 检查是否有对应的参数
            if input_model_choice_from_modeling == "Normal" and st.session_state.get("fitted_normal_params"):
                has_input_modeling = True
                input_modeling_info = {
                    "type": "Normal",
                    "params": st.session_state["fitted_normal_params"],
                    "source": "输入建模"
                }
            elif input_model_choice_from_modeling == "Student-t" and st.session_state.get("fitted_student_t_params"):
                has_input_modeling = True
                input_modeling_info = {
                    "type": "Student-t",
                    "params": st.session_state["fitted_student_t_params"],
                    "source": "输入建模"
                }
            elif input_model_choice_from_modeling == "Bootstrap" and st.session_state.get("bootstrap_returns") is not None:
                has_input_modeling = True
                bootstrap_returns = st.session_state.get("bootstrap_returns")
                # 处理不同的数据类型
                if isinstance(bootstrap_returns, np.ndarray):
                    if len(bootstrap_returns) > 0:
                        input_modeling_info = {
                            "type": "Bootstrap",
                            "params": {"historical_returns": bootstrap_returns.tolist() if hasattr(bootstrap_returns, 'tolist') else bootstrap_returns.tolist()},
                            "source": "输入建模"
                        }
                elif isinstance(bootstrap_returns, (list, tuple)):
                    if len(bootstrap_returns) > 0:
                        input_modeling_info = {
                            "type": "Bootstrap",
                            "params": {"historical_returns": list(bootstrap_returns)},
                            "source": "输入建模"
                        }
                else:
                    # 尝试转换
                    try:
                        bootstrap_array = np.array(bootstrap_returns)
                        if len(bootstrap_array) > 0:
                            input_modeling_info = {
                                "type": "Bootstrap",
                                "params": {"historical_returns": bootstrap_array.tolist()},
                                "source": "输入建模"
                            }
                    except:
                        pass
        
        input_choices = ["Normal", "Student-t", "Bootstrap"]
        
        # 如果有输入建模结果，优先使用
        if has_input_modeling and input_modeling_info:
            default_choice = input_model_choice_from_modeling
            if input_model_choice_from_modeling not in input_choices:
                default_choice = "Normal"
        else:
            default_choice = st.session_state.get("input_model_choice", "Normal")
            if default_choice not in input_choices:
                default_choice = "Normal"
        
        with c4:
            input_model_type = st.selectbox(
                "Return Dist", 
                input_choices, 
                index=input_choices.index(default_choice) if default_choice in input_choices else 0,
                help="收益分布模型：Normal(正态分布), Student-t(t分布), Bootstrap(经验分布)。如果已完成输入建模，将自动使用建模结果。"
            )
        
        # 显示输入建模状态提示
        if has_input_modeling and input_modeling_info:
            # 检查是否有回测中选择的策略
            backtest_strategy = st.session_state.get("backtest_strategy", None)
            if backtest_strategy:
                st.success(f"""
                ✅ **已应用回测结果**：
                - **策略**：{backtest_strategy}（来自回测）
                - **Input Model**：{input_model_choice_from_modeling} 分布（基于历史标的物价格数据拟合）
                - **用途**：使用此Input Model模拟未来价格走向，评估策略在未来表现
                """)
            else:
                st.success(f"✅ **已应用输入建模结果**：使用 {input_model_choice_from_modeling} 分布（基于历史数据拟合）。这将使预测更准确地反映历史市场特征。")
            
            if input_model_type != input_model_choice_from_modeling:
                st.warning(f"⚠️ **注意**：当前选择的分布（{input_model_type}）与输入建模结果（{input_model_choice_from_modeling}）不一致。建议使用输入建模推荐的分布以获得更准确的预测。")
        else:
            # 提供更详细的调试信息
            debug_info = []
            if input_model_choice_from_modeling:
                debug_info.append(f"选择的分布: {input_model_choice_from_modeling}")
                if input_model_choice_from_modeling == "Normal":
                    debug_info.append(f"Normal参数存在: {'fitted_normal_params' in st.session_state}")
                elif input_model_choice_from_modeling == "Student-t":
                    debug_info.append(f"Student-t参数存在: {'fitted_student_t_params' in st.session_state}")
                elif input_model_choice_from_modeling == "Bootstrap":
                    debug_info.append(f"Bootstrap数据存在: {'bootstrap_returns' in st.session_state}")
                    if 'bootstrap_returns' in st.session_state:
                        bootstrap_data = st.session_state.get("bootstrap_returns")
                        debug_info.append(f"Bootstrap数据类型: {type(bootstrap_data)}, 长度: {len(bootstrap_data) if hasattr(bootstrap_data, '__len__') else 'N/A'}")
            else:
                debug_info.append("未找到 input_model_choice")
            
            st.warning("⚠️ **未检测到输入建模结果**：建议先在「输入建模」功能中分析历史数据，以获得更准确的预测。当前将使用默认参数。")
            with st.expander("🔍 调试信息", expanded=False):
                st.text("\n".join(debug_info) if debug_info else "无调试信息")
        
        # 输入建模详细说明和可视化
        st.markdown("---")
        with st.expander("📊 输入建模（Input Modeling）详解", expanded=True):
            st.markdown("""
            **什么是输入建模？**
            
            输入建模是Monte Carlo模拟的核心，它基于**历史标的物价格数据**，分析收益率的分布特征，然后使用这个分布模型来**模拟未来价格走向**。
            
            **工作流程：**
            1. 📊 **分析历史数据**：从上传的标的物价格数据中提取收益率序列
            2. 📈 **拟合分布模型**：使用多种统计分布（Normal、Student-t、Bootstrap等）拟合历史收益率
            3. ✅ **选择最佳模型**：通过拟合优度指标（KS检验、AIC、BIC等）选择最合适的分布
            4. 🚀 **用于未来模拟**：在PROJECTION模式中，使用选定的分布模型生成未来收益率，模拟价格路径
            5. 📊 **评估策略表现**：基于模拟的未来价格路径，评估不同策略的表现
            
            **为什么重要？**
            
            输入建模决定了Monte Carlo模拟如何生成未来的资产收益率。
            不同的分布模型会对预测结果产生显著影响，选择合适的模型至关重要。
            """)
            
            # 三种分布模型的详细说明
            model_tabs = st.tabs(["📈 Normal（正态分布）", "📊 Student-t（t分布）", "🔄 Bootstrap（经验分布）"])
            
            with model_tabs[0]:
                st.markdown("""
                ### 📈 Normal（正态分布）
                
                **特点：**
                - ✅ **简单易用**：最常用的分布模型
                - ✅ **参数明确**：只需要均值和标准差
                - ✅ **计算快速**：适合快速模拟
                
                **适用场景：**
                - 市场波动相对稳定
                - 需要快速得到初步预测
                - 数据量较少时
                
                **局限性：**
                - ⚠️ 假设收益服从正态分布（实际市场可能有偏态和厚尾）
                - ⚠️ 可能低估极端事件（黑天鹅）的概率
                
                **参数说明：**
                - **均值（Mean）**：预期日收益率
                - **波动率（Volatility）**：收益率的标准差
                
                **如何获取参数：**
                - 从历史数据计算：上传CSV文件，系统自动拟合
                - 手动设置：根据市场预期设置
                """)
                
                # 可视化正态分布
                if uploaded_file_projection is not None or st.session_state.get("bootstrap_returns") is not None:
                    try:
                        if uploaded_file_projection is not None:
                            market_data = InvestSimBridge.load_market_data(uploaded_file_projection)
                            returns = market_data.pct_change().dropna()
                            sample_returns = returns.values.flatten()
                            sample_returns = sample_returns[~np.isnan(sample_returns)]
                        else:
                            sample_returns = st.session_state.get("bootstrap_returns", np.array([]))
                        
                        if len(sample_returns) > 0:
                            mean_ret = np.mean(sample_returns)
                            std_ret = np.std(sample_returns)
                            
                            # 生成正态分布曲线
                            x = np.linspace(sample_returns.min(), sample_returns.max(), 100)
                            y = (1 / (std_ret * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_ret) / std_ret) ** 2)
                            
                            fig_normal = go.Figure()
                            fig_normal.add_trace(go.Histogram(
                                x=sample_returns,
                                name="历史收益率",
                                opacity=0.6,
                                nbinsx=50,
                                marker_color=COLORS["blue"]
                            ))
                            fig_normal.add_trace(go.Scatter(
                                x=x,
                                y=y * len(sample_returns) * (x[1] - x[0]),
                                name="正态分布拟合",
                                line=dict(color=COLORS["gold"], width=2)
                            ))
                            fig_normal.update_layout(
                                title=f"正态分布拟合（均值={mean_ret:.4f}, 标准差={std_ret:.4f}）",
                                xaxis_title="收益率",
                                yaxis_title="频数",
                                template="plotly_dark",
                                height=300,
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                            )
                            st.plotly_chart(fig_normal, use_container_width=True)
                            
                            st.info(f"📊 **拟合参数**：均值 = {mean_ret:.4f} ({mean_ret*252:.2%} 年化), 标准差 = {std_ret:.4f} ({std_ret*np.sqrt(252):.2%} 年化波动率)")
                    except:
                        pass
            
            with model_tabs[1]:
                st.markdown("""
                ### 📊 Student-t（t分布）
                
                **特点：**
                - ✅ **考虑厚尾**：比正态分布有更厚的尾部
                - ✅ **更真实**：能更好地捕捉极端事件
                - ✅ **灵活调整**：通过自由度参数控制尾部厚度
                
                **适用场景：**
                - 市场波动较大，极端事件较多
                - 需要更保守的风险估计
                - 数据呈现明显的厚尾特征
                
                **参数说明：**
                - **均值（Mean）**：预期收益率
                - **自由度（df）**：控制尾部厚度，越小尾部越厚（默认5.0）
                - **尺度（Scale）**：收益率的标准差
                
                **与正态分布的区别：**
                - t分布的尾部更厚，极端事件概率更高
                - 适合波动较大的市场环境
                """)
                
                # 可视化t分布 vs 正态分布
                try:
                    x = np.linspace(-0.1, 0.1, 200)
                    normal_y = (1 / (0.02 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / 0.02) ** 2)
                    t_y = (1 / (0.02 * np.sqrt(5 * np.pi))) * (1 + (x / 0.02) ** 2 / 5) ** (-3)
                    
                    fig_t = go.Figure()
                    fig_t.add_trace(go.Scatter(x=x, y=normal_y, name="正态分布", line=dict(color=COLORS["blue"])))
                    fig_t.add_trace(go.Scatter(x=x, y=t_y, name="Student-t分布 (df=5)", line=dict(color=COLORS["gold"])))
                    fig_t.update_layout(
                        title="正态分布 vs Student-t分布（厚尾对比）",
                        xaxis_title="收益率",
                        yaxis_title="概率密度",
                        template="plotly_dark",
                        height=300,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                    )
                    st.plotly_chart(fig_t, use_container_width=True)
                    st.caption("💡 t分布的尾部更厚，能更好地捕捉极端事件")
                except:
                    pass
            
            with model_tabs[2]:
                st.markdown("""
                ### 🔄 Bootstrap（经验分布）
                
                **特点：**
                - ✅ **完全基于历史数据**：不假设任何分布形式
                - ✅ **保留所有特征**：包括偏态、厚尾、相关性等
                - ✅ **最真实**：直接使用历史收益率进行重采样
                
                **适用场景：**
                - 有足够的历史数据（建议至少1年）
                - 希望完全基于历史经验预测
                - 市场特征复杂，难以用参数模型描述
                
                **工作原理：**
                1. 从历史收益率中随机抽取（有放回）
                2. 保持历史数据的完整特征
                3. 生成大量模拟路径
                
                **优势：**
                - 不需要假设分布形式
                - 自动保留历史数据的偏态和厚尾
                - 更贴近实际市场行为
                
                **局限性：**
                - ⚠️ 需要足够的历史数据
                - ⚠️ 假设未来会重复历史模式
                - ⚠️ 无法预测历史未出现的情况
                """)
                
                # 显示历史数据统计
                if uploaded_file_projection is not None or st.session_state.get("bootstrap_returns") is not None:
                    try:
                        if uploaded_file_projection is not None:
                            market_data = InvestSimBridge.load_market_data(uploaded_file_projection)
                            returns = market_data.pct_change().dropna()
                            bootstrap_returns = returns.values.flatten()
                            bootstrap_returns = bootstrap_returns[~np.isnan(bootstrap_returns)]
                        else:
                            bootstrap_returns = st.session_state.get("bootstrap_returns", np.array([]))
                        
                        if len(bootstrap_returns) > 0:
                            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                            with col_stat1:
                                st.metric("数据点数", f"{len(bootstrap_returns):,}")
                            with col_stat2:
                                st.metric("均值", f"{np.mean(bootstrap_returns):.4f}")
                            with col_stat3:
                                st.metric("标准差", f"{np.std(bootstrap_returns):.4f}")
                            with col_stat4:
                                st.metric("偏度", f"{float(pd.Series(bootstrap_returns).skew()):.2f}")
                            
                            # 显示历史收益率分布
                            fig_bootstrap = go.Figure()
                            fig_bootstrap.add_trace(go.Histogram(
                                x=bootstrap_returns,
                                name="历史收益率分布",
                                nbinsx=50,
                                marker_color=COLORS["green"]
                            ))
                            fig_bootstrap.update_layout(
                                title="历史收益率分布（Bootstrap将使用此分布）",
                                xaxis_title="收益率",
                                yaxis_title="频数",
                                template="plotly_dark",
                                height=300,
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                            )
                            st.plotly_chart(fig_bootstrap, use_container_width=True)
                    except:
                        st.info("💡 上传历史数据后，将显示数据统计信息")
        
        # 分布模型选择建议
        st.markdown("#### 💡 选择建议")
        if input_model_type == "Bootstrap":
            if uploaded_file_projection is None and st.session_state.get("bootstrap_returns") is None:
                st.warning("⚠️ **Bootstrap模式需要历史数据**。请上传CSV文件，或先在BACKTEST模式运行回测以获取历史收益率。")
            else:
                st.success("✅ **Bootstrap模式已就绪**：将使用历史收益率数据生成模拟路径，保留历史数据的完整特征。")
        elif input_model_type == "Normal":
            st.info("💡 **Normal模式**：使用正态分布假设，适合大多数情况。如果上传了历史数据，系统会自动拟合参数。")
        elif input_model_type == "Student-t":
            st.info("💡 **Student-t模式**：考虑厚尾分布，适合波动较大的市场。能更好地捕捉极端事件。")
        
        # 操作检查清单
        st.markdown("#### ✅ 配置检查清单")
        checklist_items = []
        checklist_status = []
        
        if strategy_name_global:
            checklist_items.append("✅ 策略已选择")
            checklist_status.append(True)
        else:
            checklist_items.append("❌ 请选择策略")
            checklist_status.append(False)
        
        if initial_capital > 0:
            checklist_items.append("✅ 初始资金已设置")
            checklist_status.append(True)
        else:
            checklist_items.append("❌ 请设置初始资金")
            checklist_status.append(False)
        
        if sim_years > 0:
            checklist_items.append("✅ 预测年限已设置")
            checklist_status.append(True)
        else:
            checklist_items.append("❌ 请设置预测年限")
            checklist_status.append(False)
        
        # 显示检查清单
        for item in checklist_items:
            st.markdown(f"- {item}")
        
        # 状态提示
        if all(checklist_status):
            st.success("🎉 **所有配置已完成，可以运行模拟！**")
        else:
            missing_count = len([x for x in checklist_status if not x])
            st.warning(f"⚠️ 还有 {missing_count} 项配置需要完成")
        
        run_mc = st.button("🚀 RUN SIMULATION", type="primary", use_container_width=True)
        
        # 按钮提示
        if not all(checklist_status):
            st.caption("💡 请先完成所有配置项后再运行模拟")

    if run_mc:
        st.session_state["user_has_run_projection"] = True
        st.session_state["show_welcome"] = False
        with st.spinner("CALCULATING PROBABILITY PATHS..."):
            dist_name_map = {"Normal": "normal", "Student-t": "student_t", "Bootstrap": "empirical_bootstrap"}
            dist_name = dist_name_map.get(input_model_type, "normal")
            
            dist_params = {}
            used_input_modeling = False
            
            # 优先使用输入建模的结果
            if has_input_modeling and input_modeling_info and input_model_type == input_model_choice_from_modeling:
                # 使用输入建模的结果
                dist_params = input_modeling_info["params"].copy()
                used_input_modeling = True
                st.success(f"✅ **使用输入建模结果**：{input_model_choice_from_modeling} 分布（基于历史标的物价格数据拟合）")
                st.info(f"💡 将使用此分布模型生成未来收益率，模拟标的物价格走向，然后评估策略表现。")
            elif dist_name == "normal":
                # 优先使用输入建模的Normal参数
                fitted_params = st.session_state.get("fitted_normal_params")
                if fitted_params:
                    dist_params = fitted_params.copy()
                    used_input_modeling = True
                    st.success("✅ **使用输入建模的Normal分布参数**（基于历史标的物价格数据拟合）")
                    st.info("💡 将使用此分布模型生成未来收益率，模拟标的物价格走向，然后评估策略表现。")
                elif uploaded_file_projection is not None:
                    # 如果上传了数据，尝试从数据中拟合参数
                    try:
                        market_data = InvestSimBridge.load_market_data(uploaded_file_projection)
                        returns = market_data.pct_change().dropna()
                        mean_return = returns.mean().mean()
                        vol_return = returns.std().mean()
                        dist_params = {"mean": mean_return, "vol": vol_return}
                        st.session_state["fitted_normal_params"] = dist_params
                        st.info("✅ 从上传的数据中自动拟合Normal分布参数")
                    except:
                        dist_params = {"mean": 0.0005, "vol": 0.02}
                        st.warning("⚠️ 使用默认Normal分布参数")
                else:
                    dist_params = {"mean": 0.0005, "vol": 0.02}
                    st.warning("⚠️ 使用默认Normal分布参数（建议先进行输入建模）")
            elif dist_name == "student_t":
                # 优先使用输入建模的Student-t参数
                fitted_params = st.session_state.get("fitted_student_t_params")
                if fitted_params:
                    dist_params = fitted_params.copy()
                    used_input_modeling = True
                    st.success("✅ **使用输入建模的Student-t分布参数**（基于历史标的物价格数据拟合）")
                    st.info("💡 将使用此分布模型生成未来收益率，模拟标的物价格走向，然后评估策略表现。")
                else:
                    dist_params = {"mean": 0.0, "df": 5.0, "scale": 0.02}
                    st.warning("⚠️ 使用默认Student-t分布参数（建议先进行输入建模）")
            elif dist_name == "empirical_bootstrap":
                # 优先使用输入建模的Bootstrap数据
                bootstrap_returns = st.session_state.get("bootstrap_returns")
                if bootstrap_returns is not None and len(bootstrap_returns) > 0:
                    if isinstance(bootstrap_returns, np.ndarray):
                        dist_params = {"historical_returns": bootstrap_returns.tolist()}
                    else:
                        dist_params = {"historical_returns": bootstrap_returns}
                    used_input_modeling = True
                    st.success(f"✅ **使用输入建模的Bootstrap数据**（{len(bootstrap_returns):,} 个历史收益率样本，来自标的物价格数据）")
                    st.info("💡 将使用此历史收益率分布生成未来收益率，模拟标的物价格走向，然后评估策略表现。")
                elif uploaded_file_projection is not None:
                    # 如果上传了数据，尝试从数据中提取
                    try:
                        market_data = InvestSimBridge.load_market_data(uploaded_file_projection)
                        returns = market_data.pct_change().dropna()
                        bootstrap_returns = returns.values.flatten()
                        bootstrap_returns = bootstrap_returns[~np.isnan(bootstrap_returns)]
                        if len(bootstrap_returns) > 0:
                            dist_params = {"historical_returns": bootstrap_returns.tolist()}
                            st.info(f"✅ 从上传的数据中提取Bootstrap样本（{len(bootstrap_returns):,} 个）")
                        else:
                            raise ValueError("No valid returns found")
                    except Exception as e:
                        st.warning("⚠️ Bootstrap需要历史数据。使用Normal分布代替。")
                        dist_name = "normal"
                        dist_params = {"mean": 0.0005, "vol": 0.02}
                else:
                    st.warning("⚠️ Bootstrap需要历史数据。使用Normal分布代替。")
                    dist_name = "normal"
                    dist_params = {"mean": 0.0005, "vol": 0.02}
            
            # 优先使用回测中选择的策略
            backtest_strategy = st.session_state.get("backtest_strategy", None)
            backtest_strategy_params = st.session_state.get("backtest_strategy_params", {})
            backtest_params = st.session_state.get("backtest_params", {})
            
            # 如果回测中有策略，优先使用回测的策略和参数
            final_strategy = backtest_strategy if backtest_strategy else strategy_name_global
            final_strategy_params = backtest_strategy_params if backtest_strategy_params else strategy_params
            final_leverage = backtest_params.get("leverage", leverage) if backtest_params else leverage
            final_capital = backtest_params.get("capital", initial_capital) if backtest_params else initial_capital
            
            if backtest_strategy and backtest_strategy != strategy_name_global:
                st.info(f"💡 **使用回测中选择的策略**：{backtest_strategy}（当前选择的是 {strategy_name_global}，已自动切换为回测策略）")
            
            input_model_config = {"dist_name": dist_name, "params": dist_params}
            params = {
                "strategy": final_strategy,
                "leverage": final_leverage,
                "capital": final_capital,
                "duration": sim_years,
                "num_trials": num_trials,
                "annual_contribution": annual_cont,
                "input_model": input_model_config,
                **final_strategy_params
            }
            mc_res = InvestSimBridge.run_forward_simulation(params)
            st.session_state['mc_result'] = mc_res

    if 'mc_result' in st.session_state:
        # 成功提示
        st.success("✅ **模拟完成！** 已基于输入建模的分布模型生成未来价格路径，并评估了策略表现。下方显示详细预测结果和决策建议。")
        
        res = st.session_state['mc_result']
        
        # 显示输入模型信息
        if res.get("input_model"):
            input_model = res["input_model"]
            dist_name = input_model.get("dist_name", "normal")
            params = input_model.get("params", {})
            
            st.markdown("---")
            st.markdown("### 📊 输入建模信息")
            
            model_info_col1, model_info_col2 = st.columns([2, 1])
            with model_info_col1:
                dist_name_display = {
                    "normal": "Normal（正态分布）",
                    "student_t": "Student-t（t分布）",
                    "empirical_bootstrap": "Bootstrap（经验分布）"
                }.get(dist_name, dist_name)
                
                st.markdown(f"**使用的分布模型：** {dist_name_display}")
                
                if dist_name == "normal":
                    mean_val = params.get("mean", 0)
                    vol_val = params.get("vol", 0)
                    st.markdown(f"""
                    **参数：**
                    - 均值（Mean）：{mean_val:.6f} ({mean_val*252:.2%} 年化)
                    - 波动率（Volatility）：{vol_val:.6f} ({vol_val*np.sqrt(252):.2%} 年化)
                    """)
                elif dist_name == "student_t":
                    df_val = params.get("df", 5.0)
                    mean_val = params.get("mean", 0)
                    scale_val = params.get("scale", 0.02)
                    st.markdown(f"""
                    **参数：**
                    - 均值（Mean）：{mean_val:.6f} ({mean_val*252:.2%} 年化)
                    - 自由度（df）：{df_val:.2f}（控制尾部厚度）
                    - 尺度（Scale）：{scale_val:.6f} ({scale_val*np.sqrt(252):.2%} 年化)
                    """)
                elif dist_name == "empirical_bootstrap":
                    hist_returns = params.get("historical_returns", [])
                    if len(hist_returns) > 0:
                        hist_arr = np.array(hist_returns)
                        st.markdown(f"""
                        **参数：**
                        - 历史数据点数：{len(hist_returns):,}
                        - 历史均值：{np.mean(hist_arr):.6f} ({np.mean(hist_arr)*252:.2%} 年化)
                        - 历史标准差：{np.std(hist_arr):.6f} ({np.std(hist_arr)*np.sqrt(252):.2%} 年化)
                        - 历史偏度：{float(pd.Series(hist_arr).skew()):.2f}
                        """)
            
            with model_info_col2:
                st.markdown("**模型特点：**")
                if dist_name == "normal":
                    st.info("✅ 简单易用\n✅ 计算快速\n⚠️ 可能低估极端事件")
                elif dist_name == "student_t":
                    st.info("✅ 考虑厚尾\n✅ 更真实\n⚠️ 需要更多参数")
                elif dist_name == "empirical_bootstrap":
                    st.info("✅ 完全基于历史\n✅ 保留所有特征\n⚠️ 需要足够数据")
            
            st.caption(f"💡 **模拟说明**：当前使用 {dist_name_display} 来生成未来收益率，模拟标的物价格走向。这些参数来自输入建模对历史数据的分析。")
            
            # 显示模拟流程说明
            st.markdown("---")
            st.markdown("#### 🔄 模拟流程说明")
            st.markdown("""
            **本次模拟的工作流程：**
            
            1. 📊 **输入建模阶段**：基于历史标的物价格数据，分析收益率分布特征
               - 从历史价格数据中提取收益率序列
               - 拟合分布模型（当前使用：{dist_name_display}）
               - 保存分布参数
            
            2. 🚀 **价格模拟阶段**：使用输入建模的分布模型生成未来收益率
               - 每个模拟周期，从选定的分布中随机抽取收益率
               - 根据收益率更新标的物价格：`新价格 = 旧价格 × (1 + 收益率)`
               - 重复此过程，生成 {num_trials:,} 条未来价格路径
            
            3. 📈 **策略评估阶段**：基于模拟的价格路径，评估策略表现
               - 对每条价格路径，应用选定的投资策略
               - 计算策略在不同价格路径下的表现
               - 汇总所有路径的结果，得到策略的预期表现和风险指标
            
            **关键点：**
            - ✅ 模拟的未来价格走向基于历史数据的分布特征
            - ✅ 不同的输入模型会产生不同的价格路径
            - ✅ 策略表现评估基于这些模拟路径，反映策略在不同市场情景下的表现
            """.format(dist_name_display=dist_name_display, num_trials=num_trials))
        
        res = st.session_state['mc_result']
        final_values = res['paths'][-1]
        median_val = np.median(final_values)
        p05_val = np.percentile(final_values, 5)
        p95_val = np.percentile(final_values, 95)
        breakeven_balance = initial_capital + annual_cont * sim_years
        gain = (median_val / breakeven_balance) - 1
        
        # 计算更多统计指标
        mean_val = np.mean(final_values)
        std_val = np.std(final_values)
        success_prob = np.mean(final_values > breakeven_balance)
        loss_prob = np.mean(final_values < initial_capital)
        
        # 原始指标显示
        st.markdown("### 📊 预测结果概览")
        c1, c2, c3 = st.columns(3)
        with c1: 
            st.metric("Expected Outcome", f"${median_val:,.0f}", f"{gain:+.1%} vs Invested",
                     help="中位数预测结果，表示50%的概率会达到或超过此值")
        with c2: 
            st.metric("Worst Case (95% VaR)", f"${p05_val:,.0f}", delta_color="inverse",
                     help="95%置信度下的最坏情况，只有5%的概率会低于此值")
        with c3: 
            st.metric("Success Prob", f"{success_prob:.1%}",
                     help="最终价值超过投入资金（含年度贡献）的概率")

        st.markdown("---")
        
        # 多标签页图表展示
        chart_tabs = st.tabs(["📈 Path Simulation", "📊 Distribution Analysis", "📉 Probability Analysis", "📊 Scenario Analysis", "📈 Risk Metrics", "💾 Export"])
        
        with chart_tabs[0]:  # Path Simulation
            # 详细说明
            with st.expander("📖 什么是路径模拟（Path Simulation）？", expanded=False):
                st.markdown("""
                **路径模拟** 展示投资组合价值在未来时间内的可能变化路径。
                
                **这个图表展示什么？**
                - 📈 **扇形图**：显示所有模拟路径的置信区间
                - 🟡 **深色区域（50%置信区间）**：50%的模拟路径落在此范围内
                - 🟨 **浅色区域（90%置信区间）**：90%的模拟路径落在此范围内
                - 📊 **中位数路径**：所有模拟路径的中位数，代表最可能的结果
                
                **如何解读？**
                - **扇形越宽**：不确定性越大，预测结果分散
                - **扇形越窄**：不确定性越小，预测结果集中
                - **向上倾斜**：预期价值增长
                - **向下倾斜**：预期价值下降
                
                **关键观察点：**
                - ✅ **最终价值范围**：查看最终可能的价值区间
                - ✅ **增长趋势**：是否持续向上
                - ✅ **不确定性**：扇形宽度反映风险
                - ✅ **中位数路径**：最可能的结果
                
                **实际应用：**
                - 评估未来收益潜力
                - 识别可能的风险范围
                - 制定投资计划
                """)
            
            st.caption("💡 **Path Simulation**: Shows projected wealth paths with confidence intervals. Wider fan = more uncertainty.")
        st.plotly_chart(plot_monte_carlo_fan(res['dates'], res['paths'], res['median']), use_container_width=True)
        st.caption(describe_input_model(res.get("input_model")))
        
        with chart_tabs[1]:  # Distribution Analysis
            # 详细说明
            with st.expander("📖 什么是分布分析（Distribution Analysis）？", expanded=False):
                st.markdown("""
                **分布分析** 展示最终投资组合价值的概率分布，帮助理解不同结果的概率。
                
                **这个图表展示什么？**
                - 📊 **直方图**：显示最终价值的分布情况
                - 📈 **正态分布拟合**：理论上的正态分布曲线
                - 📍 **关键分位数**：5%、25%、50%、75%、95%分位数
                
                **如何解读？**
                - **分布形状**：
                  - 左偏：更多路径结果较低
                  - 右偏：更多路径结果较高
                  - 对称：结果分布均匀
                - **峰值位置**：最可能的结果
                - **分布宽度**：不确定性大小
                
                **关键指标：**
                - **均值**：所有模拟结果的平均值
                - **中位数**：50%分位数，最可能的结果
                - **标准差**：结果分散程度
                - **偏度**：分布的不对称程度
                - **峰度**：分布的尖锐程度
                
                **实际应用：**
                - 理解结果的不确定性
                - 评估不同结果的概率
                - 识别异常值风险
                """)
            
            st.caption("💡 **Distribution Analysis**: Histogram of final portfolio values showing probability distribution.")
            
            # 最终价值分布直方图
            fig_dist = go.Figure()
            
            # 直方图
            fig_dist.add_trace(go.Histogram(
                x=final_values,
                nbinsx=50,
                name='Final Value Distribution',
                marker_color=COLORS['gold'],
                opacity=0.7
            ))
            
            # 添加关键分位数线
            fig_dist.add_vline(x=median_val, line_dash="dash", line_color=COLORS['green'], 
                              annotation_text=f"Median: ${median_val:,.0f}")
            fig_dist.add_vline(x=p05_val, line_dash="dash", line_color=COLORS['red'], 
                              annotation_text=f"5%: ${p05_val:,.0f}")
            fig_dist.add_vline(x=p95_val, line_dash="dash", line_color=COLORS['blue'], 
                              annotation_text=f"95%: ${p95_val:,.0f}")
            fig_dist.add_vline(x=breakeven_balance, line_dash="dot", line_color=COLORS['text_sub'], 
                              annotation_text=f"Breakeven: ${breakeven_balance:,.0f}")
            
            # 正态分布拟合
            try:
                from scipy import stats  # pyright: ignore[reportMissingImports]
                mu, sigma = stats.norm.fit(final_values)
                x_norm = np.linspace(final_values.min(), final_values.max(), 100)
                y_norm = stats.norm.pdf(x_norm, mu, sigma) * len(final_values) * (final_values.max() - final_values.min()) / 50
                fig_dist.add_trace(go.Scatter(
                    x=x_norm,
                    y=y_norm,
                    mode='lines',
                    name='Normal Fit',
                    line=dict(color=COLORS['text_sub'], width=2, dash='dash')
                ))
            except:
                pass
            
            fig_dist.update_layout(**get_chart_layout(400))
            fig_dist.update_layout(
                title="Final Portfolio Value Distribution",
                xaxis=dict(title="Final Value ($)"),
                yaxis=dict(title="Frequency")
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # 分布统计
            col_dist1, col_dist2, col_dist3, col_dist4 = st.columns(4)
            with col_dist1:
                st.markdown("**基本统计**")
                st.metric("Mean", f"${mean_val:,.0f}")
                st.metric("Median", f"${median_val:,.0f}")
                st.metric("Std Dev", f"${std_val:,.0f}")
            
            with col_dist2:
                st.markdown("**分位数**")
                p25_val = np.percentile(final_values, 25)
                p75_val = np.percentile(final_values, 75)
                st.metric("25%", f"${p25_val:,.0f}")
                st.metric("75%", f"${p75_val:,.0f}")
                st.metric("95%", f"${p95_val:,.0f}")
            
            with col_dist3:
                st.markdown("**分布特征**")
                try:
                    from scipy import stats  # pyright: ignore[reportMissingImports]
                    skewness = stats.skew(final_values)
                    kurtosis = stats.kurtosis(final_values)
                    st.metric("Skewness", f"{skewness:.2f}")
                    st.metric("Kurtosis", f"{kurtosis:.2f}")
                    cv = std_val / mean_val if mean_val > 0 else 0
                    st.metric("CV", f"{cv:.2f}")
                except ImportError:
                    # scipy not available, use basic calculations
                    cv = std_val / mean_val if mean_val > 0 else 0
                    st.metric("CV", f"{cv:.2f}")
                    st.caption("安装scipy以查看更多统计")
                except Exception:
                    st.caption("统计计算中...")
            
            with col_dist4:
                st.markdown("**概率指标**")
                st.metric("Success Prob", f"{success_prob:.1%}")
                st.metric("Loss Prob", f"{loss_prob:.1%}")
                prob_2x = np.mean(final_values > initial_capital * 2)
                st.metric("2x Prob", f"{prob_2x:.1%}")
        
        with chart_tabs[2]:  # Probability Analysis
            # 详细说明
            with st.expander("📖 什么是概率分析（Probability Analysis）？", expanded=False):
                st.markdown("""
                **概率分析** 展示达到不同目标价值的概率，帮助制定投资目标。
                
                **这个图表展示什么？**
                - 📊 **累积分布函数（CDF）**：显示达到或超过某个价值的概率
                - 📈 **概率密度函数（PDF）**：显示不同价值的概率密度
                - 🎯 **目标概率**：达到特定目标的概率
                
                **如何解读？**
                - **CDF曲线**：
                  - 上升越快：结果越集中
                  - 上升越慢：结果越分散
                  - 50%对应的值：中位数
                - **目标概率**：
                  - 高概率：目标容易达成
                  - 低概率：目标难以达成
                
                **关键应用：**
                - 设定合理的目标
                - 评估目标达成概率
                - 制定风险应对策略
                """)
            
            st.caption("💡 **Probability Analysis**: Shows probability of achieving different target values.")
            
            # 累积分布函数
            sorted_values = np.sort(final_values)
            probabilities = np.arange(1, len(sorted_values) + 1) / len(sorted_values) * 100
            
            fig_cdf = go.Figure()
            fig_cdf.add_trace(go.Scatter(
                x=sorted_values,
                y=probabilities,
                mode='lines',
                name='CDF',
                line=dict(color=COLORS['gold'], width=2),
                fill='tozeroy',
                fillcolor='rgba(210, 153, 34, 0.1)'
            ))
            
            # 添加关键目标线
            targets = [
                (initial_capital, "Initial Capital"),
                (breakeven_balance, "Breakeven"),
                (initial_capital * 1.5, "1.5x Target"),
                (initial_capital * 2, "2x Target")
            ]
            
            for target_val, target_name in targets:
                prob_at_target = np.mean(final_values >= target_val) * 100
                fig_cdf.add_vline(x=target_val, line_dash="dash", 
                                 line_color=COLORS['text_sub'], opacity=0.5)
                fig_cdf.add_annotation(
                    x=target_val,
                    y=prob_at_target,
                    text=f"{target_name}<br>{prob_at_target:.1f}%",
                    showarrow=True,
                    arrowhead=2
                )
            
            fig_cdf.update_layout(**get_chart_layout(400))
            fig_cdf.update_layout(
                title="Cumulative Distribution Function (CDF)",
                xaxis=dict(title="Final Value ($)"),
                yaxis=dict(title="Probability (%)")
            )
            st.plotly_chart(fig_cdf, use_container_width=True)
            
            # 目标概率分析
            st.markdown("#### 🎯 目标达成概率")
            target_col1, target_col2, target_col3, target_col4 = st.columns(4)
            
            targets_analysis = [
                (initial_capital * 0.9, "90% of Initial", "保本90%"),
                (initial_capital, "Break Even", "保本"),
                (breakeven_balance, "Breakeven", "盈亏平衡"),
                (initial_capital * 1.5, "1.5x Initial", "1.5倍"),
                (initial_capital * 2, "2x Initial", "2倍"),
                (initial_capital * 3, "3x Initial", "3倍")
            ]
            
            for i, (target_val, target_name_en, target_name_cn) in enumerate(targets_analysis):
                prob = np.mean(final_values >= target_val) * 100
                col = [target_col1, target_col2, target_col3, target_col4][i % 4]
                with col:
                    st.metric(target_name_cn, f"{prob:.1f}%")
                    st.caption(f"${target_val:,.0f}")
        
        with chart_tabs[3]:  # Scenario Analysis
            # 详细说明
            with st.expander("📖 什么是情景分析（Scenario Analysis）？", expanded=False):
                st.markdown("""
                **情景分析** 展示不同概率情景下的投资路径，帮助理解各种可能的结果。
                
                **这个图表展示什么？**
                - 📈 **不同分位数的路径**：5%、25%、50%、75%、95%分位数路径
                - 📊 **情景对比**：对比乐观、中性、悲观情景
                - 🎯 **关键时间点**：不同时间点的价值分布
                
                **如何解读？**
                - **5%路径**：悲观情景，只有5%的概率会更差
                - **25%路径**：较悲观情景
                - **50%路径**：中性情景，最可能的结果
                - **75%路径**：较乐观情景
                - **95%路径**：乐观情景，只有5%的概率会更好
                
                **实际应用：**
                - 制定不同情景下的应对策略
                - 评估极端情况的影响
                - 设定风险预警线
                """)
            
            st.caption("💡 **Scenario Analysis**: Shows different percentile paths to understand various possible outcomes.")
            
            # 计算不同分位数的路径
            p05_path = np.percentile(res['paths'], 5, axis=1)
            p25_path = np.percentile(res['paths'], 25, axis=1)
            p75_path = np.percentile(res['paths'], 75, axis=1)
            p95_path = np.percentile(res['paths'], 95, axis=1)
            
            fig_scenario = go.Figure()
            
            # 添加分位数路径
            fig_scenario.add_trace(go.Scatter(
                x=res['dates'],
                y=p05_path,
                mode='lines',
                name='5% (Pessimistic)',
                line=dict(color=COLORS['red'], width=2, dash='dash')
            ))
            fig_scenario.add_trace(go.Scatter(
                x=res['dates'],
                y=p25_path,
                mode='lines',
                name='25%',
                line=dict(color=COLORS['text_sub'], width=1.5)
            ))
            fig_scenario.add_trace(go.Scatter(
                x=res['dates'],
                y=res['median'],
                mode='lines',
                name='50% (Median)',
                line=dict(color=COLORS['gold'], width=3)
            ))
            fig_scenario.add_trace(go.Scatter(
                x=res['dates'],
                y=p75_path,
                mode='lines',
                name='75%',
                line=dict(color=COLORS['text_sub'], width=1.5)
            ))
            fig_scenario.add_trace(go.Scatter(
                x=res['dates'],
                y=p95_path,
                mode='lines',
                name='95% (Optimistic)',
                line=dict(color=COLORS['green'], width=2, dash='dash')
            ))
            
            # 添加投入资金线
            contribution_path = np.array([initial_capital + annual_cont * (i / len(res['dates'])) for i in range(len(res['dates']))])
            fig_scenario.add_trace(go.Scatter(
                x=res['dates'],
                y=contribution_path,
                mode='lines',
                name='Total Invested',
                line=dict(color=COLORS['text_sub'], width=1, dash='dot')
            ))
            
            fig_scenario.update_layout(**get_chart_layout(400))
            fig_scenario.update_layout(
                title="Scenario Analysis - Percentile Paths",
                xaxis=dict(title="Time"),
                yaxis=dict(title="Portfolio Value ($)")
            )
            st.plotly_chart(fig_scenario, use_container_width=True)
            
            # 情景对比表
            st.markdown("#### 📊 情景对比")
            scenario_data = {
                "Scenario": ["Pessimistic (5%)", "Lower Quartile (25%)", "Median (50%)", "Upper Quartile (75%)", "Optimistic (95%)"],
                "Final Value": [f"${p05_val:,.0f}", f"${p25_val:,.0f}", f"${median_val:,.0f}", f"${p75_val:,.0f}", f"${p95_val:,.0f}"],
                "vs Invested": [
                    f"{((p05_val - breakeven_balance) / breakeven_balance * 100):+.1f}%",
                    f"{((p25_val - breakeven_balance) / breakeven_balance * 100):+.1f}%",
                    f"{((median_val - breakeven_balance) / breakeven_balance * 100):+.1f}%",
                    f"{((p75_val - breakeven_balance) / breakeven_balance * 100):+.1f}%",
                    f"{((p95_val - breakeven_balance) / breakeven_balance * 100):+.1f}%"
                ]
            }
            scenario_df = pd.DataFrame(scenario_data)
            st.dataframe(scenario_df, use_container_width=True, hide_index=True)
        
        with chart_tabs[4]:  # Risk Metrics
            # 详细说明
            with st.expander("📖 什么是风险指标（Risk Metrics）？", expanded=False):
                st.markdown("""
                **风险指标** 量化投资组合的风险水平，帮助评估和管理风险。
                
                **关键风险指标：**
                - **VaR (Value at Risk)**：在给定置信度下，预期最大损失
                  - VaR (95%)：95%置信度下的最大损失
                  - VaR (99%)：99%置信度下的最大损失
                - **CVaR (Conditional VaR)**：超过VaR时的平均损失
                  - 也称为Expected Shortfall
                - **最大潜在损失**：最坏情况下的损失
                - **下行标准差**：只考虑负收益的标准差
                
                **如何解读？**
                - **VaR越小**：风险越低
                - **CVaR越小**：极端损失越小
                - **下行标准差越小**：下跌风险越小
                
                **实际应用：**
                - 设定风险限额
                - 评估极端情况
                - 制定风险应对策略
                """)
            
            st.caption("💡 **Risk Metrics**: Quantifies portfolio risk levels using VaR, CVaR, and other risk measures.")
            
            # 风险指标计算
            returns_sim = (final_values - breakeven_balance) / breakeven_balance
            
            # VaR计算
            var_95 = np.percentile(returns_sim, 5)
            var_99 = np.percentile(returns_sim, 1)
            var_95_val = initial_capital * (1 + var_95)
            var_99_val = initial_capital * (1 + var_99)
            
            # CVaR计算
            cvar_95 = returns_sim[returns_sim <= np.percentile(returns_sim, 5)].mean()
            cvar_99 = returns_sim[returns_sim <= np.percentile(returns_sim, 1)].mean()
            cvar_95_val = initial_capital * (1 + cvar_95)
            cvar_99_val = initial_capital * (1 + cvar_99)
            
            # 下行标准差
            negative_returns = returns_sim[returns_sim < 0]
            downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 0
            
            # 风险指标展示
            risk_col1, risk_col2 = st.columns(2)
            
            with risk_col1:
                st.markdown("#### 📉 Value at Risk (VaR)")
                st.metric("VaR (95%)", f"{var_95:.2%}", f"${var_95_val:,.0f}", delta_color="inverse",
                         help="95%置信度下的最大损失")
                st.metric("VaR (99%)", f"{var_99:.2%}", f"${var_99_val:,.0f}", delta_color="inverse",
                         help="99%置信度下的最大损失")
            
            with risk_col2:
                st.markdown("#### 📊 Conditional VaR (CVaR)")
                st.metric("CVaR (95%)", f"{cvar_95:.2%}", f"${cvar_95_val:,.0f}", delta_color="inverse",
                         help="超过VaR(95%)时的平均损失")
                st.metric("CVaR (99%)", f"{cvar_99:.2%}", f"${cvar_99_val:,.0f}", delta_color="inverse",
                         help="超过VaR(99%)时的平均损失")
            
            # 风险分布图
            fig_risk = go.Figure()
            fig_risk.add_trace(go.Histogram(
                x=returns_sim * 100,
                nbinsx=50,
                name='Return Distribution',
                marker_color=COLORS['red'],
                opacity=0.7
            ))
            
            # 添加VaR线
            fig_risk.add_vline(x=var_95 * 100, line_dash="dash", line_color=COLORS['red'],
                              annotation_text=f"VaR(95%): {var_95:.2%}")
            fig_risk.add_vline(x=var_99 * 100, line_dash="dash", line_color=COLORS['red'],
                              annotation_text=f"VaR(99%): {var_99:.2%}")
            fig_risk.add_vline(x=0, line_dash="dot", line_color=COLORS['text_sub'])
            
            fig_risk.update_layout(**get_chart_layout(300))
            fig_risk.update_layout(
                title="Return Distribution with VaR",
                xaxis=dict(title="Return (%)"),
                yaxis=dict(title="Frequency")
            )
            st.plotly_chart(fig_risk, use_container_width=True)
            
            # 其他风险指标
            st.markdown("#### 📊 其他风险指标")
            other_risk_col1, other_risk_col2, other_risk_col3, other_risk_col4 = st.columns(4)
            
            with other_risk_col1:
                st.metric("Max Loss", f"{returns_sim.min():.2%}")
                st.caption("最大潜在损失")
            
            with other_risk_col2:
                st.metric("Downside Std", f"{downside_std:.2%}")
                st.caption("下行标准差")
            
            with other_risk_col3:
                st.metric("Loss Prob", f"{loss_prob:.1%}")
                st.caption("亏损概率")
            
            with other_risk_col4:
                tail_risk = np.mean(returns_sim < -0.2)  # 损失超过20%的概率
                st.metric("Tail Risk", f"{tail_risk:.1%}")
                st.caption("极端损失概率")
        
        with chart_tabs[5]:  # Export
            # 详细说明
            with st.expander("📖 报告导出说明", expanded=False):
                st.markdown("""
                **完整流程报告** 包含您整个工作流程的详细信息：
                
                **报告内容包括：**
                - **回测阶段**：历史数据信息、策略选择、回测结果、输入建模结果
                - **预测阶段**：使用的策略和Input Model、预测参数配置、模拟结果
                - **完整流程说明**：从数据上传到结果评估的完整过程
                
                **报告格式：**
                - Markdown格式，易于阅读和分享
                - 包含所有关键信息和指标
                - 适合保存、打印或分享给团队成员
                """)
            
            st.caption("💡 **完整流程报告**：下载包含回测和预测完整流程的详细报告。")
            
            # 生成完整流程报告
            if 'mc_result' in st.session_state:
                try:
                    # 收集回测阶段信息
                    backtest_info = {}
                    if 'bt_result' in st.session_state:
                        bt_res = st.session_state['bt_result']
                        backtest_info['has_backtest'] = True
                        backtest_info['metrics'] = bt_res.metrics if hasattr(bt_res, 'metrics') else {}
                    else:
                        backtest_info['has_backtest'] = False
                    
                    backtest_strategy = st.session_state.get("backtest_strategy", None)
                    backtest_params = st.session_state.get("backtest_params", {})
                    backtest_strategy_params = st.session_state.get("backtest_strategy_params", {})
                    
                    # 收集输入建模信息
                    input_model_choice = st.session_state.get("input_model_choice", None)
                    input_model_info = {}
                    if input_model_choice == "Normal":
                        fitted_params = st.session_state.get("fitted_normal_params")
                        if fitted_params:
                            input_model_info = {
                                "type": "Normal",
                                "params": fitted_params
                            }
                    elif input_model_choice == "Student-t":
                        fitted_params = st.session_state.get("fitted_student_t_params")
                        if fitted_params:
                            input_model_info = {
                                "type": "Student-t",
                                "params": fitted_params
                            }
                    elif input_model_choice == "Bootstrap":
                        bootstrap_returns = st.session_state.get("bootstrap_returns")
                        if bootstrap_returns is not None:
                            input_model_info = {
                                "type": "Bootstrap",
                                "params": {"historical_returns_count": len(bootstrap_returns) if hasattr(bootstrap_returns, '__len__') else 0}
                            }
                    
                    # 收集预测阶段信息
                    res = st.session_state['mc_result']
                    final_values = res['paths'][-1] if len(res['paths'].shape) > 1 else res['paths']
                    median_val = np.median(final_values) if isinstance(final_values, np.ndarray) else final_values
                    mean_val = np.mean(final_values) if isinstance(final_values, np.ndarray) else final_values
                    std_val = np.std(final_values) if isinstance(final_values, np.ndarray) else 0
                    p05_val = np.percentile(final_values, 5) if isinstance(final_values, np.ndarray) else final_values
                    p95_val = np.percentile(final_values, 95) if isinstance(final_values, np.ndarray) else final_values
                    breakeven_balance = initial_capital + (annual_cont * sim_years)
                    success_prob = np.mean(final_values > breakeven_balance) if isinstance(final_values, np.ndarray) else 0.5
                    loss_prob = np.mean(final_values < initial_capital) if isinstance(final_values, np.ndarray) else 0.5
                    gain = (median_val / breakeven_balance) - 1 if breakeven_balance > 0 else 0
                    
                    # 获取输入模型信息
                    input_model = res.get("input_model", {})
                    dist_name = input_model.get("dist_name", "normal")
                    dist_name_display = {
                        "normal": "Normal（正态分布）",
                        "student_t": "Student-t（t分布）",
                        "empirical_bootstrap": "Bootstrap（经验分布）"
                    }.get(dist_name, dist_name)
                    
                    # 生成报告
                    report = f"""# 投资策略模拟完整流程报告

**生成时间**：{datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}

---

## 📊 第一部分：回测阶段（BACKTEST）

### 1.1 历史数据上传

{"✅ 已上传历史标的物价格数据" if backtest_info['has_backtest'] else "⚠️ 未检测到回测数据"}

### 1.2 策略选择

{"✅ **选择的策略**：" + backtest_strategy if backtest_strategy else "⚠️ 未检测到策略选择"}

**策略参数**：
"""
                    
                    if backtest_strategy_params:
                        for key, value in backtest_strategy_params.items():
                            report += f"- {key}: {value}\n"
                    else:
                        report += "- 无特殊策略参数\n"
                    
                    report += f"""
**回测配置参数**：
- 初始资金：${backtest_params.get('capital', initial_capital):,.0f}
- 杠杆倍数：{backtest_params.get('leverage', leverage):.2f}
- 无风险利率：{backtest_params.get('risk_free', 0):.2%}
- 再平衡频率：{backtest_params.get('rebalance_frequency', 1)} 期

### 1.3 回测结果

"""
                    
                    if backtest_info['has_backtest'] and backtest_info['metrics']:
                        metrics = backtest_info['metrics']
                        report += f"""
**回测表现指标**：
- 总收益率：{metrics.get('total_return', 0):.2%}
- 年化收益率：{metrics.get('annualized_return', 0):.2%}
- Sharpe比率：{metrics.get('sharpe', 0):.2f}
- 最大回撤：{metrics.get('max_drawdown', 0):.2%}
- 波动率：{metrics.get('volatility', 0):.2%}
- VaR (95%)：{metrics.get('var_95', 0):.2%}
- CVaR (95%)：{metrics.get('cvar_95', 0):.2%}

"""
                    else:
                        report += "⚠️ 未检测到回测结果数据\n\n"
                    
                    report += f"""
### 1.4 输入建模（Input Modeling）

**目的**：从历史标的物价格数据中提取收益率分布特征，得到Input Model用于未来价格预测

**建模结果**：
"""
                    
                    if input_model_info:
                        report += f"""
✅ **已完成的输入建模**

**选择的分布模型**：{input_model_info['type']}

**分布参数**：
"""
                        if input_model_info['type'] == "Normal":
                            params = input_model_info['params']
                            report += f"""
- 均值（Mean）：{params.get('mean', 0):.6f} ({params.get('mean', 0)*252:.2%} 年化)
- 波动率（Volatility）：{params.get('vol', 0):.6f} ({params.get('vol', 0)*np.sqrt(252):.2%} 年化)
"""
                        elif input_model_info['type'] == "Student-t":
                            params = input_model_info['params']
                            report += f"""
- 均值（Mean）：{params.get('mean', 0):.6f} ({params.get('mean', 0)*252:.2%} 年化)
- 自由度（df）：{params.get('df', 5.0):.2f}
- 尺度（Scale）：{params.get('scale', 0.02):.6f} ({params.get('scale', 0.02)*np.sqrt(252):.2%} 年化)
"""
                        elif input_model_info['type'] == "Bootstrap":
                            params = input_model_info['params']
                            report += f"""
- 历史数据点数：{params.get('historical_returns_count', 0):,}
- 使用历史收益率的完整分布特征
"""
                    else:
                        report += "⚠️ 未检测到输入建模结果\n"
                    
                    report += f"""

**输入建模说明**：
系统从历史标的物价格数据中提取收益率序列，拟合了多种分布模型（Normal、Student-t、Bootstrap等），
通过拟合优度指标（KS检验、AIC、BIC等）选择了最适合的分布模型。
这个Input Model将用于预测阶段模拟未来价格走向。

---

## 🔮 第二部分：预测阶段（PROJECTION）

### 2.1 使用的策略和Input Model

**策略**：{backtest_strategy if backtest_strategy else strategy_name_global}（{"来自回测" if backtest_strategy else "当前选择"}）

**Input Model**：{dist_name_display}

**模型参数**：
"""
                    
                    if dist_name == "normal":
                        params = input_model.get("params", {})
                        report += f"""
- 均值（Mean）：{params.get('mean', 0):.6f} ({params.get('mean', 0)*252:.2%} 年化)
- 波动率（Volatility）：{params.get('vol', 0):.6f} ({params.get('vol', 0)*np.sqrt(252):.2%} 年化)
"""
                    elif dist_name == "student_t":
                        params = input_model.get("params", {})
                        report += f"""
- 均值（Mean）：{params.get('mean', 0):.6f} ({params.get('mean', 0)*252:.2%} 年化)
- 自由度（df）：{params.get('df', 5.0):.2f}
- 尺度（Scale）：{params.get('scale', 0.02):.6f} ({params.get('scale', 0.02)*np.sqrt(252):.2%} 年化)
"""
                    elif dist_name == "empirical_bootstrap":
                        params = input_model.get("params", {})
                        hist_returns = params.get("historical_returns", [])
                        if len(hist_returns) > 0:
                            hist_arr = np.array(hist_returns)
                            report += f"""
- 历史数据点数：{len(hist_returns):,}
- 历史均值：{np.mean(hist_arr):.6f} ({np.mean(hist_arr)*252:.2%} 年化)
- 历史标准差：{np.std(hist_arr):.6f} ({np.std(hist_arr)*np.sqrt(252):.2%} 年化)
"""
                    
                    report += f"""

### 2.2 预测参数配置

**预测时间期限**：{sim_years} 年

**蒙特卡洛模拟次数**：{num_trials:,} 次

**初始资金**：${initial_capital:,.0f}

**每年追加投资**：${annual_cont:,.0f}

**杠杆倍数**：{leverage:.2f}

**策略参数**：
"""
                    
                    if backtest_strategy_params:
                        for key, value in backtest_strategy_params.items():
                            report += f"- {key}: {value}\n"
                    else:
                        report += "- 无特殊策略参数\n"
                    
                    report += f"""

### 2.3 模拟结果

**最终价值统计**：
- 中位数（Median）：${median_val:,.0f}
- 平均值（Mean）：${mean_val:,.0f}
- 标准差（Std Dev）：${std_val:,.0f}
- 5%分位数（最坏情况）：${p05_val:,.0f}
- 95%分位数（最好情况）：${p95_val:,.0f}
- 最小值：${np.min(final_values):,.0f}
- 最大值：${np.max(final_values):,.0f}

**概率分析**：
- 成功概率（超过盈亏平衡点）：{success_prob:.1%}
- 亏损概率（低于初始资金）：{loss_prob:.1%}
- 预期收益（相对于投入）：{gain:+.1%}

**盈亏平衡点**：${breakeven_balance:,.0f}（初始资金 + 累计追加投资）

---

## 🔄 完整工作流程说明

### 流程概览

本次分析遵循以下完整工作流程：

#### 第一步：回测（BACKTEST）

1. **上传历史数据** → 上传包含标的物价格的历史数据（CSV格式）
2. **选择策略** → 选择投资策略算法（{backtest_strategy if backtest_strategy else strategy_name_global}）
3. **运行回测** → 在历史数据上测试策略表现
4. **自动进行输入建模** → 系统从标的物价格数据中提取收益率分布特征
5. **得到Input Model** → 选择最适合的分布模型（{input_model_info.get('type', '未检测到') if input_model_info else '未检测到'}）

**回测的目的**：分析历史数据，得到标的物价格的输入建模（Input Model）并选择策略

#### 第二步：预测（PROJECTION）

6. **自动使用回测的策略和Input Model** → 系统自动应用回测中选择的策略和Input Model
7. **配置预测参数** → 设置预测时间期限、模拟次数等参数
8. **运行模拟** → 使用Input Model模拟未来价格走向
9. **评估策略在未来表现** → 基于模拟结果评估策略表现

**预测的目的**：使用回测中得到的Input Model模拟未来价格走向，评估策略在未来表现

### 关键要点

✅ **数据流**：历史数据 → 输入建模 → Input Model → 未来价格模拟 → 策略评估

✅ **策略继承**：预测阶段自动使用回测中选择的策略，确保一致性

✅ **模型应用**：Input Model基于历史数据特征，用于生成未来价格路径

✅ **结果解读**：预测结果反映策略在不同市场情景下的表现，帮助做出投资决策

---

## 📈 结论与建议

### 策略表现评估

基于 {num_trials:,} 次蒙特卡洛模拟的结果：

- **预期最终价值**：${median_val:,.0f}（中位数）
- **风险水平**：标准差为 ${std_val:,.0f}，表明策略存在一定波动性
- **成功概率**：{success_prob:.1%} 的概率能够达到或超过盈亏平衡点
- **最大潜在损失**：在最坏情况下（5%分位数），最终价值可能降至 ${p05_val:,.0f}

### 建议

"""
                    
                    if success_prob > 0.7:
                        report += "✅ 策略表现良好，成功概率较高，可以考虑实施。\n"
                    elif success_prob > 0.5:
                        report += "⚠️ 策略表现中等，建议进一步优化或调整参数。\n"
                    else:
                        report += "❌ 策略风险较高，建议重新评估策略或调整配置。\n"
                    
                    if loss_prob > 0.3:
                        report += "⚠️ 亏损概率较高，需要加强风险控制。\n"
                    
                    report += f"""
- 建议定期回顾和调整策略参数
- 考虑分散投资以降低风险
- 根据市场变化及时更新Input Model

---

**报告生成时间**：{datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}

**注**：本报告基于历史数据和统计模型生成，实际投资结果可能因市场变化而有所不同。投资有风险，决策需谨慎。
"""
                    
                    # 提供下载按钮
                    st.download_button(
                        label="📄 下载完整流程报告（Markdown）",
                        data=report,
                        file_name=f"investment_simulation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                    
                    st.success("✅ 报告已生成！点击上方按钮下载完整流程报告。")
                    
                except Exception as e:
                    import traceback
                    st.error(f"报告生成失败：{str(e)}")
                    st.caption(f"错误详情：{traceback.format_exc()}")
            else:
                st.warning("⚠️ 无法生成报告：请先运行预测模拟。")
        
        # ==========================================
        # 预测结论与决策建议（放在图表之后）
        # ==========================================
        st.markdown("---")
        st.markdown("### 🎯 预测结论与决策建议")
        
        # 生成综合评估
        conclusion_col1, conclusion_col2 = st.columns([2, 1])
        
        with conclusion_col1:
            # 综合评分（0-100）
            score = 0
            score_details = []
            
            # 预期收益评分（30分）
            if gain > 0.5:
                ret_score = 30
                ret_comment = "优秀"
            elif gain > 0.2:
                ret_score = 20
                ret_comment = "良好"
            elif gain > 0:
                ret_score = 10
                ret_comment = "一般"
            else:
                ret_score = 0
                ret_comment = "亏损"
            score += ret_score
            score_details.append(f"预期收益：{ret_comment} (+{ret_score}分)")
            
            # 成功概率评分（25分）
            success_score = int(success_prob * 25)
            if success_prob > 0.8:
                success_comment = "很高"
            elif success_prob > 0.6:
                success_comment = "较高"
            elif success_prob > 0.4:
                success_comment = "中等"
            else:
                success_comment = "较低"
            score += success_score
            score_details.append(f"成功概率：{success_comment} (+{success_score}分)")
            
            # 风险控制评分（25分）
            worst_case_loss = (p05_val - breakeven_balance) / breakeven_balance
            if worst_case_loss > -0.1:
                risk_score = 25
                risk_comment = "优秀"
            elif worst_case_loss > -0.2:
                risk_score = 20
                risk_comment = "良好"
            elif worst_case_loss > -0.3:
                risk_score = 15
                risk_comment = "一般"
            else:
                risk_score = 10
                risk_comment = "较差"
            score += risk_score
            score_details.append(f"风险控制：{risk_comment} (+{risk_score}分)")
            
            # 稳定性评分（20分）
            cv = std_val / mean_val if mean_val > 0 else 1.0  # 变异系数
            stability_score = max(0, 20 - int(cv * 20))
            if cv < 0.2:
                stability_comment = "非常稳定"
            elif cv < 0.3:
                stability_comment = "较稳定"
            elif cv < 0.5:
                stability_comment = "中等波动"
            else:
                stability_comment = "高波动"
            score += stability_score
            score_details.append(f"预测稳定性：{stability_comment} (+{stability_score}分)")
            
            # 总体评价
            if score >= 80:
                overall_rating = "优秀 ⭐⭐⭐⭐⭐"
                rating_color = "#3FB950"
                recommendation = "强烈推荐"
            elif score >= 65:
                overall_rating = "良好 ⭐⭐⭐⭐"
                rating_color = "#58A6FF"
                recommendation = "推荐"
            elif score >= 50:
                overall_rating = "一般 ⭐⭐⭐"
                rating_color = "#D29922"
                recommendation = "可考虑"
            elif score >= 35:
                overall_rating = "较差 ⭐⭐"
                rating_color = "#F85149"
                recommendation = "需谨慎"
            else:
                overall_rating = "差 ⭐"
                rating_color = "#F85149"
                recommendation = "不推荐"
            
            # 显示评分卡片
            st.markdown(f"""
            <div style='background-color: rgba({int(rating_color[1:3], 16)}, {int(rating_color[3:5], 16)}, {int(rating_color[5:7], 16)}, 0.1); 
                        padding: 20px; border-radius: 10px; border-left: 4px solid {rating_color}; margin-bottom: 20px;'>
            <h3 style='color: {rating_color}; margin-top: 0;'>综合评分：{score}/100</h3>
            <h4 style='color: {rating_color};'>总体评价：{overall_rating}</h4>
            <p style='font-size: 16px;'><strong>建议：{recommendation}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # 评分详情
            with st.expander("📊 评分详情", expanded=False):
                for detail in score_details:
                    st.markdown(f"- {detail}")
                st.markdown(f"**总分：{score}/100**")
        
        with conclusion_col2:
            # 关键指标卡片
            st.markdown("#### 关键指标")
            st.metric("预期结果", f"${median_val:,.0f}", f"{gain:+.1%}")
            st.metric("成功概率", f"{success_prob:.1%}")
            st.metric("最坏情况", f"${p05_val:,.0f}")
            st.metric("最好情况", f"${p95_val:,.0f}")
        
        # 决策建议
        st.markdown("---")
        st.markdown("#### 💡 决策建议")
        
        advice_col1, advice_col2 = st.columns(2)
        
        with advice_col1:
            st.markdown("##### ✅ 预测优势")
            advantages = []
            
            if gain > 0.3:
                advantages.append("**预期收益优秀** - 预期收益率超过30%，增长潜力大")
            elif gain > 0.1:
                advantages.append("**预期收益良好** - 预期收益率超过10%，有增长空间")
            
            if success_prob > 0.7:
                advantages.append("**成功概率高** - 超过70%的概率达到目标，信心较高")
            
            if worst_case_loss > -0.15:
                advantages.append("**下行风险可控** - 最坏情况下损失可控，风险较低")
            
            if cv < 0.3:
                advantages.append("**预测稳定性好** - 结果分布集中，预测可靠性高")
            
            if not advantages:
                advantages.append("预测结果中规中矩，无明显突出优势")
            
            for adv in advantages:
                st.markdown(f"- {adv}")
        
        with advice_col2:
            st.markdown("##### ⚠️ 需要关注")
            concerns = []
            
            if gain < 0:
                concerns.append("**预期亏损** - 预期收益率为负，需要重新评估策略")
            elif gain < 0.05:
                concerns.append("**收益偏低** - 预期收益率低于5%，可能不如无风险资产")
            
            if success_prob < 0.5:
                concerns.append("**成功概率较低** - 成功概率低于50%，风险较高")
            
            if worst_case_loss < -0.3:
                concerns.append("**下行风险较大** - 最坏情况下可能损失超过30%")
            
            if cv > 0.5:
                concerns.append("**预测不确定性高** - 结果分布分散，预测可靠性较低")
            
            if loss_prob > 0.3:
                concerns.append("**亏损概率较高** - 超过30%的概率出现亏损")
            
            if not concerns:
                concerns.append("预测结果良好，无明显风险点")
            
            for concern in concerns:
                st.markdown(f"- {concern}")
        
        # 策略适用性评估
        st.markdown("---")
        st.markdown("#### 🎯 策略适用性评估")
        
        suitability_col1, suitability_col2, suitability_col3 = st.columns(3)
        
        with suitability_col1:
            st.markdown("##### 📊 适合的投资者类型")
            investor_types = []
            
            if cv < 0.25 and worst_case_loss > -0.15:
                investor_types.append("✅ **风险厌恶型** - 低波动、低风险")
            
            if gain > 0.1 and success_prob > 0.6:
                investor_types.append("✅ **平衡型** - 收益风险平衡")
            
            if gain > 0.2 and success_prob > 0.5:
                investor_types.append("✅ **成长型** - 追求较高收益")
            
            if not investor_types:
                investor_types.append("⚠️ 需要根据个人风险偏好谨慎评估")
            
            for it in investor_types:
                st.markdown(it)
        
        with suitability_col2:
            st.markdown("##### 📈 时间期限建议")
            time_horizon_advice = []
            
            if sim_years >= 10:
                time_horizon_advice.append("✅ **长期投资** - 10年以上，适合长期持有")
            elif sim_years >= 5:
                time_horizon_advice.append("✅ **中期投资** - 5-10年，平衡收益和流动性")
            else:
                time_horizon_advice.append("✅ **短期投资** - 5年以下，关注短期波动")
            
            if success_prob > 0.7:
                time_horizon_advice.append("💡 当前时间期限下成功概率较高")
            else:
                time_horizon_advice.append("💡 考虑延长投资期限以提高成功概率")
            
            for tha in time_horizon_advice:
                st.markdown(tha)
        
        with suitability_col3:
            st.markdown("##### 🔄 优化建议")
            optimizations = []
            
            if gain < 0.1:
                optimizations.append("💡 考虑调整策略参数以提高预期收益")
            
            if success_prob < 0.6:
                optimizations.append("💡 增加年度贡献以提高成功概率")
            
            if worst_case_loss < -0.2:
                optimizations.append("💡 增加风险控制措施，降低下行风险")
            
            if cv > 0.4:
                optimizations.append("💡 考虑更保守的策略以降低不确定性")
            
            if not optimizations:
                optimizations.append("✅ 策略配置良好，可继续使用")
            
            for opt in optimizations:
                st.markdown(opt)
        
        # 最终结论
        st.markdown("---")
        st.markdown("#### 📝 最终结论")
        
        conclusion_text = f"""
        **预测结果总结：**
        
        基于蒙特卡洛模拟（{num_trials}次试验），{strategy_name_global}策略在未来{sim_years}年的预测表现{'良好' if score >= 65 else '一般' if score >= 50 else '较差'}。
        
        **核心发现：**
        - 预期结果为 **${median_val:,.0f}**，相比投入资金{'增长' if gain > 0 else '减少'} **{abs(gain):.2%}**
        - 成功概率为 **{success_prob:.1%}**，{'信心较高' if success_prob > 0.7 else '信心中等' if success_prob > 0.5 else '信心较低'}
        - 最坏情况（95% VaR）为 **${p05_val:,.0f}**，{'风险可控' if worst_case_loss > -0.2 else '风险较高'}
        - 最好情况（95%分位数）为 **${p95_val:,.0f}**，潜在收益{'可观' if (p95_val - breakeven_balance) / breakeven_balance > 0.3 else '有限'}
        
        **决策建议：**
        {'✅ 该策略预测表现优秀，建议采用并长期持有' if score >= 80 else '✅ 该策略预测表现良好，可以采纳' if score >= 65 else '⚠️ 该策略预测表现一般，建议优化参数或考虑其他策略' if score >= 50 else '❌ 该策略预测表现较差，建议重新评估或更换策略'}
        
        **风险提示：**
        - 预测结果基于历史数据和统计模型，不代表实际收益
        - 市场环境变化可能影响实际表现
        - 建议定期回顾和调整投资策略
        - 投资有风险，决策需谨慎
        """
        
        st.info(conclusion_text)
    
    elif not st.session_state.get("user_has_run_projection", False):
        # 引导提示
        guide_col1, guide_col2 = st.columns([2, 1])
        with guide_col1:
            st.info("""
            📍 **开始你的第一次预测：**
            
            1. **在左侧边栏选择策略** - 推荐新手使用 "Equal Weight" 或 "Fixed Weights"
            2. **配置投资参数** - 设置初始资金、杠杆等（可使用默认值）
            3. **设置模拟参数** - 选择预测年限、模拟次数、收益分布模型
            4. **点击运行按钮** - 点击 "🚀 RUN SIMULATION" 开始预测
            
            💡 **提示**：如果不确定如何配置，可以先使用默认参数快速体验！
            """)
        with guide_col2:
            st.markdown("""
            <div style="background-color: rgba(210, 153, 34, 0.1); padding: 20px; border-radius: 10px; border-left: 4px solid #D29922;">
            <h4>🎯 快速开始</h4>
            <p><strong>推荐配置：</strong></p>
            <ul style="text-align: left;">
            <li>策略：Equal Weight</li>
            <li>初始资金：$100,000</li>
            <li>预测年限：10年</li>
            <li>模拟次数：1000次</li>
            <li>收益分布：Normal</li>
            </ul>
            <p>点击运行即可！</p>
            </div>
            """, unsafe_allow_html=True)

# ------------------------------------------
# SCENARIO C: Derivatives Lab (Refactored)
# ------------------------------------------
else:
    render_derivatives_lab()