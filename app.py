# -*- coding: utf-8 -*-
"""
毕业设计核心系统：基于大语言模型的二手平台数字商品生命周期与需求分析系统
前端实现：Streamlit + Plotly 交互式可视化分析大屏
运行命令：streamlit run app.py
"""

import os
import sys
import sqlite3
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ==============================================================================
# 1. 页面基本配置
# ==============================================================================
st.set_page_config(
    page_title="二手数字商品生命周期与需求分析系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS 样式提升视觉质感
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 4px solid #1E88E5;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        font-weight: 600;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xianyu_lifecycle.db")

# ==============================================================================
# 2. 数据库读取与缓存函数
# ==============================================================================
@st.cache_data(ttl=60)
def load_data():
    if not os.path.exists(DB_PATH):
        st.error(f"未找到数据库文件: {DB_PATH}")
        return pd.DataFrame(), pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)
    
    # 联合查询商品与最新快照
    items_query = """
    SELECT 
        i.item_id,
        i.title,
        i.price,
        i.area,
        i.seller_id,
        i.seller_nick,
        i.publish_time,
        i.category_keyword,
        i.fine_target,
        i.sub_category,
        i.risk_level,
        i.risk_type,
        i.risk_reason,
        i.pain_points_json,
        i.seo_title_suggestion,
        s.want_num,
        s.age_days,
        s.daily_velocity,
        s.profit_index,
        s.lifecycle_stage
    FROM items i
    LEFT JOIN (
        SELECT item_id, want_num, age_days, daily_velocity, profit_index, lifecycle_stage, MAX(snapshot_date)
        FROM daily_snapshots
        GROUP BY item_id
    ) s ON i.item_id = s.item_id
    """
    df_items = pd.read_sql_query(items_query, conn)

    # 读取卖家画像表
    sellers_query = "SELECT * FROM sellers ORDER BY total_wants DESC"
    df_sellers = pd.read_sql_query(sellers_query, conn)

    conn.close()

    # 衍生计算 TOS 蓝海机会指数
    # TOS = (Velocity * Price) / (ln(1 + CR4) * (1 + RiskFactor))
    # 针对各品类 CR4 映射
    cr4_map = {"简历模板": 79.92, "Python源码": 81.80, "考公资料": 71.82}
    risk_map = {"L0": 0.0, "L1": 0.3, "L2": 1.0, "L3": 3.0}

    def calc_tos(row):
        vel = float(row.get("daily_velocity") or 0.0)
        p = float(row.get("price") or 1.0)
        cat = str(row.get("category_keyword") or "简历模板")
        cr4 = cr4_map.get(cat, 75.0)
        r_lvl = str(row.get("risk_level") or "L0")
        r_factor = risk_map.get(r_lvl, 0.0)
        denom = np.log(1.0 + cr4) * (1.0 + r_factor)
        return round((vel * p) / denom, 2) if denom > 0 else 0.0

    if not df_items.empty:
        df_items["tos_score"] = df_items.apply(calc_tos, axis=1)

    return df_items, df_sellers

df_items, df_sellers = load_data()

# ==============================================================================
# 3. 页面侧边栏：多维智能筛选器与系统概览
# ==============================================================================
st.sidebar.title("🎛️ 智能筛选与控制面板")
st.sidebar.markdown("---")

if df_items.empty:
    st.warning("暂无数据，请确认数据库状态！")
    st.stop()

# 筛选条件
all_categories = ["全选"] + sorted(list(df_items["category_keyword"].dropna().unique()))
selected_cat = st.sidebar.selectbox("选择大类赛道", all_categories, index=0)

all_sub_categories = ["全选"] + sorted(list(df_items["sub_category"].dropna().unique()))
selected_sub_cat = st.sidebar.selectbox("细分二级赛道", all_sub_categories, index=0)

all_stages = ["全选"] + sorted(list(df_items["lifecycle_stage"].dropna().unique()))
selected_stage = st.sidebar.selectbox("生命周期阶段", all_stages, index=0)

all_risks = ["全选", "L0 (合规)", "L1 (疑似搬运)", "L2 (侵权灰产)", "L3 (违规引流)"]
selected_risk = st.sidebar.selectbox("合规风险等级", all_risks, index=0)

min_velocity = st.sidebar.slider("最低想要流速 (人/天)", 0.0, float(df_items["daily_velocity"].max()), 0.0, step=1.0)

# 过滤数据
filtered_df = df_items.copy()
if selected_cat != "全选":
    filtered_df = filtered_df[filtered_df["category_keyword"] == selected_cat]
if selected_sub_cat != "全选":
    filtered_df = filtered_df[filtered_df["sub_category"] == selected_sub_cat]
if selected_stage != "全选":
    filtered_df = filtered_df[filtered_df["lifecycle_stage"] == selected_stage]
if selected_risk != "全选":
    risk_code = selected_risk[:2]
    filtered_df = filtered_df[filtered_df["risk_level"] == risk_code]
filtered_df = filtered_df[filtered_df["daily_velocity"] >= min_velocity]

st.sidebar.markdown("---")
st.sidebar.subheader("📈 实时数据快照")
st.sidebar.metric("当前筛选商品数", f"{len(filtered_df)} / {len(df_items)} 件")
st.sidebar.metric("职业倒爷矩阵号", f"{len(df_sellers[df_sellers['is_matrix_seller'] == 1])} / {len(df_sellers)} 人")
st.sidebar.caption("系统内核：Playwright + SQLite + Bass ODE + DeepSeek/Qwen")

# ==============================================================================
# 4. 主看板标题区与全局关键 KPI 指标卡
# ==============================================================================
st.title("📊 二手数字商品生命周期与需求分析系统")
st.markdown("##### 基于大语言模型与 Bass 创新扩散理论的闲鱼虚拟资产智能挖掘大屏")

kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5 = st.columns(5)
with kpi_c1:
    st.metric("📦 收录商品总量", f"{len(df_items)} 件", f"+{len(df_items[df_items['age_days'] <= 15])} 萌芽新上架")
with kpi_c2:
    total_wants = int(df_items["want_num"].sum())
    st.metric("🔥 累计收割想要", f"{total_wants:,} 人", "市场总需求规模")
with kpi_c3:
    growth_count = len(df_items[df_items["lifecycle_stage"] == "成长期暴利黑马"])
    st.metric("⚡ 成长期暴利黑马", f"{growth_count} 件", f"占比 {growth_count/len(df_items)*100:.1f}%")
with kpi_c4:
    avg_price = df_items["price"].mean()
    st.metric("💰 平均客单价", f"¥{avg_price:.2f}", f"最高 ¥{df_items['price'].max():.2f}")
with kpi_c5:
    l0_ratio = (df_items["risk_level"] == "L0").sum() / len(df_items) * 100
    st.metric("🛡️ 原创合规资产率", f"{l0_ratio:.1f}%", "L0 合规占比")

st.markdown("---")

# ==============================================================================
# 5. 四大核心视窗交互 Tab
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 视窗一：暴利黑马选品雷达",
    "📈 视窗二：Bass 扩散演化与生命周期",
    "🕵️ 视窗三：职业倒爷与垄断天眼",
    "🛡️ 视窗四：LLM 语义归因与合规防火墙"
])

# ------------------------------------------------------------------------------
# 视窗一：暴利黑马选品雷达
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("🔥 暴利高流速黑马与蓝海机会排行 (TOS 指数引擎)")
    st.markdown(
        "通过 **TOS (True Opportunity Score) 蓝海机会指数** 过滤无效低价僵尸链接，"
        "优先甄别 **处于起飞期、高流速、高收益且低垄断惩罚** 的优质虚拟标的。"
    )

    # 排序与展示表
    display_cols = [
        "fine_target", "category_keyword", "sub_category", "price", 
        "daily_velocity", "profit_index", "tos_score", "lifecycle_stage", 
        "risk_level", "age_days", "want_num"
    ]
    radar_df = filtered_df.sort_values(by="tos_score", ascending=False)

    col_chart, col_table = st.columns([1, 1])

    with col_chart:
        # TOS 分数散点气泡图：流速 vs 客单价 (气泡大小为 TOS)
        fig_bubble = px.scatter(
            radar_df.head(40),
            x="daily_velocity",
            y="price",
            size="tos_score",
            color="lifecycle_stage",
            hover_name="fine_target",
            labels={"daily_velocity": "想要日流速 (人/天)", "price": "标价 (元)", "tos_score": "TOS 蓝海分"},
            title="🎯 Top 40 选品机会气泡图 (流速 × 价格 × 蓝海指数)",
            color_discrete_map={
                "成长期暴利黑马": "#FF3366",
                "成熟期现金牛": "#00CC96",
                "衰退长尾期": "#AB63FA",
                "萌芽测试期": "#FFA15A"
            }
        )
        fig_bubble.update_layout(height=420)
        st.plotly_chart(fig_bubble, use_container_width=True)

    with col_table:
        # 二级赛道均收益柱状图
        cat_profit = filtered_df.groupby("sub_category")["profit_index"].mean().reset_index()
        cat_profit = cat_profit.sort_values(by="profit_index", ascending=True)
        fig_bar = px.bar(
            cat_profit,
            x="profit_index",
            y="sub_category",
            orientation="h",
            labels={"profit_index": "预估日均收益潜力 (元/天)", "sub_category": "细分赛道"},
            title="💡 各细分二级赛道平均收益潜力排行",
            color="profit_index",
            color_continuous_scale="Viridis"
        )
        fig_bar.update_layout(height=420)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### 📋 暴利黑马实时穿透明细表")
    st.dataframe(
        radar_df[display_cols].rename(columns={
            "fine_target": "细粒度核心标的",
            "category_keyword": "大类",
            "sub_category": "二级细分赛道",
            "price": "标价(元)",
            "daily_velocity": "日流速(人/天)",
            "profit_index": "日收益指数",
            "tos_score": "TOS蓝海分",
            "lifecycle_stage": "生命周期阶段",
            "risk_level": "风险",
            "age_days": "存活(天)",
            "want_num": "想要数"
        }),
        use_container_width=True,
        height=320
    )

# ------------------------------------------------------------------------------
# 视窗二：Bass 扩散演化与生命周期
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("📈 Bass 创新扩散微分方程演化与拐点反演")
    st.markdown(
        "基于 **Bass 扩散闭式解** $N(t) = M \\frac{1 - e^{-(p+q)t}}{1 + \\frac{q}{p}e^{-(p+q)t}}$ "
        "与特征拐点定理（起飞点 $t_1$、峰值时间 $t^*$、衰竭点 $t_2$）对任意商品进行生长趋势外推。"
    )

    # 选择单品进行 Bass 曲线可视化
    item_options = {
        f"[{row['item_id'][:8]}] {row['fine_target'] or row['title'][:20]} (存活:{row['age_days']:.1f}天 | 流速:{row['daily_velocity']:.1f})": row['item_id']
        for _, row in filtered_df.head(50).iterrows()
    }
    
    selected_item_label = st.selectbox("选择待分析商品链接进行 Bass 动力学反演:", list(item_options.keys()))
    selected_id = item_options[selected_item_label]
    target_row = filtered_df[filtered_df["item_id"] == selected_id].iloc[0]

    # 获取参数模拟 S 曲线
    age_t = max(0.1, float(target_row["age_days"]))
    current_wants = float(target_row["want_num"])
    
    # 启发式估计或标准参数
    if target_row["lifecycle_stage"] == "成长期暴利黑马":
        M = max(current_wants * 5.0, 1500.0)
        p, q = 0.028, 0.273
    elif target_row["lifecycle_stage"] == "成熟期现金牛":
        M = max(current_wants * 1.05, 50000.0)
        p, q = 0.0013, 0.021
    else:
        M = max(current_wants * 1.01, 600.0)
        p, q = 0.004, 0.004

    t_peak = (1.0 / (p + q)) * np.log(q / p) if q > p else 0.0
    t_takeoff = (1.0 / (p + q)) * np.log((q / p) * (2.0 + np.sqrt(3.0))) if q > p else 0.0
    t_decay = (1.0 / (p + q)) * np.log((q / p) * (2.0 - np.sqrt(3.0))) if q > p else t_peak * 2.0

    t_series = np.linspace(0, max(age_t * 1.8, 60.0), 200)
    # Bass 闭式解计算
    exp_term = np.exp(-(p + q) * t_series)
    N_series = M * (1.0 - exp_term) / (1.0 + (q / p) * exp_term)
    S_series = M * ((p + q)**2 / p) * exp_term / ((1.0 + (q / p) * exp_term)**2)

    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        fig_bass = make_subplots(specs=[[{"secondary_y": True}]])
        # 累计想要 S 曲线
        fig_bass.add_trace(
            go.Scatter(x=t_series, y=N_series, name="累计想要数 N(t) [S曲线]", line=dict(color="#1E88E5", width=3)),
            secondary_y=False
        )
        # 瞬时流速钟形曲线
        fig_bass.add_trace(
            go.Scatter(x=t_series, y=S_series, name="即时想要流速 S(t) [钟形图]", line=dict(color="#FF8F00", width=2, dash="dot")),
            secondary_y=True
        )
        # 标记当前点
        fig_bass.add_trace(
            go.Scatter(x=[age_t], y=[current_wants], mode="markers+text", name="当前实际状态", 
                       marker=dict(size=12, color="red"), text=[f"当前 t={age_t:.1f}天"], textposition="top left"),
            secondary_y=False
        )
        # 添加特征线
        if t_peak > 0:
            fig_bass.add_vline(x=t_peak, line_width=1.5, line_dash="dash", line_color="green", annotation_text=f"峰值拐点 t*={t_peak:.1f}d")
        
        fig_bass.update_layout(
            title=f"🔬 《{target_row['fine_target']}》Bass 扩散微分方程动力学演化曲线",
            xaxis_title="商品上架时间 t (天)",
            height=450,
            hovermode="x unified"
        )
        fig_bass.update_yaxes(title_text="累计想要人数 N(t)", secondary_y=False)
        fig_bass.update_yaxes(title_text="即时新增流速 S(t) (人/天)", secondary_y=True)
        st.plotly_chart(fig_bass, use_container_width=True)

    with col_s2:
        st.markdown("#### 📐 计量参数与拐点判据")
        st.markdown(f"""
        - **潜在市场天花板 $M$**：`{M:.1f}` 人
        - **外部创新系数 $p$ (搜索刚需)**：`{p:.5f}`
        - **内部模仿系数 $q$ (算法加权)**：`{q:.5f}`
        - **当前市场饱和度 $F(t)$**：`{min(100.0, current_wants/M*100):.1f}%`
        - **起飞拐点 $t_1$**：`{t_takeoff:.1f}` 天
        - **峰值爆发点 $t^*$**：`{t_peak:.1f}` 天
        - **衰竭拐点 $t_2$**：`{t_decay:.1f}` 天
        ---
        **判定结论**：当前商品处于 **{target_row['lifecycle_stage']}**。
        """)

# ------------------------------------------------------------------------------
# 视窗三：职业倒爷与垄断天眼
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("🕵️ 供给端竞争格局、行业集中度与矩阵号监控")
    st.markdown("运用产业经济学 **CR4 / CR8 行业集中度** 与 **HHI 赫芬达尔指数**，透视闲鱼虚拟品职业倒爷垄断态势。")

    c_m1, c_m2, c_m3 = st.columns(3)
    with c_m1:
        st.metric("简历模板集中度", "CR4 = 79.92%", "HHI: 3873.3 (极高垄断)")
    with c_m2:
        st.metric("Python源码集中度", "CR4 = 81.80%", "HHI: 1982.0 (高集中寡占)")
    with c_m3:
        st.metric("考公资料集中度", "CR4 = 71.82%", "HHI: 1721.0 (中度寡占/易突围)")

    st.markdown("---")
    
    col_sel_chart, col_sel_table = st.columns([1, 1])
    with col_sel_chart:
        # 矩阵倒爷 Top 10 想要数分布
        top_sellers = df_sellers.head(10)
        fig_sellers = px.bar(
            top_sellers,
            x="total_wants",
            y="seller_nick",
            orientation="h",
            color="is_matrix_seller",
            labels={"total_wants": "累计收割想要数", "seller_nick": "卖家昵称"},
            title="🏆 头部职业倒爷收割规模排行 (橙色为矩阵号)",
            color_discrete_map={1: "#FF6600", 0: "#0099FF"}
        )
        fig_sellers.update_layout(height=400)
        st.plotly_chart(fig_sellers, use_container_width=True)

    with col_sel_table:
        # 卖家商品覆盖数量与预估日流水散点
        fig_scatter = px.scatter(
            df_sellers,
            x="total_items",
            y="est_daily_revenue",
            color="is_matrix_seller",
            hover_name="seller_nick",
            labels={"total_items": "在架爆款商品数", "est_daily_revenue": "预估日均总流水 (元/天)"},
            title="💼 卖家矩阵规模 vs 预估日收益分布",
            color_discrete_map={1: "#FF3366", 0: "#33CC33"}
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("### 📋 职业倒爷/矩阵大号核心特征画像明细")
    st.dataframe(
        df_sellers.rename(columns={
            "seller_nick": "卖家昵称",
            "total_items": "爆款商品数",
            "total_wants": "累计想要总数",
            "avg_price": "平均标价",
            "est_daily_revenue": "预估日总流水(元/天)",
            "is_matrix_seller": "是否矩阵号"
        }),
        use_container_width=True,
        height=260
    )

# ------------------------------------------------------------------------------
# 视窗四：LLM 语义归因与合规防火墙
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("🛡️ 大模型细粒度实体抽取与四级合规风险防火墙")
    st.markdown("展示大语言模型（DeepSeek / Qwen）对非结构化标题的实体解构成果与风险评级分布。")

    col_r1, col_r2 = st.columns([1, 1])
    with col_r1:
        # 四级合规风险占比饼图
        risk_counts = df_items["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["risk_level", "count"]
        fig_pie = px.pie(
            risk_counts,
            values="count",
            names="risk_level",
            title="🛡️ 全库四级合规与擦边风险等级分布",
            color="risk_level",
            color_discrete_map={
                "L0": "#00CC96",  # 绿色合规
                "L1": "#FFA15A",  # 橙色搬运
                "L2": "#FF3366",  # 红色侵权
                "L3": "#7F7F7F"   # 灰色违规
            }
        )
        fig_pie.update_layout(height=380)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_r2:
        # 买家五维痛点命中频次统计
        all_pains = []
        for p_json in df_items["pain_points_json"].dropna():
            try:
                plist = json.loads(p_json)
                if isinstance(plist, list):
                    all_pains.extend(plist)
            except Exception:
                pass
        
        if all_pains:
            pain_df = pd.Series(all_pains).value_counts().reset_index()
            pain_df.columns = ["pain_point", "count"]
            fig_pain = px.bar(
                pain_df,
                x="count",
                y="pain_point",
                orientation="h",
                labels={"count": "出现频次", "pain_point": "买家核心痛点维度"},
                title="🎯 买家五维核心痛点覆盖频度 (5D Demand Ontology)",
                color="count",
                color_continuous_scale="Teal"
            )
            fig_pain.update_layout(height=380)
            st.plotly_chart(fig_pain, use_container_width=True)

    st.markdown("### 📝 大模型动态 SEO 标题重排与合规审计样例对比")
    sample_audit = filtered_df[[
        "fine_target", "title", "seo_title_suggestion", "risk_level", "risk_reason"
    ]].head(15).rename(columns={
        "fine_target": "抽取真实标的",
        "title": "原始标题党文案",
        "seo_title_suggestion": "动态SEO重构标题",
        "risk_level": "风险等级",
        "risk_reason": "合规审计理由"
    })
    st.dataframe(sample_audit, use_container_width=True, height=280)

# ==============================================================================
# 页脚
# ==============================================================================
st.markdown("---")
st.caption("《基于大语言模型的二手平台数字商品生命周期与需求分析系统》 | 毕业设计项目工程资产归档")
