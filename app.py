from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Vietnam Egg Price Intelligence | HealthyFarm",
    page_icon="🥚",
    layout="wide",
)


# ============================================================
# HEALTHYFARM-INSPIRED BRAND PALETTE
# ------------------------------------------------------------
# The dashboard uses the same visual direction as HealthyFarm:
# natural greens, warm cream backgrounds, and a yellow accent.
# ============================================================
HF_DARK_GREEN = "#234B3A"
HF_GREEN = "#4F775D"
HF_SAGE = "#8FAE91"
HF_LIGHT_GREEN = "#E8F0E7"
HF_CREAM = "#F7F3E8"
HF_WARM_YELLOW = "#E8B84A"
HF_SOFT_YELLOW = "#F6E8B4"
HF_TERRACOTTA = "#C9774D"
HF_TEXT = "#26332B"
HF_MUTED = "#68736C"
HF_WHITE = "#FFFFFF"
HF_BORDER = "#DDE5DC"
HF_ERROR = "#B94A48"

LEVEL_COLOR_MAP = {
    "Market": HF_DARK_GREEN,
    "Farmgate": HF_WARM_YELLOW,
    "Retail": HF_TERRACOTTA,
}

REGION_COLOR_MAP = {
    "North": HF_DARK_GREEN,
    "Central": HF_WARM_YELLOW,
    "South": HF_TERRACOTTA,
    "Unknown": HF_SAGE,
}

# Apply a HealthyFarm-inspired Streamlit theme with CSS.
st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {HF_CREAM};
            color: {HF_TEXT};
        }}

        [data-testid="stSidebar"] {{
            background-color: {HF_LIGHT_GREEN};
            border-right: 1px solid {HF_BORDER};
        }}

        [data-testid="stSidebar"] * {{
            color: {HF_TEXT};
        }}

        h1, h2, h3, h4 {{
            color: {HF_DARK_GREEN};
        }}

        .hf-hero {{
            background: linear-gradient(135deg, {HF_DARK_GREEN} 0%, {HF_GREEN} 100%);
            border-radius: 18px;
            padding: 24px 28px;
            margin-bottom: 18px;
            color: white;
        }}

        .hf-hero h1 {{
            color: white;
            margin: 0 0 6px 0;
            font-size: 2.15rem;
            line-height: 1.15;
        }}

        .hf-hero p {{
            color: #F5F7F4;
            margin: 0;
            font-size: 1rem;
        }}

        .hf-tag {{
            display: inline-block;
            background-color: {HF_WARM_YELLOW};
            color: {HF_DARK_GREEN};
            font-weight: 700;
            padding: 5px 10px;
            border-radius: 999px;
            margin-bottom: 10px;
            font-size: 0.82rem;
        }}

        [data-testid="stMetric"] {{
            background-color: {HF_WHITE};
            border: 1px solid {HF_BORDER};
            border-radius: 14px;
            padding: 14px 16px;
            box-shadow: 0 1px 3px rgba(35, 75, 58, 0.05);
        }}

        [data-testid="stMetricLabel"] {{
            color: {HF_MUTED};
        }}

        [data-testid="stMetricValue"] {{
            color: {HF_DARK_GREEN};
        }}

        div[data-testid="stExpander"] {{
            background-color: {HF_WHITE};
            border: 1px solid {HF_BORDER};
            border-radius: 12px;
        }}

        .stButton > button,
        .stDownloadButton > button {{
            background-color: {HF_DARK_GREEN};
            color: white;
            border: none;
            border-radius: 10px;
        }}

        .stButton > button:hover,
        .stDownloadButton > button:hover {{
            background-color: {HF_GREEN};
            color: white;
        }}

        hr {{
            border-color: {HF_BORDER};
        }}

        .hf-note {{
            background-color: {HF_SOFT_YELLOW};
            border-left: 5px solid {HF_WARM_YELLOW};
            padding: 12px 14px;
            border-radius: 8px;
            color: {HF_TEXT};
            margin: 8px 0 16px 0;
        }}

        .hf-small {{
            color: {HF_MUTED};
            font-size: 0.9rem;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA SOURCE
# ------------------------------------------------------------
# Expected GitHub / local project structure:
#
# egg_price_project_vietnam/
# ├── app.py
# ├── preprocessing_clean_analytics.ipynb
# ├── ANALYTICS_EGG_PRICE_DATA.csv
# └── requirements.txt
#
# The analytics CSV lives in the SAME folder as app.py.
# Using __file__ keeps the path portable on Windows, GitHub,
# and Streamlit deployment.
# ============================================================
DATA_FILE_NAME = "ANALYTICS_EGG_PRICE_DATA.csv"

# Folder where app.py is located
BASE_DIR = Path(__file__).resolve().parent

# Analytics-ready database created by preprocessing
DATA_PATH = BASE_DIR / DATA_FILE_NAME

# ============================================================
# ANALYTICS COLUMN DEFINITIONS
# ============================================================
DATE_COL = "Date Clean"
PRICE_COL = "Price Per Egg VND Clean"
PRICE_LEVEL_COL = "Price Level"
REGION_COL = "Region Normalized"
PROVINCE_COL = "Province Normalized"
CITY_COL = "City Normalized"
SOURCE_COL = "Source"
QUALITY_COL = "Quality Status"
DATA_ORIGIN_COL = "Data Origin"
EGG_TYPE_COL = "Egg Type Label"
EGG_TYPE_FALLBACK = "Egg Type Normalized"
PRODUCTION_COL = "Production System"
BRAND_COL = "Brand"
PRODUCT_COL = "Product Name"
STORE_COL = "Store Name"
PACK_PRICE_COL = "Pack Price VND"
EGG_COUNT_COL = "Egg Count Clean"


# ============================================================
# DATA LOAD
# ------------------------------------------------------------
# Data engineering belongs in preprocessing_clean_analytics.ipynb.
# Streamlit only performs light type checking and a final safety check.
# ============================================================
@st.cache_data(show_spinner=False)
def load_analytics_data(path: str, file_modified_time: float) -> pd.DataFrame:
    """
    Load the preprocessed analytics CSV.

    file_modified_time is part of the cache key, so Streamlit
    reloads the data automatically whenever preprocessing
    creates a newer analytics CSV.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")

    required_columns = [DATE_COL, PRICE_COL, PRICE_LEVEL_COL, REGION_COL, SOURCE_COL]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            "The analytics file is missing required columns: " + ", ".join(missing)
        )

    # Parse types used by the dashboard.
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    numeric_cols = [
        PRICE_COL,
        PACK_PRICE_COL,
        EGG_COUNT_COL,
        "Buying Price VND",
        "Selling Price VND",
        "Quantity Sold",
        "Stock Quantity",
        "Feed Cost VND",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Final dashboard safety checks.
    # The preprocessing notebook should already have removed these rows,
    # but keeping this guard prevents accidental display of known hard errors.
    if "Analytics Eligible" in df.columns:
        df = df[df["Analytics Eligible"].astype(str).str.lower().eq("yes")].copy()

    if QUALITY_COL in df.columns:
        df = df[~df[QUALITY_COL].astype(str).str.upper().eq("ERROR")].copy()

    df = df[df[DATE_COL].notna() & df[PRICE_COL].notna()].copy()
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    # Friendly egg type display if the label field is unavailable.
    if EGG_TYPE_COL not in df.columns and EGG_TYPE_FALLBACK in df.columns:
        df[EGG_TYPE_COL] = df[EGG_TYPE_FALLBACK]

    return df


try:
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    df = load_analytics_data(
        str(DATA_PATH),
        DATA_PATH.stat().st_mtime,
    )

except FileNotFoundError:
    st.error(
        f"Could not find **{DATA_FILE_NAME}** at:\n\n"
        f"`{DATA_PATH}`\n\n"
        "Run `preprocessing_clean_analytics.ipynb` first so the file is created "
        "in the same project folder as `app.py`."
    )
    st.stop()
except Exception as exc:
    st.error(f"Unable to load the analytics dataset: {exc}")
    st.stop()

if df.empty:
    st.error("The analytics dataset contains no usable observations.")
    st.stop()


# ============================================================
# HELPERS
# ============================================================
def format_vnd(value) -> str:
    if pd.isna(value):
        return "—"
    return f"₫{value:,.0f}"


def safe_pct_change(previous: float, current: float) -> float:
    if pd.isna(previous) or pd.isna(current) or previous == 0:
        return np.nan
    return (current - previous) / previous * 100.0


def direction_from_change(pct: float) -> str:
    if pd.isna(pct):
        return "Not enough data"
    if pct > 1.0:
        return "↑ Up"
    if pct < -1.0:
        return "↓ Down"
    return "→ Sideways"


def options_for(data: pd.DataFrame, column: str) -> list:
    if column not in data.columns:
        return ["(All)"]
    values = (
        data[column]
        .dropna()
        .astype(str)
        .str.strip()
    )
    values = sorted(v for v in values.unique().tolist() if v and v.lower() != "nan")
    return ["(All)"] + values


def apply_single_filter(data: pd.DataFrame, column: str, selected: str) -> pd.DataFrame:
    if selected != "(All)" and column in data.columns:
        return data[data[column].astype(str) == selected]
    return data


def style_figure(fig, legend_title=None):
    fig.update_layout(
        paper_bgcolor=HF_CREAM,
        plot_bgcolor=HF_WHITE,
        font=dict(color=HF_TEXT),
        title_font=dict(color=HF_DARK_GREEN, size=18),
        margin=dict(l=20, r=20, t=55, b=20),
        hoverlabel=dict(bgcolor=HF_WHITE, font_color=HF_TEXT),
    )
    fig.update_xaxes(gridcolor="#EEF1EC", linecolor=HF_BORDER)
    fig.update_yaxes(gridcolor="#EEF1EC", linecolor=HF_BORDER)
    if legend_title is not None:
        fig.update_layout(legend_title_text=legend_title)
    return fig


def aggregation_rule(data: pd.DataFrame):
    if data.empty:
        return "W", "Weekly"
    span_days = (data[DATE_COL].max() - data[DATE_COL].min()).days
    if span_days > 730:
        return "MS", "Monthly"
    if span_days > 120:
        return "W", "Weekly"
    return "D", "Daily"


# ============================================================
# HERO
# ============================================================
st.markdown(
    """
    <div class="hf-hero">
        <div class="hf-tag">HealthyFarm Market Intelligence</div>
        <h1>Vietnam Egg Price Dashboard</h1>
        <p>Track market, farmgate, and retail egg prices across Vietnam using one clean analytics dataset.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR FILTERS
# ============================================================
st.sidebar.header("Filters")
st.sidebar.caption(f"Data source: {DATA_FILE_NAME}")

min_date = df[DATE_COL].min().date()
max_date = df[DATE_COL].max().date()

# Default to the latest calendar year's YTD view.
default_start = date(max_date.year, 1, 1)
if default_start < min_date:
    default_start = min_date

selected_dates = st.sidebar.date_input(
    "Date range",
    value=(default_start, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
else:
    start_date = selected_dates
    end_date = selected_dates

# Start with date filter so all following options reflect the selected period.
df_scope = df[
    (df[DATE_COL].dt.date >= start_date)
    & (df[DATE_COL].dt.date <= end_date)
].copy()

price_levels = options_for(df_scope, PRICE_LEVEL_COL)
selected_price_level = st.sidebar.selectbox(
    "Price level",
    price_levels,
    index=0,
    help="Market, Farmgate, and Retail are kept separate because they represent different points in the supply chain.",
)

df_option_scope = apply_single_filter(df_scope, PRICE_LEVEL_COL, selected_price_level)

selected_region = st.sidebar.selectbox(
    "Region",
    options_for(df_option_scope, REGION_COL),
    index=0,
)

df_option_scope = apply_single_filter(df_option_scope, REGION_COL, selected_region)

selected_province = st.sidebar.selectbox(
    "Province",
    options_for(df_option_scope, PROVINCE_COL),
    index=0,
)

df_option_scope = apply_single_filter(df_option_scope, PROVINCE_COL, selected_province)

selected_source = st.sidebar.selectbox(
    "Source",
    options_for(df_option_scope, SOURCE_COL),
    index=0,
)

df_option_scope = apply_single_filter(df_option_scope, SOURCE_COL, selected_source)

selected_egg_type = st.sidebar.selectbox(
    "Egg type",
    options_for(df_option_scope, EGG_TYPE_COL),
    index=0,
)

with st.sidebar.expander("More filters"):
    selected_production = st.selectbox(
        "Production system",
        options_for(df_option_scope, PRODUCTION_COL),
        index=0,
    )
    selected_brand = st.selectbox(
        "Brand",
        options_for(df_option_scope, BRAND_COL),
        index=0,
    )
    selected_store = st.selectbox(
        "Store",
        options_for(df_option_scope, STORE_COL),
        index=0,
    )
    selected_product = st.selectbox(
        "Product",
        options_for(df_option_scope, PRODUCT_COL),
        index=0,
    )
    selected_quality = st.selectbox(
        "Quality status",
        options_for(df_option_scope, QUALITY_COL),
        index=0,
        help="REVIEW observations remain analytically usable but are kept visible for monitoring.",
    )
    selected_origin = st.selectbox(
        "Data origin",
        options_for(df_option_scope, DATA_ORIGIN_COL),
        index=0,
    )

# ============================================================
# APPLY FILTERS
# ============================================================
df_f = df_scope.copy()
filter_pairs = [
    (PRICE_LEVEL_COL, selected_price_level),
    (REGION_COL, selected_region),
    (PROVINCE_COL, selected_province),
    (SOURCE_COL, selected_source),
    (EGG_TYPE_COL, selected_egg_type),
    (PRODUCTION_COL, selected_production),
    (BRAND_COL, selected_brand),
    (STORE_COL, selected_store),
    (PRODUCT_COL, selected_product),
    (QUALITY_COL, selected_quality),
    (DATA_ORIGIN_COL, selected_origin),
]

for column, selected in filter_pairs:
    df_f = apply_single_filter(df_f, column, selected)

if df_f.empty:
    st.warning("No data found for the selected filters.")
    st.stop()

st.sidebar.caption(
    f"Selected data range: {df_f[DATE_COL].min().date()} to {df_f[DATE_COL].max().date()}"
)
st.sidebar.caption(f"Usable observations: {len(df_f):,}")


# ============================================================
# TOP KPIs
# ============================================================
st.subheader("Price Snapshot")

levels_in_scope = [
    level for level in ["Market", "Farmgate", "Retail"]
    if level in df_f[PRICE_LEVEL_COL].dropna().unique()
]

if selected_price_level == "(All)":
    kpi_cols = st.columns(5)

    for idx, level in enumerate(["Market", "Farmgate", "Retail"]):
        level_data = df_f[df_f[PRICE_LEVEL_COL] == level]
        value = level_data[PRICE_COL].mean() if not level_data.empty else np.nan
        kpi_cols[idx].metric(f"Avg {level} Price", format_vnd(value))

    kpi_cols[3].metric("Observations", f"{len(df_f):,}")
    kpi_cols[4].metric("Latest Observation", df_f[DATE_COL].max().strftime("%d %b %Y"))
else:
    latest_date = df_f[DATE_COL].max()
    latest_avg = df_f.loc[df_f[DATE_COL] == latest_date, PRICE_COL].mean()
    avg_price = df_f[PRICE_COL].mean()
    min_price = df_f[PRICE_COL].min()
    max_price = df_f[PRICE_COL].max()
    std_price = df_f[PRICE_COL].std()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(f"Avg {selected_price_level} Price", format_vnd(avg_price))
    k2.metric("Latest Daily Avg", format_vnd(latest_avg), latest_date.strftime("%d %b %Y"))
    k3.metric("Minimum", format_vnd(min_price))
    k4.metric("Maximum", format_vnd(max_price))
    k5.metric("Volatility (Std)", format_vnd(std_price))

# Simple market-to-retail spread for the selected period.
market_avg = df_f.loc[df_f[PRICE_LEVEL_COL] == "Market", PRICE_COL].mean()
retail_avg = df_f.loc[df_f[PRICE_LEVEL_COL] == "Retail", PRICE_COL].mean()

if selected_price_level == "(All)" and pd.notna(market_avg) and pd.notna(retail_avg):
    spread = retail_avg - market_avg
    spread_pct = spread / market_avg * 100 if market_avg != 0 else np.nan
    st.markdown(
        f"""
        <div class="hf-note">
        <b>Selected-period retail premium:</b> {format_vnd(spread)} per egg
        ({spread_pct:,.1f}% above the simple average market price).<br>
        <span class="hf-small">This is a descriptive price spread, not a retailer margin calculation.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()


# ============================================================
# 1. MARKET / FARMGATE / RETAIL COMPARISON
# ============================================================
st.subheader("1) Price Level Comparison")

level_summary = (
    df_f.groupby(PRICE_LEVEL_COL, as_index=False)
    .agg(
        Average_Price=(PRICE_COL, "mean"),
        Minimum_Price=(PRICE_COL, "min"),
        Maximum_Price=(PRICE_COL, "max"),
        Observations=(PRICE_COL, "size"),
    )
)

if level_summary.empty:
    st.info("No price-level data is available for the selected filters.")
else:
    fig_level = px.bar(
        level_summary,
        x=PRICE_LEVEL_COL,
        y="Average_Price",
        color=PRICE_LEVEL_COL,
        color_discrete_map=LEVEL_COLOR_MAP,
        text=level_summary["Average_Price"].round(0),
        title="Average Price per Egg by Price Level",
    )
    fig_level.update_traces(texttemplate="₫%{text:,.0f}", textposition="outside")
    fig_level.update_layout(
        xaxis_title="Price level",
        yaxis_title="Average price per egg (VND)",
        yaxis_tickformat=",",
        showlegend=False,
    )
    style_figure(fig_level)
    st.plotly_chart(fig_level, use_container_width=True)

    st.caption(
        "Market, farmgate, and retail observations are shown separately to avoid blending different supply-chain price levels."
    )

st.divider()


# ============================================================
# 2. REGIONAL PRICE COMPARISON
# ============================================================
st.subheader("2) Regional Price Comparison")

df_region = df_f[
    df_f[REGION_COL].notna()
    & ~df_f[REGION_COL].astype(str).eq("Unknown")
].copy()

if df_region.empty:
    st.info("Not enough region data to show a regional comparison.")
else:
    if selected_price_level == "(All)":
        region_summary = (
            df_region.groupby([REGION_COL, PRICE_LEVEL_COL], as_index=False)[PRICE_COL]
            .mean()
        )
        fig_region = px.bar(
            region_summary,
            x=REGION_COL,
            y=PRICE_COL,
            color=PRICE_LEVEL_COL,
            barmode="group",
            color_discrete_map=LEVEL_COLOR_MAP,
            title="Average Price per Egg by Region and Price Level",
        )
        fig_region.update_layout(legend_title_text="Price level")
    else:
        region_summary = (
            df_region.groupby(REGION_COL, as_index=False)[PRICE_COL]
            .mean()
            .sort_values(PRICE_COL, ascending=False)
        )
        fig_region = px.bar(
            region_summary,
            x=REGION_COL,
            y=PRICE_COL,
            color=REGION_COL,
            color_discrete_map=REGION_COLOR_MAP,
            text=region_summary[PRICE_COL].round(0),
            title=f"Average {selected_price_level} Price by Region",
        )
        fig_region.update_traces(texttemplate="₫%{text:,.0f}", textposition="outside")
        fig_region.update_layout(showlegend=False)

    fig_region.update_layout(
        xaxis_title="Region",
        yaxis_title="Average price per egg (VND)",
        yaxis_tickformat=",",
    )
    style_figure(fig_region)
    st.plotly_chart(fig_region, use_container_width=True)

st.divider()


# ============================================================
# 3. PRICE TREND
# ------------------------------------------------------------
# When All price levels are selected, trends are separated by
# price level. When one level is selected, trends are separated
# by region.
# ============================================================
st.subheader("3) Price Trend")

trend_data = df_f[df_f[DATE_COL].notna()].copy()
rule, aggregation_label = aggregation_rule(trend_data)

if trend_data.empty:
    st.info("Not enough dated observations to show a trend.")
else:
    if selected_price_level == "(All)":
        trend = (
            trend_data.set_index(DATE_COL)
            .groupby(PRICE_LEVEL_COL)[PRICE_COL]
            .resample(rule)
            .mean()
            .reset_index()
            .dropna(subset=[PRICE_COL])
        )
        fig_trend = px.line(
            trend,
            x=DATE_COL,
            y=PRICE_COL,
            color=PRICE_LEVEL_COL,
            markers=True,
            color_discrete_map=LEVEL_COLOR_MAP,
            title=f"{aggregation_label} Average Price Trend by Price Level",
        )
        legend_title = "Price level"
    else:
        trend_scope = trend_data[
            trend_data[REGION_COL].notna()
            & ~trend_data[REGION_COL].astype(str).eq("Unknown")
        ]
        trend = (
            trend_scope.set_index(DATE_COL)
            .groupby(REGION_COL)[PRICE_COL]
            .resample(rule)
            .mean()
            .reset_index()
            .dropna(subset=[PRICE_COL])
        )
        fig_trend = px.line(
            trend,
            x=DATE_COL,
            y=PRICE_COL,
            color=REGION_COL,
            markers=True,
            color_discrete_map=REGION_COLOR_MAP,
            title=f"{aggregation_label} {selected_price_level} Price Trend by Region",
        )
        legend_title = "Region"

    if trend.empty:
        st.info("Not enough observations are available after filtering to build the trend chart.")
    else:
        fig_trend.update_layout(
            xaxis_title="Date",
            yaxis_title="Average price per egg (VND)",
            yaxis_tickformat=",",
        )
        style_figure(fig_trend, legend_title)
        st.plotly_chart(fig_trend, use_container_width=True)

        # Short-term direction table for the series displayed.
        series_col = PRICE_LEVEL_COL if selected_price_level == "(All)" else REGION_COL
        direction_rows = []
        for series_name in sorted(trend[series_col].dropna().unique()):
            ts = trend.loc[trend[series_col] == series_name, [DATE_COL, PRICE_COL]].sort_values(DATE_COL)
            values = ts[PRICE_COL].dropna()
            if len(values) < 4:
                pct = np.nan
            else:
                previous = values.iloc[-4:-2].mean()
                current = values.iloc[-2:].mean()
                pct = safe_pct_change(previous, current)

            direction_rows.append(
                {
                    series_col: series_name,
                    "Short-term direction": direction_from_change(pct),
                    "Change (%)": round(pct, 1) if pd.notna(pct) else np.nan,
                }
            )

        if direction_rows:
            st.dataframe(
                pd.DataFrame(direction_rows),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                f"Direction compares the latest two {aggregation_label.lower()} observations with the previous two."
            )

st.divider()


# ============================================================
# 4. VOLATILITY / PRICE STABILITY
# ------------------------------------------------------------
# Volatility is only meaningful when one price level is selected.
# ============================================================
st.subheader("4) Price Stability by Region")

if selected_price_level == "(All)":
    st.info(
        "Select one Price Level (Market, Farmgate, or Retail) to compare volatility without mixing different supply-chain prices."
    )
else:
    vol_data = df_f[
        df_f[DATE_COL].notna()
        & df_f[REGION_COL].notna()
        & ~df_f[REGION_COL].astype(str).eq("Unknown")
    ].copy()

    if vol_data.empty:
        st.info("Not enough regional data is available for a volatility calculation.")
    else:
        weekly = (
            vol_data.set_index(DATE_COL)
            .groupby(REGION_COL)[PRICE_COL]
            .resample("W")
            .mean()
            .reset_index()
            .dropna(subset=[PRICE_COL])
        )

        vol = (
            weekly.groupby(REGION_COL, as_index=False)[PRICE_COL]
            .agg(std="std", mean="mean", observations="count")
        )

        valid_std = vol["std"].dropna()
        if len(valid_std) >= 3:
            q1 = valid_std.quantile(0.33)
            q2 = valid_std.quantile(0.66)
        elif not valid_std.empty:
            q1 = valid_std.median()
            q2 = valid_std.median()
        else:
            q1 = q2 = np.nan

        def volatility_label(value):
            if pd.isna(value):
                return "Not enough data"
            if pd.isna(q1) or pd.isna(q2):
                return "Not enough data"
            if value <= q1:
                return "Stable"
            if value <= q2:
                return "Moderate"
            return "Risky"

        vol["Volatility level"] = vol["std"].apply(volatility_label)

        VOL_COLOR_MAP = {
            "Stable": HF_GREEN,
            "Moderate": HF_WARM_YELLOW,
            "Risky": HF_TERRACOTTA,
            "Not enough data": HF_SAGE,
        }

        fig_vol = px.bar(
            vol.sort_values("std", ascending=False),
            x=REGION_COL,
            y="std",
            color="Volatility level",
            color_discrete_map=VOL_COLOR_MAP,
            title=f"Weekly {selected_price_level} Price Volatility by Region",
        )
        fig_vol.update_layout(
            xaxis_title="Region",
            yaxis_title="Standard deviation (VND per egg)",
            yaxis_tickformat=",",
        )
        style_figure(fig_vol, "Volatility level")
        st.plotly_chart(fig_vol, use_container_width=True)

        st.caption(
            "Lower volatility suggests more stable observed prices. The labels are relative to the regions available in the selected dataset."
        )


# ============================================================
# 5. RETAIL MARKET INTELLIGENCE
# ------------------------------------------------------------
# This section appears whenever retail observations remain in the
# current filter scope.
# ============================================================
retail = df_f[df_f[PRICE_LEVEL_COL] == "Retail"].copy()

if not retail.empty:
    st.divider()
    st.subheader("5) Retail Market Intelligence")

    retail_avg = retail[PRICE_COL].mean()
    retail_pack_avg = (
        retail[PACK_PRICE_COL].mean()
        if PACK_PRICE_COL in retail.columns and retail[PACK_PRICE_COL].notna().any()
        else np.nan
    )
    retail_brands = retail[BRAND_COL].nunique(dropna=True) if BRAND_COL in retail.columns else 0
    retail_stores = retail[STORE_COL].nunique(dropna=True) if STORE_COL in retail.columns else 0

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Avg Retail Price / Egg", format_vnd(retail_avg))
    r2.metric("Avg Retail Pack Price", format_vnd(retail_pack_avg))
    r3.metric("Brands", f"{retail_brands:,}")
    r4.metric("Stores", f"{retail_stores:,}")

    left, right = st.columns(2)

    with left:
        if BRAND_COL in retail.columns and retail[BRAND_COL].notna().any():
            brand_summary = (
                retail.dropna(subset=[BRAND_COL])
                .groupby(BRAND_COL, as_index=False)
                .agg(
                    Average_Price=(PRICE_COL, "mean"),
                    Observations=(PRICE_COL, "size"),
                )
                .sort_values(["Observations", "Average_Price"], ascending=[False, False])
                .head(12)
            )
            fig_brand = px.bar(
                brand_summary.sort_values("Average_Price"),
                x="Average_Price",
                y=BRAND_COL,
                orientation="h",
                color_discrete_sequence=[HF_GREEN],
                title="Average Retail Price by Brand",
            )
            fig_brand.update_layout(
                xaxis_title="Average price per egg (VND)",
                yaxis_title="Brand",
                xaxis_tickformat=",",
            )
            style_figure(fig_brand)
            st.plotly_chart(fig_brand, use_container_width=True)
        else:
            st.info("Brand information is not available for the selected retail records.")

    with right:
        if STORE_COL in retail.columns and retail[STORE_COL].notna().any():
            store_summary = (
                retail.dropna(subset=[STORE_COL])
                .groupby(STORE_COL, as_index=False)
                .agg(
                    Average_Price=(PRICE_COL, "mean"),
                    Observations=(PRICE_COL, "size"),
                )
                .sort_values("Observations", ascending=False)
                .head(12)
            )
            fig_store = px.bar(
                store_summary.sort_values("Average_Price"),
                x="Average_Price",
                y=STORE_COL,
                orientation="h",
                color_discrete_sequence=[HF_WARM_YELLOW],
                title="Average Retail Price by Store",
            )
            fig_store.update_layout(
                xaxis_title="Average price per egg (VND)",
                yaxis_title="Store",
                xaxis_tickformat=",",
            )
            style_figure(fig_store)
            st.plotly_chart(fig_store, use_container_width=True)
        else:
            st.info("Store information is not available for the selected retail records.")

    st.markdown("#### Latest Retail Product Observations")
    retail_table_cols = [
        DATE_COL,
        SOURCE_COL,
        REGION_COL,
        PROVINCE_COL,
        STORE_COL,
        BRAND_COL,
        PRODUCT_COL,
        EGG_COUNT_COL,
        PACK_PRICE_COL,
        PRICE_COL,
        QUALITY_COL,
    ]
    retail_table_cols = [c for c in retail_table_cols if c in retail.columns]

    retail_table = (
        retail[retail_table_cols]
        .sort_values(DATE_COL, ascending=False)
        .head(100)
        .copy()
    )

    rename_map = {
        DATE_COL: "Date",
        REGION_COL: "Region",
        PROVINCE_COL: "Province",
        EGG_COUNT_COL: "Egg Count",
        PACK_PRICE_COL: "Pack Price (VND)",
        PRICE_COL: "Price / Egg (VND)",
        QUALITY_COL: "Quality",
    }
    retail_table = retail_table.rename(columns=rename_map)
    st.dataframe(retail_table, use_container_width=True, hide_index=True)


# ============================================================
# DATA COVERAGE AND DOWNLOAD
# ============================================================
st.divider()
st.subheader("Data Coverage")

coverage_left, coverage_right = st.columns(2)

with coverage_left:
    source_summary = (
        df_f.groupby(SOURCE_COL, as_index=False)
        .agg(
            Observations=(PRICE_COL, "size"),
            First_Date=(DATE_COL, "min"),
            Last_Date=(DATE_COL, "max"),
        )
        .sort_values("Observations", ascending=False)
    )
    fig_source = px.bar(
        source_summary.head(15).sort_values("Observations"),
        x="Observations",
        y=SOURCE_COL,
        orientation="h",
        color_discrete_sequence=[HF_DARK_GREEN],
        title="Observations by Source",
    )
    style_figure(fig_source)
    st.plotly_chart(fig_source, use_container_width=True)

with coverage_right:
    level_counts = (
        df_f.groupby(PRICE_LEVEL_COL, as_index=False)
        .agg(Observations=(PRICE_COL, "size"))
        .sort_values("Observations", ascending=False)
    )
    fig_counts = px.pie(
        level_counts,
        names=PRICE_LEVEL_COL,
        values="Observations",
        color=PRICE_LEVEL_COL,
        color_discrete_map=LEVEL_COLOR_MAP,
        hole=0.55,
        title="Current Filter Coverage by Price Level",
    )
    style_figure(fig_counts, "Price level")
    st.plotly_chart(fig_counts, use_container_width=True)

with st.expander("View filtered data"):
    display_cols = [
        DATE_COL,
        PRICE_LEVEL_COL,
        SOURCE_COL,
        REGION_COL,
        PROVINCE_COL,
        CITY_COL,
        EGG_TYPE_COL,
        PRODUCTION_COL,
        BRAND_COL,
        PRODUCT_COL,
        STORE_COL,
        PACK_PRICE_COL,
        PRICE_COL,
        QUALITY_COL,
    ]
    display_cols = [c for c in display_cols if c in df_f.columns]
    st.dataframe(
        df_f[display_cols].sort_values(DATE_COL, ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    csv = df_f.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download filtered data (CSV)",
        data=csv,
        file_name="ANALYTICS_EGG_PRICE_DATA_filtered.csv",
        mime="text/csv",
    )

st.caption(
    "Dashboard calculations use the preprocessed Price Per Egg VND Clean field. "
    "Market, farmgate, and retail observations are kept separate unless a comparison view explicitly shows them side by side."
)
