from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Vietnam Egg Price Intelligence | HealthyFarm",
    page_icon="🥚",
    layout="wide",
)

# ============================================================
# HEALTHYFARM BRAND KIT
# ------------------------------------------------------------
# Official client palette (HEALTHYFARM BRAND KIT):
# #8C1E14 red, #FF8325 orange, #FFD230 yellow,
# #192E6D navy, #0E9E8B teal. Typography: Nunito.
# ============================================================
HF_RED = "#8C1E14"
HF_ORANGE = "#FF8325"
HF_YELLOW = "#FFD230"
HF_NAVY = "#192E6D"
HF_TEAL = "#0E9E8B"
HF_WHITE = "#FFFFFF"
HF_BG = "#F8FAF9"
HF_LIGHT_TEAL = "#E8F6F4"
HF_LIGHT_YELLOW = "#FFF7D8"
HF_TEXT = "#24304A"
HF_MUTED = "#667085"
HF_BORDER = "#DCE6E4"

LEVEL_COLOR_MAP = {
    "Market": HF_NAVY,
    "Farmgate": HF_YELLOW,
    "Retail": HF_TEAL,
}
HOUSING_COLOR_MAP = {
    "Caged": HF_RED,
    "Cage-Free": HF_TEAL,
    "Free-Range": HF_ORANGE,
}
REGION_COLOR_MAP = {
    "North": HF_NAVY,
    "Central": HF_ORANGE,
    "South": HF_TEAL,
}
HOUSING_ORDER = ["Caged", "Cage-Free", "Free-Range"]
LEVEL_ORDER = ["Market", "Farmgate", "Retail"]

st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;700;900&display=swap');
        .stApp {{ background-color: {HF_BG}; color: {HF_TEXT}; font-family: 'Nunito', sans-serif; }}
        .stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{ font-family: 'Nunito', sans-serif; }}
        [data-testid="stSidebar"] {{
            background-color: {HF_LIGHT_TEAL};
            border-right: 1px solid {HF_BORDER};
        }}
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{ color: {HF_NAVY}; }}
        h1, h2, h3, h4 {{ color: {HF_NAVY}; }}
        .hf-hero {{
            background: {HF_NAVY};
            border-radius: 18px; padding: 24px 28px; margin: 8px 0 20px 0; color: white;
            border-bottom: 5px solid {HF_TEAL};
        }}
        .hf-hero h1 {{ color: white; margin: 0 0 6px 0; font-size: 2.15rem; font-weight: 900; }}
        .hf-hero p {{ color: white; margin: 0; font-size: 1rem; }}
        .hf-tag {{
            display: inline-block; background-color: {HF_YELLOW};
            color: {HF_NAVY}; font-weight: 900; padding: 5px 10px;
            border-radius: 999px; margin-bottom: 10px; font-size: 0.82rem;
        }}
        [data-testid="stMetric"] {{
            background-color: {HF_WHITE}; border: 1px solid {HF_BORDER};
            border-radius: 14px; padding: 14px 16px;
            box-shadow: 0 2px 8px rgba(25,46,109,0.05);
        }}
        [data-testid="stMetricLabel"] {{ color: {HF_MUTED}; font-weight: 700; }}
        [data-testid="stMetricValue"] {{ color: {HF_NAVY}; font-weight: 900; }}
        div[data-testid="stExpander"] {{
            background-color: {HF_WHITE}; border: 1px solid {HF_BORDER}; border-radius: 12px;
        }}
        .hf-note {{
            background-color: {HF_LIGHT_YELLOW}; border-left: 5px solid {HF_YELLOW};
            padding: 12px 14px; border-radius: 8px; color: {HF_TEXT}; margin: 8px 0 16px 0;
        }}
        .hf-info {{
            background-color: {HF_LIGHT_TEAL}; border-left: 5px solid {HF_TEAL};
            padding: 12px 14px; border-radius: 8px; color: {HF_TEXT}; margin: 8px 0 16px 0;
        }}
        .hf-small {{ color: {HF_MUTED}; font-size: 0.9rem; }}
        hr {{ border-color: {HF_BORDER}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DATA
# ============================================================
DATA_FILE_NAME = "ANALYTICS_EGG_PRICE_DATA.csv"
BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "assets" / "healthyfarm_logo.png"
DATA_CANDIDATES = [
    BASE_DIR / DATA_FILE_NAME,
    BASE_DIR / "processed_data" / DATA_FILE_NAME,
]

DATE_COL = "Date Clean"
PRICE_COL = "Price Per Egg VND Clean"
PRICE_LEVEL_COL = "Price Level"
REGION_COL = "Region Normalized"
PROVINCE_COL = "Province Normalized"
CITY_COL = "City Normalized"
SOURCE_COL = "Source"
QUALITY_COL = "Quality Status"
EGG_TYPE_COL = "Egg Type Label"
HOUSING_COL = "Housing System"
HOUSING_STATUS_COL = "Housing Label Status"
BRAND_COL = "Brand"
PRODUCT_COL = "Product Name"
STORE_COL = "Store Name"
PACK_PRICE_COL = "Pack Price VND"
EGG_COUNT_COL = "Egg Count Clean"


def find_data_path() -> Path:
    for candidate in DATA_CANDIDATES:
        if candidate.exists():
            return candidate
    return DATA_CANDIDATES[0]


@st.cache_data(show_spinner=False)
def load_data(path: str, modified: float) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    required = [DATE_COL, PRICE_COL, PRICE_LEVEL_COL, SOURCE_COL, HOUSING_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Analytics file is missing required columns: " + ", ".join(missing)
            + ". Run the updated preprocessing notebook first."
        )

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce", format="mixed")
    for col in [PRICE_COL, PACK_PRICE_COL, EGG_COUNT_COL]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Analytics Eligible" in df.columns:
        df = df[df["Analytics Eligible"].astype(str).str.casefold().eq("yes")].copy()
    if QUALITY_COL in df.columns:
        df = df[~df[QUALITY_COL].astype(str).str.upper().eq("ERROR")].copy()

    df = df[df[DATE_COL].notna() & df[PRICE_COL].notna()].copy()
    df = df[df[PRICE_COL].between(500, 20_000)].copy()
    return df.sort_values(DATE_COL).reset_index(drop=True)


DATA_PATH = find_data_path()
try:
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)
    df = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
except FileNotFoundError:
    st.error(
        f"Could not find **{DATA_FILE_NAME}**. Run the updated preprocessing notebook first."
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


def format_pct(value) -> str:
    if pd.isna(value):
        return "—"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:,.1f}%"


def values_for(data: pd.DataFrame, column: str) -> list[str]:
    if column not in data.columns:
        return []
    values = data[column].dropna().astype(str).str.strip()
    return sorted(v for v in values.unique().tolist() if v and v.lower() != "nan")


def options_for(data: pd.DataFrame, column: str) -> list[str]:
    return ["(All)"] + values_for(data, column)


def apply_single_filter(data: pd.DataFrame, column: str, selected: str) -> pd.DataFrame:
    if selected != "(All)" and column in data.columns:
        return data[data[column].astype(str) == selected]
    return data


def style_figure(fig, legend_title=None):
    fig.update_layout(
        paper_bgcolor=HF_BG,
        plot_bgcolor=HF_WHITE,
        font=dict(color=HF_TEXT),
        title_font=dict(color=HF_NAVY, size=18),
        margin=dict(l=20, r=20, t=55, b=20),
        hoverlabel=dict(bgcolor=HF_WHITE, font_color=HF_TEXT),
    )
    fig.update_xaxes(gridcolor="#EEF2F1", linecolor=HF_BORDER)
    fig.update_yaxes(gridcolor="#EEF2F1", linecolor=HF_BORDER)
    if legend_title:
        fig.update_layout(legend_title_text=legend_title)
    return fig


def aggregation_rule(data: pd.DataFrame):
    span = (data[DATE_COL].max() - data[DATE_COL].min()).days if not data.empty else 0
    if span > 730:
        return "MS", "Monthly"
    if span > 120:
        return "W", "Weekly"
    return "D", "Daily"


def level_average(data: pd.DataFrame, level: str) -> float:
    scoped = data[data[PRICE_LEVEL_COL] == level]
    return scoped[PRICE_COL].mean() if not scoped.empty else np.nan


def spread_metrics(data: pd.DataFrame, base_level="Market", target_level="Retail"):
    base = level_average(data, base_level)
    target = level_average(data, target_level)
    if pd.isna(base) or pd.isna(target) or base == 0:
        return base, target, np.nan, np.nan
    spread = target - base
    pct = spread / base * 100
    return base, target, spread, pct


# ============================================================
# HERO
# ============================================================
if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=300)

st.markdown(
    """
    <div class="hf-hero">
        <div class="hf-tag">HealthyFarm Egg Price Intelligence</div>
        <h1>Vietnam Egg Price Dashboard</h1>
        <p>Track market, farmgate and retail egg prices, with transparent comparisons for Caged, Cage-Free and explicitly identified Free-Range eggs.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hf-info">
    <b>Current methodology:</b> FeedIn and AGROINFO are treated as <b>Caged provisionally</b> while source methodology is being confirmed.
    Cage-Free and Free-Range are only shown when supported by the product/source classification.
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# FILTERS
# ============================================================
st.sidebar.header("Filters")
st.sidebar.caption(f"Data source: {DATA_PATH.name}")

min_date = df[DATE_COL].min().date()
max_date = df[DATE_COL].max().date()
default_start = max(min_date, date(max_date.year, 1, 1))
selected_dates = st.sidebar.date_input(
    "Date range",
    value=(default_start, max_date),
    min_value=min_date,
    max_value=max_date,
)
if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
else:
    start_date = end_date = selected_dates

scope = df[
    (df[DATE_COL].dt.date >= start_date)
    & (df[DATE_COL].dt.date <= end_date)
].copy()

selected_price_level = st.sidebar.selectbox(
    "Price level",
    options_for(scope, PRICE_LEVEL_COL),
    help="Market, Farmgate and Retail remain separate supply-chain price levels.",
)

available_housing = [h for h in HOUSING_ORDER if h in values_for(scope, HOUSING_COL)]
selected_housing = st.sidebar.multiselect(
    "Housing system",
    options=available_housing,
    default=[],
    help=(
        "Leave empty to use all price observations. Housing comparison charts only use correctly labeled "
        "Caged, Cage-Free and explicit Free-Range records; Unknown is never used in housing comparisons."
    ),
)

selected_region = st.sidebar.selectbox("Region", options_for(scope, REGION_COL))
region_scope = apply_single_filter(scope, REGION_COL, selected_region)
selected_province = st.sidebar.selectbox("Province", options_for(region_scope, PROVINCE_COL))
province_scope = apply_single_filter(region_scope, PROVINCE_COL, selected_province)
selected_source = st.sidebar.selectbox("Source", options_for(province_scope, SOURCE_COL))

with st.sidebar.expander("More filters"):
    selected_egg_type = st.selectbox("Raw egg type (optional)", options_for(province_scope, EGG_TYPE_COL), help="Underlying source egg-type label. The client-facing comparison uses Housing system.")
    selected_brand = st.selectbox("Brand", options_for(province_scope, BRAND_COL))
    selected_store = st.selectbox("Store", options_for(province_scope, STORE_COL))
    selected_product = st.selectbox("Product", options_for(province_scope, PRODUCT_COL))

# Apply all filters.
df_f = scope.copy()
for column, selected in [
    (PRICE_LEVEL_COL, selected_price_level),
    (REGION_COL, selected_region),
    (PROVINCE_COL, selected_province),
    (SOURCE_COL, selected_source),
    (EGG_TYPE_COL, selected_egg_type),
    (BRAND_COL, selected_brand),
    (STORE_COL, selected_store),
    (PRODUCT_COL, selected_product),
]:
    df_f = apply_single_filter(df_f, column, selected)

if selected_housing:
    df_f = df_f[df_f[HOUSING_COL].isin(selected_housing)].copy()

if df_f.empty:
    st.warning("No data found for the selected filters.")
    st.stop()

st.sidebar.caption(f"Selected range: {df_f[DATE_COL].min().date()} to {df_f[DATE_COL].max().date()}")
st.sidebar.caption(f"Usable observations: {len(df_f):,}")

# ============================================================
# PRICE SNAPSHOT
# ============================================================
st.subheader("Price Snapshot")
market_avg = level_average(df_f, "Market")
farmgate_avg = level_average(df_f, "Farmgate")
retail_avg = level_average(df_f, "Retail")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Avg Market Price", format_vnd(market_avg))
k2.metric("Avg Farmgate Price", format_vnd(farmgate_avg))
k3.metric("Avg Retail Price", format_vnd(retail_avg))
k4.metric("Observations", f"{len(df_f):,}")
k5.metric("Latest Observation", df_f[DATE_COL].max().strftime("%d %b %Y"))

st.divider()

# ============================================================
# 1. RETAIL VS MARKET / FARMGATE
# ============================================================
st.subheader("1) Retail vs Market / Farmgate Prices")
level_summary = (
    df_f.groupby(PRICE_LEVEL_COL, as_index=False)
    .agg(Average_Price=(PRICE_COL, "mean"), Observations=(PRICE_COL, "size"))
)
level_summary[PRICE_LEVEL_COL] = pd.Categorical(
    level_summary[PRICE_LEVEL_COL], categories=LEVEL_ORDER, ordered=True
)
level_summary = level_summary.sort_values(PRICE_LEVEL_COL)

fig_level = px.bar(
    level_summary,
    x=PRICE_LEVEL_COL,
    y="Average_Price",
    color=PRICE_LEVEL_COL,
    color_discrete_map=LEVEL_COLOR_MAP,
    text=level_summary["Average_Price"].round(0),
    title="Average Price per Egg by Supply-Chain Level",
)
fig_level.update_traces(texttemplate="₫%{text:,.0f}", textposition="outside")
fig_level.update_layout(
    xaxis_title="Price level", yaxis_title="Average price per egg (VND)",
    yaxis_tickformat=",", showlegend=False,
)
style_figure(fig_level)
st.plotly_chart(fig_level, use_container_width=True)

# ============================================================
# 2. MARKET -> RETAIL SPREAD
# ============================================================
st.subheader("2) Market → Retail Spread")
base, target, spread, spread_pct = spread_metrics(df_f, "Market", "Retail")

s1, s2, s3, s4 = st.columns(4)
s1.metric("Market Price", format_vnd(base))
s2.metric("Retail Price", format_vnd(target))
s3.metric("Retail Spread", format_vnd(spread) if pd.notna(spread) else "—")
s4.metric("Spread %", format_pct(spread_pct))

if pd.notna(spread):
    direction = "above" if spread >= 0 else "below"
    st.markdown(
        f"""
        <div class="hf-note">
        Retail is <b>{format_vnd(abs(spread))} per egg</b> ({abs(spread_pct):,.1f}%) {direction} the selected-period market average.
        <br><span class="hf-small">This is a descriptive price spread, not retailer margin or profit.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info("Both Market and Retail observations are required to calculate the selected-period spread.")

# If farmgate data is available, show the equivalent Farmgate -> Retail spread.
f_base, f_target, f_spread, f_spread_pct = spread_metrics(df_f, "Farmgate", "Retail")
if pd.notna(f_spread):
    st.markdown("#### Farmgate → Retail Spread")
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Farmgate Price", format_vnd(f_base))
    f2.metric("Retail Price", format_vnd(f_target))
    f3.metric("Retail Spread", format_vnd(f_spread))
    f4.metric("Spread %", format_pct(f_spread_pct))

# Housing-matched spread. Unknown housing is deliberately excluded.
housing_for_spread = df_f[df_f[HOUSING_COL].isin(HOUSING_ORDER)].copy()
if not housing_for_spread.empty:
    housing_levels = (
        housing_for_spread.groupby([HOUSING_COL, PRICE_LEVEL_COL], as_index=False)
        .agg(Average_Price=(PRICE_COL, "mean"), Observations=(PRICE_COL, "size"))
    )
    pivot = housing_levels.pivot(index=HOUSING_COL, columns=PRICE_LEVEL_COL, values="Average_Price")
    counts = housing_levels.pivot(index=HOUSING_COL, columns=PRICE_LEVEL_COL, values="Observations")
    rows = []
    for housing in HOUSING_ORDER:
        if housing not in pivot.index:
            continue
        market = pivot.loc[housing].get("Market", np.nan)
        farmgate = pivot.loc[housing].get("Farmgate", np.nan)
        retail = pivot.loc[housing].get("Retail", np.nan)
        h_spread = retail - market if pd.notna(retail) and pd.notna(market) else np.nan
        h_pct = h_spread / market * 100 if pd.notna(h_spread) and market else np.nan
        rows.append({
            "Housing System": housing,
            "Market (VND/egg)": market,
            "Farmgate (VND/egg)": farmgate,
            "Retail (VND/egg)": retail,
            "Market → Retail Spread": h_spread,
            "Spread %": h_pct,
            "Retail Observations": counts.loc[housing].get("Retail", 0) if housing in counts.index else 0,
        })
    if rows:
        spread_table = pd.DataFrame(rows)
        st.markdown("#### Spread by Housing System")
        st.dataframe(
            spread_table.style.format({
                "Market (VND/egg)": "₫{:,.0f}",
                "Farmgate (VND/egg)": "₫{:,.0f}",
                "Retail (VND/egg)": "₫{:,.0f}",
                "Market → Retail Spread": "₫{:,.0f}",
                "Spread %": "{:+,.1f}%",
                "Retail Observations": "{:,.0f}",
            }, na_rep="—"),
            use_container_width=True,
            hide_index=True,
        )

st.divider()

# ============================================================
# 3. CAGED VS CAGE-FREE VS FREE-RANGE
# ============================================================
st.subheader("3) Housing-System Price Comparison")
known_housing = df_f[df_f[HOUSING_COL].isin(HOUSING_ORDER)].copy()

if known_housing.empty:
    st.info("No correctly labeled Caged, Cage-Free or Free-Range observations are available for the current filters.")
else:
    housing_level_summary = (
        known_housing.groupby([HOUSING_COL, PRICE_LEVEL_COL], as_index=False)
        .agg(Average_Price=(PRICE_COL, "mean"), Observations=(PRICE_COL, "size"))
    )
    fig_housing = px.bar(
        housing_level_summary,
        x=HOUSING_COL,
        y="Average_Price",
        color=PRICE_LEVEL_COL,
        barmode="group",
        color_discrete_map=LEVEL_COLOR_MAP,
        text=housing_level_summary["Average_Price"].round(0),
        category_orders={HOUSING_COL: HOUSING_ORDER, PRICE_LEVEL_COL: LEVEL_ORDER},
        title="Average Price by Housing System and Price Level",
    )
    fig_housing.update_traces(texttemplate="₫%{text:,.0f}", textposition="outside")
    fig_housing.update_layout(
        xaxis_title="Housing system", yaxis_title="Average price per egg (VND)", yaxis_tickformat=","
    )
    style_figure(fig_housing, "Price level")
    st.plotly_chart(fig_housing, use_container_width=True)

    retail_known = known_housing[known_housing[PRICE_LEVEL_COL] == "Retail"].copy()
    if not retail_known.empty:
        retail_housing = (
            retail_known.groupby(HOUSING_COL, as_index=False)
            .agg(Average_Retail_Price=(PRICE_COL, "mean"), Observations=(PRICE_COL, "size"))
        )
        retail_housing[HOUSING_COL] = pd.Categorical(
            retail_housing[HOUSING_COL], categories=HOUSING_ORDER, ordered=True
        )
        retail_housing = retail_housing.sort_values(HOUSING_COL)

        st.markdown("#### Retail: Is Cage-Free / Free-Range More Expensive?")
        caged_rows = retail_housing[retail_housing[HOUSING_COL] == "Caged"]
        caged_avg = caged_rows["Average_Retail_Price"].iloc[0] if not caged_rows.empty else np.nan

        # Client-facing headline premium: only appears when the comparison category exists.
        cage_free_rows = retail_housing[retail_housing[HOUSING_COL] == "Cage-Free"]
        free_range_rows = retail_housing[retail_housing[HOUSING_COL] == "Free-Range"]
        if pd.notna(caged_avg) and not cage_free_rows.empty:
            cf_avg = cage_free_rows["Average_Retail_Price"].iloc[0]
            cf_diff = cf_avg - caged_avg
            cf_pct = cf_diff / caged_avg * 100 if caged_avg else np.nan
            p1, p2, p3 = st.columns(3)
            p1.metric("Caged Retail", format_vnd(caged_avg))
            p2.metric("Cage-Free Retail", format_vnd(cf_avg))
            p3.metric("Cage-Free Premium vs Caged", format_vnd(cf_diff), format_pct(cf_pct))
        if pd.notna(caged_avg) and not free_range_rows.empty:
            fr_avg = free_range_rows["Average_Retail_Price"].iloc[0]
            fr_diff = fr_avg - caged_avg
            fr_pct = fr_diff / caged_avg * 100 if caged_avg else np.nan
            p1, p2, p3 = st.columns(3)
            p1.metric("Caged Retail", format_vnd(caged_avg))
            p2.metric("Free-Range Retail", format_vnd(fr_avg))
            p3.metric("Free-Range Premium vs Caged", format_vnd(fr_diff), format_pct(fr_pct))

        comparison_rows = []
        for _, row in retail_housing.iterrows():
            housing = str(row[HOUSING_COL])
            avg = row["Average_Retail_Price"]
            if pd.notna(caged_avg) and caged_avg != 0:
                diff = avg - caged_avg
                pct = diff / caged_avg * 100
            else:
                diff = pct = np.nan
            comparison_rows.append({
                "Housing System": housing,
                "Avg Retail Price (VND/egg)": avg,
                "Difference vs Caged": diff,
                "% vs Caged": pct,
                "Observations": row["Observations"],
            })

        comparison = pd.DataFrame(comparison_rows)
        st.dataframe(
            comparison.style.format({
                "Avg Retail Price (VND/egg)": "₫{:,.0f}",
                "Difference vs Caged": "{:+,.0f}",
                "% vs Caged": "{:+,.1f}%",
                "Observations": "{:,.0f}",
            }, na_rep="—"),
            use_container_width=True,
            hide_index=True,
        )

        sparse = comparison[(comparison["Housing System"] != "Caged") & (comparison["Observations"] < 5)]
        if not sparse.empty:
            st.warning(
                "One or more non-caged categories have fewer than 5 observations in the current filters. "
                "Treat the price difference as directional until more Cage-Free / Free-Range data is collected."
            )

        non_caged_present = set(retail_housing[HOUSING_COL].astype(str)) - {"Caged"}
        if not non_caged_present:
            st.info(
                "No Cage-Free or explicit Free-Range retail observations are available yet for the current filters. "
                "They will appear automatically when the new retail scrapers add correctly labeled records."
            )

st.divider()

# ============================================================
# 4. PRICE TREND
# ============================================================
st.subheader("4) Price Trend")
trend_mode = st.radio("Trend view", ["Price Level", "Housing System"], horizontal=True)
rule, aggregation_label = aggregation_rule(df_f)

if trend_mode == "Price Level":
    trend = (
        df_f.set_index(DATE_COL)
        .groupby(PRICE_LEVEL_COL)[PRICE_COL]
        .resample(rule).mean().reset_index().dropna(subset=[PRICE_COL])
    )
    if trend.empty:
        st.info("Not enough observations to build the price-level trend.")
    else:
        fig_trend = px.line(
            trend, x=DATE_COL, y=PRICE_COL, color=PRICE_LEVEL_COL, markers=True,
            color_discrete_map=LEVEL_COLOR_MAP,
            category_orders={PRICE_LEVEL_COL: LEVEL_ORDER},
            title=f"{aggregation_label} Price Trend by Price Level",
        )
        fig_trend.update_layout(xaxis_title="Date", yaxis_title="Average price per egg (VND)", yaxis_tickformat=",")
        style_figure(fig_trend, "Price level")
        st.plotly_chart(fig_trend, use_container_width=True)
else:
    trend_scope = df_f[df_f[HOUSING_COL].isin(HOUSING_ORDER)].copy()
    trend = (
        trend_scope.set_index(DATE_COL)
        .groupby(HOUSING_COL)[PRICE_COL]
        .resample(rule).mean().reset_index().dropna(subset=[PRICE_COL])
    ) if not trend_scope.empty else pd.DataFrame()
    if trend.empty:
        st.info("No correctly labeled housing-system data is available for this trend.")
    else:
        fig_trend = px.line(
            trend, x=DATE_COL, y=PRICE_COL, color=HOUSING_COL, markers=True,
            color_discrete_map=HOUSING_COLOR_MAP,
            category_orders={HOUSING_COL: HOUSING_ORDER},
            title=f"{aggregation_label} Price Trend by Housing System",
        )
        fig_trend.update_layout(xaxis_title="Date", yaxis_title="Average price per egg (VND)", yaxis_tickformat=",")
        style_figure(fig_trend, "Housing system")
        st.plotly_chart(fig_trend, use_container_width=True)

st.divider()

# ============================================================
# 5. REGIONAL COMPARISON
# ============================================================
st.subheader("5) Regional Price Comparison")
region_data = df_f[df_f[REGION_COL].notna()].copy()
if region_data.empty:
    st.info("No regional data is available for the current filters.")
else:
    region_summary = (
        region_data.groupby([REGION_COL, PRICE_LEVEL_COL], as_index=False)[PRICE_COL].mean()
    )
    fig_region = px.bar(
        region_summary, x=REGION_COL, y=PRICE_COL, color=PRICE_LEVEL_COL,
        barmode="group", color_discrete_map=LEVEL_COLOR_MAP,
        category_orders={PRICE_LEVEL_COL: LEVEL_ORDER},
        title="Average Price by Region and Price Level",
    )
    fig_region.update_layout(xaxis_title="Region", yaxis_title="Average price per egg (VND)", yaxis_tickformat=",")
    style_figure(fig_region, "Price level")
    st.plotly_chart(fig_region, use_container_width=True)

st.divider()

# ============================================================
# 6. RETAIL PRODUCT INTELLIGENCE
# ============================================================
retail = df_f[df_f[PRICE_LEVEL_COL] == "Retail"].copy()
st.subheader("6) Retail Product Intelligence")
if retail.empty:
    st.info("No retail observations are available for the current filters.")
else:
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Avg Retail Price / Egg", format_vnd(retail[PRICE_COL].mean()))
    r2.metric("Retail Observations", f"{len(retail):,}")
    r3.metric("Brands", f"{retail[BRAND_COL].nunique(dropna=True):,}" if BRAND_COL in retail else "—")
    r4.metric("Stores", f"{retail[STORE_COL].nunique(dropna=True):,}" if STORE_COL in retail else "—")

    if BRAND_COL in retail.columns and retail[BRAND_COL].notna().any():
        brand_summary = (
            retail.dropna(subset=[BRAND_COL])
            .groupby([BRAND_COL, HOUSING_COL], as_index=False)
            .agg(Average_Price=(PRICE_COL, "mean"), Observations=(PRICE_COL, "size"))
            .sort_values("Observations", ascending=False).head(20)
        )
        fig_brand = px.bar(
            brand_summary, x="Average_Price", y=BRAND_COL, color=HOUSING_COL,
            orientation="h", color_discrete_map=HOUSING_COLOR_MAP,
            title="Retail Price by Brand and Housing System",
        )
        fig_brand.update_layout(xaxis_title="Average price per egg (VND)", yaxis_title="Brand", xaxis_tickformat=",")
        style_figure(fig_brand, "Housing system")
        st.plotly_chart(fig_brand, use_container_width=True)

st.divider()

# ============================================================
# DATA COVERAGE / QA
# ============================================================
st.subheader("Data Coverage")
left, right = st.columns(2)
with left:
    source_summary = (
        df_f.groupby(SOURCE_COL, as_index=False)
        .agg(Observations=(PRICE_COL, "size"))
        .sort_values("Observations", ascending=False).head(15)
    )
    fig_source = px.bar(
        source_summary.sort_values("Observations"), x="Observations", y=SOURCE_COL,
        orientation="h", color_discrete_sequence=[HF_NAVY], title="Observations by Source"
    )
    style_figure(fig_source)
    st.plotly_chart(fig_source, use_container_width=True)

with right:
    housing_counts = (
        df_f.groupby(HOUSING_COL, as_index=False)
        .agg(Observations=(PRICE_COL, "size"))
        .sort_values("Observations", ascending=False)
    )
    fig_housing_count = px.bar(
        housing_counts, x=HOUSING_COL, y="Observations", color=HOUSING_COL,
        color_discrete_map={**HOUSING_COLOR_MAP, "Unknown": HF_MUTED},
        category_orders={HOUSING_COL: HOUSING_ORDER + ["Unknown"]},
        title="Housing-Label Coverage",
    )
    fig_housing_count.update_layout(showlegend=False, xaxis_title="Housing system", yaxis_title="Observations")
    style_figure(fig_housing_count)
    st.plotly_chart(fig_housing_count, use_container_width=True)

known_count = int(df_f[HOUSING_COL].isin(HOUSING_ORDER).sum())
unknown_count = int(df_f[HOUSING_COL].eq("Unknown").sum())
st.caption(
    f"Housing-labeled observations in current filters: {known_count:,}. "
    f"Unclassified/Unknown: {unknown_count:,}. Unknown rows are excluded from housing-system comparisons."
)

with st.expander("View filtered data"):
    display_cols = [
        DATE_COL, PRICE_LEVEL_COL, SOURCE_COL, REGION_COL, PROVINCE_COL,
        EGG_TYPE_COL, HOUSING_COL, HOUSING_STATUS_COL, BRAND_COL, PRODUCT_COL,
        STORE_COL, EGG_COUNT_COL, PACK_PRICE_COL, PRICE_COL, QUALITY_COL,
    ]
    display_cols = [c for c in display_cols if c in df_f.columns]
    st.dataframe(
        df_f[display_cols].sort_values(DATE_COL, ascending=False),
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "Dashboard prices use the cleaned VND/egg field. Market, Farmgate and Retail are never blended into one supply-chain price. "
    "Housing comparisons only use Caged, Cage-Free and explicitly identified Free-Range records."
)
