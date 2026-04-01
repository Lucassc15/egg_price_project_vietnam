import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =============================
# Page / Branding
# =============================
st.set_page_config(page_title="Egg Price Dashboard — Vietnam (Farmers)", layout="wide")

st.title("Egg Price Dashboard — Vietnam")
st.caption(
    "Farmer-friendly signals: compare average prices, see short-term direction, and understand price stability."
)

DATA_PATH = "database.csv"  # keep the CSV in the same folder as this app

# =============================
# Data Load & Cleaning
# ✅ DO NOT drop rows except missing price
# ✅ Fill missing egg_type as "General egg prices"
# ✅ Fix price scale (kVND -> VND) automatically, always output VND
# =============================
@st.cache_data
def load_and_clean(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)

    # Intended schema (16)
    intended_cols = [
        "blank", "date", "unit", "buying_price_vnd", "selling_price_vnd",
        "quantity_sold", "market", "region", "egg_type", "feed_cost_vnd",
        "buyer_type", "weather", "event_impact", "source", "notes", "extra"
    ]

    # Align columns by count (robust to 15/16/etc.)
    df = raw.copy()
    if df.shape[1] == len(intended_cols):
        df.columns = intended_cols
    elif df.shape[1] < len(intended_cols):
        df.columns = intended_cols[: df.shape[1]]
    else:
        df.columns = intended_cols + [f"extra_{i}" for i in range(df.shape[1] - len(intended_cols))]

    # Remove “internal header row” ONLY if first row looks like headers
    if len(df) > 0:
        try:
            first_row = df.iloc[0]
            header_like_tokens = (
                "date", "region", "selling", "selling_price",
                "selling_price_vnd", "market", "egg"
            )

            first_row_strings = [str(cell).strip().lower() for cell in first_row.tolist()]

            is_header = any(
                any(tok in cell for tok in header_like_tokens)
                for cell in first_row_strings
            )

            if is_header:
                df = df.iloc[1:].copy()

        except Exception:
            # If anything goes wrong, just skip header detection safely
            pass

    # Convert date (keep rows even if date is missing)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Convert numerics
    for col in ["selling_price_vnd", "feed_cost_vnd", "quantity_sold", "buying_price_vnd"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalize text columns
    text_cols = ["market", "region", "egg_type", "buyer_type", "weather", "event_impact", "unit"]
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
            df.loc[df[c].str.lower().isin(["nan", "none", ""]), c] = pd.NA

    # Fill missing egg_type
    if "egg_type" not in df.columns:
        df["egg_type"] = "General egg prices"
    else:
        df["egg_type"] = df["egg_type"].fillna("General egg prices")

    # Price scale fix -> ALWAYS produce VND in ⁠ price_vnd ⁠
    if "selling_price_vnd" not in df.columns:
        df["price_vnd"] = np.nan
    else:
        price_med = df["selling_price_vnd"].dropna().median()
        if pd.isna(price_med):
            df["price_vnd"] = np.nan
        elif price_med < 100:  # likely kVND (e.g., 3.65)
            df["price_vnd"] = df["selling_price_vnd"] * 1000
        else:  # already VND (e.g., 3490)
            df["price_vnd"] = df["selling_price_vnd"]

    # Drop ONLY rows missing price
    df = df.dropna(subset=["price_vnd"]).copy()

    # Sort by date if available (does not drop missing dates)
    if "date" in df.columns:
        df = df.sort_values("date")

    return df

df = load_and_clean(DATA_PATH)
PRICE_COL = "price_vnd"  # ✅ ALWAYS VND

# =============================
# Sidebar Filters (simple)
# =============================
st.sidebar.header("Filters")

max_date = df["date"].max() if ("date" in df.columns and df["date"].notna().any()) else pd.NaT

window_choice = st.sidebar.selectbox(
    "Time window",
    ["Last 30 days", "Last 60 days", "Last 90 days", "All time"],
    index=3
)
window_days = {"Last 30 days": 30, "Last 60 days": 60, "Last 90 days": 90, "All time": None}[window_choice]

# ✅ Region = dropdown like Market/Location
regions = ["(All)"] + sorted(df["region"].dropna().unique().tolist()) if "region" in df.columns else ["(All)"]
selected_region = st.sidebar.selectbox("Region", regions, index=0)

with st.sidebar.expander("More filters (optional)"):
    egg_types = ["(All)"] + sorted(df["egg_type"].dropna().unique().tolist()) if "egg_type" in df.columns else ["(All)"]
    egg_type = st.sidebar.selectbox("Egg type", egg_types, index=0)

    markets = ["(All)"]
    if "market" in df.columns:
        markets += sorted(df["market"].dropna().unique().tolist())
    market = st.sidebar.selectbox("Market / Location", markets, index=0)

# =============================
# Apply filters
# =============================
df_f = df.copy()

# Region filter
if selected_region != "(All)" and "region" in df_f.columns:
    df_f = df_f[df_f["region"] == selected_region]

# Time window filter (only applied to rows with valid dates)
if window_days is not None and "date" in df_f.columns and pd.notna(max_date):
    start_date = max_date - pd.Timedelta(days=window_days)
    df_f = df_f[(df_f["date"].isna()) | (df_f["date"] >= start_date)]

# Egg type filter
if egg_type != "(All)" and "egg_type" in df_f.columns:
    df_f = df_f[df_f["egg_type"] == egg_type]

# Market filter
if market != "(All)" and "market" in df_f.columns:
    df_f = df_f[df_f["market"] == market]

if df_f.empty:
    st.warning("No data found for the selected filters.")
    st.stop()

# Sidebar data range (dated rows only)
if "date" in df_f.columns and df_f["date"].notna().any():
    st.sidebar.caption(f"Data range (dated rows): {df_f['date'].min().date()} to {df_f['date'].max().date()}")
else:
    st.sidebar.caption("Data range: no valid dates in current filter.")

# =============================
# Helper functions
# =============================
def direction_from_change(pct: float) -> str:
    if np.isnan(pct):
        return "→ Not enough data"
    if pct > 1.0:
        return "↑ Up"
    if pct < -1.0:
        return "↓ Down"
    return "→ Sideways"

def safe_pct_change(prev: float, last: float) -> float:
    if pd.isna(prev) or pd.isna(last) or prev == 0:
        return np.nan
    return (last - prev) / prev * 100.0

# =============================
# Top summary KPIs
# =============================
c1, c2, c3 = st.columns(3)
overall_mean = df_f[PRICE_COL].mean()
overall_std = df_f[PRICE_COL].std()

c1.metric("Overall Avg Price (VND)", f"{overall_mean:,.0f}")
c2.metric("Overall Volatility (Std, VND)", f"{overall_std:,.0f}" if pd.notna(overall_std) else "—")
c3.metric("Data Points (priced rows)", f"{len(df_f):,}")

st.divider()


# =============================
# 1) Average price comparison by region
# Uses rows WITH region (because chart needs a region axis)
# =============================
st.subheader("1) Average Price Comparison by Region")

df_region = df_f[df_f["region"].notna()].copy() if "region" in df_f.columns else df_f.iloc[0:0]

if df_region.empty:
    st.info("Not enough region data to show the comparison chart.")
else:
    region_means = (
        df_region.groupby("region", as_index=False)[PRICE_COL]
                 .mean()
                 .sort_values(PRICE_COL, ascending=False)
    )

    fig_mean = px.bar(
        region_means,
        x="region",
        y=PRICE_COL,
        text=region_means[PRICE_COL].round(0),
        title="Average Selling Price by Region (VND)"
    )
    fig_mean.update_traces(textposition="outside")
    fig_mean.update_layout(
        xaxis_title="Region",
        yaxis_title="Average price (VND)",
        showlegend=False,
        yaxis_tickformat=","  # ✅ show 2,419 not 2.4
    )
    st.plotly_chart(fig_mean, use_container_width=True)

    if len(region_means) >= 2:
        top = region_means.iloc[0]
        low = region_means.iloc[-1]
        st.info(
            f"**Interpretation:** In the selected period, **{top['region']}** has the highest average price "
            f"({top[PRICE_COL]:,.0f} VND). **{low['region']}** has the lowest average price "
            f"({low[PRICE_COL]:,.0f} VND)."
        )
    else:
        st.info("**Interpretation:** Only one region is available in the current filter.")

st.divider()


# =============================
# 2) Price trend and short-term direction
# Uses rows WITH date AND region (needed for weekly time-series)
# =============================
st.subheader("2) Price Trend and Short-Term Direction")

df_trend = df_f.copy()
if "date" in df_trend.columns:
    df_trend = df_trend[df_trend["date"].notna()]
if "region" in df_trend.columns:
    df_trend = df_trend[df_trend["region"].notna()]

if df_trend.empty:
    st.info("Not enough dated + region data to show trends.")
else:
    weekly = (
        df_trend.set_index("date")
                .groupby("region")[PRICE_COL]
                .resample("W")
                .mean()
                .reset_index()
                .dropna()
    )

    fig_trend = px.line(
        weekly,
        x="date",
        y=PRICE_COL,
        color="region",
        markers=True,
        title="Weekly Average Price Trend by Region (VND)"
    )
    fig_trend.update_layout(
        xaxis_title="Date",
        yaxis_title="Average price (VND)",
        yaxis_tickformat=","
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # Direction table: if a single region is selected, show one row; else show all in current filter
    regions_for_calc = (
        [selected_region]
        if selected_region != "(All)"
        else sorted(weekly["region"].dropna().unique())
    )

    rows = []
    for r in regions_for_calc:
        ts = weekly[weekly["region"] == r].set_index("date")[PRICE_COL].dropna()
        if len(ts) < 4:
            rows.append({"Region": r, "Short-term direction": "→ Not enough data", "Change (%)": np.nan})
            continue

        last2 = ts.tail(2).mean()
        prev2 = ts.iloc[-4:-2].mean()
        pct = safe_pct_change(prev2, last2)
        rows.append({"Region": r, "Short-term direction": direction_from_change(pct), "Change (%)": pct})

    direction_df = pd.DataFrame(rows)
    direction_df["Change (%)"] = direction_df["Change (%)"].round(1)
    st.dataframe(direction_df, use_container_width=True, hide_index=True)

    st.info(
        "**Interpretation:** The direction compares the most recent **2 weeks** to the previous **2 weeks**. "
        "**Up** means prices increased, **Down** means prices decreased, and **Sideways** means prices stayed similar."
    )

st.divider()

# =============================
# 3) Volatility indicator (stability / risk) by region
# (Based on weekly averages)
# =============================
st.subheader("3) Price Volatility Indicator (Stability vs Risk)")

# --- Build weekly series for ALL regions (baseline), using the SAME time window + egg_type + market filters ---
# We use df_trend (already date+region valid) but do NOT filter by selected_region here
df_trend_all = df[df["date"].notna() & df["region"].notna()].copy()

# Apply the SAME window_days / egg_type / market filters to baseline
if window_days is not None and pd.notna(max_date):
    start_date = max_date - pd.Timedelta(days=window_days)
    df_trend_all = df_trend_all[(df_trend_all["date"].isna()) | (df_trend_all["date"] >= start_date)]

if egg_type != "(All)":
    df_trend_all = df_trend_all[df_trend_all["egg_type"] == egg_type]

if market != "(All)" and "market" in df_trend_all.columns:
    df_trend_all = df_trend_all[df_trend_all["market"] == market]

weekly_all = (
    df_trend_all.set_index("date")
                .groupby("region")[PRICE_COL]
                .resample("W")
                .mean()
                .reset_index()
                .dropna()
)

# --- Compute volatility per region (baseline thresholds) ---
vol_all = (
    weekly_all.groupby("region", as_index=False)[PRICE_COL]
              .agg(std="std", mean="mean", count="count")
)

# Baseline thresholds computed from ALL regions (so labels don’t change when user filters regions)
valid_std = vol_all["std"].dropna()
if len(valid_std) >= 3:
    q1 = valid_std.quantile(0.33)
    q2 = valid_std.quantile(0.66)
else:
    # fallback: avoid weird behavior on tiny datasets
    q1 = valid_std.median()
    q2 = valid_std.median()

def vol_label(s):
    if pd.isna(s):
        return "Not enough data"
    if s <= q1:
        return "Stable"
    elif s <= q2:
        return "Moderate"
    return "Risky"

vol_all["Volatility level"] = vol_all["std"].apply(vol_label)

# --- Now show only selected regions in the chart (but with baseline labels) ---
vol_show = vol_all.copy()
if selected_region != "(All)":
    vol_show = vol_show[vol_show["region"] == selected_region]
else:
    # when "(All)", show only regions that exist in the current filtered dataset df_f
    regions_in_scope = sorted(df_f["region"].dropna().unique().tolist())
    vol_show = vol_show[vol_show["region"].isin(regions_in_scope)]

# Last chart (volatility) colors
VOL_COLOR_MAP = {
    "Risky": "#e03131",        # softer red
    "Moderate": "#fab005",     # amber 
    "Stable": "#2f9e44",       # deeper green
    "Not enough data": "#adb5bd"
}

fig_vol = px.bar(
    vol_show.sort_values("std", ascending=False),
    x="region",
    y="std",
    color="Volatility level",
    color_discrete_map=VOL_COLOR_MAP,   # ✅ FIX
    title="Price Volatility by Region (weekly averages, VND)"
)

fig_vol.update_layout(
    xaxis_title="Region",
    yaxis_title="Volatility (Std Dev, VND)",
    yaxis_tickformat=",",
    legend_title_text="Volatility level"
)

st.plotly_chart(fig_vol, use_container_width=True)

# Create a dynamic example from the currently displayed volatility chart
example_std_text = "N/A"
example_region_text = ""

if not vol_show.empty and vol_show["std"].notna().any():
    # If a single region is selected, use that region's value
    if selected_region != "(All)":
        row_example = vol_show.iloc[0]
    else:
        # Otherwise use the first region shown in the chart
        row_example = vol_show.sort_values("std", ascending=False).iloc[0]

    example_std = row_example["std"]
    example_region = row_example["region"]

    if pd.notna(example_std):
        example_std_text = f"{example_std:,.0f}"
        example_region_text = f" for *{example_region}*"

st.info(
    "*Interpretation:*\n"
    "Volatility shows how much prices move up and down. "
    "*Stable* regions change less (lower risk and easier planning), while "
    "*Risky* regions change more from week to week (harder to plan).\n\n"
    f"The volatility value (for example *±{example_std_text} VND*{example_region_text}) "
    f"means prices typically move about *{example_std_text} VND up or down per week* "
    "around the average price.\n\n"
    "Risk labels are calculated using a consistent baseline, so they do *not change* "
    "when you filter regions."
)

st.caption("Note: Trends and volatility rely on available dated observations. Missing weeks/months can affect signals.")