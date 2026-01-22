import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="GOES Flare Explorer", layout="wide")

st.title("GOES Flare Explorer")
st.caption("Visualize detected flare events from your GOES XRS-derived table.")

# -----------------------------
# USER PATH (your file)
# -----------------------------
DATA_PATH = "goes_flares_processed.csv"


# -----------------------------
# Helpers
# -----------------------------
CLASS_ORDER = ["A", "B", "C", "M", "X"]

def parse_goes_class(s: str):
    """
    "C1.9" -> ("C", 1.9)
    "B8"   -> ("B", 8.0)
    """
    if pd.isna(s):
        return (None, np.nan)
    s = str(s).strip().upper()
    m = re.match(r"^([ABCMX])\s*([0-9]*\.?[0-9]+)?$", s)
    if not m:
        return (None, np.nan)
    letter = m.group(1)
    mag = float(m.group(2)) if m.group(2) is not None else np.nan
    return (letter, mag)

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # parse datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # numeric columns
    for c in ["secs_since_2000","GoesNr","Flux","Flare","IntegratedFlux","BackgroundFlux"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # parse class
    letter_mag = df["Class"].apply(parse_goes_class) if "Class" in df.columns else [(None, np.nan)] * len(df)
    df["ClassLetter"] = [lm[0] for lm in letter_mag]
    df["ClassMag"] = [lm[1] for lm in letter_mag]

    return df.dropna(subset=["Date"]).sort_values("Date")


# -----------------------------
# Load data
# -----------------------------
try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Could not load data from: {DATA_PATH}")
    st.exception(e)
    st.stop()

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Data")
st.sidebar.write(f"Loaded: `{DATA_PATH}`")
st.sidebar.write(f"Rows: {len(df):,}")

min_date, max_date = df["Date"].min(), df["Date"].max()

st.sidebar.header("Filters")

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date()
)

start = pd.to_datetime(date_range[0])
end = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

available_classes = [c for c in CLASS_ORDER if c in set(df["ClassLetter"].dropna())]
class_choices = st.sidebar.multiselect(
    "Flare classes",
    options=available_classes,
    default=available_classes
)

min_flux_default = float(np.nanmin(df["Flux"])) if df["Flux"].notna().any() else 0.0
min_flux = st.sidebar.number_input(
    "Minimum peak flux (W/m²)",
    value=min_flux_default,
    format="%.3e"
)

only_peaks = st.sidebar.checkbox("Only EVENT_PEAK rows", value=True)

# Apply filters
f = df.copy()
f = f[(f["Date"] >= start) & (f["Date"] <= end)]

if class_choices:
    f = f[f["ClassLetter"].isin(class_choices)]

if "Flux" in f.columns:
    f = f[f["Flux"] >= min_flux]

if only_peaks and "Status" in f.columns:
    f = f[f["Status"].astype(str).str.upper().eq("EVENT_PEAK")]

# -----------------------------
# Top KPIs
# -----------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Events", f.shape[0])
k2.metric("Max peak flux", f"{f['Flux'].max():.3e}" if f["Flux"].notna().any() else "—")
k3.metric("Top class", f"{f['ClassLetter'].dropna().max()}" if f["ClassLetter"].notna().any() else "—")
k4.metric("Max integrated", f"{f['IntegratedFlux'].max():.3e}" if f["IntegratedFlux"].notna().any() else "—")

st.divider()

# -----------------------------
# Charts + table
# -----------------------------
left, right = st.columns([2, 1])

with left:
    st.subheader("Peak Flux over Time (log scale)")
    if f.empty:
        st.warning("No rows match the current filters.")
    else:
        fig = px.scatter(
            f,
            x="Date",
            y="Flux",
            color="ClassLetter",
            hover_data=["Class", "IntegratedFlux", "BackgroundFlux", "GoesNr", "secs_since_2000"],
        )
        fig.update_yaxes(type="log", title="Flux (W/m²)")
        fig.update_xaxes(title="Date")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Flux vs Background (ratio)")
    if (not f.empty) and ("BackgroundFlux" in f.columns) and f["BackgroundFlux"].notna().any():
        tmp = f.copy()
        tmp["Flux_to_Background"] = tmp["Flux"] / tmp["BackgroundFlux"].replace(0, np.nan)
        fig2 = px.scatter(
            tmp,
            x="Date",
            y="Flux_to_Background",
            color="ClassLetter",
            hover_data=["Class", "Flux", "BackgroundFlux"]
        )
        fig2.update_yaxes(type="log", title="Flux / Background (log)")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.caption("BackgroundFlux missing/NA in filtered data.")

with right:
    st.subheader("Class distribution")
    if not f.empty:
        counts = (
            f["ClassLetter"]
            .value_counts()
            .reindex(CLASS_ORDER)
            .dropna()
            .reset_index()
        )
        counts.columns = ["Class", "Count"]
        fig3 = px.bar(counts, x="Class", y="Count")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.caption("No data to show.")

    st.subheader("Events table")
    cols = ["Date","secs_since_2000","GoesNr","Flux","Status","Class","IntegratedFlux","BackgroundFlux"]
    cols = [c for c in cols if c in f.columns]
    st.dataframe(
        f.sort_values("Date", ascending=False)[cols],
        use_container_width=True,
        height=420
    )

st.divider()

# -----------------------------
# Event drilldown
# -----------------------------
st.subheader("Event drilldown")
if f.empty:
    st.caption("No events to drill into.")
else:
    options = f.sort_values("Date", ascending=False).reset_index(drop=True)
    label = options.apply(
        lambda r: f"{r['Date']} — {r.get('Class','')} — {r['Flux']:.2e}",
        axis=1
    )

    idx = st.selectbox(
        "Select an event",
        options=range(len(options)),
        format_func=lambda i: label[i]
    )

    ev = options.loc[idx]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Date", str(ev["Date"]))
    c2.metric("Class", str(ev.get("Class", "—")))
    c3.metric("Peak Flux", f"{ev['Flux']:.3e}" if pd.notna(ev["Flux"]) else "—")
    c4.metric("Background", f"{ev['BackgroundFlux']:.3e}" if pd.notna(ev.get("BackgroundFlux", np.nan)) else "—")

    st.write("Row data:")
    st.json(ev.to_dict())
