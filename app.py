import io
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------------
# PAGE SETUP
# ------------------------------------------------------------------
st.set_page_config(page_title="Easy Data Analyzer", page_icon="📊", layout="wide")

APP_NAME = "📊 Easy Data Analyzer"          # <- change to your own app name
CONTACT = "WhatsApp: +91 XXXXXXXXXX"        # <- put your number here


# ------------------------------------------------------------------
# OPTIONAL ACCESS CODE (for paid customers). Only active if you set
# ACCESS_CODE in Streamlit "Secrets". Otherwise the app is open.
# ------------------------------------------------------------------
def check_access():
    try:
        code = st.secrets.get("ACCESS_CODE", None)
    except Exception:
        code = None
    if not code:
        return
    entered = st.sidebar.text_input("🔑 Access code", type="password")
    if entered != code:
        st.title(APP_NAME)
        st.info(f"Please enter your access code in the left panel.\n\nNo code yet? Contact: {CONTACT}")
        st.stop()


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def fmt(x):
    if pd.isna(x):
        return "-"
    if abs(x) >= 100:
        return f"{x:,.0f}"
    return f"{x:,.2f}"


def sample_data():
    rng = np.random.default_rng(42)
    n = 400
    dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    products = rng.choice(["Rice", "Oil", "Sugar", "Soap", "Tea", "Biscuits", "Milk"], n)
    city = rng.choice(["Tirupati", "Chittoor", "Nellore"], n)
    qty = rng.integers(1, 20, n)
    price = pd.Series(products).map(
        {"Rice": 60, "Oil": 140, "Sugar": 45, "Soap": 35, "Tea": 90, "Biscuits": 20, "Milk": 30}
    ).values
    df = pd.DataFrame(
        {"Date": dates, "Product": products, "City": city,
         "Quantity": qty, "Price": price, "Sales": qty * price}
    ).sort_values("Date")
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def read_file(file_bytes, filename, sheet=None):
    name = filename.lower()
    if name.endswith(".csv"):
        for enc in ("utf-8", "latin-1"):
            try:
                return pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
            except Exception:
                continue
        raise ValueError("Could not read this CSV file.")
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet or 0)


def sheet_names(file_bytes):
    return pd.ExcelFile(io.BytesIO(file_bytes)).sheet_names


def clean_basic(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all").dropna(axis=1, how="all")
    # make duplicate column names unique
    seen = {}
    new_cols = []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            new_cols.append(c)
    df.columns = new_cols
    # try to detect date columns stored as text
    for c in df.columns:
        if df[c].dtype == object:
            sample = df[c].dropna().astype(str).head(50)
            if len(sample) == 0:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(sample, errors="coerce", dayfirst=True)
            looks_like_date = any(k in c.lower() for k in ["date", "time", "day", "month", "year"])
            if parsed.notna().mean() > 0.8 and (looks_like_date or sample.str.contains(r"[-/]").mean() > 0.8):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
    return df


def column_types(df):
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    date = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    cat = [c for c in df.columns
           if c not in num and c not in date and 2 <= df[c].nunique() <= 50]
    return num, cat, date


def make_insights(df, num, cat, date):
    tips = []
    tips.append(f"Your file has **{len(df):,} rows** and **{len(df.columns)} columns**.")
    missing = int(df.isna().sum().sum())
    if missing:
        tips.append(f"⚠️ There are **{missing:,} empty cells**. You can fix them in the *Clean Data* tab.")
    dups = int(df.duplicated().sum())
    if dups:
        tips.append(f"⚠️ There are **{dups:,} duplicate rows**. You can remove them in the *Clean Data* tab.")
    if num:
        main = num[-1] if len(num) > 1 else num[0]
        tips.append(f"Total **{main}** is **{fmt(df[main].sum())}**, the average is **{fmt(df[main].mean())}**.")
        tips.append(f"Highest **{main}** is **{fmt(df[main].max())}**, lowest is **{fmt(df[main].min())}**.")
        if cat:
            c = cat[0]
            g = df.groupby(c)[main].sum().sort_values(ascending=False)
            if len(g) >= 2 and g.sum() != 0:
                share = g.iloc[0] / g.sum() * 100
                tips.append(f"Best **{c}** is **{g.index[0]}** with {fmt(g.iloc[0])} ({share:.0f}% of total).")
                tips.append(f"Lowest **{c}** is **{g.index[-1]}** with {fmt(g.iloc[-1])}.")
        if date:
            d = date[0]
            m = df.dropna(subset=[d]).groupby(df[d].dt.to_period("M"))[main].sum()
            if len(m) >= 2:
                best, worst = m.idxmax(), m.idxmin()
                tips.append(f"Best month: **{best}** ({fmt(m.max())}). Weakest month: **{worst}** ({fmt(m.min())}).")
                if m.iloc[0] != 0:
                    change = (m.iloc[-1] - m.iloc[0]) / abs(m.iloc[0]) * 100
                    word = "grew" if change >= 0 else "fell"
                    tips.append(f"From the first to the last month, **{main}** {word} by **{abs(change):.0f}%**.")
    return tips


def build_excel(df, num):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Your Data", index=False)
        if num:
            df[num].describe().T.round(2).to_excel(writer, sheet_name="Summary")
        miss = df.isna().sum().reset_index()
        miss.columns = ["Column", "Empty cells"]
        miss.to_excel(writer, sheet_name="Empty Cells", index=False)
    return buf.getvalue()


# ------------------------------------------------------------------
# APP
# ------------------------------------------------------------------
check_access()

st.title(APP_NAME)
st.caption("Upload your Excel or CSV file. Get a clear report in seconds. No technical skills needed.")

with st.sidebar:
    st.header("How to use")
    st.markdown(
        "**1.** Upload your file below\n\n"
        "**2.** Look at the Summary and Charts\n\n"
        "**3.** Download your report"
    )
    uploaded = st.file_uploader("📁 Upload Excel or CSV", type=["csv", "xlsx", "xls"])
    use_sample = st.button("✨ Try with sample data")
    st.divider()
    st.caption("🔒 Your file is used only to make this report. It is not saved.")
    st.caption(f"Need help? {CONTACT}")

if use_sample:
    st.session_state["sample"] = True
if uploaded:
    st.session_state["sample"] = False

df = None
if uploaded:
    try:
        data = uploaded.getvalue()
        sheet = None
        if uploaded.name.lower().endswith((".xlsx", ".xls")):
            sheets = sheet_names(data)
            if len(sheets) > 1:
                sheet = st.sidebar.selectbox("Which sheet?", sheets)
        df = clean_basic(read_file(data, uploaded.name, sheet))
    except Exception:
        st.error("Sorry, we could not read this file. Please check it is a normal Excel or CSV file and try again.")
        st.stop()
elif st.session_state.get("sample"):
    df = clean_basic(sample_data())
    st.info("You are looking at sample data. Upload your own file in the left panel.")

if df is None:
    st.markdown("### 👋 Welcome!")
    st.markdown(
        "This tool turns your messy sheet into **simple numbers, charts and tips**.\n\n"
        "- Works for sales, expenses, marks, attendance, stock, surveys and more\n"
        "- Just upload a file in the **left panel** to start\n"
        "- Or click **Try with sample data** to see how it works"
    )
    st.stop()

if df.empty:
    st.error("This file has no data. Please upload another file.")
    st.stop()

# keep a working copy that the user can clean
if "work_sig" not in st.session_state or st.session_state["work_sig"] != (uploaded.name if uploaded else "sample", df.shape):
    st.session_state["work_sig"] = (uploaded.name if uploaded else "sample", df.shape)
    st.session_state["work_df"] = df
df = st.session_state["work_df"]

num, cat, date = column_types(df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", len(df.columns))
c3.metric("Empty cells", f"{int(df.isna().sum().sum()):,}")
c4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["💡 Quick Tips", "📈 Auto Charts", "🛠️ Make Your Own Chart", "🧹 Clean Data", "⬇️ Download"]
)

# ---------------- TAB 1: INSIGHTS ----------------
with tab1:
    st.subheader("What your data is telling you")
    for tip in make_insights(df, num, cat, date):
        st.markdown(f"- {tip}")
    if num:
        st.subheader("Number summary")
        st.dataframe(df[num].describe().T.round(2), use_container_width=True)
    with st.expander("👀 See my data"):
        st.dataframe(df.head(200), use_container_width=True)

# ---------------- TAB 2: AUTO CHARTS ----------------
with tab2:
    if not num:
        st.warning("No number columns found, so charts are limited. Make sure your numbers are plain numbers (no text like 'Rs.').")
    else:
        measure = st.selectbox("Which number do you want to study?", num, index=len(num) - 1)
        if cat:
            col = st.selectbox("Group by", cat)
            g = df.groupby(col)[measure].sum().sort_values(ascending=False).head(10).reset_index()
            st.markdown(f"**Top 10 {col} by {measure}**")
            st.plotly_chart(px.bar(g, x=col, y=measure, text_auto=".2s"), use_container_width=True)
            if df[col].nunique() <= 8:
                st.markdown(f"**Share of each {col}**")
                st.plotly_chart(px.pie(g, names=col, values=measure), use_container_width=True)
        if date:
            d = date[0]
            t = df.dropna(subset=[d]).copy()
            t["Month"] = t[d].dt.to_period("M").dt.to_timestamp()
            m = t.groupby("Month")[measure].sum().reset_index()
            st.markdown(f"**{measure} month by month**")
            st.plotly_chart(px.line(m, x="Month", y=measure, markers=True), use_container_width=True)
        st.markdown(f"**How {measure} is spread**")
        st.plotly_chart(px.histogram(df, x=measure), use_container_width=True)

# ---------------- TAB 3: CUSTOM CHART ----------------
with tab3:
    st.write("Choose what you want to see. The chart builds itself.")
    kind = st.radio("Chart type", ["Bar", "Line", "Pie", "Scatter (compare two numbers)"], horizontal=True)
    all_cols = list(df.columns)
    try:
        if kind.startswith("Scatter"):
            if len(num) < 2:
                st.warning("You need at least two number columns for this chart.")
            else:
                x = st.selectbox("Horizontal (X)", num, key="sx")
                y = st.selectbox("Vertical (Y)", num, index=1, key="sy")
                st.plotly_chart(px.scatter(df, x=x, y=y), use_container_width=True)
        else:
            x = st.selectbox("Group by / Names", all_cols, key="cx")
            y_opts = ["Count of rows"] + num
            y = st.selectbox("What to measure", y_opts, key="cy")
            how = st.selectbox("How to combine", ["Sum", "Average", "Highest", "Lowest"], key="how") if y != "Count of rows" else None
            if y == "Count of rows":
                g = df.groupby(x).size().reset_index(name="Count")
                yy = "Count"
            else:
                fn = {"Sum": "sum", "Average": "mean", "Highest": "max", "Lowest": "min"}[how]
                g = df.groupby(x)[y].agg(fn).reset_index()
                yy = y
            g = g.sort_values(yy, ascending=False).head(30)
            if kind == "Bar":
                st.plotly_chart(px.bar(g, x=x, y=yy, text_auto=".2s"), use_container_width=True)
            elif kind == "Line":
                st.plotly_chart(px.line(g.sort_values(x), x=x, y=yy, markers=True), use_container_width=True)
            else:
                st.plotly_chart(px.pie(g.head(10), names=x, values=yy), use_container_width=True)
    except Exception:
        st.warning("This combination could not be drawn. Please try different choices.")

# ---------------- TAB 4: CLEAN ----------------
with tab4:
    st.write("Fix common problems in one click. Your original file is never changed.")
    colA, colB = st.columns(2)
    with colA:
        if st.button("Remove duplicate rows"):
            before = len(df)
            st.session_state["work_df"] = df.drop_duplicates()
            st.success(f"Removed {before - len(st.session_state['work_df'])} rows.")
            st.rerun()
    with colB:
        if st.button("Remove rows with empty cells"):
            before = len(df)
            st.session_state["work_df"] = df.dropna()
            st.success(f"Removed {before - len(st.session_state['work_df'])} rows.")
            st.rerun()
    if num and st.button("Fill empty number cells with the average"):
        d2 = df.copy()
        d2[num] = d2[num].fillna(d2[num].mean())
        st.session_state["work_df"] = d2
        st.success("Done.")
        st.rerun()
    st.markdown("**Empty cells per column**")
    miss = df.isna().sum().reset_index()
    miss.columns = ["Column", "Empty cells"]
    st.dataframe(miss, use_container_width=True)

# ---------------- TAB 5: DOWNLOAD ----------------
with tab5:
    st.write("Take your report with you.")
    st.download_button(
        "⬇️ Download Excel report",
        data=build_excel(df, num),
        file_name="my_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.download_button(
        "⬇️ Download cleaned data (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_data.csv",
        mime="text/csv",
    )
    tips_text = "\n".join(t.replace("**", "") for t in make_insights(df, num, cat, date))
    st.download_button("⬇️ Download quick tips (text)", data=tips_text, file_name="quick_tips.txt")
