import io
import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import URL

# ------------------------------------------------------------------
# PAGE SETUP
# ------------------------------------------------------------------
st.set_page_config(page_title="Easy Data Analyzer", page_icon="📊", layout="wide")

APP_NAME = "📊 Easy Data Analyzer"          # <- change to your own app name
CONTACT = "WhatsApp: +91 XXXXXXXXXX"        # <- put your number here

COUNT = "Count of rows"
MAXROWS = 200000
AGG = {"Sum": "sum", "Average": "mean", "Max": "max", "Min": "min", "Count": "count"}

S = st.session_state
S.setdefault("sources", {})
S.setdefault("loaded", set())
S.setdefault("clear_n", 0)


# ------------------------------------------------------------------
# OPTIONAL ACCESS CODE (set ACCESS_CODE in Streamlit "Secrets")
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
    try:
        if pd.isna(x):
            return "-"
        if abs(x) >= 100:
            return f"{x:,.0f}"
        return f"{x:,.2f}"
    except Exception:
        return str(x)


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
    sales = pd.DataFrame(
        {"Date": dates, "Product": products, "City": city,
         "Quantity": qty, "Price": price, "Sales": qty * price}
    ).sort_values("Date").reset_index(drop=True)
    info = pd.DataFrame(
        {"Product": ["Rice", "Oil", "Sugar", "Soap", "Tea", "Biscuits", "Milk"],
         "Category": ["Food", "Food", "Food", "Home", "Drinks", "Food", "Drinks"],
         "Cost": [50, 120, 40, 28, 75, 15, 25]}
    )
    return sales, info


def clean_basic(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all").dropna(axis=1, how="all")
    seen, new_cols = {}, []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            new_cols.append(c)
    df.columns = new_cols
    for c in df.columns:
        if df[c].dtype == object:
            sample = df[c].dropna().astype(str).head(50)
            if len(sample) == 0:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(sample, errors="coerce", dayfirst=True)
            looks = any(k in c.lower() for k in ["date", "time", "day", "month", "year"])
            if parsed.notna().mean() > 0.8 and (looks or sample.str.contains(r"[-/]").mean() > 0.8):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
    return df.reset_index(drop=True)


def column_types(df):
    num = [c for c in df.columns
           if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    date = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    cat = [c for c in df.columns
           if c not in num and c not in date and 2 <= df[c].nunique() <= 50]
    return num, cat, date


def read_upload(f):
    data = f.getvalue()
    name = f.name
    out = {}
    if name.lower().endswith(".csv"):
        df = None
        for enc in ("utf-8", "latin-1"):
            try:
                df = pd.read_csv(io.BytesIO(data), encoding=enc)
                break
            except Exception:
                continue
        if df is None:
            raise ValueError("bad csv")
        out[name] = clean_basic(df)
    else:
        xl = pd.ExcelFile(io.BytesIO(data))
        for sh in xl.sheet_names:
            d = xl.parse(sh)
            if d.empty:
                continue
            key = name if len(xl.sheet_names) == 1 else f"{name} - {sh}"
            out[key] = clean_basic(d)
    if not out:
        raise ValueError("empty")
    return out


def make_insights(df, num, cat, date):
    tips = [f"Your data has **{len(df):,} rows** and **{len(df.columns)} columns**."]
    missing = int(df.isna().sum().sum())
    if missing:
        tips.append(f"⚠️ There are **{missing:,} empty cells**. Fix them in the *Data & Combine* tab.")
    dups = int(df.duplicated().sum())
    if dups:
        tips.append(f"⚠️ There are **{dups:,} duplicate rows**. Remove them in the *Data & Combine* tab.")
    if num:
        main = num[-1] if len(num) > 1 else num[0]
        tips.append(f"Total **{main}** is **{fmt(df[main].sum())}**, average is **{fmt(df[main].mean())}**.")
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
                tips.append(f"Best month: **{m.idxmax()}** ({fmt(m.max())}). Weakest month: **{m.idxmin()}** ({fmt(m.min())}).")
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


def agg_table(d, by, measure, agg):
    if measure == COUNT or agg == "Count":
        t = d.groupby(by).size().reset_index(name=COUNT)
        return t, COUNT
    t = d.groupby(by)[measure].agg(AGG[agg]).reset_index()
    return t, measure


def selected_values(key):
    ev = S.get(key)
    try:
        pts = ev["selection"]["points"]
    except Exception:
        return []
    vals = []
    for p in pts:
        v = p.get("x")
        if v is not None:
            vals.append(str(v))
    return vals


def keep_selected(d, col, vals):
    if col and vals:
        return d[d[col].astype(str).isin(vals)]
    return d


def apply_slicers(df, cols, active):
    out = df
    if not cols:
        return out
    boxes = st.columns(min(4, len(cols)))
    for i, c in enumerate(cols):
        with boxes[i % len(boxes)]:
            key = f"sl_{active}_{c}"
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                valid = df[c].dropna()
                if valid.empty:
                    continue
                lo, hi = valid.min().date(), valid.max().date()
                if lo == hi:
                    continue
                r = st.date_input(c, value=(lo, hi), min_value=lo, max_value=hi, key=key)
                if isinstance(r, (list, tuple)) and len(r) == 2 and (r[0] != lo or r[1] != hi):
                    out = out[(out[c] >= pd.Timestamp(r[0])) &
                              (out[c] < pd.Timestamp(r[1]) + pd.Timedelta(days=1))]
            elif pd.api.types.is_numeric_dtype(df[c]):
                valid = df[c].dropna()
                if valid.empty:
                    continue
                lo, hi = float(valid.min()), float(valid.max())
                if lo == hi:
                    continue
                r = st.slider(c, lo, hi, (lo, hi), key=key)
                if r != (lo, hi):
                    out = out[(out[c] >= r[0]) & (out[c] <= r[1])]
            else:
                opts = sorted(df[c].dropna().astype(str).unique().tolist())
                chosen = st.multiselect(c, opts, key=key, placeholder="All")
                if chosen:
                    out = out[out[c].astype(str).isin(chosen)]
    return out


# ------------------------------------------------------------------
# SQL HELPERS
# ------------------------------------------------------------------
def connect_engine(engine):
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        S["engine"] = engine
        S["tables"] = inspect(engine).get_table_names()
        st.success("✅ Connected!")
    except Exception:
        st.error("Could not connect. Please check host, port, name, user and password. "
                 "Also check that your database allows connections from the internet.")


def load_sql(query, name):
    q = query.strip().rstrip(";").strip()
    if not q.lower().startswith(("select", "with")) or ";" in q:
        st.error("Only a single SELECT query is allowed (this keeps your database safe).")
        return
    try:
        with S["engine"].connect() as conn:
            df = pd.read_sql_query(text(q), conn)
        for c in df.columns:
            if df[c].dtype == object:
                try:
                    df[c] = pd.to_numeric(df[c])
                except (ValueError, TypeError):
                    pass
        if df.empty:
            st.warning("The query worked but returned no rows.")
            return
        key = f"SQL: {name}"
        S["sources"][key] = clean_basic(df)
        S["next_active"] = key
        st.success(f"Loaded {len(df):,} rows.")
    except Exception:
        st.error("The query failed. Please check the table name or SQL and try again.")


# ------------------------------------------------------------------
# DASHBOARD
# ------------------------------------------------------------------
def render_dashboard(active):
    df = S["sources"][active]
    num, cat, date = column_types(df)

    with st.expander("⚙️ Customize this dashboard", expanded=False):
        c1, c2, c3 = st.columns(3)
        measures = num + [COUNT] if num else [COUNT]
        measure = c1.selectbox("Number to analyze", measures,
                               index=max(0, len(measures) - 2) if num else 0, key=f"m_{active}")
        agg = c2.selectbox("How to combine", list(AGG.keys()), key=f"a_{active}")
        grain = c3.selectbox("Time view", ["Month", "Day", "Week", "Quarter", "Year"], key=f"g_{active}")
        c4, c5, c6 = st.columns(3)
        cat_opts = ["(none)"] + cat
        cat1 = c4.selectbox("Group 1", cat_opts, index=1 if cat else 0, key=f"c1_{active}")
        cat2 = c5.selectbox("Group 2", cat_opts, index=2 if len(cat) > 1 else 0, key=f"c2_{active}")
        date_opts = ["(none)"] + date
        dcol = c6.selectbox("Date column", date_opts, index=1 if date else 0, key=f"d_{active}")
        filter_opts = cat + date + num
        default_filters = (cat[:2] + date[:1])
        filter_cols = st.multiselect("Filters (slicers) to show", filter_opts,
                                     default=default_filters, key=f"f_{active}")

    cat1 = None if cat1 == "(none)" else cat1
    cat2 = None if cat2 == "(none)" else cat2
    dcol = None if dcol == "(none)" else dcol

    st.markdown("#### 🎛️ Filters")
    base = apply_slicers(df, filter_cols, active)

    n = S["clear_n"]
    k1, k2 = f"x1_{active}_{n}", f"x2_{active}_{n}"
    sel1 = selected_values(k1) if cat1 else []
    sel2 = selected_values(k2) if cat2 else []
    df_all = keep_selected(keep_selected(base, cat1, sel1), cat2, sel2)

    if sel1 or sel2:
        msg = []
        if sel1:
            msg.append(f"{cat1}: {', '.join(sel1)}")
        if sel2:
            msg.append(f"{cat2}: {', '.join(sel2)}")
        cc1, cc2 = st.columns([6, 2])
        cc1.info("👆 Click-filter active → " + " | ".join(msg))
        if cc2.button("❌ Clear click-filters"):
            S["clear_n"] += 1
            st.rerun()
    else:
        st.caption("💡 Tip: click any bar in the bar charts to filter the whole dashboard (like Power BI). Hold Shift to pick more than one.")

    if df_all.empty:
        st.warning("No data matches these filters. Please change the filters.")
        return

    # KPI cards
    if measure == COUNT or agg == "Count":
        main_val = len(df_all)
    else:
        main_val = getattr(df_all[measure], AGG[agg])()
    k = st.columns(4)
    k[0].metric(f"{agg} of {measure}" if measure != COUNT else COUNT, fmt(main_val))
    k[1].metric("Rows shown", f"{len(df_all):,}", f"of {len(df):,}", delta_color="off")
    if measure != COUNT:
        k[2].metric(f"Average {measure}", fmt(df_all[measure].mean()))
        k[3].metric(f"Highest {measure}", fmt(df_all[measure].max()))
    else:
        if cat1:
            k[2].metric(f"Different {cat1}", f"{df_all[cat1].nunique():,}")
        if cat2:
            k[3].metric(f"Different {cat2}", f"{df_all[cat2].nunique():,}")

    label = f"{agg} of {measure}" if measure != COUNT else COUNT

    # Trend
    if dcol:
        freq = {"Day": "D", "Week": "W", "Month": "M", "Quarter": "Q", "Year": "Y"}[grain]
        t = df_all.dropna(subset=[dcol]).copy()
        if not t.empty:
            t["Period"] = t[dcol].dt.to_period(freq).dt.to_timestamp()
            tt, val = agg_table(t, "Period", measure, agg)
            st.markdown(f"#### 📈 {label} by {grain.lower()}")
            fig = px.area(tt, x="Period", y=val, markers=True)
            fig.update_layout(hovermode="x unified", margin=dict(t=10, b=10))
            st.plotly_chart(fig)

    # Two clickable bar charts
    left, right = st.columns(2)
    with left:
        if cat1:
            d1 = keep_selected(base, cat2, sel2)
            g1, val = agg_table(d1, cat1, measure, agg)
            g1 = g1.sort_values(val, ascending=False).head(15)
            st.markdown(f"#### 📊 {label} by {cat1}")
            fig = px.bar(g1, x=cat1, y=val, text_auto=".2s", color_discrete_sequence=["#4C78A8"])
            fig.update_layout(clickmode="event+select", margin=dict(t=10, b=10))
            st.plotly_chart(fig, key=k1, on_select="rerun", selection_mode="points")
        else:
            st.info("No grouping column found for chart 1.")
    with right:
        if cat2:
            d2 = keep_selected(base, cat1, sel1)
            g2, val = agg_table(d2, cat2, measure, agg)
            g2 = g2.sort_values(val, ascending=False).head(15)
            st.markdown(f"#### 📊 {label} by {cat2}")
            fig = px.bar(g2, x=cat2, y=val, text_auto=".2s", color_discrete_sequence=["#F58518"])
            fig.update_layout(clickmode="event+select", margin=dict(t=10, b=10))
            st.plotly_chart(fig, key=k2, on_select="rerun", selection_mode="points")
        else:
            st.info("Choose 'Group 2' in Customize to see a second chart.")

    # Treemap / pie + scatter
    left, right = st.columns(2)
    with left:
        if cat1 and cat2 and cat1 != cat2:
            tm, val = agg_table(df_all, [cat1, cat2], measure, agg)
            tm = tm[tm[val] > 0]
            if not tm.empty:
                st.markdown(f"#### 🧩 Drill-down: {cat1} → {cat2}")
                st.caption("Click a box to zoom in. Click the top bar to come back.")
                st.plotly_chart(px.treemap(tm, path=[cat1, cat2], values=val))
        elif cat1:
            pg, val = agg_table(df_all, cat1, measure, agg)
            pg = pg[pg[val] > 0].sort_values(val, ascending=False).head(10)
            if not pg.empty:
                st.markdown(f"#### 🥧 Share of each {cat1}")
                st.plotly_chart(px.pie(pg, names=cat1, values=val, hole=0.45))
    with right:
        if len(num) >= 2:
            sx = st.selectbox("Compare (horizontal)", num, index=0, key=f"sx_{active}")
            sy = st.selectbox("with (vertical)", num, index=1, key=f"sy_{active}")
            st.plotly_chart(px.scatter(df_all.head(5000), x=sx, y=sy, color=cat1 if cat1 else None))
        elif num:
            st.markdown(f"#### 📉 How {num[0]} is spread")
            st.plotly_chart(px.histogram(df_all, x=num[0]))

    with st.expander("💡 Quick tips from your data"):
        for tip in make_insights(df_all, num, cat, date):
            st.markdown(f"- {tip}")

    with st.expander("📋 See the data table"):
        st.dataframe(df_all.head(1000))

    st.markdown("#### ⬇️ Download")
    d1_, d2_ = st.columns(2)
    d1_.download_button("⬇️ Excel report (with filters)", data=build_excel(df_all, num),
                        file_name="my_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    d2_.download_button("⬇️ Filtered data (CSV)", data=df_all.to_csv(index=False).encode("utf-8"),
                        file_name="filtered_data.csv", mime="text/csv")


# ------------------------------------------------------------------
# DATA & COMBINE TAB
# ------------------------------------------------------------------
def render_data_tab(active):
    src = S["sources"]
    st.subheader("Your data sources")
    for name, d in list(src.items()):
        c1, c2, c3 = st.columns([5, 3, 1])
        c1.write(f"**{name}**")
        c2.caption(f"{len(d):,} rows × {len(d.columns)} columns")
        if c3.button("🗑️", key=f"rm_{name}"):
            src.pop(name)
            st.rerun()

    st.divider()
    st.subheader("🔗 Combine your sources")
    names = list(src.keys())
    mode = st.radio("How do you want to combine?",
                    ["Match by a common column (Join)", "Put rows one below another (Stack)"],
                    horizontal=True)
    if mode.startswith("Match"):
        if len(names) < 2:
            st.info("Add at least 2 sources (files or SQL tables) to join them.")
        else:
            c1, c2 = st.columns(2)
            ln = c1.selectbox("First table", names, key="jl")
            rn = c2.selectbox("Second table", [n for n in names if n != ln], key="jr")
            c3, c4 = st.columns(2)
            lk = c3.selectbox("Matching column in first table", list(src[ln].columns), key="jlk")
            rk = c4.selectbox("Matching column in second table", list(src[rn].columns), key="jrk")
            how_label = st.selectbox("Which rows to keep?", [
                "Keep everything from the first table",
                "Only rows that match in both",
                "Keep everything from both"])
            how = {"Keep everything from the first table": "left",
                   "Only rows that match in both": "inner",
                   "Keep everything from both": "outer"}[how_label]
            newname = st.text_input("Name for the combined table", value="Combined")
            if st.button("🔗 Create combined table"):
                try:
                    left, right = src[ln].copy(), src[rn].copy()
                    if left[lk].dtype != right[rk].dtype:
                        left[lk] = left[lk].astype(str).str.strip()
                        right[rk] = right[rk].astype(str).str.strip()
                    merged = pd.merge(left, right, left_on=lk, right_on=rk, how=how, suffixes=("", "_2"))
                    src[newname] = merged
                    S["next_active"] = newname
                    st.rerun()
                except Exception:
                    st.error("Could not combine these tables. Please try different matching columns.")
    else:
        picks = st.multiselect("Choose tables to stack (they should have similar columns)", names)
        newname = st.text_input("Name for the stacked table", value="Stacked")
        if st.button("📚 Create stacked table"):
            if len(picks) < 2:
                st.warning("Please choose at least 2 tables.")
            else:
                parts = []
                for p in picks:
                    t = src[p].copy()
                    t["Source"] = p
                    parts.append(t)
                src[newname] = pd.concat(parts, ignore_index=True)
                S["next_active"] = newname
                st.rerun()

    st.divider()
    st.subheader(f"🧹 Clean: {active}")
    d = src[active]
    num, _, _ = column_types(d)
    cA, cB, cC = st.columns(3)
    if cA.button("Remove duplicate rows"):
        src[active] = d.drop_duplicates().reset_index(drop=True)
        st.rerun()
    if cB.button("Remove rows with empty cells"):
        src[active] = d.dropna().reset_index(drop=True)
        st.rerun()
    if num and cC.button("Fill empty numbers with average"):
        d2 = d.copy()
        d2[num] = d2[num].fillna(d2[num].mean())
        src[active] = d2
        st.rerun()
    miss = d.isna().sum().reset_index()
    miss.columns = ["Column", "Empty cells"]
    with st.expander("Empty cells per column"):
        st.dataframe(miss)
    with st.expander("Preview of the data"):
        st.dataframe(d.head(200))


# ------------------------------------------------------------------
# MAIN APP
# ------------------------------------------------------------------
check_access()

st.title(APP_NAME)
st.caption("Upload files or connect a database. Combine them. Explore with clickable, interactive charts.")

with st.sidebar:
    st.header("1️⃣ Add your data")
    tab_files, tab_sql = st.tabs(["📁 Files", "🗄️ SQL database"])

    with tab_files:
        files = st.file_uploader("Excel or CSV (you can choose many files)",
                                 type=["csv", "xlsx", "xls"], accept_multiple_files=True)
        for f in files or []:
            sig = (f.name, f.size)
            if sig not in S["loaded"]:
                try:
                    loaded = read_upload(f)
                    S["sources"].update(loaded)
                    S["loaded"].add(sig)
                    S["next_active"] = list(loaded.keys())[0]
                except Exception:
                    st.error(f"Could not read {f.name}. Please check the file.")
        if st.button("✨ Try sample data"):
            a, b = sample_data()
            S["sources"]["Sample Sales"] = clean_basic(a)
            S["sources"]["Sample Products"] = clean_basic(b)
            S["next_active"] = "Sample Sales"

    with tab_sql:
        st.caption("🔒 Tip: use a **read-only** database user. We never save your password.")
        db_type = st.selectbox("Database type", ["SQLite file", "MySQL", "PostgreSQL"])
        if db_type == "SQLite file":
            dbfile = st.file_uploader("Upload .db / .sqlite file", type=["db", "sqlite", "sqlite3"], key="dbfile")
            if dbfile and st.button("Connect", key="conn_sqlite"):
                tmp = os.path.join(tempfile.gettempdir(), f"upload_{abs(hash(dbfile.name + str(dbfile.size)))}.db")
                with open(tmp, "wb") as fh:
                    fh.write(dbfile.getvalue())
                connect_engine(create_engine(f"sqlite:///{tmp}"))
        else:
            host = st.text_input("Host (server address)", key=f"h_{db_type}")
            port = st.number_input("Port", value=3306 if db_type == "MySQL" else 5432, step=1, key=f"p_{db_type}")
            dbname = st.text_input("Database name", key=f"n_{db_type}")
            user = st.text_input("User name", key=f"u_{db_type}")
            pwd = st.text_input("Password", type="password", key=f"w_{db_type}")
            if st.button("Connect", key=f"conn_{db_type}"):
                driver = "mysql+pymysql" if db_type == "MySQL" else "postgresql+psycopg2"
                try:
                    url = URL.create(driver, username=user, password=pwd, host=host,
                                     port=int(port), database=dbname)
                    connect_engine(create_engine(url, connect_args={"connect_timeout": 10}))
                except Exception:
                    st.error("Could not connect. Please check your details.")

        eng = S.get("engine")
        if eng is not None:
            st.success("Database is connected")
            if S.get("tables"):
                tbl = st.selectbox("Choose a table", S["tables"])
                if st.button("Load this table"):
                    qname = eng.dialect.identifier_preparer.quote(tbl)
                    load_sql(f"SELECT * FROM {qname} LIMIT {MAXROWS}", tbl)
            else:
                st.info("No tables found. You can still write your own SQL below.")
            with st.expander("Advanced: write my own SQL (SELECT only)"):
                sql_text = st.text_area("SQL", value="SELECT * FROM your_table LIMIT 1000", height=100)
                sql_name = st.text_input("Name for this result", value="My query")
                if st.button("Run SQL"):
                    load_sql(sql_text, sql_name)
            if st.button("Disconnect"):
                S.pop("engine", None)
                S.pop("tables", None)
                st.rerun()

    st.divider()
    st.caption("🔒 Files are used only to make your report. They are not saved.")
    st.caption(f"Need help? {CONTACT}")

names = list(S["sources"].keys())

if not names:
    st.markdown("### 👋 Welcome!")
    st.markdown(
        "Turn your data into an **interactive dashboard** in 3 steps:\n\n"
        "1. **Add data** in the left panel (Excel, CSV, or a SQL database)\n"
        "2. **Combine** several sources if you want (Data & Combine tab)\n"
        "3. **Explore**: click on charts, use filters, download reports\n\n"
        "New here? Click **✨ Try sample data** in the left panel."
    )
    st.stop()

if "next_active" in S:
    na = S.pop("next_active")
    if na in names:
        S["active"] = na
if S.get("active") not in names:
    S.pop("active", None)

active = st.selectbox("📂 Which data do you want to analyze?", names, key="active")

tab_dash, tab_data = st.tabs(["📊 Dashboard", "🔗 Data & Combine"])
with tab_dash:
    render_dashboard(active)
with tab_data:
    render_data_tab(active)
