import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="WIREQ | Quality Monitoring", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:Inter,sans-serif}
.stApp{background:#F7F9FA;color:#344451}
[data-testid="stSidebar"]{background:#FFFFFF;border-right:1px solid #E7ECEF}
.block-container{padding-top:1.5rem;max-width:1500px}
h1,h2,h3{color:#344451}
.hero{background:linear-gradient(120deg,#FFFFFF,#EEF4F5);border:1px solid #E5ECEF;border-radius:18px;padding:28px;margin-bottom:20px}
.eyebrow{color:#7895A8;font-size:.76rem;font-weight:700;letter-spacing:1.4px}
.hero-title{font-size:2rem;font-weight:700;color:#344451;margin:7px 0}
.muted{color:#84919A}
.card{background:#FFF;border:1px solid #E7ECEF;border-radius:14px;padding:17px 19px}
.klabel{color:#84919A;font-size:.75rem;font-weight:600;letter-spacing:.7px;text-transform:uppercase}
.kvalue{font-size:1.75rem;font-weight:700;color:#344451;margin-top:6px}
.status{padding:11px 15px;border-radius:10px;font-weight:700;display:inline-block}
.good{background:#EAF2ED;color:#547C65;border:1px solid #D6E6DC}
.bad{background:#F8ECEB;color:#A85F5F;border:1px solid #EFD9D7}
.warn{background:#F7F1E5;color:#9A7B43;border:1px solid #EEE2C8}
.stButton>button,.stDownloadButton>button{background:#7895A8;color:white;border:0;border-radius:9px;font-weight:600}
.stButton>button:hover,.stDownloadButton>button:hover{background:#607F93;color:white}
</style>
""", unsafe_allow_html=True)

MONTHS = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli",
          "Agustus", "September", "Oktober", "November", "Desember"]

ALIASES = {
    "Tanggal": ["tanggal", "date", "periode"],
    "No_Subgrup": ["no_subgrup", "subgroup", "subgrup", "no_subgroup"],
    "Jumlah_Produksi": ["jumlah_produksi", "total_produksi", "produksi", "ukuran_sampel", "sample_size", "n"],
    "Total_Cacat": ["total_cacat", "jumlah_cacat", "produk_cacat", "defect", "cacat", "d"],
}
DEFECTS = {
    "KTC_Kepala_Tidak_Center": ["ktc_kepala_tidak_center", "ktc"],
    "Kepala_Minus": ["kepala_minus", "k-"],
    "UC_Ujung_Cacat": ["uc_ujung_cacat", "uc"],
    "BC_Badan_Cacat": ["bc_badan_cacat", "bc"],
}
DEFECT_LABELS = {
    "KTC_Kepala_Tidak_Center": "KTC (Kepala Tidak Center)",
    "Kepala_Minus": "Kepala Minus",
    "UC_Ujung_Cacat": "UC (Ujung Cacat)",
    "BC_Badan_Cacat": "BC (Badan Cacat)",
}

C_MAIN, C_LIMIT, C_BAD, C_GOOD = "#7895A8", "#C9A66B", "#A85F5F", "#547C65"


# ---------------------------------------------------------------- data
def standardize(df):
    df = df.copy()
    lookup = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    ren = {}
    for group in (ALIASES, DEFECTS):
        for target, aliases in group.items():
            for a in aliases:
                if a in lookup and lookup[a] not in ren:
                    ren[lookup[a]] = target
                    break
    return df.rename(columns=ren)


def prepare(df):
    df = standardize(df)
    missing = [c for c in ALIASES if c not in df.columns]
    if missing:
        raise ValueError("Kolom wajib tidak ditemukan: " + ", ".join(missing))

    df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce", dayfirst=True)
    for c in ["No_Subgrup", "Jumlah_Produksi", "Total_Cacat"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Tanggal", "No_Subgrup", "Jumlah_Produksi", "Total_Cacat"]).copy()
    if df.empty:
        raise ValueError("Tidak ada baris data valid.")
    if (df["Jumlah_Produksi"] <= 0).any():
        raise ValueError("Jumlah produksi harus lebih dari nol.")
    if (df["Total_Cacat"] < 0).any() or (df["Total_Cacat"] > df["Jumlah_Produksi"]).any():
        raise ValueError("Jumlah cacat harus antara 0 dan jumlah produksi.")

    for c in DEFECTS:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["No_Subgrup"] = df["No_Subgrup"].astype(int)
    df["Jumlah_Produksi"] = df["Jumlah_Produksi"].astype(int)
    df["Total_Cacat"] = df["Total_Cacat"].astype(int)
    df["Tahun"] = df["Tanggal"].dt.year
    df["Bulan_No"] = df["Tanggal"].dt.month
    df["Bulan"] = df["Bulan_No"].map(lambda x: MONTHS[x - 1])
    df["Proporsi_Cacat"] = df["Total_Cacat"] / df["Jumlah_Produksi"]
    df["Jumlah_Produk_Baik"] = df["Jumlah_Produksi"] - df["Total_Cacat"]
    return df.sort_values(["Tanggal", "No_Subgrup"]).reset_index(drop=True)


@st.cache_data
def dummy():
    rng = np.random.default_rng(204324)
    rates = {1: .025, 2: .030, 3: .042, 4: .028, 5: .026, 6: .035,
             7: .024, 8: .029, 9: .031, 10: .023, 11: .027, 12: .025}
    rows = []
    for m in range(1, 13):
        for s in range(1, 26):
            n = int(rng.integers(950, 1251))
            p = rates[m]
            if (m, s) in [(3, 8), (3, 19), (6, 12), (9, 22)]:
                p += .035
            if (m, s) in [(7, 5), (10, 17)]:
                p = max(.005, p - .018)
            d = int(rng.binomial(n, p))
            parts = rng.multinomial(d, rng.dirichlet([4, 2.5, 2, 1.5]))
            rows.append({
                "Tanggal": pd.Timestamp(2024, m, s),
                "No_Subgrup": s,
                "Jumlah_Produksi": n,
                "Total_Cacat": d,
                "KTC_Kepala_Tidak_Center": int(parts[0]),
                "Kepala_Minus": int(parts[1]),
                "UC_Ujung_Cacat": int(parts[2]),
                "BC_Badan_Cacat": int(parts[3]),
            })
    return pd.DataFrame(rows)


@st.cache_data
def load_upload(content, name):
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(content))
    return pd.read_csv(io.BytesIO(content), sep=None, engine="python")


# ---------------------------------------------------------------- p-chart
def pchart(df, revise=False, k=3.0):
    """p-chart ukuran sampel variabel: batas dihitung per subgrup."""
    df = df.copy()
    base = df
    pbar = df["Total_Cacat"].sum() / df["Jumlah_Produksi"].sum()
    for _ in range(20):
        pbar = base["Total_Cacat"].sum() / base["Jumlah_Produksi"].sum()
        sigma = np.sqrt(pbar * (1 - pbar) / df["Jumlah_Produksi"])
        ucl = pbar + k * sigma
        lcl = (pbar - k * sigma).clip(lower=0)
        out = (df["Proporsi_Cacat"] > ucl) | (df["Proporsi_Cacat"] < lcl)
        if not revise:
            break
        new_base = df[~out]
        if new_base.empty or len(new_base) == len(base):
            break
        base = new_base
    df["CL"], df["UCL"], df["LCL"], df["Out"] = pbar, ucl, lcl, out
    df["Arah"] = np.where(df["Proporsi_Cacat"] > df["UCL"], "Di atas UCL",
                          np.where(df["Proporsi_Cacat"] < df["LCL"], "Di bawah LCL", "Terkendali"))
    return df, pbar


def kpi(col, label, value):
    col.markdown(f'<div class="card"><div class="klabel">{label}</div>'
                 f'<div class="kvalue">{value}</div></div>', unsafe_allow_html=True)


def style_fig(fig, height=430):
    fig.update_layout(
        height=height, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family="Inter", color="#344451"),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", y=1.12, x=0),
    )
    fig.update_xaxes(gridcolor="#EEF1F3", zeroline=False)
    fig.update_yaxes(gridcolor="#EEF1F3", zeroline=False)
    return fig


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan")
    source = st.radio("Sumber data", ["Data contoh", "Unggah file"])
    raw = None
    if source == "Unggah file":
        up = st.file_uploader("CSV / Excel", type=["csv", "xlsx", "xls"])
        if up is not None:
            try:
                raw = load_upload(up.getvalue(), up.name)
            except Exception as e:
                st.error(f"Gagal membaca file: {e}")
        template = dummy().head(5).to_csv(index=False).encode("utf-8")
        st.download_button("Unduh template CSV", template, "template_wireq.csv", "text/csv")
    else:
        raw = dummy()

if raw is None:
    st.markdown('<div class="hero"><div class="eyebrow">WIREQ</div>'
                '<div class="hero-title">Quality Monitoring</div>'
                '<div class="muted">Unggah file data produksi di sidebar untuk memulai. '
                'Kolom wajib: Tanggal, No_Subgrup, Jumlah_Produksi, Total_Cacat.</div></div>',
                unsafe_allow_html=True)
    st.stop()

try:
    data = prepare(raw)
except ValueError as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    years = sorted(data["Tahun"].unique())
    year = st.selectbox("Tahun", years, index=len(years) - 1)
    months_avail = (data[data["Tahun"] == year][["Bulan_No", "Bulan"]]
                    .drop_duplicates().sort_values("Bulan_No")["Bulan"].tolist())
    sel_months = st.multiselect("Bulan", months_avail, default=months_avail)
    revise = st.checkbox("Revisi batas kendali", value=False,
                         help="Buang titik di luar batas, lalu hitung ulang CL secara iteratif.")
    k = st.slider("Lebar batas (σ)", 2.0, 4.0, 3.0, 0.5)

view = data[(data["Tahun"] == year) & (data["Bulan"].isin(sel_months))].reset_index(drop=True)
if view.empty:
    st.warning("Tidak ada data untuk filter yang dipilih.")
    st.stop()

res, pbar = pchart(view, revise, k)
res["Urutan"] = np.arange(1, len(res) + 1)
res["Label"] = res["Tanggal"].dt.strftime("%d %b %Y") + " · SG " + res["No_Subgrup"].astype(str)
n_out = int(res["Out"].sum())

# ---------------------------------------------------------------- header
st.markdown(f'<div class="hero"><div class="eyebrow">WIREQ · QUALITY MONITORING</div>'
            f'<div class="hero-title">Peta Kendali Proporsi Cacat (p-chart)</div>'
            f'<div class="muted">Periode {year} · {len(res)} subgrup · '
            f'batas kendali ±{k:g}σ</div></div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
kpi(c1, "Total Produksi", f"{res['Jumlah_Produksi'].sum():,}".replace(",", "."))
kpi(c2, "Total Cacat", f"{res['Total_Cacat'].sum():,}".replace(",", "."))
kpi(c3, "Proporsi Cacat (CL)", f"{pbar*100:.2f}%")
kpi(c4, "Yield", f"{(1-res['Total_Cacat'].sum()/res['Jumlah_Produksi'].sum())*100:.2f}%")
kpi(c5, "Titik di Luar Batas", str(n_out))

st.write("")
if n_out == 0:
    st.markdown('<span class="status good">✔ Proses terkendali secara statistik</span>', unsafe_allow_html=True)
elif n_out / len(res) <= 0.1:
    st.markdown(f'<span class="status warn">⚠ {n_out} subgrup di luar batas kendali, perlu investigasi</span>',
                unsafe_allow_html=True)
else:
    st.markdown(f'<span class="status bad">✖ Proses tidak terkendali: {n_out} subgrup di luar batas</span>',
                unsafe_allow_html=True)
st.write("")

tab1, tab2, tab3, tab4 = st.tabs(["📈 p-chart", "📊 Jenis Cacat", "🗓️ Ringkasan Bulanan", "📋 Data"])

# ---------------------------------------------------------------- tab 1
with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res["Urutan"], y=res["UCL"] * 100, name="UCL", mode="lines",
                             line=dict(color=C_LIMIT, dash="dash", shape="hv")))
    fig.add_trace(go.Scatter(x=res["Urutan"], y=res["LCL"] * 100, name="LCL", mode="lines",
                             line=dict(color=C_LIMIT, dash="dash", shape="hv")))
    fig.add_trace(go.Scatter(x=res["Urutan"], y=res["CL"] * 100, name="CL", mode="lines",
                             line=dict(color=C_GOOD, width=1.5)))
    fig.add_trace(go.Scatter(
        x=res["Urutan"], y=res["Proporsi_Cacat"] * 100, name="Proporsi cacat",
        mode="lines+markers", line=dict(color=C_MAIN, width=2),
        marker=dict(size=7, color=np.where(res["Out"], C_BAD, C_MAIN)),
        customdata=np.stack([res["Label"], res["Jumlah_Produksi"], res["Total_Cacat"], res["Arah"]], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>Proporsi: %{y:.2f}%<br>"
                      "Produksi: %{customdata[1]}<br>Cacat: %{customdata[2]}<br>%{customdata[3]}<extra></extra>"))
    oo = res[res["Out"]]
    if not oo.empty:
        fig.add_trace(go.Scatter(x=oo["Urutan"], y=oo["Proporsi_Cacat"] * 100, mode="markers",
                                 name="Di luar batas", marker=dict(size=12, color=C_BAD, symbol="x"),
                                 hoverinfo="skip"))
    fig.update_layout(title="p-chart: Proporsi Cacat per Subgrup",
                      xaxis_title="Urutan subgrup", yaxis_title="Proporsi cacat (%)")
    st.plotly_chart(style_fig(fig, 460), use_container_width=True)

    if revise:
        st.caption("Mode revisi aktif: CL dan batas dihitung ulang tanpa titik di luar batas.")
    if n_out:
        st.markdown("**Subgrup di luar batas kendali**")
        show = oo[["Tanggal", "No_Subgrup", "Jumlah_Produksi", "Total_Cacat", "Proporsi_Cacat", "UCL", "LCL", "Arah"]].copy()
        show["Tanggal"] = show["Tanggal"].dt.strftime("%d-%m-%Y")
        for c in ["Proporsi_Cacat", "UCL", "LCL"]:
            show[c] = (show[c] * 100).round(2).astype(str) + "%"
        st.dataframe(show, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- tab 2
with tab2:
    dcols = [c for c in DEFECTS if c in res.columns]
    if not dcols:
        st.info("Data tidak memiliki kolom rincian jenis cacat (KTC, Kepala_Minus, UC, BC).")
    else:
        tot = res[dcols].sum().sort_values(ascending=False)
        cum = tot.cumsum() / tot.sum() * 100
        labels = [DEFECT_LABELS[c] for c in tot.index]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=labels, y=tot.values, name="Jumlah", marker_color=C_MAIN))
        fig.add_trace(go.Scatter(x=labels, y=cum.values, name="Kumulatif (%)", yaxis="y2",
                                 mode="lines+markers", line=dict(color=C_LIMIT, width=2)))
        fig.update_layout(title="Diagram Pareto Jenis Cacat",
                          yaxis=dict(title="Jumlah cacat"),
                          yaxis2=dict(title="Kumulatif (%)", overlaying="y", side="right",
                                      range=[0, 105], showgrid=False))
        st.plotly_chart(style_fig(fig, 420), use_container_width=True)
        top = tot.index[0]
        st.caption(f"Jenis cacat terbesar: **{DEFECT_LABELS[top]}** "
                   f"({tot.iloc[0]/tot.sum()*100:.1f}% dari seluruh cacat).")

# ---------------------------------------------------------------- tab 3
with tab3:
    g = (res.groupby(["Bulan_No", "Bulan"], as_index=False)
         .agg(Produksi=("Jumlah_Produksi", "sum"), Cacat=("Total_Cacat", "sum"),
              Subgrup=("No_Subgrup", "count"), Di_Luar_Batas=("Out", "sum")))
    g["Proporsi_Cacat_%"] = (g["Cacat"] / g["Produksi"] * 100).round(2)
    fig = go.Figure(go.Bar(x=g["Bulan"], y=g["Proporsi_Cacat_%"], marker_color=C_MAIN,
                           text=g["Proporsi_Cacat_%"].astype(str) + "%", textposition="outside"))
    fig.add_hline(y=pbar * 100, line_dash="dash", line_color=C_GOOD,
                  annotation_text=f"CL {pbar*100:.2f}%")
    fig.update_layout(title="Proporsi Cacat per Bulan", yaxis_title="Proporsi cacat (%)")
    st.plotly_chart(style_fig(fig, 400), use_container_width=True)
    st.dataframe(g.drop(columns="Bulan_No"), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- tab 4
with tab4:
    out_df = res.drop(columns=["Urutan", "Label"]).copy()
    out_df["Tanggal"] = out_df["Tanggal"].dt.strftime("%Y-%m-%d")
    st.dataframe(out_df, use_container_width=True, hide_index=True)
    st.download_button("Unduh hasil (CSV)", out_df.to_csv(index=False).encode("utf-8"),
                       "hasil_pchart_wireq.csv", "text/csv")
