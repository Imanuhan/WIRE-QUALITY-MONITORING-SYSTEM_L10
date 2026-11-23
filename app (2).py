"""QC & Process Capability Monitoring - Paku L10 (PT Surabaya Wire). Jalankan: streamlit run app.py"""
import io, math, re
import numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from scipy.stats import norm

st.set_page_config(page_title="QC & Process Capability - Paku L10", page_icon="📊", layout="wide")
FAKTOR, ALPHA, K = 4926 / 30, 0.0027, 4
CAT = ["KTC", "K-", "UC", "BC"]; W = np.ones(4)
AB = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
MON = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
MI = dict(jan=0, feb=1, mar=2, apr=3, mei=4, may=4, jun=5, jul=6, agu=7, aug=7, sep=8, okt=9, oct=9, nov=10, des=11, dec=11)
Z = norm.ppf((1 - ALPHA) ** (1 / (K + 1)))
DEFAULT = [["Jan 2024",1,1142605,1806,0,1205,0],["Feb 2024",1,882406,1221,419,0,0],["Mar 2024",1,882329,677,0,0,0],["Apr 2024",1,598873,0,0,0,547],["Mei 2024",1,926591,3100,0,584,0],["Jun 2024",1,866858,622,0,789,0],["Jul 2024",1,1055019,0,670,688,0],["Agu 2024",1,1017862,568,0,1848,626],["Sep 2024",1,909643,1379,1128,598,0],["Okt 2024",1,1135519,636,0,1924,694],["Nov 2024",1,978176,0,1158,637,0],["Des 2024",1,876531,620,610,0,0],["Jan 2025",1,949057,1197,578,0,0],["Feb 2025",1,950099,0,0,1105,0],["Mar 2025",1,997630,0,0,0,536],["Apr 2025",1,772453,0,0,566,679],["Mei 2025",1,952963,0,0,1064,621],["Jun 2025",1,849628,0,0,0,1195],["Jul 2025",1,1168833,1800,647,511,0],["Agu 2025",1,1043728,2537,607,598,0],["Sep 2025",1,998330,1071,0,1294,0],["Nov 2025",2,1031962,596,0,1861,0],["Des 2025",2,784868,0,0,1170,0],["Jan 2026",2,259036,1839,623,559,0],["Feb 2026",2,244298,0,0,1270,0],["Mar 2026",2,227898,595,1185,0,0],["Apr 2026",2,310140,0,0,0,606],["Mei 2026",2,244093,2433,0,1178,0],["Jun 2026",2,248199,1680,1143,1651,0]]
PAGES = ["1 · User Guide", "2 · Quality Control Dashboard"]

def fm(v, d=0):
    return f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")

def pn(v):
    if pd.isna(v): return None
    if hasattr(v, "month") and hasattr(v, "year"): return f"{AB[v.month - 1]} {v.year}"
    s = str(v).strip().lower()
    m = re.match(r"(\d{4})-(\d{1,2})", s)
    if m and 1 <= int(m[2]) <= 12: return f"{AB[int(m[2]) - 1]} {m[1]}"
    m = re.search(r"([a-z]{3})[a-z]*\.?\s*[-/]?\s*'?(\d{4})", s)
    return f"{AB[MI[m[1]]]} {m[2]}" if m and m[1] in MI else None

def read_any(f):
    return pd.read_csv(f) if f.name.lower().endswith(".csv") else pd.read_excel(f)

def read_prod(f):
    df = read_any(f)
    key = lambda pat: next((c for c in df.columns if re.match(pat, re.sub(r"[\s_]", "", str(c).lower()))), None)
    col = {"p": key(r"(periode|bulan|period|tanggal|date)"), "f": key(r"(fase|phase)"), "q": key(r"(produksi|production)")}
    col.update({c: key(p) for c, p in zip(CAT, [r"ktc", r"k[-−–]", r"uc", r"bc"])})
    miss = [n for n, k in [("Periode", col["p"]), ("Produksi_kg", col["q"])] + [(c, col[c]) for c in CAT] if k is None]
    if miss: raise ValueError("Kolom tidak ditemukan: " + ", ".join(miss))
    out = pd.DataFrame({"Periode": df[col["p"]].map(pn),
                        "Fase": df[col["f"]].map(lambda v: 2 if re.search(r"2|ii", str(v), re.I) else 1) if col["f"] else 1,
                        "Produksi_kg": pd.to_numeric(df[col["q"]], errors="coerce"),
                        **{c: pd.to_numeric(df[col[c]], errors="coerce").fillna(0) for c in CAT}})
    out = out[out.Periode.notna() & (out.Produksi_kg > 0)].drop_duplicates("Periode", keep="last")
    if (out.Fase == 1).sum() < 6: raise ValueError("Minimal 6 periode Fase I diperlukan.")
    return out

def prep(df):
    d = df.copy(); d["m"] = d.Periode.str[:3].map(AB.index); d["y"] = d.Periode.str[4:].astype(int)
    d = d.sort_values(["y", "m"]).reset_index(drop=True)
    d["N"] = d.Produksi_kg * FAKTOR
    P = (d[CAT] * FAKTOR).div(d.N, axis=0).to_numpy()
    d["delta"] = P @ W; d["Cacat_kg"] = d[CAT].sum(axis=1)
    return d, P

def phase1(d, P, mode):
    f1 = (d.Fase == 1).to_numpy(); s = np.sort(np.maximum(P[f1].mean(0), 1e-12))
    nC = int(math.ceil(max(5 / s[1], 1 / s[0])))
    nof = np.full(len(d), float(nC)) if mode == "fixed" else d.N.to_numpy()
    idx = np.where(f1)[0]; rem = []; dl = d.delta.to_numpy()
    for _ in range(11):
        pb = P[idx].mean(0); GT = W @ pb; v = (W ** 2) @ pb - GT ** 2
        ooc = np.abs(dl[idx] - GT) > Z * np.sqrt(v / nof[idx])
        if not ooc.any() or len(idx) - ooc.sum() < 5: break
        rem += list(idx[ooc]); idx = idx[~ooc]
    return dict(pb=pb, GT=GT, v=v, rem=[d.Periode[i] for i in rem], used=len(idx), n1=int(f1.sum()), ok=not ooc.any(), nC=nC, nof=nof)

def cat(x):
    return ("Capable", "success") if x >= 1.33 else ("Marginal", "warning") if x >= 1 else ("Not Capable", "error")

def capability(meas, usl, lsl, period):
    why = []
    if meas is None: why.append("Dataset awal tidak memuat pengukuran individual/subgrup karakteristik kualitas (hanya total produksi dan berat cacat per kategori). Cp/Cpk tidak boleh dihitung dari jumlah atau persentase cacat.")
    if usl is None or lsl is None: why.append("USL dan LSL belum diisi (batas spesifikasi tidak ada pada data).")
    elif usl <= lsl: why.append("USL harus lebih besar dari LSL.")
    v = np.array([])
    if meas is not None and period:
        v = meas.loc[meas.Periode == period, "Nilai"].dropna().to_numpy(float)
        if len(v) < 2: why.append(f"Tidak ada (atau kurang dari 2) data ukur untuk {period}.")
    if why: return dict(na=True, why=why)
    n, mu, sd = len(v), v.mean(), v.std(ddof=1)
    if not sd > 0: return dict(na=True, why=["Standar deviasi data ukur = 0."])
    cp = (usl - lsl) / (6 * sd); cpk = min((usl - mu) / (3 * sd), (mu - lsl) / (3 * sd))
    sk = (((v - mu) / sd) ** 3).mean(); ku = (((v - mu) / sd) ** 4).mean() - 3
    pn_ = math.exp(-n / 6 * (sk ** 2 + ku ** 2 / 4) / 2)
    return dict(na=False, v=v, n=n, mu=mu, sd=sd, cp=cp, cpk=cpk, usl=usl, lsl=lsl, pn=pn_, normal=n >= 8 and pn_ >= .05)

def badge(txt, kind):
    getattr(st, kind)(txt)

# ------------------------------------------------------------------ sidebar
st.session_state.setdefault("page", PAGES[0])
st.sidebar.title("QC & PROCESS CAPABILITY")
st.sidebar.caption("PT Surabaya Wire · Paku L10")
page = st.sidebar.radio("Navigasi", PAGES, key="page")

# ------------------------------------------------------------------ page 1
if page == PAGES[0]:
    st.title("QUALITY CONTROL & PROCESS CAPABILITY MONITORING")
    st.caption("Dokumentasi pengguna · Produk: paku kawat baja L10 · PT Surabaya Wire")
    st.markdown("""
### 1. Deskripsi dan Tujuan
Dashboard menganalisis kualitas produksi paku L10 per bulan: (a) kestabilan proses dengan peta kendali multivariat P (Mp Chart) dan (b) kapabilitas proses terhadap batas spesifikasi (Cp, Cpk). Seluruh angka dihitung dari data produksi (notebook `Mp_Chart_Paku_L10_Bab2.ipynb` atau file Excel yang Anda unggah).

### 2. Sumber dan Struktur Data
Data bawaan: bulanan Januari 2024 – Juni 2026 (29 periode; Oktober 2025 tidak dimasukkan seperti pada notebook). Fase I = penentuan batas kendali, Fase II = monitoring.

| Kolom | Keterangan |
|---|---|
| Periode, Fase | Bulan produksi (mis. Jan 2024) dan fase analisis (1/2) |
| Produksi_kg | Total produksi; 1 kg = 4926/30 = 164,2 biji |
| KTC, K-, UC, BC | Berat cacat (kg) tiap kategori |

Data tidak memuat pengukuran individual karakteristik kualitas maupun USL/LSL, sehingga Cp/Cpk memerlukan data tambahan (bagian Process Capability).

### 3. Metode Analisis
- Proporsi cacat p̂ᵢⱼ = xᵢⱼ / nⱼ; statistik gabungan δⱼ = Σ dᵢ p̂ᵢⱼ (dᵢ = 1, asumsi notebook).
- CL = Σ dᵢ p̄ᵢ; batas kendali = CL ± z √{ (1/nⱼ) [ Σ dᵢ² p̄ᵢ − (Σ dᵢ p̄ᵢ)² ] }, z = Φ⁻¹[(1−α)^(1/(k+1))] (Šidák), α = 0,0027, k = 4.
- nⱼ = jumlah biji diperiksa pada subgrup ke-j. Default: n tetap aturan Cochran (sesuai notebook). Opsi lain: nⱼ aktual per subgrup; karena n sangat besar (puluhan hingga ratusan juta biji), batas menjadi sangat sempit dan hampir semua titik tampak di luar batas.
- Fase I diulang: titik di luar batas dikeluarkan, batas dihitung ulang sampai terkendali; batas final dipakai memonitor seluruh periode.
- DPO = cacat / (unit × k); DPMO = DPO × 10⁶; level sigma = Φ⁻¹(1 − DPO) + 1,5.

### 4. Indikator
**Total Production** (kg) · **Total Defect** (jumlah 4 kategori, kg) · **Good Products** = produksi − cacat · **Defect Percentage** = cacat / produksi × 100%.

### 5. CL, UCL, LCL, Cp, Cpk
| Istilah | Arti |
|---|---|
| CL | Garis tengah: rata-rata δ saat proses terkendali |
| UCL / LCL | Batas kendali atas / bawah. Di atas UCL = pemburukan; di bawah LCL = perbaikan tak wajar yang tetap perlu ditelusuri |
| USL / LSL | Batas spesifikasi (dari pelanggan/standar, bukan dihitung dari data proses) |
| Cp | (USL − LSL) / 6σ |
| Cpk | min[(USL − μ)/3σ, (μ − LSL)/3σ] |

### 6. Kriteria Interpretasi
| Cp / Cpk | Kategori |
|---|---|
| ≥ 1,33 | Capable |
| 1,00 – < 1,33 | Marginal |
| < 1,00 | Not Capable |

**Stabilitas ≠ kapabilitas.** Proses terkendali hanya berarti variasinya konsisten; belum tentu memenuhi spesifikasi.

### 7. Langkah Penggunaan
1. Klik OPEN DASHBOARD. 2. (Opsional) unggah Excel/CSV data produksi di bagian atas dashboard. 3. Pilih tahun dan bulan (default Januari 2024). 4. Baca KPI dan control chart (titik merah = di luar batas, lingkaran oranye = periode terpilih; hover untuk detail, seret untuk zoom). 5. Isi USL/LSL dan unggah data ukur (kolom Periode, Nilai) untuk Cp/Cpk. 6. Baca Process Evaluation dan Analysis Summary.
""")
    st.button("OPEN DASHBOARD", type="primary", on_click=lambda: st.session_state.update(page=PAGES[1]))
    st.stop()

# ------------------------------------------------------------------ page 2
st.title("Quality Control Dashboard")
with st.expander("📁 Data produksi (unggah Excel/CSV)", expanded=False):
    up = st.file_uploader("File data produksi", type=["xlsx", "xls", "csv"])
    tpl = io.BytesIO(); pd.DataFrame(DEFAULT, columns=["Periode", "Fase", "Produksi_kg"] + CAT).to_excel(tpl, index=False)
    st.download_button("Unduh template/data bawaan (.xlsx)", tpl.getvalue(), "template_data_produksi.xlsx")
    st.caption("Kolom: Periode (mis. Jan 2024 atau tanggal), Fase (1/2), Produksi_kg, KTC, K-, UC, BC (kg). Minimal 6 periode Fase I.")
src = "Data bawaan (notebook)"; raw = pd.DataFrame(DEFAULT, columns=["Periode", "Fase", "Produksi_kg"] + CAT)
if up is not None:
    try: raw = read_prod(up); src = up.name
    except Exception as e: st.error(f"Gagal memuat file: {e}. Memakai data bawaan.")
d, P = prep(raw); st.caption(f"Sumber data: {src} · {len(d)} periode (Fase I: {(d.Fase == 1).sum()}, Fase II: {(d.Fase == 2).sum()})")

years = sorted(d.y.unique()); f = st.columns(4)
yr = f[0].selectbox("Tahun produksi", years, index=years.index(2024) if 2024 in years else 0)
mo = f[1].selectbox("Bulan produksi", range(12), format_func=lambda i: MON[i], index=0)
mode = f[2].selectbox("Ukuran sampel batas kendali", ["fixed", "actual"], format_func=lambda x: "Aktual per subgrup (nⱼ)" if x == "actual" else "Tetap (Cochran, seperti notebook)")
rng = f[3].selectbox("Rentang grafik", ["Seluruh periode", "Tahun terpilih"])

S = phase1(d, P, mode); h = Z * np.sqrt(S["v"] / S["nof"])
d["CL"] = S["GT"]; d["UCL"] = S["GT"] + h; d["LCL"] = S["GT"] - h
d["Status"] = np.where(d.delta > d.UCL, "OOC ▲ di atas UCL", np.where(d.delta < d.LCL, "OOC ▼ di bawah LCL", "Terkendali"))
sm = d[(d.y == yr) & (d.m == mo)]; sel = sm.iloc[0] if len(sm) else None; si = sm.index[0] if len(sm) else None
period = f"{AB[mo]} {yr}"
if sel is None: st.warning(f"Tidak ada data produksi untuk {MON[mo]} {yr} pada dataset ini.")

st.subheader(f"A. Production Overview — {MON[mo]} {yr}")
k = st.columns(4)
if sel is not None:
    dpo = sel.Cacat_kg * FAKTOR / (sel.N * K); dpmo = dpo * 1e6; sg = norm.ppf(1 - dpo) + 1.5 if dpo > 0 else float("nan")
    k[0].metric("Total Production", f"{fm(sel.Produksi_kg)} kg"); k[1].metric("Total Defect", f"{fm(sel.Cacat_kg)} kg")
    k[2].metric("Good Products", f"{fm(sel.Produksi_kg - sel.Cacat_kg)} kg"); k[3].metric("Defect Percentage", f"{fm(sel.Cacat_kg / sel.Produksi_kg * 100, 3)} %")
    st.caption("Cacat per kategori (kg): " + " · ".join(f"{c} {fm(sel[c])}" for c in CAT) + f" · DPMO {fm(dpmo)} · ≈ {fm(sg, 2)} sigma")
else:
    for c, n in zip(k, ["Total Production", "Total Defect", "Good Products", "Defect Percentage"]): c.metric(n, "—")

st.subheader("B. Statistical Process Control — Mp Chart")
st.caption(f"δ = Σdᵢp̂ᵢ · CL = {fm(S['GT'], 6)} · z (Šidák) = {fm(Z, 4)} · batas Fase I final dari {S['used']} subgrup"
           + (f" (dikeluarkan: {', '.join(S['rem'])})" if S["rem"] else "") + ("" if S["ok"] else " · belum konvergen")
           + f" · ukuran sampel: {'tetap n = ' + fm(S['nC']) if mode == 'fixed' else 'aktual per subgrup'}")
v = d if rng == "Seluruh periode" else d[d.y == yr]
fig = go.Figure()
for n, c, dash in [("UCL", "#c62828", "dash"), ("CL", "#1e8e3e", "solid"), ("LCL", "#c62828", "dash")]:
    fig.add_trace(go.Scatter(x=v.Periode, y=v[n], name=n, mode="lines", line=dict(color=c, dash=dash, width=1.5), hoverinfo="skip"))
ooc = v.Status != "Terkendali"
fig.add_trace(go.Scatter(x=v.Periode, y=v.delta, name="δ", mode="lines+markers", line=dict(color="#2563eb", width=1, dash="dot"),
    marker=dict(size=8, color=np.where(ooc, "#e53935", "#2563eb"), symbol=np.where(ooc, "diamond", "circle")),
    customdata=np.c_[v.Fase, v.UCL, v.LCL, v.Produksi_kg, v.Cacat_kg, v.Status],
    hovertemplate="%{x} (Fase %{customdata[0]})<br>δ = %{y:.6f}<br>UCL = %{customdata[1]:.6f}<br>LCL = %{customdata[2]:.6f}<br>Produksi %{customdata[3]:,.0f} kg · cacat %{customdata[4]:,.0f} kg<br>%{customdata[5]}<extra></extra>"))
if sel is not None and si in v.index:
    fig.add_trace(go.Scatter(x=[sel.Periode], y=[sel.delta], name="Periode terpilih", mode="markers", hoverinfo="skip",
                             marker=dict(size=18, symbol="circle-open", color="#f59e0b", line=dict(width=3))))
fig.update_layout(height=420, margin=dict(l=60, r=20, t=30, b=60), xaxis=dict(type="category", tickangle=-60), yaxis=dict(title="δ", tickformat=".5f"), legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig, width="stretch")
tb = v[["Periode", "Fase", "Produksi_kg", "Cacat_kg", "delta", "LCL", "CL", "UCL", "Status"]]
st.dataframe(tb.style.apply(lambda r: ["background-color:#fff4d6;color:#000" if r.Periode == period else ""] * len(r), axis=1)
             .format({"Produksi_kg": "{:,.0f}", "Cacat_kg": "{:,.0f}", "delta": "{:.6f}", "LCL": "{:.6f}", "CL": "{:.6f}", "UCL": "{:.6f}"}),
             hide_index=True, width="stretch", height=300)

st.subheader("C. Process Capability Analysis")
c1 = st.columns([1, 1, 2])
usl = c1[0].number_input("USL", value=None, format="%.4f", placeholder="isi USL"); lsl = c1[1].number_input("LSL", value=None, format="%.4f", placeholder="isi LSL")
mf = c1[2].file_uploader("Data ukur (Excel/CSV, kolom Periode dan Nilai)", type=["xlsx", "xls", "csv"])
meas = None
if mf is not None:
    try:
        m = read_any(mf); m = m.iloc[:, :2]; m.columns = ["Periode", "Nilai"]
        m["Periode"] = m.Periode.map(pn); m["Nilai"] = pd.to_numeric(m.Nilai.astype(str).str.replace(",", "."), errors="coerce"); meas = m.dropna()
    except Exception as e: st.error(f"Gagal membaca data ukur: {e}")
C = capability(meas, usl, lsl, period if sel is not None else None)
cc = st.columns(4)
if C["na"]:
    cc[0].metric("Cp", "Not Available"); cc[1].metric("Cpk", "Not Available"); cc[2].metric("USL", fm(usl, 3) if usl is not None else "—"); cc[3].metric("LSL", fm(lsl, 3) if lsl is not None else "—")
    st.warning("**Not Available.** Alasan:\n\n" + "\n".join(f"- {w}" for w in C["why"]))
else:
    a, b = cat(C["cp"]), cat(C["cpk"])
    cc[0].metric("Cp", fm(C["cp"], 3)); cc[1].metric("Cpk", fm(C["cpk"], 3)); cc[2].metric("USL", fm(usl, 3)); cc[3].metric("LSL", fm(lsl, 3))
    hc, ic = st.columns(2)
    vv = C["v"]; lo, hi = min(vv.min(), lsl), max(vv.max(), usl); nb = min(20, max(6, math.ceil(math.sqrt(C["n"]))))
    cnt, edges = np.histogram(vv, bins=nb, range=(lo, hi)); w = edges[1] - edges[0]
    hf = go.Figure(go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=cnt, width=w * .95, name="Frekuensi", marker_color="#6fa8dc"))
    if C["normal"]:
        xs = np.linspace(lo, hi, 120); hf.add_trace(go.Scatter(x=xs, y=C["n"] * w * norm.pdf(xs, C["mu"], C["sd"]), mode="lines", name="Normal", line=dict(color="#e53935")))
    for x, t, c in [(lsl, "LSL", "#c62828"), (usl, "USL", "#c62828"), (C["mu"], "μ", "#1e8e3e")]: hf.add_vline(x=x, line=dict(color=c, dash="dash"), annotation_text=t)
    hf.update_layout(height=320, margin=dict(l=40, r=10, t=30, b=40), showlegend=False, yaxis_title="Frekuensi"); hc.plotly_chart(hf, width="stretch")
    with ic:
        badge(f"Kapabilitas (Cpk): {b[0]}", b[1])
        st.write(f"n = {C['n']} · μ = {fm(C['mu'], 4)} · σ = {fm(C['sd'], 4)}. Cp {fm(C['cp'], 3)} ({a[0]}), Cpk {fm(C['cpk'], 3)} ({b[0]}).")
        if C["cp"] - C["cpk"] > .1: st.write("Cpk lebih kecil dari Cp: rata-rata proses bergeser dari titik tengah spesifikasi.")
        if C["n"] < 30: st.write("**n < 30: estimasi kurang andal.**")
        st.caption(f"Uji normalitas (Jarque–Bera) p ≈ {fm(C['pn'], 3)}: " + ("asumsi normal dapat diterima, kurva normal ditampilkan." if C["normal"] else "asumsi normal tidak didukung/data terlalu sedikit; kurva normal tidak ditampilkan, tafsirkan Cp/Cpk dengan hati-hati."))

st.subheader("D. Process Evaluation")
e1, e2 = st.columns(2)
with e1:
    st.markdown("**1. Process Stability**")
    if sel is None: badge("Tidak ada data periode terpilih.", "info")
    elif sel.Status == "Terkendali": badge("Terkendali (in control): δ berada di antara LCL dan UCL.", "success")
    elif "▲" in sel.Status: badge("Tidak terkendali: δ di atas UCL (pemburukan kualitas).", "error")
    else: badge("Tidak terkendali: δ di bawah LCL (perbaikan tak wajar; tetap telusuri penyebabnya).", "warning")
    p2 = d[d.Fase == 2]; st.caption(f"Konteks: Fase I final memakai {S['used']} dari {S['n1']} subgrup; Fase II: {(p2.Status != 'Terkendali').sum()} dari {len(p2)} bulan di luar batas.")
with e2:
    st.markdown("**2. Process Capability**")
    if C["na"]: badge("Not Available: Cp/Cpk belum dapat dihitung (lihat alasan di bagian C).", "info")
    else: badge(f"{cat(C['cpk'])[0]}: Cpk = {fm(C['cpk'], 3)}, Cp = {fm(C['cp'], 3)}.", cat(C["cpk"])[1])
    st.caption("Proses yang terkendali belum tentu capable; keduanya dinilai terpisah.")

st.subheader("E. Analysis Summary")
if sel is None: st.info(f"Tidak ada data produksi pada {MON[mo]} {yr}, sehingga analisis tidak dapat dibuat.")
else:
    ct = W * (P[si] - S["pb"]); top = CAT[int(ct.argmax())]; big = max(zip(CAT, [sel[c] for c in CAT]), key=lambda t: t[1])
    est = (f"berada di dalam batas kendali ({fm(sel.LCL, 6)} – {fm(sel.UCL, 6)}), sehingga proses dinilai **terkendali secara statistik**." if sel.Status == "Terkendali"
           else f"berada **di atas UCL** ({fm(sel.UCL, 6)}); proses **tidak terkendali** dengan indikasi pemburukan, penyumbang terbesar: **{top}**." if "▲" in sel.Status
           else f"berada **di bawah LCL** ({fm(sel.LCL, 6)}); proses **tidak terkendali** (perbaikan tak wajar), perlu ditelusuri penyebabnya.")
    capt = ("Kapabilitas proses **Not Available** karena data ukur individual dan/atau USL/LSL belum tersedia; kapabilitas tidak dapat disimpulkan dari data cacat." if C["na"]
            else f"Kapabilitas proses: Cp = {fm(C['cp'], 3)}, Cpk = {fm(C['cpk'], 3)} → **{cat(C['cpk'])[0]}**.")
    st.info(f"Pada **{MON[mo]} {yr}** (Fase {'I' if sel.Fase == 1 else 'II'}), produksi mencapai **{fm(sel.Produksi_kg)} kg** dengan total cacat **{fm(sel.Cacat_kg)} kg ({fm(sel.Cacat_kg / sel.Produksi_kg * 100, 3)}%)**; "
            f"kategori cacat terbesar adalah **{big[0]}** ({fm(big[1])} kg). Level kualitas setara **{fm(dpmo)} DPMO**" + (f" (≈ {fm(sg, 2)} sigma)" if sg == sg else "") +
            f". Nilai δ = {fm(sel.delta, 6)} {est} {capt} Stabilitas dan kapabilitas adalah dua penilaian berbeda dan tidak boleh disamakan.")
