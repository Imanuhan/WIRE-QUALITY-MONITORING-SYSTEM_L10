import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

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

MONTHS=["Januari","Februari","Maret","April","Mei","Juni","Juli","Agustus","September","Oktober","November","Desember"]
ALIASES={
 "Tanggal":["tanggal","date","periode"],
 "No_Subgrup":["no_subgrup","subgroup","subgrup","no_subgroup"],
 "Jumlah_Produksi":["jumlah_produksi","total_produksi","produksi","ukuran_sampel","sample_size","n"],
 "Total_Cacat":["total_cacat","jumlah_cacat","produk_cacat","defect","cacat","d"]
}
DEFECTS={
 "KTC_Kepala_Tidak_Center":["ktc_kepala_tidak_center","ktc"],
 "Kepala_Minus":["kepala_minus","k-"],
 "UC_Ujung_Cacat":["uc_ujung_cacat","uc"],
 "BC_Badan_Cacat":["bc_badan_cacat","bc"]
}

def standardize(df):
    df=df.copy()
    lookup={str(c).strip().lower().replace(" ","_"):c for c in df.columns}
    ren={}
    for target,aliases in ALIASES.items():
        for a in aliases:
            if a in lookup: ren[lookup[a]]=target; break
    for target,aliases in DEFECTS.items():
        for a in aliases:
            if a in lookup: ren[lookup[a]]=target; break
    return df.rename(columns=ren)

def prepare(df):
    df=standardize(df)
    required=list(ALIASES)
    missing=[c for c in required if c not in df.columns]
    if missing: raise ValueError("Kolom wajib tidak ditemukan: "+", ".join(missing))
    df["Tanggal"]=pd.to_datetime(df["Tanggal"],errors="coerce",dayfirst=True)
    for c in ["No_Subgrup","Jumlah_Produksi","Total_Cacat"]:
        df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.dropna(subset=["Tanggal","No_Subgrup","Jumlah_Produksi","Total_Cacat"]).copy()
    if df.empty: raise ValueError("Tidak ada baris data valid.")
    if (df["Jumlah_Produksi"]<=0).any(): raise ValueError("Jumlah produksi harus lebih dari nol.")
    if (df["Total_Cacat"]<0).any() or (df["Total_Cacat"]>df["Jumlah_Produksi"]).any():
        raise ValueError("Jumlah cacat harus antara 0 dan jumlah produksi.")
    for c in DEFECTS:
        if c in df: df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0)
    df["No_Subgrup"]=df["No_Subgrup"].astype(int)
    df["Jumlah_Produksi"]=df["Jumlah_Produksi"].astype(int)
    df["Total_Cacat"]=df["Total_Cacat"].astype(int)
    df["Tahun"]=df["Tanggal"].dt.year
    df["Bulan_No"]=df["Tanggal"].dt.month
    df["Bulan"]=df["Bulan_No"].map(lambda x:MONTHS[x-1])
    df["Proporsi_Cacat"]=df["Total_Cacat"]/df["Jumlah_Produksi"]
    df["Jumlah_Produk_Baik"]=df["Jumlah_Produksi"]-df["Total_Cacat"]
    return df.sort_values(["Tanggal","No_Subgrup"]).reset_index(drop=True)

def dummy():
    rng=np.random.default_rng(204324)
    rates={1:.025,2:.030,3:.042,4:.028,5:.026,6:.035,7:.024,8:.029,9:.031,10:.023,11:.027,12:.025}
    rows=[]
    for m in range(1,13):
        for s in range(1,26):
            n=int(rng.integers(950,1251)); p=rates[m]
            if (m,s) in [(3,8),(3,19),(6,12),(9,22)]: p+=.035
            if (m,s) in [(7,5),(10,17)]: p=max(.005,p-.018)
            d=int(rng.binomial(n,p)); parts=rng.multinomial(d,rng.dirichlet([4,2.5,2,1.5]))
            rows.append({"Tanggal":pd.Timestamp(2024,m,min(s,28)),"No_Subgrup":s,
             "Jumlah_Produksi":n,"Total_Cacat":d,
             "KTC_Kepala_Tidak_Center":parts[0],"Kepala_Minus":parts[1],
