"""
Crypto Scanner Pro - Streamlit + CoinGecko
Jalankan:  py -m streamlit run app.py
Butuh:     py -m pip install streamlit requests pandas plotly
"""
import os
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Crypto Scanner Pro", page_icon="🚀", layout="wide")

API = "https://api.coingecko.com/api/v3"

# ----------------------------------------------------------------- STYLE
st.markdown(
    """
<style>
.stApp {background: radial-gradient(circle at 15% 0%, #1b1f3b 0%, #0b0e1a 55%);}
.hero {padding: 22px 28px; border-radius: 18px; margin-bottom: 18px;
  background: linear-gradient(120deg, #6366f1 0%, #8b5cf6 45%, #06b6d4 100%);
  box-shadow: 0 10px 40px rgba(99,102,241,.35);}
.hero h1 {margin: 0; color: #fff; font-size: 2rem;}
.hero p {margin: 4px 0 0; color: #e0e7ff;}
.card {padding: 16px 18px; border-radius: 16px; background: rgba(255,255,255,.05);
  border: 1px solid rgba(255,255,255,.09); backdrop-filter: blur(8px);}
.card .lbl {font-size: .78rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .06em;}
.card .val {font-size: 1.55rem; font-weight: 700; color: #f8fafc;}
.card .sub {font-size: .85rem;}
.up {color: #22c55e;} .down {color: #ef4444;} .mid {color: #f59e0b;}
div[data-testid="stTabs"] button {font-weight: 600;}
section[data-testid="stSidebar"] {background: #0f1224;}
</style>
""",
    unsafe_allow_html=True,
)


def card(label, value, sub="", cls=""):
    return (
        f'<div class="card"><div class="lbl">{label}</div>'
        f'<div class="val">{value}</div><div class="sub {cls}">{sub}</div></div>'
    )


# ------------------------------------------------------------------ DATA
@st.cache_data(ttl=120, show_spinner=False)
def get_markets(vs, pages):
    rows = []
    for p in range(1, pages + 1):
        r = requests.get(
            f"{API}/coins/markets",
            params={
                "vs_currency": vs, "order": "market_cap_desc", "per_page": 100,
                "page": p, "sparkline": "true",
                "price_change_percentage": "1h,24h,7d",
            },
            timeout=25,
        )
        r.raise_for_status()
        rows += r.json()
    return pd.DataFrame(rows)


@st.cache_data(ttl=300, show_spinner=False)
def get_global():
    try:
        return requests.get(f"{API}/global", timeout=15).json()["data"]
    except Exception:
        return {}


@st.cache_data(ttl=600, show_spinner=False)
def get_fng():
    try:
        d = requests.get("https://api.alternative.me/fng/", timeout=15).json()["data"][0]
        return int(d["value"]), d["value_classification"]
    except Exception:
        return None, "N/A"


@st.cache_data(ttl=300, show_spinner=False)
def get_trending():
    try:
        coins = requests.get(f"{API}/search/trending", timeout=15).json()["coins"]
        return [c["item"] for c in coins]
    except Exception:
        return []


@st.cache_data(ttl=300, show_spinner=False)
def get_ohlc(coin_id, vs, days):
    r = requests.get(
        f"{API}/coins/{coin_id}/ohlc",
        params={"vs_currency": vs, "days": days}, timeout=20,
    )
    r.raise_for_status()
    df = pd.DataFrame(r.json(), columns=["t", "open", "high", "low", "close"])
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    return df


def rsi(prices, n=14):
    s = pd.Series(prices, dtype=float)
    if len(s) < n + 2:
        return np.nan
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return float((100 - 100 / (1 + rs)).iloc[-1])


def make_signal(r):
    tags = []
    if pd.notna(r.rsi):
        if r.rsi < 30:
            tags.append("🟢 Oversold")
        elif r.rsi > 70:
            tags.append("🔴 Overbought")
    if r.h1 > 0 and r.h24 > 0 and r.d7 > 0:
        tags.append("🚀 Bullish")
    if r.h1 < 0 and r.h24 < 0 and r.d7 < 0:
        tags.append("🔻 Bearish")
    if r.vol_mcap > 25:
        tags.append("🔥 Volume Spike")
    if r.ath_pct > -10:
        tags.append("👑 Dekat ATH")
    return " · ".join(tags) if tags else "—"


def prep(df):
    d = df.copy()
    d["spark"] = d["sparkline_in_7d"].apply(
        lambda s: s.get("price", []) if isinstance(s, dict) else []
    )
    d["rsi"] = d["spark"].apply(rsi)
    d["h1"] = d["price_change_percentage_1h_in_currency"].fillna(0)
    d["h24"] = d["price_change_percentage_24h_in_currency"].fillna(0)
    d["d7"] = d["price_change_percentage_7d_in_currency"].fillna(0)
    d["ath_pct"] = d["ath_change_percentage"].fillna(-100)
    d["vol_mcap"] = (d["total_volume"] / d["market_cap"].replace(0, np.nan) * 100).fillna(0)
    score = (
        50 + 2 * d.h24.clip(-10, 10) + 0.5 * d.d7.clip(-20, 20)
        + 3 * d.h1.clip(-3, 3) + d.vol_mcap.clip(0, 30) / 3
        - np.where(d.rsi > 80, 10, 0)
    )
    d["skor"] = score.clip(0, 100).round(0)
    d["sinyal"] = d.apply(make_signal, axis=1)
    d["symbol"] = d["symbol"].str.upper()
    return d


def send_telegram(token, chat_id, text):
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data={"chat_id": chat_id, "text": text}, timeout=15,
    )
    return r.ok


# --------------------------------------------------------------- SIDEBAR
with st.sidebar:
    st.markdown("## ⚙️ Pengaturan")
    vs = st.selectbox("Mata uang", ["usd", "idr", "eur"], index=0)
    pages = st.select_slider("Jumlah koin dipindai", [100, 200, 300], value=200) // 100
    st.markdown("### 🔎 Filter")
    q = st.text_input("Cari koin (nama / simbol)")
    min_vol = st.number_input("Min. volume 24j (USD/IDR)", 0, value=0, step=1_000_000)
    min_mcap = st.number_input("Min. market cap", 0, value=0, step=10_000_000)
    rsi_rng = st.slider("Rentang RSI", 0, 100, (0, 100))
    min_h24 = st.slider("Min. perubahan 24j (%)", -50, 50, -50)
    sig_opts = ["Oversold", "Overbought", "Bullish", "Bearish", "Volume Spike", "Dekat ATH"]
    sig_f = st.multiselect("Hanya tampilkan sinyal", sig_opts)
    st.markdown("### ⭐ Watchlist")
    watch = st.session_state.setdefault("watch", ["BTC", "ETH", "SOL"])
    st.markdown("### 🔄 Auto refresh")
    auto = st.checkbox("Refresh otomatis")
    every = st.select_slider("Interval (detik)", [60, 120, 300], value=120)
    st.markdown("### 📲 Telegram")
    tg_token = st.text_input("Bot token", value=os.getenv("TG_TOKEN", ""), type="password")
    tg_chat = st.text_input("Chat ID", value=os.getenv("TG_CHAT_ID", ""))
    if st.button("🔄 Refresh sekarang", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ------------------------------------------------------------------ LOAD
try:
    with st.spinner("Memindai pasar..."):
        data = prep(get_markets(vs, pages))
except Exception as e:
    st.error(f"Gagal mengambil data CoinGecko (mungkin kena rate limit, tunggu 1 menit). Detail: {e}")
    st.stop()

sym = "$" if vs == "usd" else ("Rp" if vs == "idr" else "€")

st.markdown(
    '<div class="hero"><h1>🚀 Crypto Scanner Pro</h1>'
    f'<p>Memindai {len(data)} koin · data CoinGecko · update {time.strftime("%H:%M:%S")}</p></div>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------- OVERVIEW
g = get_global()
fng_v, fng_c = get_fng()
mc = g.get("total_market_cap", {}).get(vs, 0)
mcc = g.get("market_cap_change_percentage_24h_usd", 0)
btc_d = g.get("market_cap_percentage", {}).get("btc", 0)
adv = int((data.h24 > 0).sum())

c1, c2, c3, c4, c5 = st.columns(5)
c1.markdown(card("Total Market Cap", f"{sym}{mc/1e12:,.2f}T", f"{mcc:+.2f}% 24j", "up" if mcc >= 0 else "down"), unsafe_allow_html=True)
c2.markdown(card("Dominasi BTC", f"{btc_d:.1f}%", "ETH " + f"{g.get('market_cap_percentage', {}).get('eth', 0):.1f}%"), unsafe_allow_html=True)
fcls = "down" if (fng_v or 50) < 40 else ("up" if (fng_v or 50) > 60 else "mid")
c3.markdown(card("Fear & Greed", fng_v if fng_v is not None else "–", fng_c, fcls), unsafe_allow_html=True)
c4.markdown(card("Koin Naik 24j", f"{adv}/{len(data)}", f"{adv/len(data)*100:.0f}% pasar hijau", "up" if adv > len(data) / 2 else "down"), unsafe_allow_html=True)
c5.markdown(card("Sinyal Aktif", int((data.sinyal != "—").sum()), "koin dengan sinyal"), unsafe_allow_html=True)
st.write("")

# ------------------------------------------------------------- FILTERING
f = data.copy()
if q:
    f = f[f["name"].str.contains(q, case=False) | f["symbol"].str.contains(q, case=False)]
f = f[(f.total_volume >= min_vol) & (f.market_cap >= min_mcap) & (f.h24 >= min_h24)]
f = f[(f.rsi.fillna(50) >= rsi_rng[0]) & (f.rsi.fillna(50) <= rsi_rng[1])]
for s in sig_f:
    f = f[f.sinyal.str.contains(s)]

with st.sidebar:
    watch = st.multiselect("Koin favorit", sorted(data.symbol.unique()), default=watch, key="watch")

# ------------------------------------------------------------------ TABS
t1, t2, t3, t4, t5 = st.tabs(["📊 Scanner", "🗺️ Heatmap", "📈 Chart Detail", "⭐ Watchlist", "🔥 Trending"])

COLS = {
    "market_cap_rank": st.column_config.NumberColumn("#", width="small"),
    "name": "Nama", "symbol": "Simbol",
    "current_price": st.column_config.NumberColumn("Harga", format=f"{sym}%.6g"),
    "h1": st.column_config.NumberColumn("1j %", format="%.2f"),
    "h24": st.column_config.NumberColumn("24j %", format="%.2f"),
    "d7": st.column_config.NumberColumn("7h %", format="%.2f"),
    "rsi": st.column_config.NumberColumn("RSI", format="%.0f"),
    "total_volume": st.column_config.NumberColumn("Volume 24j", format="%,.0f"),
    "market_cap": st.column_config.NumberColumn("Market Cap", format="%,.0f"),
    "spark": st.column_config.LineChartColumn("7 Hari", width="medium"),
    "skor": st.column_config.ProgressColumn("Skor", min_value=0, max_value=100, format="%d"),
    "sinyal": st.column_config.TextColumn("Sinyal", width="large"),
}
SHOW = ["market_cap_rank", "name", "symbol", "current_price", "h1", "h24", "d7",
        "rsi", "total_volume", "market_cap", "spark", "skor", "sinyal"]


def style_table(df):
    sty = df[SHOW].style.map(
        lambda v: "color:#22c55e" if isinstance(v, (int, float)) and v > 0
        else ("color:#ef4444" if isinstance(v, (int, float)) and v < 0 else ""),
        subset=["h1", "h24", "d7"],
    )
    return sty


with t1:
    st.caption(f"{len(f)} koin lolos filter")
    a, b = st.columns(2)
    with a:
        st.markdown("#### 🟢 Top Gainers 24j")
        st.dataframe(f.nlargest(5, "h24")[["symbol", "current_price", "h24", "skor"]],
                     hide_index=True, use_container_width=True,
                     column_config={"h24": st.column_config.NumberColumn("24j %", format="%.2f"),
                                    "current_price": st.column_config.NumberColumn("Harga", format=f"{sym}%.6g"),
                                    "skor": st.column_config.ProgressColumn("Skor", min_value=0, max_value=100, format="%d")})
    with b:
        st.markdown("#### 🔴 Top Losers 24j")
        st.dataframe(f.nsmallest(5, "h24")[["symbol", "current_price", "h24", "skor"]],
                     hide_index=True, use_container_width=True,
                     column_config={"h24": st.column_config.NumberColumn("24j %", format="%.2f"),
                                    "current_price": st.column_config.NumberColumn("Harga", format=f"{sym}%.6g"),
                                    "skor": st.column_config.ProgressColumn("Skor", min_value=0, max_value=100, format="%d")})
    sort_by = st.selectbox("Urutkan", ["skor", "h24", "h1", "d7", "rsi", "total_volume", "market_cap_rank"])
    asc = st.toggle("Naik (ascending)", value=(sort_by == "market_cap_rank"))
    view = f.sort_values(sort_by, ascending=asc)
    st.dataframe(style_table(view), column_config=COLS, hide_index=True,
                 use_container_width=True, height=560)
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Unduh CSV", view[SHOW[:-3] + ["skor", "sinyal"]].to_csv(index=False).encode(),
                       "crypto_scan.csv", "text/csv", use_container_width=True)
    if d2.button("📲 Kirim 5 sinyal teratas ke Telegram", use_container_width=True):
        if not (tg_token and tg_chat):
            st.warning("Isi Bot token dan Chat ID di sidebar dulu.")
        else:
            top = view.nlargest(5, "skor")
            msg = "🚀 Crypto Scanner - Top 5\n" + "\n".join(
                f"{r.symbol}: {sym}{r.current_price:,.6g} | 24j {r.h24:+.2f}% | skor {int(r.skor)} | {r.sinyal}"
                for r in top.itertuples()
            )
            st.success("Terkirim ✅") if send_telegram(tg_token, tg_chat, msg) else st.error("Gagal kirim, cek token/chat ID.")

with t2:
    n = st.slider("Jumlah koin di heatmap", 20, 150, 60)
    hm = f.nsmallest(n, "market_cap_rank")
    fig = px.treemap(hm, path=["symbol"], values="market_cap", color="h24",
                     color_continuous_scale=["#ef4444", "#1f2937", "#22c55e"],
                     color_continuous_midpoint=0, range_color=[-10, 10],
                     hover_data={"name": True, "current_price": ":.6g", "h24": ":.2f"})
    fig.update_traces(texttemplate="<b>%{label}</b><br>%{color:.1f}%")
    fig.update_layout(height=620, margin=dict(t=10, l=0, r=0, b=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    sc = px.scatter(hm, x="d7", y="h24", size="market_cap", color="rsi", hover_name="name",
                    color_continuous_scale="Turbo", labels={"d7": "7 hari %", "h24": "24 jam %"},
                    size_max=55, title="Momentum 24j vs 7h (warna = RSI)")
    sc.update_layout(height=450, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(sc, use_container_width=True)

with t3:
    names = f.sort_values("market_cap_rank")
    if names.empty:
        st.info("Tidak ada koin yang lolos filter.")
    else:
        pick = st.selectbox("Pilih koin", names["id"],
                            format_func=lambda i: f"{names.set_index('id').loc[i, 'name']} ({names.set_index('id').loc[i, 'symbol']})")
        days = st.radio("Periode", [1, 7, 14, 30, 90, 180], index=2, horizontal=True)
        row = names.set_index("id").loc[pick]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Harga", f"{sym}{row.current_price:,.6g}", f"{row.h24:+.2f}% (24j)")
        m2.metric("RSI (7h, per jam)", f"{row.rsi:.0f}" if pd.notna(row.rsi) else "–")
        m3.metric("Skor momentum", int(row.skor))
        m4.metric("Dari ATH", f"{row.ath_pct:.1f}%")
        try:
            oh = get_ohlc(pick, vs, days)
            oh["MA20"] = oh["close"].rolling(20).mean()
            fig = go.Figure(go.Candlestick(x=oh.t, open=oh.open, high=oh.high, low=oh.low,
                                           close=oh.close, name="Harga",
                                           increasing_line_color="#22c55e", decreasing_line_color="#ef4444"))
            fig.add_trace(go.Scatter(x=oh.t, y=oh.MA20, name="MA20", line=dict(color="#f59e0b", width=1.6)))
            fig.update_layout(height=560, template="plotly_dark", xaxis_rangeslider_visible=False,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              margin=dict(t=20, l=0, r=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Chart belum bisa dimuat (rate limit?). Coba lagi sebentar. ({e})")
        st.markdown(f"**Sinyal:** {row.sinyal}")

with t4:
    w = data[data.symbol.isin(watch)].sort_values("market_cap_rank")
    if w.empty:
        st.info("Pilih koin favorit di sidebar (bagian Watchlist).")
    else:
        st.dataframe(style_table(w), column_config=COLS, hide_index=True, use_container_width=True)
        st.markdown("#### 🔔 Alert harga")
        a1, a2, a3 = st.columns(3)
        who = a1.selectbox("Koin", w.symbol)
        mode = a2.selectbox("Kondisi", ["Naik di atas", "Turun di bawah"])
        lvl = a3.number_input("Level harga", min_value=0.0, value=float(w.set_index("symbol").loc[who, "current_price"]), format="%.6f")
        cur = float(w.set_index("symbol").loc[who, "current_price"])
        hit = cur >= lvl if mode == "Naik di atas" else cur <= lvl
        st.success(f"✅ Tercapai! {who} = {sym}{cur:,.6g}") if hit else st.info(f"Belum tercapai. {who} sekarang {sym}{cur:,.6g}")
        if hit and tg_token and tg_chat and st.button("Kirim alert ke Telegram"):
            send_telegram(tg_token, tg_chat, f"🔔 {who} {mode.lower()} {lvl:g} → sekarang {cur:,.6g}")
            st.success("Alert terkirim")

with t5:
    tr = get_trending()
    if not tr:
        st.info("Data trending belum tersedia.")
    else:
        cols = st.columns(4)
        for i, c in enumerate(tr[:8]):
            cols[i % 4].markdown(
                card(f"#{i+1} Trending", f"{c['symbol']}", c["name"] + (f" · rank {c['market_cap_rank']}" if c.get("market_cap_rank") else "")),
                unsafe_allow_html=True,
            )
            if i % 4 == 3:
                st.write("")

st.caption("⚠️ Bukan saran investasi. Sinyal dihitung otomatis dari data publik; selalu riset sendiri (DYOR).")

if auto:
    time.sleep(every)
    st.cache_data.clear()
    st.rerun()