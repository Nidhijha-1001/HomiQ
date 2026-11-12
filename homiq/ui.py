"""Shared visual theme and small UI helpers for the Streamlit app."""

import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
        :root {
            --navy: #101c36;
            --ink: #17243d;
            --muted: #71809a;
            --blue: #4263eb;
            --cyan: #45c7d8;
            --line: #e8edf5;
            --surface: #ffffff;
            --soft: #f5f8fc;
        }
        html { scroll-behavior: smooth; }
        body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
        .stApp { background: #f6f8fc; }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stToolbar"], footer { visibility: hidden; }
        [data-testid="stMainBlockContainer"] { max-width: 1240px; padding-top: 1.1rem; }
        h1, h2, h3 { font-family: 'Manrope', sans-serif; letter-spacing: -0.035em; color: var(--navy); }
        h1 { font-weight: 800; }
        .brand-row { display:flex; align-items:center; gap:10px; padding: 2px 0 12px; }
        .brand-mark { width:38px; height:38px; border-radius:13px; display:grid; place-items:center;
            color:#fff; background:linear-gradient(145deg,#4263eb,#45c7d8); font-size:21px;
            box-shadow:0 7px 18px rgba(66,99,235,.22); }
        .brand-lockup { display:flex; flex-direction:column; align-items:flex-start; line-height:1.1; }
        .brand-name { font-family:'Manrope',sans-serif; color:var(--navy); font-size:20px; font-weight:800; letter-spacing:-.04em; }
        .brand-tag { color:var(--muted); font-size:9px; margin-top:4px; font-weight:700; letter-spacing:.11em; }
        .eyebrow { color:#4263eb; font-size:11px; font-weight:700; letter-spacing:.14em; text-transform:uppercase; }
        .hero-shell { position:relative; overflow:hidden; min-height:310px; padding:42px 44px;
            border-radius:27px; color:#fff; background:radial-gradient(ellipse at 86% 6%,rgba(69,199,216,.22),transparent 36%),
            linear-gradient(115deg,#101c36 0%,#17294f 64%,#1b3764 100%); box-shadow:0 20px 50px rgba(16,28,54,.15);
            animation: rise-in .55s ease both; }
        .hero-copy { position:relative; z-index:2; max-width:590px; }
        .hero-copy .eyebrow { color:#7ee1eb; }
        .hero-copy h1 { color:white; font-size:clamp(34px,4.2vw,56px); line-height:1.04; margin:14px 0 15px; }
        .hero-copy p { color:#c2cde0; font-size:15px; max-width:480px; line-height:1.7; margin:0; }
        .hero-art { position:absolute; width:45%; height:100%; right:0; top:0; opacity:.96; }
        .hero-art svg { width:100%; height:100%; }
        .hero-glow { filter: drop-shadow(0 14px 24px rgba(69,199,216,.18)); }
        .section-kicker { color:var(--muted); font-size:13px; line-height:1.6; margin-top:-8px; }
        .stat-card { background:#fff; padding:17px 19px; border:1px solid var(--line); border-radius:17px;
            box-shadow:0 5px 16px rgba(30,53,92,.035); min-height:90px; animation: rise-in .45s ease both; }
        .stat-label { color:var(--muted); font-size:12px; font-weight:600; }
        .stat-value { color:var(--navy); font-family:'Manrope',sans-serif; font-size:25px; font-weight:800; margin-top:5px; letter-spacing:-.04em; }
        .stat-note { color:#8190a8; font-size:11px; margin-top:2px; }
        .feature-card { height:100%; padding:21px; border-radius:17px; background:#fff; border:1px solid var(--line);
            transition:transform .18s ease, box-shadow .18s ease; }
        .feature-card:hover { transform:translateY(-3px); box-shadow:0 12px 28px rgba(25,46,83,.09); }
        .feature-icon { display:grid; place-items:center; width:40px; height:40px; border-radius:13px;
            color:#4263eb; background:#edf1ff; font-size:18px; margin-bottom:13px; }
        .feature-card h3 { font-size:15px; margin:0 0 7px; letter-spacing:-.02em; }
        .feature-card p { color:var(--muted); font-size:12px; line-height:1.65; margin:0; }
        .info-strip { padding:13px 16px; border:1px solid #dfeaf7; border-radius:13px; color:#53627b;
            background:linear-gradient(110deg,#f0f6ff,#f2fbfc); font-size:12px; line-height:1.6; }
        .summary-card { padding:15px 16px; border-radius:15px; background:#fff; border:1px solid var(--line); }
        .summary-label { color:var(--muted); font-size:11px; font-weight:600; }
        .summary-value { color:var(--navy); font-family:'Manrope',sans-serif; font-size:19px; font-weight:800; margin-top:5px; }
        .result-card { padding:27px 28px; border-radius:21px; color:#fff;
            background:linear-gradient(115deg,#101c36,#1a3562); box-shadow:0 15px 34px rgba(16,28,54,.15);
            animation: rise-in .4s ease both; }
        .result-label { color:#c4d2e7; font-size:13px; }
        .result-price { font-family:'Manrope',sans-serif; font-weight:800; font-size:clamp(36px,5vw,54px);
            line-height:1.15; letter-spacing:-.05em; margin:8px 0; animation: count-in .45s ease both; }
        .result-note { color:#bdd0e7; font-size:12px; }
        .empty-card { padding:25px; text-align:center; color:var(--muted); border:1px dashed #cfd9e8;
            border-radius:17px; background:rgba(255,255,255,.66); font-size:13px; }
        .footer-note { color:#8995a8; font-size:11px; text-align:center; padding:18px 0 8px; }
        div.stButton > button { border-radius:12px; min-height:43px; font-weight:700; transition:transform .15s ease, box-shadow .15s ease; }
        div.stButton > button:hover { transform:translateY(-2px); box-shadow:0 8px 19px rgba(66,99,235,.18); }
        div.stButton > button[kind="primary"] { border:0; background:linear-gradient(110deg,#4263eb,#5277f0); color:#fff; }
        div.stButton > button[kind="secondary"] { background:#fff; color:var(--navy); border:1px solid var(--line); }
        div.stButton > button[kind="secondary"]:hover { background:#eef3ff; border-color:#cdd8ff; }
        div[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); padding:14px 16px; border-radius:15px; }
        div[data-testid="stMetricLabel"] { color:var(--muted); }
        div[data-testid="stSelectbox"] label, div[data-testid="stSlider"] label, div[data-testid="stNumberInput"] label { font-weight:600; color:#263753; }
        [data-testid="stExpander"] { border:1px solid var(--line); border-radius:14px; background:#fff; }
        [data-testid="stVerticalBlockBorderWrapper"] { border-color:var(--line); border-radius:17px; }
        @keyframes rise-in { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
        @keyframes count-in { from { opacity:.3; transform:translateY(5px); } to { opacity:1; transform:translateY(0); } }
        @media (max-width:760px) {
            [data-testid="stMainBlockContainer"] { padding: .7rem 1rem 2rem; }
            .hero-shell { min-height:300px; padding:28px 24px 145px; }
            .hero-art { width:62%; height:54%; top:auto; bottom:-16px; opacity:.78; }
            .hero-copy h1 { font-size:36px; }
            .hero-copy p { max-width:100%; font-size:13px; }
            .stat-card { padding:14px; min-height:82px; }
            .stat-value { font-size:21px; }
            .result-card { padding:23px 20px; }
        }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after { scroll-behavior:auto !important; animation-duration:.01ms !important; transition-duration:.01ms !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_brand() -> None:
    st.markdown(
        """
        <div class="brand-row">
          <div class="brand-mark">⌂</div>
          <div class="brand-lockup"><span class="brand-name">HomiQ</span><span class="brand-tag">PROPERTY INTELLIGENCE</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat(label: str, value: str, note: str) -> None:
    st.markdown(
        f'<div class="stat-card"><div class="stat-label">{label}</div>'
        f'<div class="stat-value">{value}</div><div class="stat-note">{note}</div></div>',
        unsafe_allow_html=True,
    )
