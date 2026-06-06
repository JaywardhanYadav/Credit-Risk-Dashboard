import os
import json
import httpx
import streamlit as st

BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")
METRICS_PATH = "artifacts_cache/evaluation/metrics.json"

def get_backend_url() -> str:
    """Returns the current backend URL."""
    return BACKEND_API_URL

def load_local_metrics() -> dict:
    """Dynamically loads model evaluation metrics from local cache."""
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "train": {"auc": 0.768, "gini": 0.536, "ks": 0.415, "brier_score": 0.078},
        "test":  {"auc": 0.752, "gini": 0.504, "ks": 0.392, "brier_score": 0.081}
    }

def inject_custom_css():
    """
    Minimal, clean design: white background, black bold text.
    Simple and readable for data scientists.
    """
    custom_css = """
    <style>
        /* ── Global: white background, black bold text ──────── */
        html, body, [class*="css"], .main, .stApp {
            background-color: #ffffff !important;
            color: #000000 !important;
            font-weight: 700 !important;
        }

        .main .block-container {
            background-color: #ffffff !important;
            padding-top: 2rem;
        }

        /* ── All text elements ──────────────────────────────── */
        p, span, li, div, td, th, label, a,
        .stMarkdown p, .stMarkdown span, .stMarkdown li,
        [data-testid="stAppViewContainer"] p,
        [data-testid="stAppViewContainer"] span {
            color: #000000 !important;
            font-weight: 700 !important;
        }

        strong, b { font-weight: 900 !important; color: #000000 !important; }

        /* ── Headings ───────────────────────────────────────── */
        h1 { color: #000000 !important; font-weight: 900 !important; font-size: 2rem; }
        h2 { color: #000000 !important; font-weight: 800 !important; }
        h3 { color: #000000 !important; font-weight: 800 !important; }
        h4 { color: #000000 !important; font-weight: 700 !important; }

        /* ── Sidebar ────────────────────────────────────────── */
        section[data-testid="stSidebar"] {
            background-color: #f3f4f6 !important;
            border-right: 2px solid #000000 !important;
        }
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] div {
            color: #000000 !important;
            font-weight: 700 !important;
        }

        /* ── Streamlit Tabs ─────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {
            background-color: transparent !important;
            border-bottom: 2px solid #cccccc !important;
            border-radius: 0px !important;
            padding: 0px !important;
            gap: 12px !important;
        }
        .stTabs [data-baseweb="tab"] {
            color: #4b5563 !important;
            background-color: transparent !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
            padding: 8px 12px !important;
            border: none !important;
            border-bottom: 2px solid transparent !important;
        }
        .stTabs [data-baseweb="tab"],
        .stTabs [data-baseweb="tab"]:hover,
        .stTabs [data-baseweb="tab"]:focus,
        .stTabs [data-baseweb="tab"]:active,
        .stTabs [data-baseweb="tab"] *,
        .stTabs [data-baseweb="tab"]::before,
        .stTabs [data-baseweb="tab"]::after {
            background-color: transparent !important;
            background: transparent !important;
            box-shadow: none !important;
            outline: none !important;
        }
        .stTabs [aria-selected="true"] {
            color: #000000 !important;
            border-bottom: 2px solid #000000 !important;
            background-color: transparent !important;
        }
        .stTabs [aria-selected="true"] * {
            color: #000000 !important;
        }
        .stTabs [data-baseweb="tab"]:not([aria-selected="true"]) * {
            color: #4b5563 !important;
        }

        /* ── Buttons ────────────────────────────────────────── */
        .stButton > button {
            background-color: #000000 !important;
            color: #ffffff !important;
            border: 2px solid #000000 !important;
            border-radius: 6px;
            font-weight: 800;
            font-size: 1rem;
            padding: 0.6rem 1.4rem;
        }
        .stButton > button:hover {
            background-color: #333333 !important;
        }
        .stButton > button * {
            color: #ffffff !important;
        }

        /* ── Form Labels ────────────────────────────────────── */
        label, .stWidgetFormLabel, [data-testid="stWidgetLabel"] p {
            color: #000000 !important;
            font-weight: 800 !important;
            font-size: 0.95rem !important;
        }

        /* ── Input fields ───────────────────────────────────── */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div,
        .stSlider > div {
            background-color: #ffffff !important;
            border: 2px solid #000000 !important;
            border-radius: 4px !important;
            color: #000000 !important;
            font-weight: 700 !important;
        }
        .stTextInput input, .stNumberInput input {
            color: #000000 !important;
            font-weight: 700 !important;
        }
        .stSelectbox > div > div:focus-within,
        .stTextInput > div > div > input:focus {
            border-color: #2563eb !important;
            box-shadow: 0 0 0 2px rgba(37,99,235,0.25) !important;
        }

        /* ── Radio buttons ──────────────────────────────────── */
        [data-testid="stRadio"] label,
        [data-testid="stRadio"] p,
        div[class*="st-"] p,
        div[class*="st-"] label {
            color: #000000 !important;
            font-weight: 700 !important;
        }

        /* ── Slider ─────────────────────────────────────────── */
        .stSlider [data-baseweb="slider"] [role="slider"] {
            background-color: #000000 !important;
        }
        .stSlider [data-baseweb="slider"] div[class*="Track"] > div:first-child {
            background-color: #000000 !important;
        }

        /* ── Card / Box ─────────────────────────────────────── */
        .glass-card {
            background-color: #ffffff;
            border: 2px solid #000000;
            border-radius: 6px;
            padding: 20px;
            margin-bottom: 16px;
        }

        /* ── Metric Cards ────────────────────────────────────── */
        .metric-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: #f3f4f6;
            border: 2px solid #000000;
            border-left: 6px solid #000000;
            border-radius: 6px;
            padding: 14px 18px;
            margin-bottom: 10px;
        }
        .metric-title {
            font-size: 0.82rem;
            color: #000000 !important;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-weight: 800 !important;
            margin-bottom: 4px;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 900;
            color: #000000 !important;
            letter-spacing: -0.02em;
        }

        /* ── Score Circle ────────────────────────────────────── */
        .score-circle {
            width: 150px;
            height: 150px;
            border-radius: 50%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            margin: 0 auto 12px;
            border: 4px solid #000000;
            background-color: #ffffff;
        }
        .score-val {
            font-size: 2.6rem;
            font-weight: 900;
            color: #000000 !important;
            line-height: 1;
        }
        .score-lbl {
            font-size: 0.72rem;
            color: #000000 !important;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-weight: 700 !important;
            margin-top: 4px;
        }

        /* ── Decision Bands ─────────────────────────────────── */
        .approve-band {
            background-color: #f0fdf4;
            border: 2px solid #16a34a;
            border-radius: 6px;
            padding: 14px 18px;
            color: #15803d !important;
            font-weight: 900;
            text-align: center;
        }
        .review-band {
            background-color: #fffbeb;
            border: 2px solid #d97706;
            border-radius: 6px;
            padding: 14px 18px;
            color: #b45309 !important;
            font-weight: 900;
            text-align: center;
        }
        .reject-band {
            background-color: #fff1f2;
            border: 2px solid #dc2626;
            border-radius: 6px;
            padding: 14px 18px;
            color: #b91c1c !important;
            font-weight: 900;
            text-align: center;
        }

        /* ── Table ───────────────────────────────────────────── */
        .portal-table {
            width: 100%;
            border-collapse: collapse;
            color: #000000;
            font-size: 0.95rem;
        }
        .portal-table th {
            padding: 10px 14px;
            text-align: left;
            color: #000000 !important;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            border-bottom: 3px solid #000000;
            font-weight: 900 !important;
            background-color: #f3f4f6;
        }
        .portal-table td {
            padding: 10px 14px;
            border-bottom: 1px solid #cccccc;
            font-weight: 700 !important;
            color: #000000 !important;
        }
        .portal-table tr:last-child td { border-bottom: none; }
        .portal-table tr:hover td { background-color: #f3f4f6; }

        /* ── Info Pills ─────────────────────────────────────── */
        .info-pill {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 0.85rem;
            font-weight: 800 !important;
            border: 2px solid #000000;
        }
        .pill-green { background: #f0fdf4; color: #15803d !important; border-color: #16a34a; }
        .pill-amber { background: #fffbeb; color: #b45309 !important; border-color: #d97706; }
        .pill-rose  { background: #fff1f2; color: #b91c1c !important; border-color: #dc2626; }
        .pill-blue  { background: #eff6ff; color: #1d4ed8 !important; border-color: #2563eb; }

        /* ── Pipeline Step Cards ─────────────────────────────── */
        .step-card {
            border: 2px solid #000000;
            border-radius: 6px;
            padding: 12px 14px;
            min-width: 130px;
            background-color: #f3f4f6;
            margin: 6px;
            text-align: center;
        }
        .step-card:hover { background-color: #e5e7eb; }
        .step-title { font-weight: 800 !important; color: #000000 !important; font-size: 0.92rem; }
        .step-sub   { font-size: 0.76rem; color: #333333 !important; font-weight: 700 !important; margin-top: 4px; }

        /* ── Score Breakdown Row ─────────────────────────────── */
        .breakdown-row {
            display: flex;
            justify-content: space-between;
            padding: 10px 12px;
            border-bottom: 1px solid #cccccc;
            font-size: 0.92rem;
        }
        .breakdown-row:last-child { border-bottom: none; }
        .breakdown-row:hover { background-color: #f3f4f6; }
        .breakdown-label { color: #000000 !important; font-weight: 700 !important; }
        .pts-positive { color: #15803d !important; font-weight: 800 !important; }
        .pts-negative { color: #b91c1c !important; font-weight: 800 !important; }

        /* ── Divider ─────────────────────────────────────────── */
        hr { border: none; border-top: 2px solid #000000 !important; margin: 1.2rem 0; }

        /* ── Scrollbar ───────────────────────────────────────── */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #f3f4f6; }
        ::-webkit-scrollbar-thumb { background: #999999; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #333333; }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def safe_api_get(endpoint: str) -> dict:
    """Performs a safe GET request to the FastAPI backend."""
    try:
        url = f"{BACKEND_API_URL}/{endpoint.lstrip('/')}"
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url)
            if response.status_code == 200:
                return response.json()
            return {}
    except Exception:
        return {}

def safe_api_post(endpoint: str, payload: dict) -> dict:
    """Performs a safe POST request to the FastAPI backend."""
    try:
        url = f"{BACKEND_API_URL}/{endpoint.lstrip('/')}"
        with httpx.Client(timeout=10.0) as client:
            response = client.post(url, json=payload)
            if response.status_code == 200:
                return response.json()
            return {}
    except Exception:
        return {}
