import streamlit as st
from portal_utils import inject_custom_css, load_local_metrics

st.set_page_config(
    page_title="Credit Risk Scorecard Portal",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_css()

# ── Header ──────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:8px;">
    <span style="
        background:linear-gradient(135deg,rgba(79,142,247,.15),rgba(79,142,247,.04));
        border:1px solid rgba(79,142,247,.25);
        border-radius:20px;padding:4px 14px;
        font-size:0.78rem;color:#4f8ef7;font-weight:600;letter-spacing:.05em;
    ">PRODUCTION SCORECARD v1.0</span>
</div>
""", unsafe_allow_html=True)

st.title("🛡️ Credit Risk Decisioning Portal")
st.markdown(
    "<p style='color:#cbd5e1;font-size:1.05rem;margin-top:-6px;font-weight:700;'>"
    "Home Credit Default Risk · WoE Scorecard · Logistic Regression + Platt Calibration"
    "</p>",
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)

# ── Layout ───────────────────────────────────────────────────────────
col_metrics, col_info = st.columns([2, 3], gap="large")

with col_metrics:
    st.subheader("📊 Model Performance")

    metrics_data = load_local_metrics()
    tab_test, tab_train = st.tabs(["  Test Partition  ", "  Train Partition  "])

    ACCENT_COLORS = ["#10b981", "#4f8ef7", "#f59e0b", "#f43f5e"]

    def render_metrics(m):
        rows = [
            ("Area Under ROC (AUC)",        m.get("auc",         0.752), ACCENT_COLORS[0]),
            ("Gini Coefficient",             m.get("gini",        0.504), ACCENT_COLORS[1]),
            ("Kolmogorov-Smirnov (KS)",      m.get("ks",          0.392), ACCENT_COLORS[2]),
            ("Brier Score Loss",             m.get("brier_score", 0.081), ACCENT_COLORS[3]),
        ]
        html = ""
        for title, val, color in rows:
            html += f"""
            <div class="metric-container" style="border-left-color:{color};">
                <div>
                    <div class="metric-title">{title}</div>
                    <div class="metric-value">{val:.4f}</div>
                </div>
                <div style="width:42px;height:42px;border-radius:50%;
                     background:linear-gradient(135deg,{color}22,{color}08);
                     border:2px solid {color}44;display:flex;
                     align-items:center;justify-content:center;
                     font-size:1.2rem;">
                </div>
            </div>
            """
        return html

    with tab_test:
        st.markdown(render_metrics(metrics_data.get("test", {})), unsafe_allow_html=True)
    with tab_train:
        st.markdown(render_metrics(metrics_data.get("train", {})), unsafe_allow_html=True)

with col_info:
    st.subheader("⚖️ Credit Decision Bands")
    st.markdown(
        "<p style='color:#cbd5e1;font-size:0.95rem;margin-top:-4px;font-weight:700;'>Cutoff: <b style='color:#3b82f6;font-weight:900;'>725 points</b> · Range 300–850</p>",
        unsafe_allow_html=True
    )

    bands = [
        ("≥ 748",    "Very Low Risk",  "Approve",             "#10b981", "#10b981"),
        ("733–747",  "Low Risk",       "Approve",             "#34d399", "#34d399"),
        ("725–732",  "Medium Risk",    "Approve",             "#4f8ef7", "#4f8ef7"),
        ("705–724",  "Elevated Risk",  "Manual Review",       "#f59e0b", "#f59e0b"),
        ("< 705",    "Very High Risk", "Reject",              "#f43f5e", "#f43f5e"),
    ]

    rows_html = ""
    for score, tier, action, score_color, action_color in bands:
        rows_html += f"""<tr>
<td style="padding:11px 14px;font-weight:800;color:{score_color} !important;font-size:1.05rem;">{score}</td>
<td style="padding:11px 14px;color:#000000 !important;font-weight:700;">{tier}</td>
<td style="padding:11px 14px;font-weight:800;color:{action_color} !important;">{action}</td>
</tr>"""

    st.markdown(f"""<div class="glass-card" style="padding:0;overflow:hidden;">
<table class="portal-table">
<thead>
<tr>
<th>Score Band</th>
<th>Risk Tier</th>
<th>Action</th>
</tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>
</div>""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Architecture Flow ─────────────────────────────────────────────────
st.subheader("🏗️ MLOps Pipeline Architecture")

steps = [
    ("1", "Raw Ingestion",    "7 Tables · 10% Sample",    "🗂️"),
    ("2", "Feature Eng.",     "Bureau · POS · CC · Install","⚙️"),
    ("3", "WoE Binning",      "82 Features · IV Filter",   "📊"),
    ("4", "Bin Optimisation", "Corr Prune · VIF · 36 Feat","🔬"),
    ("5", "Training",         "LogReg + Platt Scaling",    "🧠"),
    ("6", "Scorecard API",    "FastAPI Real-time API",     "🚀"),
]

arrows = "".join([
    f"""<div class="step-card">
<div style="font-size:1.4rem;margin-bottom:6px;">{icon}</div>
<div class="step-title">{name}</div>
<div class="step-sub">{sub}</div>
</div>
{"<div style='font-size:1.3rem;align-self:center;color:#1e2d4a;'>›</div>" if i < len(steps)-1 else ""}"""
    for i, (num, name, sub, icon) in enumerate(steps)
])

st.markdown(f"""<div class="glass-card">
<div style="display:flex;justify-content:center;flex-wrap:wrap;align-items:center;">
{arrows}
</div>
</div>""", unsafe_allow_html=True)

# ── ECL Formula ───────────────────────────────────────────────────────
st.markdown("""
<div class="glass-card" style="border-left:4px solid #4f8ef7;">
    <div style="display:flex;gap:40px;flex-wrap:wrap;align-items:center;">
        <div>
            <div class="metric-title">Expected Credit Loss Formula</div>
            <div style="font-size:1.4rem;font-weight:700;color:#e2e8f0;margin-top:6px;">
                ECL = PD &times; LGD &times; EAD
            </div>
        </div>
        <div style="display:flex;gap:24px;flex-wrap:wrap;">
            <div>
                <div class="metric-title">PD</div>
                <div style="color:#4f8ef7;font-weight:600;">Calibrated via Platt Scaling</div>
            </div>
            <div>
                <div class="metric-title">LGD</div>
                <div style="color:#f59e0b;font-weight:600;">45% (Retail Unsecured)</div>
            </div>
            <div>
                <div class="metric-title">EAD</div>
                <div style="color:#10b981;font-weight:600;">Requested Loan Amount</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
