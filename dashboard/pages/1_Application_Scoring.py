import streamlit as st
from portal_utils import inject_custom_css, safe_api_post

st.set_page_config(page_title="Application Scoring — Credit Portal", page_icon="📑", layout="wide")
inject_custom_css()

# ── Header ───────────────────────────────────────────────────────────
st.title("📑 Applicant Scoring Engine")
st.markdown(
    "<p style='color:#cbd5e1;font-size:1.05rem;margin-top:-6px;font-weight:700;'>"
    "Enter applicant details to compute credit score, calibrated PD, and lending decision.</p>",
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)

# ── Demo Presets ─────────────────────────────────────────────────────
st.markdown("**⚡ Quick Presets**")
preset = st.radio(
    "Select profile:",
    ["✏️ Manual Input", "🟢 Low Risk (Superprime)", "🟡 Medium Risk (Prime)", "🔴 High Risk (Subprime)"],
    horizontal=True
)

# Shared scorecard fields used by all presets (risk-specific values overridden per profile)
_COMMON_FIELDS = {
    "NAME_FAMILY_STATUS": "Married",
    "NAME_HOUSING_TYPE": "House / apartment",
    "DAYS_REGISTRATION": -3000.0,
    "DAYS_ID_PUBLISH": -2000.0,
    "REGION_POPULATION_RELATIVE": 0.018,
    "REGION_RATING_CLIENT": 2.0,
    "REG_CITY_NOT_LIVE_CITY": 0.0,
    "REG_CITY_NOT_WORK_CITY": 0.0,
    "LANDAREA_AVG": 0.04,
    "LIVINGAREA_AVG": 0.07,
    "FLAG_DOCUMENT_3": 1.0,
    "AMT_REQ_CREDIT_BUREAU_YEAR": 1.0,
}

presets_data = {
    "🟢 Low Risk (Superprime)": {
        **_COMMON_FIELDS,
        "AMT_INCOME_TOTAL": 300000.0, "AMT_CREDIT": 1000000.0, "AMT_ANNUITY": 45000.0,
        "AMT_GOODS_PRICE": 900000.0,
        "CNT_CHILDREN": 0, "CNT_FAM_MEMBERS": 2.0, "AGE": 45, "EMPLOYED_YRS": 15,
        "DAYS_LAST_PHONE_CHANGE": -1500, "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_INCOME_TYPE": "Commercial associate", "CODE_GENDER": "F",
        "FLAG_OWN_CAR": "Y", "FLAG_OWN_REALTY": "Y", "OWN_CAR_AGE": 5.0,
        "OCCUPATION_TYPE": "Managers", "ORGANIZATION_TYPE": "Bank",
        "EXT_SOURCE_1": 0.55, "EXT_SOURCE_2": 0.72, "EXT_SOURCE_3": 0.68,
        "BUREAU_LOANS_COUNT": 5.0, "BUREAU_DAYS_CREDIT_MAX": -300.0,
        "BUREAU_DEBT_SUM": 50000.0, "BUREAU_CREDIT_SUM": 400000.0, "BUREAU_ACTIVE_LOANS": 1.0,
        "PREV_APP_COUNT": 3.0, "PREV_ANNUITY_AVG": 15000.0, "PREV_REJECT_COUNT": 0.0,
        "DPD_MAX_INSTALL": 0.0, "DPD_MEAN_INSTALL": 0.0, "UNDERPAY_MAX_INSTALL": 0.0,
        "INSTALL_LATE_PAY_RATIO": 0.0, "DPD_MAX_CC": 0.0, "CC_UTILIZATION_AVG": 0.15,
        "DPD_MAX_POS": 0.0, "POS_FUTURE_INSTALMENTS_AVG": 10.0, "loan_amount": 100000.0
    },
    "🟡 Medium Risk (Prime)": {
        **_COMMON_FIELDS,
        "AMT_INCOME_TOTAL": 135000.0, "AMT_CREDIT": 450000.0, "AMT_ANNUITY": 22000.0,
        "AMT_GOODS_PRICE": 420000.0,
        "CNT_CHILDREN": 1, "CNT_FAM_MEMBERS": 3.0, "AGE": 32, "EMPLOYED_YRS": 4,
        "DAYS_LAST_PHONE_CHANGE": -400, "NAME_EDUCATION_TYPE": "Secondary / special",
        "NAME_INCOME_TYPE": "Working", "CODE_GENDER": "M",
        "FLAG_OWN_CAR": "N", "FLAG_OWN_REALTY": "Y", "OWN_CAR_AGE": 0.0,
        "OCCUPATION_TYPE": "Core staff", "ORGANIZATION_TYPE": "Business Entity Type 1",
        "EXT_SOURCE_1": 0.38, "EXT_SOURCE_2": 0.42, "EXT_SOURCE_3": 0.40,
        "BUREAU_LOANS_COUNT": 3.0, "BUREAU_DAYS_CREDIT_MAX": -50.0,
        "BUREAU_DEBT_SUM": 120000.0, "BUREAU_CREDIT_SUM": 300000.0, "BUREAU_ACTIVE_LOANS": 1.0,
        "PREV_APP_COUNT": 2.0, "PREV_ANNUITY_AVG": 18000.0, "PREV_REJECT_COUNT": 0.0,
        "DPD_MAX_INSTALL": 5.0, "DPD_MEAN_INSTALL": 1.2, "UNDERPAY_MAX_INSTALL": 100.0,
        "INSTALL_LATE_PAY_RATIO": 0.12, "DPD_MAX_CC": 0.0, "CC_UTILIZATION_AVG": 0.45,
        "DPD_MAX_POS": 0.0, "POS_FUTURE_INSTALMENTS_AVG": 6.0, "loan_amount": 50000.0
    },
    "🔴 High Risk (Subprime)": {
        **_COMMON_FIELDS,
        "NAME_FAMILY_STATUS": "Single / not married",
        "NAME_HOUSING_TYPE": "Rented apartment",
        "DAYS_REGISTRATION": -200.0,
        "DAYS_ID_PUBLISH": -100.0,
        "REGION_POPULATION_RELATIVE": 0.005,
        "REGION_RATING_CLIENT": 3.0,
        "REG_CITY_NOT_LIVE_CITY": 1.0,
        "REG_CITY_NOT_WORK_CITY": 1.0,
        "LANDAREA_AVG": 0.01,
        "LIVINGAREA_AVG": 0.02,
        "FLAG_DOCUMENT_3": 0.0,
        "AMT_REQ_CREDIT_BUREAU_YEAR": 3.0,
        "AMT_INCOME_TOTAL": 80000.0, "AMT_CREDIT": 350000.0, "AMT_ANNUITY": 25000.0,
        "AMT_GOODS_PRICE": 340000.0,
        "CNT_CHILDREN": 2, "CNT_FAM_MEMBERS": 4.0, "AGE": 25, "EMPLOYED_YRS": 1,
        "DAYS_LAST_PHONE_CHANGE": -50, "NAME_EDUCATION_TYPE": "Secondary / special",
        "NAME_INCOME_TYPE": "Working", "CODE_GENDER": "M",
        "FLAG_OWN_CAR": "N", "FLAG_OWN_REALTY": "N", "OWN_CAR_AGE": 0.0,
        "OCCUPATION_TYPE": "Low-skill Laborers", "ORGANIZATION_TYPE": "Industry: type 13",
        "EXT_SOURCE_1": 0.12, "EXT_SOURCE_2": 0.10, "EXT_SOURCE_3": 0.15,
        "BUREAU_LOANS_COUNT": 6.0, "BUREAU_DAYS_CREDIT_MAX": -15.0,
        "BUREAU_DEBT_SUM": 200000.0, "BUREAU_CREDIT_SUM": 220000.0, "BUREAU_ACTIVE_LOANS": 3.0,
        "PREV_APP_COUNT": 4.0, "PREV_ANNUITY_AVG": 20000.0, "PREV_REJECT_COUNT": 2.0,
        "DPD_MAX_INSTALL": 45.0, "DPD_MEAN_INSTALL": 12.5, "UNDERPAY_MAX_INSTALL": 1500.0,
        "INSTALL_LATE_PAY_RATIO": 0.65, "DPD_MAX_CC": 15.0, "CC_UTILIZATION_AVG": 0.95,
        "DPD_MAX_POS": 4.0, "POS_FUTURE_INSTALMENTS_AVG": 1.2, "loan_amount": 30000.0
    }
}

_manual_defaults = {
    **_COMMON_FIELDS,
    "AMT_GOODS_PRICE": 405000.0,
    "OWN_CAR_AGE": 0.0,
    "OCCUPATION_TYPE": "Core staff",
    "ORGANIZATION_TYPE": "Business Entity Type 1",
    "EXT_SOURCE_1": 0.40,
    "EXT_SOURCE_2": 0.45,
    "EXT_SOURCE_3": 0.42,
}

data_fill = {**_manual_defaults, **presets_data.get(preset, {})}
st.markdown("<br>", unsafe_allow_html=True)

# ── Application Form ─────────────────────────────────────────────────
with st.form("scoring_form"):
    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown("#### 👤 Personal Details")
        income = st.number_input("Annual Income (₹)", min_value=1000.0, max_value=10_000_000.0,
                                 value=data_fill.get("AMT_INCOME_TOTAL", 150000.0), step=5000.0)
        age_yrs = st.number_input("Age (Years)", min_value=18, max_value=100,
                                  value=data_fill.get("AGE", 35), step=1)
        emp_yrs = st.number_input("Employment Duration (Years)", min_value=0, max_value=60,
                                  value=data_fill.get("EMPLOYED_YRS", 5), step=1)
        education = st.selectbox("Education Level",
            ["Secondary / special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"],
            index=["Secondary / special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"]
                   .index(data_fill.get("NAME_EDUCATION_TYPE", "Secondary / special")))
        income_type = st.selectbox("Income Source",
            ["Working", "Commercial associate", "State servant", "Pensioner", "Unemployed"],
            index=["Working", "Commercial associate", "State servant", "Pensioner", "Unemployed"]
                   .index(data_fill.get("NAME_INCOME_TYPE", "Working")))
        gender   = st.selectbox("Gender", ["F", "M"], index=["F","M"].index(data_fill.get("CODE_GENDER","F")))
        own_car  = st.radio("Owns Car?",     ["N", "Y"], index=["N","Y"].index(data_fill.get("FLAG_OWN_CAR","N")),  horizontal=True)
        own_realty = st.radio("Owns Realty?",  ["N", "Y"], index=["N","Y"].index(data_fill.get("FLAG_OWN_REALTY","Y")), horizontal=True)

    with col2:
        st.markdown("#### 💳 Credit Request")
        credit   = st.number_input("Loan Credit Amount (₹)", min_value=1000.0, max_value=10_000_000.0,
                                   value=data_fill.get("AMT_CREDIT", 450000.0), step=10000.0)
        annuity  = st.number_input("Annual Repayment (₹)", min_value=100.0, max_value=1_000_000.0,
                                   value=data_fill.get("AMT_ANNUITY", 22000.0), step=1000.0)
        loan_ead = st.number_input("Cash Requested / EAD (₹)", min_value=100.0, max_value=10_000_000.0,
                                   value=data_fill.get("loan_amount", 50000.0), step=5000.0)

        st.markdown("#### 🏛️ Bureau History")
        bureau_loans  = st.number_input("Total Bureau Loans",     min_value=0, max_value=100, value=int(data_fill.get("BUREAU_LOANS_COUNT", 3)))
        bureau_active = st.number_input("Active Bureau Loans",    min_value=0, max_value=50,  value=int(data_fill.get("BUREAU_ACTIVE_LOANS", 1)))
        bureau_debt   = st.number_input("Total Bureau Debt (₹)",  min_value=0.0, max_value=50_000_000.0,
                                        value=float(data_fill.get("BUREAU_DEBT_SUM", 60000.0)), step=10000.0)
        bureau_credit = st.number_input("Total Bureau Limit (₹)", min_value=0.0, max_value=50_000_000.0,
                                        value=float(data_fill.get("BUREAU_CREDIT_SUM", 200000.0)), step=10000.0)
        bureau_days   = st.number_input("Last Bureau Loan (Days Ago, negative)", min_value=-10000, max_value=0,
                                        value=int(data_fill.get("BUREAU_DAYS_CREDIT_MAX", -300)))

    with col3:
        st.markdown("#### ⚠️ Delinquency & Behaviour")
        phone_days = st.slider("Days Since Phone Change", min_value=-5000, max_value=0,
                               value=int(data_fill.get("DAYS_LAST_PHONE_CHANGE", -400)))
        prev_apps    = st.number_input("Previous Applications",  min_value=0, max_value=100, value=int(data_fill.get("PREV_APP_COUNT", 2)))
        prev_rejects = st.number_input("Applications Rejected",  min_value=0, max_value=50,  value=int(data_fill.get("PREV_REJECT_COUNT", 0)))
        prev_annuity = st.number_input("Avg Previous Annuity (₹)", min_value=0.0, max_value=1_000_000.0,
                                       value=float(data_fill.get("PREV_ANNUITY_AVG", 10000.0)), step=1000.0)
        st.markdown("<br>", unsafe_allow_html=True)
        late_ratio   = st.slider("Installment Late Payment Ratio", 0.0, 1.0,
                                 value=float(data_fill.get("INSTALL_LATE_PAY_RATIO", 0.05)), step=0.01)
        install_dpd  = st.number_input("Max Installment Delay (Days)", min_value=0, max_value=1000, value=int(data_fill.get("DPD_MAX_INSTALL", 0)))
        install_undr = st.number_input("Max Installment Underpay (₹)", min_value=0.0, max_value=500000.0,
                                       value=float(data_fill.get("UNDERPAY_MAX_INSTALL", 0.0)), step=100.0)
        cc_util = st.slider("Avg Credit Card Utilization", 0.0, 1.5,
                            value=float(data_fill.get("CC_UTILIZATION_AVG", 0.0)), step=0.05)
        cc_dpd  = st.number_input("Max Credit Card Delay (Days)", min_value=0, max_value=1000, value=int(data_fill.get("DPD_MAX_CC", 0)))
        pos_dpd = st.number_input("Max POS Delay (Days)", min_value=0, max_value=1000, value=int(data_fill.get("DPD_MAX_POS", 0)))
        pos_fut = st.number_input("Avg POS Remaining Instalments", min_value=0.0, max_value=100.0,
                                  value=float(data_fill.get("POS_FUTURE_INSTALMENTS_AVG", 6.0)))

    submit = st.form_submit_button("🛡️  Score Applicant", use_container_width=True)

# ── Results ───────────────────────────────────────────────────────────
if submit:
    days_birth    = -int(age_yrs  * 365.25)
    days_employed = -int(emp_yrs  * 365.25)

    payload = {
        "AMT_INCOME_TOTAL": income, "AMT_CREDIT": credit, "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": float(data_fill.get("AMT_GOODS_PRICE", credit * 0.9)),
        "CNT_CHILDREN": int(data_fill.get("CNT_CHILDREN", 0)),
        "CNT_FAM_MEMBERS": float(data_fill.get("CNT_FAM_MEMBERS", 2.0)),
        "DAYS_BIRTH": days_birth, "DAYS_EMPLOYED": days_employed,
        "DAYS_LAST_PHONE_CHANGE": float(phone_days),
        "DAYS_REGISTRATION": float(data_fill.get("DAYS_REGISTRATION", -3000.0)),
        "DAYS_ID_PUBLISH": float(data_fill.get("DAYS_ID_PUBLISH", -2000.0)),
        "NAME_EDUCATION_TYPE": education, "NAME_INCOME_TYPE": income_type,
        "NAME_FAMILY_STATUS": data_fill.get("NAME_FAMILY_STATUS", "Married"),
        "NAME_HOUSING_TYPE": data_fill.get("NAME_HOUSING_TYPE", "House / apartment"),
        "CODE_GENDER": gender, "FLAG_OWN_CAR": own_car, "FLAG_OWN_REALTY": own_realty,
        "OWN_CAR_AGE": float(data_fill.get("OWN_CAR_AGE", 0.0 if own_car == "N" else 3.0)),
        "OCCUPATION_TYPE": data_fill.get("OCCUPATION_TYPE", "Core staff"),
        "ORGANIZATION_TYPE": data_fill.get("ORGANIZATION_TYPE", "Business Entity Type 1"),
        "REGION_POPULATION_RELATIVE": float(data_fill.get("REGION_POPULATION_RELATIVE", 0.018)),
        "REGION_RATING_CLIENT": float(data_fill.get("REGION_RATING_CLIENT", 2.0)),
        "REG_CITY_NOT_LIVE_CITY": float(data_fill.get("REG_CITY_NOT_LIVE_CITY", 0.0)),
        "REG_CITY_NOT_WORK_CITY": float(data_fill.get("REG_CITY_NOT_WORK_CITY", 0.0)),
        "EXT_SOURCE_1": float(data_fill.get("EXT_SOURCE_1", 0.40)),
        "EXT_SOURCE_2": float(data_fill.get("EXT_SOURCE_2", 0.45)),
        "EXT_SOURCE_3": float(data_fill.get("EXT_SOURCE_3", 0.42)),
        "LANDAREA_AVG": float(data_fill.get("LANDAREA_AVG", 0.04)),
        "LIVINGAREA_AVG": float(data_fill.get("LIVINGAREA_AVG", 0.07)),
        "FLAG_DOCUMENT_3": float(data_fill.get("FLAG_DOCUMENT_3", 1.0)),
        "AMT_REQ_CREDIT_BUREAU_YEAR": float(data_fill.get("AMT_REQ_CREDIT_BUREAU_YEAR", 1.0)),
        "BUREAU_LOANS_COUNT": float(bureau_loans), "BUREAU_DAYS_CREDIT_MAX": float(bureau_days),
        "BUREAU_DEBT_SUM": float(bureau_debt), "BUREAU_CREDIT_SUM": float(bureau_credit),
        "BUREAU_ACTIVE_LOANS": float(bureau_active),
        "PREV_APP_COUNT": float(prev_apps), "PREV_ANNUITY_AVG": float(prev_annuity),
        "PREV_REJECT_COUNT": float(prev_rejects),
        "DPD_MAX_INSTALL": float(install_dpd), "DPD_MEAN_INSTALL": float(install_dpd * 0.2),
        "UNDERPAY_MAX_INSTALL": float(install_undr), "UNDERPAY_MEAN_INSTALL": float(install_undr * 0.5),
        "INSTALL_PAYMENTS_COUNT": 15.0, "INSTALL_LATE_PAY_RATIO": float(late_ratio),
        "DPD_MAX_CC": float(cc_dpd), "CC_UTILIZATION_AVG": float(cc_util),
        "CC_BALANCE_AVG": float(cc_util * 10000), "CC_UTILIZATION_OVER_90_RATIO": 1.0 if cc_util > 0.9 else 0.0,
        "DPD_MAX_POS": float(pos_dpd), "POS_FUTURE_INSTALMENTS_AVG": float(pos_fut),
        "loan_amount": float(loan_ead)
    }

    with st.spinner("Calculating credit score…"):
        res = safe_api_post("/predict", payload)

    if res:
        st.markdown("<hr>", unsafe_allow_html=True)

        score_val = res.get("score", 600)
        decision  = res.get("decision", "Unknown")
        risk_tier = res.get("risk_tier", "Unknown")
        pd_val    = res.get("probability_of_default", 0.0)

        # ── Score + Decision + PD row ─────────────────────────
        rc1, rc2, rc3 = st.columns([1, 1, 1], gap="medium")

        # Determine decision band styling
        if decision == "Approve":
            band_class = "approve-band"
        elif decision == "Manual Review":
            band_class = "review-band"
        else:
            band_class = "reject-band"

        with rc1:
            st.markdown(f"""<div style="text-align:center;padding:20px 0;">
<div class="score-circle">
<div class="score-val">{score_val}</div>
<div class="score-lbl">Credit Score</div>
</div>
</div>""", unsafe_allow_html=True)

        with rc2:
            st.markdown(f"""<div class="glass-card" style="text-align:center;padding:24px 16px;">
<div class="metric-title" style="margin-bottom:10px;">Decision &amp; Risk Tier</div>
<div class="{band_class}" style="font-size:1.15rem;padding:16px 20px;">{decision}<br>
<span style="font-size:0.85rem;font-weight:400;opacity:.85;">{risk_tier}</span>
</div>
</div>""", unsafe_allow_html=True)

        with rc3:
            st.markdown(f"""<div class="glass-card" style="text-align:center;padding:24px 16px;">
<div class="metric-title" style="margin-bottom:10px;">Probability of Default</div>
<div style="font-size:2.4rem;font-weight:800;color:#000000;letter-spacing:-0.02em;margin-top:6px;">{pd_val*100:.2f}%</div>
<div style="color:#000000;font-weight:700;font-size:0.9rem;margin-top:8px;">Calibrated logistic model (Platt scaling)</div>
</div>""", unsafe_allow_html=True)

        # ── Score Breakdown ───────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🔍 Score Point Breakdown")
        breakdown = res.get("score_breakdown", {})
        intercept_val = breakdown.get("Intercept", 0)

        st.markdown(f"""<div class="glass-card" style="padding:16px 20px;border-left:6px solid #3b82f6;margin-bottom:8px;">
<div style="display:flex;justify-content:space-between;align-items:center;">
<div>
<div class="metric-title">Scorecard Intercept</div>
<div style="color:#ffffff;font-weight:700;font-size:0.95rem;">Base offset added to all scores</div>
</div>
<div style="font-size:1.7rem;font-weight:900;color:#60a5fa;">+{intercept_val} pts</div>
</div>
</div>""", unsafe_allow_html=True)

        items = [(k, v) for k, v in sorted(breakdown.items(), key=lambda x: x[1], reverse=True) if k != "Intercept"]
        half  = len(items) // 2 + len(items) % 2
        col_l, col_r = st.columns(2, gap="medium")

        def render_row(name, val):
            cls  = "pts-positive" if val >= 0 else "pts-negative"
            sign = "+" if val >= 0 else ""
            bar_w = min(abs(val), 60)
            bar_col = "#10b981" if val >= 0 else "#f43f5e"
            return f"""<div class="breakdown-row"><div class="breakdown-label">{name}</div><div style="display:flex;align-items:center;gap:10px;"><div style="width:{bar_w}px;height:6px;border-radius:3px;background:{bar_col};opacity:.8;"></div><div class="{cls}">{sign}{val} pts</div></div></div>"""

        with col_l:
            st.markdown('<div class="glass-card" style="padding:8px 16px;">' + "".join(render_row(n, v) for n, v in items[:half]) + "</div>", unsafe_allow_html=True)
        with col_r:
            st.markdown('<div class="glass-card" style="padding:8px 16px;">' + "".join(render_row(n, v) for n, v in items[half:]) + "</div>", unsafe_allow_html=True)

    else:
        st.error("⚠️ No response from the scoring backend. Ensure the FastAPI service is running on port 8000.")
