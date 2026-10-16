# 💳 Credit Risk Application Scorecard & Underwriting System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.24%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-3.0%2B-945DD6?style=for-the-badge&logo=dvc&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)

<p align="center">
  <b>A production-grade, regulatory-compliant (Basel II/III) Credit Risk Application Scorecard and Underwriting System engineered with reproducible DVC pipelines, calibrated logistic regression, and containerized FastAPI + Streamlit microservices.</b>
</p>

</div>

---

## 📌 Executive Summary & Architectural Overview

The **Credit Risk Application Scorecard System** is an end-to-end financial risk engine trained on the **Home Credit Default Risk** benchmark. It automates credit underwriting by translating raw financial, transactional, and bureau histories into an explainable, standardized credit score and Probability of Default ($PD$).

Unlike black-box neural networks that violate financial regulatory explainability standards (Basel Committee & Fair Lending guidelines), this system implements an industry-standard **Weight of Evidence (WoE)** and **Information Value (IV)** transformation pipeline coupled with **Calibrated Logistic Regression** and **FICO-style Scorecard Scaling**.

┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │ DVC End-to-End Pipeline (dvc.yaml) │ ├─────────────────┬──────────────────┬─────────────────┬─────────────────┬────────────────────┤ │ Raw Ingestion │ Schema Validation│ Feature Engg. │ Quality Filter │ WoE & IV Mapping │ │ & Caching │ (Type & Bounds) │ (Debt/Annuity) │ (Null Rate<.95) │ (Min IV >= 0.02) │ └────────┬────────┴────────┬─────────┴────────┬────────┴────────┬────────┴─────────┬──────────┘ │ │ │ │ │ └─────────────────┴──────────────────┼─────────────────┴──────────────────┘ ▼ ┌───────────────────────────────────┐ │ Bin Optimization & Stability │ │ (Corr <= 0.7, VIF <= 5, PSI<.25) │ └────────────────┬──────────────────┘ ▼ ┌───────────────────────────────────┐ │ Calibrated Logistic Regression │ │ (L2 Regularization, L-BFGS) │ └────────────────┬──────────────────┘ ▼ ┌───────────────────────────────────┐ │ FICO Scorecard Scaling │ │ (Base=600@50:1, PDO=20, Cutoff=725│ └────────────────┬──────────────────┘ │ ┌─────────────────────────┴─────────────────────────┐ ▼ ▼ ┌────────────────────────┐ ┌────────────────────────┐ │ FastAPI Engine │ ◄────── HTTP / JSON ────►│ Streamlit Dashboard │ │ (Port 8000) │ │ (Port 8501) │ │ • Real-Time Scoring │ │ • Applicant Underwriter│ │ • Points Lookup Map │ │ • What-If Simulator │ └────────────────────────┘ └────────────────────────┘

---
## ⚙️ Key Methodologies & Mathematical Foundations
### 1. Weight of Evidence (WoE) & Information Value (IV)
Continuous variables are binned into monotonic buckets and mapped using Weight of Evidence:
$$\text{WoE}_i = \ln \left( \frac{\% \text{ Non-Defaults}_i}{\% \text{ Defaults}_i} \right)$$
Features are screened using Information Value ($IV$), filtering out weak predictors:
$$IV = \sum_{i=1}^{k} \left( \% \text{ Non-Defaults}_i - \% \text{ Defaults}_i \right) \times \text{WoE}_i$$
* **Filtering Threshold:** Variables with $IV < 0.02$ are dropped, retaining strong risk discriminators.
### 2. Multicollinearity & Stability Optimization
* **Variance Inflation Factor (VIF):** Limits multicollinearity by enforcing $VIF \le 5.0$ and cross-correlation $r \le 0.70$.
* **Population Stability Index (PSI):** Enforces bucket stability across train/test splits ($PSI \le 0.25$) to prevent temporal feature drift.
### 3. FICO-Style Scorecard Scaling
Model log-odds ($\ln(\text{Odds})$) are linearly scaled into standard credit points:
$$\text{Score} = \text{Offset} - \text{Factor} \times \ln(\text{Odds})$$
$$\text{Factor} = \frac{\text{PDO}}{\ln(2)}, \quad \text{Offset} = \text{Base Score} - \text{Factor} \times \ln(\text{Base Odds})$$
* **Base Score:** 600 points at 50:1 odds
* **Points to Double the Odds (PDO):** 20
* **Decision Cutoff:** 725 points ($\ge 725 \rightarrow \text{Approved}$, $< 725 \rightarrow \text{Rejected}$)
---
## 📂 Repository Topology
Credit-Risk-Dashboard/ ├── api/ │ ├── assets/ # Scaling metadata, numerical & categorical points maps │ ├── Dockerfile # FastAPI container definition │ └── main.py # Real-time scoring API endpoints ├── dashboard/ │ ├── Dockerfile # Streamlit container definition │ └── app.py # Underwriting dashboard & interactive simulator ├── engine/ │ ├── artifacts/ # Cache and generated intermediate parquets │ ├── exception/ # Custom exception handling abstractions │ ├── logger/ # Production logging configuration │ ├── settings/ # Typed configuration loaders │ └── stages/ # Modular pipeline stages: │ ├── stage_raw_ingestion.py │ ├── stage_schema_validation.py │ ├── stage_feature_engineering.py │ ├── stage_quality_filtration.py │ ├── stage_woe_mapping.py │ ├── stage_bin_optimization.py │ ├── stage_estimator_training.py │ ├── stage_performance_evaluation.py │ └── stage_scorecard_scaling.py ├── tests/ # Pytest test suite ├── docker-compose.yml # Multi-container orchestration (API + Dashboard) ├── dvc.yaml # DVC pipeline DAG definition ├── params.yaml # Hyperparameters, scaling, and threshold configs ├── pyproject.toml # Package build and dependency metadata └── README.md

---
## 🚀 Quick Start Guide
### Option A: Multi-Container Run via Docker Compose (Recommended)
Run the entire platform (FastAPI scoring engine + Streamlit UI) in one command:
```bash
docker compose up -d --build
Access the services:

Interactive Dashboard: http://localhost:8501
FastAPI Swagger Docs: http://localhost:8000/docs
bash
# View live application logs
docker compose logs -f
Option B: Local Environment Setup
1. Prerequisites
Python 3.9+ or Python 3.10+
Git & Git LFS
2. Virtual Environment & Dependencies
bash
# Clone the repository
git clone https://github.com/JaywardhanYadav/Credit-Risk-Dashboard.git
cd Credit-Risk-Dashboard
# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate
# Install dependencies
pip install -r requirements.txt
3. Reproducing the DVC Pipeline
To execute all stages from ingestion to scorecard scaling:

bash
dvc repro
4. Launching the Microservices
In separate terminal instances:

bash
# Terminal 1: Launch FastAPI Backend
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# Terminal 2: Launch Streamlit Dashboard
streamlit run dashboard/app.py --server.port 8501
🧪 Testing & Validation
Run unit and integration tests across pipeline stages:

bash
pytest tests/ -v
📜 Regulatory Standards & Disclaimers
Basel II / III Compliance: The pipeline's use of logistic regression and monotonic WoE preserves direct auditability, feature attribution, and reason-code generation for adverse credit action notices.
Demonstration Purpose: Built for research, credit risk education, and algorithmic demonstration. Not intended for commercial credit lending without localized financial regulatory compliance.
👤 Author
Jaywardhan Yadav

GitHub: @JaywardhanYadav
LinkedIn: in/jaywardhanyadav
Portfolio: jaywardhanportfolio.netlify.app
