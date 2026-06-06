import os
import pickle
import json
import numpy as np
import pandas as pd
from typing import Dict, List
from fastapi import FastAPI, HTTPException, status
from api.schemas.scoring import ApplicantRequest, ScoringResponse
from engine.logger.logger import get_logger

logger = get_logger("api_backend")

app = FastAPI(
    title="Credit Risk Scorecard API",
    description="Production-grade Credit Scoring and Real-time Drift Monitoring API",
    version="1.0.0"
)

# Global variables for caching model assets
numerical_points_map = {}
categorical_points_map = {}
scaling_metadata = {}
woe_bin_maps = {}
calibrated_model = None

# Asset file locations
NUM_MAP_PATH = "api/assets/numerical_points_map.pkl"
CAT_MAP_PATH = "api/assets/categorical_points_map.pkl"
METADATA_PATH = "api/assets/scaling_metadata.json"
WOE_MAPS_PATH = "artifacts_cache/woe/woe_bin_maps.pkl"
CALIBRATED_MODEL_PATH = "artifacts_cache/model/calibrated_model.pkl"

def build_decision_bands(train_scores: List[int]) -> Dict:
    """Build decision bands from training score percentiles."""
    arr = np.array(train_scores, dtype=float)
    p10, p25, p50, p75, p90 = np.percentile(arr, [10, 25, 50, 75, 90])
    return {
        "very_low_risk": {"min": int(p90), "max": 1000, "action": "Approve", "label": "Very Low Risk"},
        "low_risk": {"min": int(p75), "max": int(p90) - 1, "action": "Approve", "label": "Low Risk"},
        "medium_risk": {"min": int(p50), "max": int(p75) - 1, "action": "Approve", "label": "Medium Risk"},
        "elevated_risk": {"min": int(p25), "max": int(p50) - 1, "action": "Manual Review", "label": "Elevated Risk"},
        "very_high_risk": {"min": 0, "max": int(p25) - 1, "action": "Reject", "label": "Very High Risk"},
    }


@app.on_event("startup")
def startup_event():
    global numerical_points_map, categorical_points_map, scaling_metadata, woe_bin_maps, calibrated_model
    
    logger.info("Initializing API backend...")
    
    # Load scorecard assets
    try:
        if os.path.exists(NUM_MAP_PATH):
            with open(NUM_MAP_PATH, "rb") as f:
                numerical_points_map = pickle.load(f)
            logger.info("Loaded numerical points map.")
        else:
            logger.warning(f"Numerical points map not found at {NUM_MAP_PATH}")

        if os.path.exists(CAT_MAP_PATH):
            with open(CAT_MAP_PATH, "rb") as f:
                categorical_points_map = pickle.load(f)
            logger.info("Loaded categorical points map.")
        else:
            logger.warning(f"Categorical points map not found at {CAT_MAP_PATH}")

        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "r") as f:
                scaling_metadata = json.load(f)
            logger.info("Loaded scaling metadata.")
        else:
            logger.warning(f"Scaling metadata not found at {METADATA_PATH}")

        if os.path.exists(WOE_MAPS_PATH):
            with open(WOE_MAPS_PATH, "rb") as f:
                woe_bin_maps = pickle.load(f)
            logger.info("Loaded WoE bin maps.")
        else:
            logger.warning(f"WoE bin maps not found at {WOE_MAPS_PATH}")

        if os.path.exists(CALIBRATED_MODEL_PATH):
            with open(CALIBRATED_MODEL_PATH, "rb") as f:
                calibrated_model = pickle.load(f)
            logger.info("Loaded calibrated model for PD estimation.")
        else:
            logger.warning(f"Calibrated model not found at {CALIBRATED_MODEL_PATH}")

        baseline = scaling_metadata.get("train_scores_baseline", [])
        if baseline:
            scaling_metadata["decision_bands"] = build_decision_bands(baseline)
            logger.info("Decision bands recalibrated from training score distribution.")
            
    except Exception as e:
        logger.error(f"Error during API startup asset load: {e}")

def get_applicant_features(req: ApplicantRequest) -> dict:
    """Computes derived variables from raw applicant inputs."""
    data = req.dict()
    
    # Base derived features (matching pipeline logic)
    data['INCOME_CREDIT_PERC'] = req.AMT_INCOME_TOTAL / (req.AMT_CREDIT + 1e-5)
    data['ANNUITY_INCOME_PERC'] = req.AMT_ANNUITY / (req.AMT_INCOME_TOTAL + 1e-5)
    data['PAYMENT_RATE'] = req.AMT_ANNUITY / (req.AMT_CREDIT + 1e-5)
    data['CHILDREN_RATIO'] = req.CNT_CHILDREN / (req.CNT_FAM_MEMBERS + 1e-5)
    
    # Negative days variables to positive years
    data['AGE_YEARS'] = -req.DAYS_BIRTH / 365.25
    
    employed_yrs = -req.DAYS_EMPLOYED / 365.25
    data['EMPLOYED_YEARS'] = np.nan if employed_yrs < 0 or req.DAYS_EMPLOYED == 365243 else employed_yrs

    if data.get("AMT_GOODS_PRICE") is None:
        data["AMT_GOODS_PRICE"] = req.AMT_CREDIT * 0.9

    if req.BUREAU_DEBT_SUM is not None and req.BUREAU_CREDIT_SUM is not None:
        data["BUREAU_DEBT_CREDIT_RATIO"] = req.BUREAU_DEBT_SUM / (req.BUREAU_CREDIT_SUM + 1e-5)

    if req.PREV_APP_COUNT is not None and req.PREV_REJECT_COUNT is not None:
        data["PREV_REJECT_RATE"] = req.PREV_REJECT_COUNT / (req.PREV_APP_COUNT + 1e-5)

    if req.FLAG_OWN_CAR == "N" and data.get("OWN_CAR_AGE") is None:
        data["OWN_CAR_AGE"] = 0.0
    
    return data

def calculate_scorecard_points(features: dict) -> tuple:
    """Maps applicant features to scorecard points and returns score and breakdown."""
    if not scaling_metadata or not numerical_points_map or not categorical_points_map:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scorecard model assets are not loaded on the backend."
        )
        
    intercept_pts = scaling_metadata["intercept_points"]
    selected_features = scaling_metadata["selected_features"]
    
    score = intercept_pts
    breakdown = {"Intercept": intercept_pts}
    
    for col in selected_features:
        val = features.get(col, None)
        
        if col in numerical_points_map:
            edges = numerical_points_map[col]["edges"]
            pts_dict = numerical_points_map[col]["points"]
            
            pts = pts_dict.get("Missing", 0)
            if val is not None and not pd.isna(val):
                for i in range(len(edges) - 1):
                    if val >= edges[i] and val < edges[i+1]:
                        label = f"{edges[i]} to {edges[i+1]}"
                        pts = pts_dict.get(label, pts)
                        break
            
            breakdown[col] = pts
            score += pts
            
        elif col in categorical_points_map:
            group_map = categorical_points_map[col]["group_map"]
            pts_dict = categorical_points_map[col]["points"]
            
            val_str = "Missing" if val is None or pd.isna(val) else str(val)
            if group_map:
                group = group_map.get(val_str, "Other")
                pts = pts_dict.get(group, pts_dict.get("Other", pts_dict.get("Missing", 0)))
            else:
                pts = pts_dict.get(val_str, pts_dict.get("Missing", pts_dict.get("Other", 0)))
            
            breakdown[col] = pts
            score += pts
            
        else:
            breakdown[col] = 0
            
    return int(score), breakdown

def map_woe_features(features: dict) -> pd.DataFrame:
    """Transforms raw features to Weight of Evidence (WoE) values for calibrated estimation."""
    selected_features = scaling_metadata["selected_features"]
    woe_features = {}
    
    for col in selected_features:
        val = features.get(col, None)
        map_info = woe_bin_maps.get(col)
        
        if not map_info:
            woe_features[col] = 0.0
            continue
            
        woe_map = map_info["woe_map"]
        
        if map_info["type"] == "numerical":
            edges = map_info["edges"]
            woe = woe_map.get("Missing", 0.0)
            
            if val is not None and not pd.isna(val):
                for i in range(len(edges) - 1):
                    if val >= edges[i] and val < edges[i+1]:
                        label = f"{edges[i]} to {edges[i+1]}"
                        woe = woe_map.get(label, woe)
                        break
            woe_features[col] = float(woe)
        else:
            val_str = "Missing" if val is None or pd.isna(val) else str(val)
            group_map = map_info["group_map"]
            group = group_map.get(val_str, "Other")
            woe = woe_map.get(group, woe_map.get("Other", woe_map.get("Missing", 0.0)))
            woe_features[col] = float(woe)
            
    return pd.DataFrame([woe_features])


def predict_probability_of_default(features: dict, score: int) -> float:
    """Estimate PD from the Platt-calibrated logistic model on WoE features."""
    if calibrated_model is not None and woe_bin_maps:
        woe_df = map_woe_features(features)
        pd_val = float(calibrated_model.predict_proba(woe_df)[0, 1])
        return float(np.clip(pd_val, 0.0, 1.0))

    factor = float(scaling_metadata.get("factor", 28.85))
    offset = float(scaling_metadata.get("offset", 487.12))
    log_odds = (offset - score) / factor
    odds = np.exp(log_odds)
    return float(np.clip(odds / (1.0 + odds), 0.0, 1.0))


@app.post("/predict", response_model=ScoringResponse)
def predict_score(request: ApplicantRequest):
    """Calculates the scorecard score and probability of default for an applicant."""
    try:
        # 1. Compute derived features
        features = get_applicant_features(request)
        
        # 2. Scorecard scoring points
        score, score_breakdown = calculate_scorecard_points(features)
        
        # 3. Calibrated PD from WoE logistic model (Platt scaling)
        pd_val = predict_probability_of_default(features, score)
        
        # 4. Determine Decision Band
        bands = scaling_metadata["decision_bands"]
        decision = "Reject"
        risk_tier = "Very High Risk"
        
        for band_key, band in bands.items():
            if score >= band["min"] and score <= band["max"]:
                decision = band["action"]
                risk_tier = band["label"]
                break
                
        return ScoringResponse(
            sk_id_curr=request.SK_ID_CURR,
            score=score,
            probability_of_default=pd_val,
            decision=decision,
            risk_tier=risk_tier,
            score_breakdown=score_breakdown
        )
        
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failure: {e}"
        )

@app.get("/run_score_simulator/{feature}")
def run_score_simulator(feature: str):
    """Returns scorecard points boundaries and groups for an input feature."""
    if feature in numerical_points_map:
        raw_edges = numerical_points_map[feature]["edges"]
        clean_edges = [str(x) if (isinstance(x, (int, float)) and np.isinf(x)) else x for x in raw_edges]
        return {
            "type": "numerical",
            "feature": feature,
            "edges": clean_edges,
            "points": numerical_points_map[feature]["points"]
        }
    elif feature in categorical_points_map:
        return {
            "type": "categorical",
            "feature": feature,
            "group_map": categorical_points_map[feature]["group_map"],
            "points": categorical_points_map[feature]["points"]
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Feature '{feature}' not found in scorecard configuration."
        )

@app.get("/feature_importance")
def get_feature_importance():
    """Returns the coefficients of the base scoring model (from scaling metadata)."""
    if not scaling_metadata:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model metadata is not loaded."
        )
    return {
        "features": scaling_metadata["selected_features"],
        "intercept_points": scaling_metadata["intercept_points"]
    }

