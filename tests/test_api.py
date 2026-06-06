import pytest
import numpy as np
from fastapi.testclient import TestClient
import api.main as api_main
from api.main import app

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_mock_model_assets():
    """Fixture to mock model and scorecard assets in the API module on setup."""
    api_main.numerical_points_map = {
        "EXT_SOURCE_2": {
            "edges": [-float('inf'), 0.3, 0.5, 0.7, float('inf')],
            "points": {
                "-inf to 0.3": -20,
                "0.3 to 0.5": -5,
                "0.5 to 0.7": 15,
                "0.7 to inf": 35,
                "Missing": 0
            }
        }
    }
    
    api_main.categorical_points_map = {
        "NAME_EDUCATION_TYPE": {
            "group_map": {
                "Academic degree": "Higher education",
                "Higher education": "Higher education",
                "Secondary / special": "Secondary education"
            },
            "points": {
                "Higher education": 25,
                "Secondary education": 5,
                "Other": 0
            }
        }
    }
    
    api_main.scaling_metadata = {
        "base_score": 600,
        "base_odds": 50.0,
        "pdo": 20,
        "cutoff": 725,
        "intercept_points": 580,
        "selected_features": ["EXT_SOURCE_2", "NAME_EDUCATION_TYPE"],
        "train_scores_baseline": [600, 620, 640, 580, 710, 730, 750, 760, 770, 780],
        "decision_bands": {
            "very_low_risk": {"min": 748, "max": 1000, "action": "Approve", "label": "Very Low Risk"},
            "low_risk": {"min": 733, "max": 747, "action": "Approve", "label": "Low Risk"},
            "medium_risk": {"min": 725, "max": 732, "action": "Approve", "label": "Medium Risk"},
            "elevated_risk": {"min": 705, "max": 724, "action": "Manual Review", "label": "Elevated Risk"},
            "very_high_risk": {"min": 0, "max": 704, "action": "Reject", "label": "Very High Risk"}
        }
    }
    
    # Mock Platt-scaled classifier
    class MockCalibratedModel:
        def predict_proba(self, X):
            import numpy as np
            # Returns a probability of default (PD) of 0.05 (5%)
            return np.array([[0.95, 0.05]])
            
    api_main.calibrated_model = MockCalibratedModel()
    
    # Mock WoE bin maps
    api_main.woe_bin_maps = {
        "EXT_SOURCE_2": {
            "type": "numerical",
            "edges": [-float('inf'), 0.3, 0.5, 0.7, float('inf')],
            "woe_map": {
                "-inf to 0.3": -0.8,
                "0.3 to 0.5": -0.2,
                "0.5 to 0.7": 0.4,
                "0.7 to inf": 1.2,
                "Missing": 0.0
            }
        },
        "NAME_EDUCATION_TYPE": {
            "type": "categorical",
            "group_map": {
                "Academic degree": "Higher education",
                "Higher education": "Higher education",
                "Secondary / special": "Secondary education"
            },
            "woe_map": {
                "Higher education": 0.8,
                "Secondary education": 0.1,
                "Other": 0.0
            }
        }
    }

def test_api_feature_importance():
    """Tests GET /feature_importance endpoint."""
    response = client.get("/feature_importance")
    assert response.status_code == 200
    res_data = response.json()
    assert "features" in res_data
    assert "intercept_points" in res_data
    assert res_data["intercept_points"] == 580
    assert "EXT_SOURCE_2" in res_data["features"]

def test_api_run_score_simulator():
    """Tests GET /run_score_simulator/{feature} for valid and invalid features."""
    # Valid feature
    response = client.get("/run_score_simulator/EXT_SOURCE_2")
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["type"] == "numerical"
    assert "edges" in res_data
    assert "points" in res_data
    
    # Invalid feature
    response_invalid = client.get("/run_score_simulator/NON_EXISTENT_FEATURE")
    assert response_invalid.status_code == 404

def test_api_predict():
    """Tests POST /predict applicant scoring and decision logic."""
    payload = {
        "AMT_INCOME_TOTAL": 150000.0,
        "AMT_CREDIT": 500000.0,
        "AMT_ANNUITY": 25000.0,
        "DAYS_BIRTH": -15000,
        "DAYS_EMPLOYED": -2000,
        "DAYS_LAST_PHONE_CHANGE": -500.0,
        "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_INCOME_TYPE": "Working",
        "CODE_GENDER": "F",
        "FLAG_OWN_CAR": "N",
        "FLAG_OWN_REALTY": "Y",
        "loan_amount": 50000.0,
        "EXT_SOURCE_2": 0.8 # should map to '0.7 to inf' bin which is +35 points
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    res_data = response.json()
    
    # Check predictions structure
    assert "score" in res_data
    assert "probability_of_default" in res_data
    assert "decision" in res_data
    assert "score_breakdown" in res_data
    
    # Check value calculations
    # Score = Intercept (580) + EXT_SOURCE_2 (35) + education Higher education (25) = 640 points
    assert res_data["score"] == 640
    assert res_data["decision"] == "Reject" # score 640 is below mocked p25 threshold
    assert np.isclose(res_data["probability_of_default"], 0.05)

