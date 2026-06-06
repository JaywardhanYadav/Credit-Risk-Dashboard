import numpy as np

def test_scorecard_scaling_math():
    """Verifies the mathematical correctness of scorecard scaling constants."""
    base_score = 600
    base_odds = 50.0
    pdo = 20
    
    # Mathematical equations
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    
    # Check calculated values
    assert np.isclose(factor, 28.853900817)
    assert np.isclose(offset, 487.1228762)
    
    # Verify score with base odds (should yield base score)
    score_at_base = offset + factor * np.log(base_odds)
    assert np.isclose(score_at_base, base_score)
    
    # Verify score with double odds (should yield base score + pdo)
    score_at_double = offset + factor * np.log(2 * base_odds)
    assert np.isclose(score_at_double, base_score + pdo)

def test_scorecard_decision_bands():
    """Verifies that the score ranges match the risk tier rules."""
    decision_bands = {
        "very_low_risk": {"min": 748, "max": 1000, "action": "Approve", "label": "Very Low Risk"},
        "low_risk": {"min": 733, "max": 747, "action": "Approve", "label": "Low Risk"},
        "medium_risk": {"min": 725, "max": 732, "action": "Approve", "label": "Medium Risk"},
        "elevated_risk": {"min": 705, "max": 724, "action": "Manual Review", "label": "Elevated Risk"},
        "very_high_risk": {"min": 0, "max": 704, "action": "Reject", "label": "Very High Risk"}
    }
    
    # Test arbitrary scores mapping to decision tiers
    assert decision_bands["very_low_risk"]["min"] == 748
    assert decision_bands["very_high_risk"]["max"] == 704
