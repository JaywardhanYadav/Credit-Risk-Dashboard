import os
import pickle
import json
import yaml
import numpy as np
import pandas as pd
from engine.logger.logger import get_logger
from engine.exception.exceptions import ScorecardScalingError

logger = get_logger("scorecard_scaling")

class ScorecardScalingStage:
    def __init__(self, params_path: str = "params.yaml"):
        with open(params_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.woe_dir = self.config["woe_mapping"]["output_dir"]
        self.opt_dir = self.config["bin_optimization"]["output_dir"]
        self.model_path = self.config["estimator_training"]["model_output"]
        
        self.num_map_path = self.config["scorecard_scaling"]["numerical_points_map"]
        self.cat_map_path = self.config["scorecard_scaling"]["categorical_points_map"]
        self.metadata_path = self.config["scorecard_scaling"]["metadata_output"]
        
        # Scaling parameters
        self.base_score = self.config["scorecard_scaling"].get("base_score", 600)
        self.base_odds = self.config["scorecard_scaling"].get("base_odds", 50.0)
        self.pdo = self.config["scorecard_scaling"].get("pdo", 20)
        self.cutoff = self.config["scorecard_scaling"].get("cutoff", 725)

    def run(self):
        try:
            logger.info("Starting scorecard scaling stage...")
            os.makedirs(os.path.dirname(self.num_map_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.cat_map_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
            
            # Load inputs
            maps_pkl_path = os.path.join(self.woe_dir, "woe_bin_maps.pkl")
            features_pkl_path = os.path.join(self.opt_dir, "selected_features.pkl")
            
            if not os.path.exists(maps_pkl_path):
                raise FileNotFoundError(f"WoE bin maps not found: {maps_pkl_path}")
            if not os.path.exists(features_pkl_path):
                raise FileNotFoundError(f"Selected features not found: {features_pkl_path}")
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Base model not found: {self.model_path}")
                
            with open(maps_pkl_path, "rb") as f:
                woe_bin_maps = pickle.load(f)
            with open(features_pkl_path, "rb") as f:
                selected_features = pickle.load(f)
            with open(self.model_path, "rb") as f:
                model = pickle.load(f)
                
            # Perform scorecard scaling calculations
            # Factor = PDO / ln(2)
            factor = float(self.pdo / np.log(2))
            # Offset = Base Score - Factor * ln(Base Odds)
            offset = float(self.base_score - factor * np.log(self.base_odds))
            
            intercept = float(model.intercept_[0])
            coefs = model.coef_[0]
            
            # Intercept Points = Offset - Factor * beta_0
            # Note: credit risk scorecard uses: Intercept Points = Offset - Factor * Intercept
            intercept_points = float(offset - factor * intercept)
            
            logger.info(f"Scaling Math Summary - Factor: {factor:.4f}, Offset: {offset:.4f}, Intercept Points: {intercept_points:.4f}")
            
            # Create feature coefficients map
            coef_map = dict(zip(selected_features, coefs))
            
            numerical_points = {}
            categorical_points = {}
            
            # Distribute points to features
            for col in selected_features:
                feature_map = woe_bin_maps[col]
                beta = coef_map[col]
                
                # Points_i = - beta * WoE * Factor
                feature_points = {}
                for bin_label, woe in feature_map["woe_map"].items():
                    pts = -beta * woe * factor
                    feature_points[bin_label] = int(round(pts))
                    
                if feature_map["type"] == "numerical":
                    numerical_points[col] = {
                        "edges": feature_map["edges"],
                        "points": feature_points
                    }
                else:
                    categorical_points[col] = {
                        "group_map": feature_map["group_map"],
                        "points": feature_points
                    }
            
            # Save scorecard maps
            with open(self.num_map_path, "wb") as f:
                pickle.dump(numerical_points, f)
            logger.info(f"Saved numerical points map to {self.num_map_path}")
            
            with open(self.cat_map_path, "wb") as f:
                pickle.dump(categorical_points, f)
            logger.info(f"Saved categorical points map to {self.cat_map_path}")
            
            # Calculate train set scores for drift monitoring baseline
            train_opt_path = os.path.join(self.opt_dir, "train_opt.parquet")
            train_scores = []
            if os.path.exists(train_opt_path):
                logger.info("Calculating train set scores baseline for drift monitoring...")
                df_train_opt = pd.read_parquet(train_opt_path)
                scores = np.zeros(len(df_train_opt)) + intercept_points
                for col in selected_features:
                    beta = coef_map[col]
                    scores += -beta * df_train_opt[col].values * factor
                train_scores = np.round(scores).astype(int).tolist()

            # Metadata Export
            baseline_scores = train_scores
            if baseline_scores:
                arr = np.array(baseline_scores, dtype=float)
                p10, p25, p50, p75, p90 = np.percentile(arr, [10, 25, 50, 75, 90])
                decision_bands = {
                    "very_low_risk": {"min": int(p90), "max": 1000, "action": "Approve", "label": "Very Low Risk"},
                    "low_risk": {"min": int(p75), "max": int(p90) - 1, "action": "Approve", "label": "Low Risk"},
                    "medium_risk": {"min": int(p50), "max": int(p75) - 1, "action": "Approve", "label": "Medium Risk"},
                    "elevated_risk": {"min": int(p25), "max": int(p50) - 1, "action": "Manual Review", "label": "Elevated Risk"},
                    "very_high_risk": {"min": 0, "max": int(p25) - 1, "action": "Reject", "label": "Very High Risk"},
                }
            else:
                decision_bands = {
                    "very_low_risk": {"min": 748, "max": 1000, "action": "Approve", "label": "Very Low Risk"},
                    "low_risk": {"min": 733, "max": 747, "action": "Approve", "label": "Low Risk"},
                    "medium_risk": {"min": 725, "max": 732, "action": "Approve", "label": "Medium Risk"},
                    "elevated_risk": {"min": 705, "max": 724, "action": "Manual Review", "label": "Elevated Risk"},
                    "very_high_risk": {"min": 0, "max": 704, "action": "Reject", "label": "Very High Risk"},
                }

            scaling_metadata = {
                "base_score": self.base_score,
                "base_odds": self.base_odds,
                "pdo": self.pdo,
                "cutoff": self.cutoff,
                "factor": factor,
                "offset": offset,
                "intercept": intercept,
                "intercept_points": int(round(intercept_points)),
                "selected_features": selected_features,
                "train_scores_baseline": train_scores,
                "decision_bands": decision_bands,
            }
            
            with open(self.metadata_path, "w") as f:
                json.dump(scaling_metadata, f, indent=2)
            logger.info(f"Saved scaling metadata to {self.metadata_path}")
            
        except Exception as e:
            logger.error(f"Error in scorecard scaling: {e}")
            raise ScorecardScalingError(f"Failed to scale scorecard: {e}")

if __name__ == "__main__":
    stage = ScorecardScalingStage()
    stage.run()
