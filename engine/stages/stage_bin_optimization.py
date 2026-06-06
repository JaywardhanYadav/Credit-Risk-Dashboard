import os
import json
import pickle
import yaml
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from engine.logger.logger import get_logger
from engine.exception.exceptions import ScorecardException

logger = get_logger("bin_optimization")

class BinOptimizationStage:
    def __init__(self, params_path: str = "params.yaml"):
        with open(params_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.woe_dir = self.config["woe_mapping"]["output_dir"]
        self.output_dir = self.config["bin_optimization"]["output_dir"]
        self.max_correlation = self.config["bin_optimization"].get("max_correlation", 0.7)
        self.max_vif = self.config["bin_optimization"].get("max_vif", 5.0)
        self.max_psi = self.config["bin_optimization"].get("max_psi", 0.25)

    def calculate_psi(self, train_series: pd.Series, test_series: pd.Series) -> float:
        """Calculates the Population Stability Index (PSI) between train and test partitions."""
        # Find unique WoE values (which represent bins)
        train_counts = train_series.value_counts(dropna=False, normalize=True)
        test_counts = test_series.value_counts(dropna=False, normalize=True)
        
        # Combine index keys
        all_bins = set(train_counts.index).union(set(test_counts.index))
        
        psi_val = 0.0
        for b in all_bins:
            # Expected (train), Actual (test)
            expected = train_counts.get(b, 0.0)
            actual = test_counts.get(b, 0.0)
            
            # Apply small smoothing to avoid zero divisions
            expected_smooth = max(expected, 1e-4)
            actual_smooth = max(actual, 1e-4)
            
            psi_val += (actual_smooth - expected_smooth) * np.log(actual_smooth / expected_smooth)
            
        return float(psi_val)

    def calculate_vifs(self, df: pd.DataFrame) -> dict:
        """Calculates Variance Inflation Factors (VIFs) for features in a DataFrame."""
        vifs = {}
        cols = df.columns
        if len(cols) <= 1:
            return {c: 1.0 for c in cols}
            
        for col in cols:
            X = df.drop(columns=[col])
            y = df[col]
            reg = LinearRegression()
            reg.fit(X, y)
            r2 = reg.score(X, y)
            vif = 1.0 / (1.0 - r2 + 1e-10)
            vifs[col] = float(vif)
        return vifs

    def run(self):
        try:
            logger.info("Starting bin optimization and feature pruning stage...")
            os.makedirs(self.output_dir, exist_ok=True)
            
            train_path = os.path.join(self.woe_dir, "train_woe.parquet")
            test_path = os.path.join(self.woe_dir, "test_woe.parquet")
            iv_path = os.path.join(self.woe_dir, "iv_report.json")
            
            if not os.path.exists(train_path) or not os.path.exists(test_path):
                raise FileNotFoundError("WoE-transformed datasets not found. Ensure stage_woe_mapping ran.")
                
            df_train = pd.read_parquet(train_path)
            df_test = pd.read_parquet(test_path)
            
            with open(iv_path, "r") as f:
                iv_report = json.load(f)
                
            logger.info(f"Initial WoE features count: {df_train.shape[1] - 2}")
            
            # 1. PSI Pruning
            logger.info("Calculating feature PSI...")
            features = [c for c in df_train.columns if c not in ["SK_ID_CURR", "TARGET"]]
            psi_report = {}
            stable_features = []
            
            for col in features:
                psi_val = self.calculate_psi(df_train[col], df_test[col])
                psi_report[col] = psi_val
                if psi_val <= self.max_psi:
                    stable_features.append(col)
                else:
                    logger.warning(f"Pruned unstable feature: {col} (PSI: {psi_val:.4f})")
            
            logger.info(f"Features after PSI pruning: {len(stable_features)}")
            
            # 2. Correlation Pruning
            logger.info("Performing correlation pruning...")
            df_corr = df_train[stable_features].corr().abs()
            
            dropped_corr = set()
            # Sort stable features by IV descending to prioritize retaining high-IV features
            sorted_features = sorted(stable_features, key=lambda x: iv_report.get(x, 0.0), reverse=True)
            
            for i in range(len(sorted_features)):
                col_i = sorted_features[i]
                if col_i in dropped_corr:
                    continue
                for j in range(i + 1, len(sorted_features)):
                    col_j = sorted_features[j]
                    if col_j in dropped_corr:
                        continue
                    if df_corr.loc[col_i, col_j] >= self.max_correlation:
                        dropped_corr.add(col_j)
                        logger.info(f"Pruned collinear feature: {col_j} (Highly correlated with {col_i}: {df_corr.loc[col_i, col_j]:.4f}. Retained {col_i} due to higher IV)")
            
            uncorrelated_features = [f for f in stable_features if f not in dropped_corr]
            logger.info(f"Features after correlation pruning: {len(uncorrelated_features)}")
            
            # 3. VIF Pruning (Iterative)
            logger.info("Performing VIF pruning...")
            vif_features = uncorrelated_features.copy()
            
            while len(vif_features) > 1:
                vifs = self.calculate_vifs(df_train[vif_features])
                max_vif_feature = max(vifs, key=vifs.get)
                max_vif_val = vifs[max_vif_feature]
                
                if max_vif_val > self.max_vif:
                    logger.warning(f"Pruned high VIF feature: {max_vif_feature} (VIF: {max_vif_val:.4f})")
                    vif_features.remove(max_vif_feature)
                else:
                    break
            
            final_features = vif_features
            logger.info(f"Final selected features ({len(final_features)}): {final_features}")
            
            # Save final train/test datasets
            df_train_opt = df_train[["SK_ID_CURR", "TARGET"] + final_features]
            df_test_opt = df_test[["SK_ID_CURR", "TARGET"] + final_features]
            
            train_opt_path = os.path.join(self.output_dir, "train_opt.parquet")
            test_opt_path = os.path.join(self.output_dir, "test_opt.parquet")
            
            df_train_opt.to_parquet(train_opt_path, index=False)
            df_test_opt.to_parquet(test_opt_path, index=False)
            logger.info(f"Saved optimized WoE datasets to {train_opt_path} and {test_opt_path}")
            
            # Save optimization report
            report = {
                "initial_count": len(features),
                "psi_pruned_count": len(features) - len(stable_features),
                "corr_pruned_count": len(stable_features) - len(uncorrelated_features),
                "vif_pruned_count": len(uncorrelated_features) - len(final_features),
                "final_count": len(final_features),
                "selected_features": final_features,
                "psi_metrics": psi_report,
                "vif_metrics": self.calculate_vifs(df_train_opt[final_features]) if len(final_features) > 0 else {}
            }
            
            report_path = os.path.join(self.output_dir, "optimization_report.json")
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved optimization report to {report_path}")
            
            # Serialize selected features list
            features_pkl_path = os.path.join(self.output_dir, "selected_features.pkl")
            with open(features_pkl_path, "wb") as f:
                pickle.dump(final_features, f)
            logger.info(f"Saved selected features list to {features_pkl_path}")
            
        except Exception as e:
            logger.error(f"Error in bin optimization: {e}")
            raise ScorecardException(f"Failed to run bin optimization: {e}")

if __name__ == "__main__":
    stage = BinOptimizationStage()
    stage.run()
