import os
import json
import yaml
import numpy as np
import pandas as pd
from engine.logger.logger import get_logger
from engine.exception.exceptions import QualityFiltrationError

logger = get_logger("quality_filtration")

# List of critical features that should NOT be dropped even if they exceed null or mode thresholds
WHITELIST_FEATURES = {
    "SK_ID_CURR",
    "TARGET",
    "DPD_MAX_CC",
    "DPD_MAX_POS",
    "DPD_MAX_INSTALL",
    "INSTALL_LATE_PAY_RATIO",
    "PREV_REJECT_RATE",
    "BUREAU_DEBT_CREDIT_RATIO",
    "BUREAU_DAYS_CREDIT_MAX"
}

class QualityFiltrationStage:
    def __init__(self, params_path: str = "params.yaml"):
        with open(params_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.input_file = self.config["feature_engineering"]["output_file"]
        self.output_file = self.config["quality_filtration"]["output_file"]
        self.report_file = self.config["quality_filtration"]["report_file"]
        self.max_null_rate = self.config["quality_filtration"].get("max_null_rate", 0.95)

    def run(self):
        try:
            logger.info("Starting quality filtration stage...")
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            
            if not os.path.exists(self.input_file):
                raise FileNotFoundError(f"Input file not found: {self.input_file}")
                
            df = pd.read_parquet(self.input_file)
            logger.info(f"Loaded engineered features. Shape: {df.shape}")
            
            total_rows = len(df)
            dropped_features = {}
            report = {
                "initial_features": df.shape[1],
                "total_rows": total_rows,
                "feature_metrics": {}
            }
            
            features_to_drop = []
            
            for col in df.columns:
                if col in WHITELIST_FEATURES:
                    continue
                
                # 1. Null rate check
                null_count = df[col].isnull().sum()
                null_rate = float(null_count / total_rows)
                
                # 2. Variance check (only for numerical columns)
                variance = None
                is_zero_variance = False
                if pd.api.types.is_numeric_dtype(df[col]):
                    variance = float(df[col].var())
                    if pd.isnull(variance) or variance <= 1e-8:
                        is_zero_variance = True
                
                # 3. High-frequency mode check
                mode_frac = 0.0
                if total_rows > 0:
                    val_counts = df[col].value_counts(dropna=True)
                    if len(val_counts) > 0:
                        mode_frac = float(val_counts.iloc[0] / total_rows)
                
                # Log metrics in report
                report["feature_metrics"][col] = {
                    "null_rate": null_rate,
                    "variance": variance,
                    "mode_fraction": mode_frac,
                    "dropped": False
                }
                
                # Apply drops
                if null_rate > self.max_null_rate:
                    features_to_drop.append(col)
                    dropped_features[col] = f"Null rate {null_rate:.4f} > {self.max_null_rate}"
                    report["feature_metrics"][col]["dropped"] = True
                    report["feature_metrics"][col]["reason"] = dropped_features[col]
                elif is_zero_variance:
                    features_to_drop.append(col)
                    dropped_features[col] = f"Zero/near-zero variance ({variance})"
                    report["feature_metrics"][col]["dropped"] = True
                    report["feature_metrics"][col]["reason"] = dropped_features[col]
                elif mode_frac > 0.99:
                    features_to_drop.append(col)
                    dropped_features[col] = f"High-frequency mode {mode_frac:.4f} > 0.99"
                    report["feature_metrics"][col]["dropped"] = True
                    report["feature_metrics"][col]["reason"] = dropped_features[col]
            
            # Drop features
            df_filtered = df.drop(columns=features_to_drop)
            logger.info(f"Dropped {len(features_to_drop)} features. Remaining: {df_filtered.shape[1]}")
            logger.info(f"Dropped feature reasons: {json.dumps(dropped_features, indent=2)}")
            
            # Save results
            report["final_features"] = df_filtered.shape[1]
            report["dropped_count"] = len(features_to_drop)
            
            df_filtered.to_parquet(self.output_file, index=False)
            logger.info(f"Saved filtered features to Parquet: {self.output_file}")
            
            with open(self.report_file, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved quality report to JSON: {self.report_file}")
            
        except Exception as e:
            logger.error(f"Error in quality filtration: {e}")
            raise QualityFiltrationError(f"Failed to run quality filtration: {e}")

if __name__ == "__main__":
    stage = QualityFiltrationStage()
    stage.run()
