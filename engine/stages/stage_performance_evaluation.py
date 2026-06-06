import os
import json
import pickle
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.stats import ks_2samp
import mlflow
from engine.logger.logger import get_logger
from engine.exception.exceptions import ScorecardException

logger = get_logger("performance_evaluation")

class PerformanceEvaluationStage:
    def __init__(self, params_path: str = "params.yaml"):
        with open(params_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.input_dir = self.config["bin_optimization"]["output_dir"]
        self.calibrated_model_path = self.config["estimator_training"]["calibrated_model_output"]
        self.metrics_output = self.config["performance_evaluation"]["metrics_output"]

    def evaluate_partition(self, X: pd.DataFrame, y: pd.Series, model) -> dict:
        """Evaluates model performance metrics on a given partition."""
        # Calibrated Probability of Default (PD)
        pds = model.predict_proba(X)[:, 1]
        
        # 1. AUC-ROC
        auc = float(roc_auc_score(y, pds))
        
        # 2. Gini Coefficient
        gini = float(2 * auc - 1.0)
        
        # 3. Kolmogorov-Smirnov (KS) statistic
        pds_good = pds[y == 0]
        pds_bad = pds[y == 1]
        if len(pds_good) > 0 and len(pds_bad) > 0:
            ks_res = ks_2samp(pds_good, pds_bad)
            ks = float(ks_res.statistic)
        else:
            ks = 0.0
            
        # 4. Brier Score
        brier = float(brier_score_loss(y, pds))
        
        return {
            "auc": round(auc, 4),
            "gini": round(gini, 4),
            "ks": round(ks, 4),
            "brier_score": round(brier, 4)
        }

    def run(self):
        try:
            logger.info("Starting performance evaluation stage...")
            os.makedirs(os.path.dirname(self.metrics_output), exist_ok=True)
            
            # Load datasets
            train_path = os.path.join(self.input_dir, "train_opt.parquet")
            test_path = os.path.join(self.input_dir, "test_opt.parquet")
            
            if not os.path.exists(train_path) or not os.path.exists(test_path):
                raise FileNotFoundError("Optimized datasets not found. Ensure stage_bin_optimization ran.")
                
            df_train = pd.read_parquet(train_path)
            df_test = pd.read_parquet(test_path)
            
            X_train = df_train.drop(columns=["SK_ID_CURR", "TARGET"])
            y_train = df_train["TARGET"]
            X_test = df_test.drop(columns=["SK_ID_CURR", "TARGET"])
            y_test = df_test["TARGET"]
            
            # Load calibrated model
            if not os.path.exists(self.calibrated_model_path):
                raise FileNotFoundError(f"Calibrated model not found: {self.calibrated_model_path}")
                
            with open(self.calibrated_model_path, "rb") as f:
                calibrated_model = pickle.load(f)
                
            # Evaluate partitions
            train_metrics = self.evaluate_partition(X_train, y_train, calibrated_model)
            test_metrics = self.evaluate_partition(X_test, y_test, calibrated_model)
            
            metrics = {
                "train": train_metrics,
                "test": test_metrics
            }
            
            # Save metrics to JSON
            with open(self.metrics_output, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Saved evaluation metrics to JSON: {self.metrics_output}")
            
            # Log metrics in MLflow
            mlflow.set_experiment("Credit_Risk_Scorecard")
            with mlflow.start_run(run_name="performance_evaluation"):
                # Train metrics
                mlflow.log_metric("train_auc", train_metrics["auc"])
                mlflow.log_metric("train_gini", train_metrics["gini"])
                mlflow.log_metric("train_ks", train_metrics["ks"])
                mlflow.log_metric("train_brier_score", train_metrics["brier_score"])
                
                # Test metrics
                mlflow.log_metric("test_auc", test_metrics["auc"])
                mlflow.log_metric("test_gini", test_metrics["gini"])
                mlflow.log_metric("test_ks", test_metrics["ks"])
                mlflow.log_metric("test_brier_score", test_metrics["brier_score"])
                
                mlflow.log_artifact(self.metrics_output, artifact_path="evaluation")
                
            logger.info("Performance evaluation stage completed successfully.")
            
        except Exception as e:
            logger.error(f"Error in performance evaluation: {e}")
            raise ScorecardException(f"Failed to run performance evaluation: {e}")

if __name__ == "__main__":
    stage = PerformanceEvaluationStage()
    stage.run()
