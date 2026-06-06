import os
import pickle
import yaml
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import mlflow
from engine.logger.logger import get_logger
from engine.exception.exceptions import ModelTrainingError

logger = get_logger("estimator_training")

class EstimatorTrainingStage:
    def __init__(self, params_path: str = "params.yaml"):
        with open(params_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.input_dir = self.config["bin_optimization"]["output_dir"]
        self.model_output = self.config["estimator_training"]["model_output"]
        self.calibrated_model_output = self.config["estimator_training"]["calibrated_model_output"]
        
        # Logistic Regression hyperparameters
        self.penalty = self.config["estimator_training"].get("penalty", "l2")
        self.C = self.config["estimator_training"].get("C", 0.1)
        self.solver = self.config["estimator_training"].get("solver", "lbfgs")
        self.max_iter = self.config["estimator_training"].get("max_iter", 1000)
        self.random_state = self.config["data"].get("random_state", 42)

    def run(self):
        try:
            logger.info("Starting estimator training stage...")
            os.makedirs(os.path.dirname(self.model_output), exist_ok=True)
            os.makedirs(os.path.dirname(self.calibrated_model_output), exist_ok=True)
            
            train_path = os.path.join(self.input_dir, "train_opt.parquet")
            test_path = os.path.join(self.input_dir, "test_opt.parquet")
            
            if not os.path.exists(train_path) or not os.path.exists(test_path):
                raise FileNotFoundError("Optimized datasets not found. Ensure stage_bin_optimization ran.")
                
            df_train = pd.read_parquet(train_path)
            df_test = pd.read_parquet(test_path)
            
            # Extract features and targets
            X_train = df_train.drop(columns=["SK_ID_CURR", "TARGET"])
            y_train = df_train["TARGET"]
            X_test = df_test.drop(columns=["SK_ID_CURR", "TARGET"])
            y_test = df_test["TARGET"]
            
            feature_names = X_train.columns.tolist()
            logger.info(f"Training on features: {feature_names}")
            
            # Setup MLflow local run
            mlflow.set_experiment("Credit_Risk_Scorecard")
            
            with mlflow.start_run() as run:
                # Log hyperparameters
                mlflow.log_param("penalty", self.penalty)
                mlflow.log_param("C", self.C)
                mlflow.log_param("solver", self.solver)
                mlflow.log_param("max_iter", self.max_iter)
                mlflow.log_param("num_features", len(feature_names))
                
                # 1. Fit Base Logistic Regression (needed for Scorecard coefficients)
                logger.info("Fitting base Logistic Regression...")
                lr = LogisticRegression(
                    penalty=self.penalty,
                    C=self.C,
                    solver=self.solver,
                    max_iter=self.max_iter,
                    random_state=self.random_state
                )
                lr.fit(X_train, y_train)
                
                # Save base model
                with open(self.model_output, "wb") as f:
                    pickle.dump(lr, f)
                logger.info(f"Saved base model to {self.model_output}")
                
                # Extract and log coefficients
                coefs = lr.coef_[0]
                intercept = lr.intercept_[0]
                
                importance_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Coefficient": coefs
                }).sort_values(by="Coefficient", ascending=False)
                
                # Save feature importance CSV
                importance_path = os.path.join(os.path.dirname(self.model_output), "feature_importance.csv")
                importance_df.to_csv(importance_path, index=False)
                logger.info(f"Saved feature importance coefficients to {importance_path}")
                
                # Log coefficients to MLflow
                mlflow.log_metric("intercept", float(intercept))
                for _, row in importance_df.iterrows():
                    mlflow.log_param(f"coef_{row['Feature']}", float(row["Coefficient"]))
                
                # 2. Fit Calibrated Estimator using Platt Scaling
                logger.info("Fitting calibrated classifier using Platt scaling (CalibratedClassifierCV)...")
                # Using 5-fold cross-validation calibration
                calibrated_model = CalibratedClassifierCV(
                    estimator=LogisticRegression(
                        penalty=self.penalty,
                        C=self.C,
                        solver=self.solver,
                        max_iter=self.max_iter,
                        random_state=self.random_state
                    ),
                    method="sigmoid",
                    cv=5
                )
                calibrated_model.fit(X_train, y_train)
                
                # Save calibrated model
                with open(self.calibrated_model_output, "wb") as f:
                    pickle.dump(calibrated_model, f)
                logger.info(f"Saved calibrated model to {self.calibrated_model_output}")
                
                # Log model artifacts in MLflow
                mlflow.log_artifact(self.model_output, artifact_path="models")
                mlflow.log_artifact(self.calibrated_model_output, artifact_path="models")
                mlflow.log_artifact(importance_path, artifact_path="evaluation")
                
                logger.info(f"Model training stage completed successfully. MLflow Run ID: {run.info.run_id}")
                
        except Exception as e:
            logger.error(f"Error in estimator training: {e}")
            raise ModelTrainingError(f"Failed to train estimator: {e}")

if __name__ == "__main__":
    stage = EstimatorTrainingStage()
    stage.run()
