import os
import yaml
import pandas as pd
from engine.logger.logger import get_logger
from engine.exception.exceptions import SchemaValidationError

logger = get_logger("schema_validation")

class SchemaValidationStage:
    def __init__(self, params_path: str = "params.yaml"):
        with open(params_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.raw_cache_dir = self.config["data"]["raw_cache_dir"]
        self.status_file = self.config["schema_validation"]["status_file"]

    def run(self):
        try:
            logger.info("Starting schema validation stage...")
            os.makedirs(os.path.dirname(self.status_file), exist_ok=True)
            
            validation_results = {
                "is_valid": True,
                "errors": [],
                "checks": {}
            }

            # 1. Validate application_train.csv
            train_path = os.path.join(self.raw_cache_dir, "application_train.csv")
            if not os.path.exists(train_path):
                raise FileNotFoundError(f"Sampled application_train.csv not found at {train_path}")

            logger.info("Validating application_train.csv...")
            df_train = pd.read_csv(train_path)
            
            # Check unique SK_ID_CURR
            is_unique = df_train["SK_ID_CURR"].is_unique
            validation_results["checks"]["unique_sk_id_curr"] = bool(is_unique)
            if not is_unique:
                validation_results["errors"].append("SK_ID_CURR is not unique in application_train.csv")
                validation_results["is_valid"] = False

            # Check TARGET nulls
            target_nulls = df_train["TARGET"].isnull().sum()
            validation_results["checks"]["target_nulls"] = int(target_nulls)
            if target_nulls > 0:
                validation_results["errors"].append("TARGET column contains null values")
                validation_results["is_valid"] = False

            # Check TARGET values (should be 0 or 1)
            target_vals = df_train["TARGET"].dropna().unique().tolist()
            validation_results["checks"]["target_values"] = target_vals
            if not all(val in [0, 1] for val in target_vals):
                validation_results["errors"].append("TARGET contains values other than 0 and 1")
                validation_results["is_valid"] = False

            # 2. Check other files exist and are loadable
            relational_files = [
                "bureau.csv",
                "bureau_balance.csv",
                "previous_application.csv",
                "installments_payments.csv",
                "credit_card_balance.csv",
                "POS_CASH_balance.csv"
            ]
            for file_name in relational_files:
                file_path = os.path.join(self.raw_cache_dir, file_name)
                exists = os.path.exists(file_path)
                validation_results["checks"][f"{file_name}_exists"] = exists
                if not exists:
                    logger.warning(f"File {file_name} does not exist at {file_path}")
                    continue
                
                # Check loadable
                try:
                    df_sample = pd.read_csv(file_path, nrows=5)
                    validation_results["checks"][f"{file_name}_loadable"] = True
                except Exception as e:
                    validation_results["errors"].append(f"Failed to load {file_name}: {e}")
                    validation_results["is_valid"] = False
                    validation_results["checks"][f"{file_name}_loadable"] = False

            # Save status YAML
            with open(self.status_file, "w") as f:
                yaml.safe_dump(validation_results, f)
            
            logger.info(f"Schema validation completed. Status saved to {self.status_file}")
            if not validation_results["is_valid"]:
                raise SchemaValidationError(f"Schema validation failed: {validation_results['errors']}")
            
        except Exception as e:
            logger.error(f"Error in schema validation: {e}")
            raise SchemaValidationError(f"Failed schema validation: {e}")

if __name__ == "__main__":
    stage = SchemaValidationStage()
    stage.run()
