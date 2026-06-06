import os
import shutil
import pandas as pd
import yaml
from engine.logger.logger import get_logger
from engine.exception.exceptions import DataIngestionError

logger = get_logger("raw_ingestion")

class RawIngestionStage:
    def __init__(self, params_path: str = "params.yaml"):
        with open(params_path, "r") as f:
            self.params = yaml.safe_load(f)["data"]
        
        self.raw_dir = self.params.get("raw_dir", "home-credit-default-risk")
        self.cache_dir = self.params.get("raw_cache_dir", "artifacts_cache/raw")
        self.sample_fraction = self.params.get("sample_fraction", 0.1)
        self.random_state = self.params.get("random_state", 42)

    def run(self):
        try:
            logger.info("Starting raw ingestion stage...")
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # 1. Ingest & Sample application_train.csv
            train_path = os.path.join(self.raw_dir, "application_train.csv")
            if not os.path.exists(train_path):
                raise FileNotFoundError(f"Missing base training file: {train_path}")

            logger.info(f"Reading base training data from {train_path}")
            df_train = pd.read_csv(train_path)
            
            if self.sample_fraction < 1.0:
                logger.info(f"Applying sampling fraction of {self.sample_fraction}...")
                df_train = df_train.sample(frac=self.sample_fraction, random_state=self.random_state).reset_index(drop=True)
                logger.info(f"Sampled train set size: {df_train.shape}")
            else:
                logger.info(f"No sampling applied. Running on full dataset. Rows: {df_train.shape[0]}")
            
            # Save sampled train
            out_train_path = os.path.join(self.cache_dir, "application_train.csv")
            df_train.to_csv(out_train_path, index=False)
            logger.info(f"Saved application_train.csv to {out_train_path}")
            
            sampled_ids = set(df_train["SK_ID_CURR"])

            # 2. Filter relational tables using sampled IDs
            relational_files = [
                "previous_application.csv",
                "installments_payments.csv",
                "credit_card_balance.csv",
                "POS_CASH_balance.csv"
            ]
            
            for file_name in relational_files:
                src_file = os.path.join(self.raw_dir, file_name)
                dest_file = os.path.join(self.cache_dir, file_name)
                if not os.path.exists(src_file):
                    logger.warning(f"File {src_file} does not exist. Skipping.")
                    continue
                
                logger.info(f"Filtering and copying {file_name}...")
                # Process in chunks to minimize memory usage
                chunk_list = []
                for chunk in pd.read_csv(src_file, chunksize=100000):
                    filtered_chunk = chunk[chunk["SK_ID_CURR"].isin(sampled_ids)]
                    chunk_list.append(filtered_chunk)
                
                df_filtered = pd.concat(chunk_list, ignore_index=True)
                df_filtered.to_csv(dest_file, index=False)
                logger.info(f"Saved filtered {file_name} (Rows: {len(df_filtered)}) to {dest_file}")

            # 3. Filter Bureau and Bureau Balance
            bureau_src = os.path.join(self.raw_dir, "bureau.csv")
            bureau_dest = os.path.join(self.cache_dir, "bureau.csv")
            bureau_balance_src = os.path.join(self.raw_dir, "bureau_balance.csv")
            bureau_balance_dest = os.path.join(self.cache_dir, "bureau_balance.csv")

            sampled_bureau_ids = set()
            if os.path.exists(bureau_src):
                logger.info("Filtering and copying bureau.csv...")
                chunk_list = []
                for chunk in pd.read_csv(bureau_src, chunksize=100000):
                    filtered_chunk = chunk[chunk["SK_ID_CURR"].isin(sampled_ids)]
                    chunk_list.append(filtered_chunk)
                df_bureau = pd.concat(chunk_list, ignore_index=True)
                df_bureau.to_csv(bureau_dest, index=False)
                logger.info(f"Saved filtered bureau.csv (Rows: {len(df_bureau)}) to {bureau_dest}")
                sampled_bureau_ids = set(df_bureau["SK_ID_BUREAU"])

            if os.path.exists(bureau_balance_src) and len(sampled_bureau_ids) > 0:
                logger.info("Filtering and copying bureau_balance.csv...")
                chunk_list = []
                for chunk in pd.read_csv(bureau_balance_src, chunksize=100000):
                    filtered_chunk = chunk[chunk["SK_ID_BUREAU"].isin(sampled_bureau_ids)]
                    chunk_list.append(filtered_chunk)
                df_bb = pd.concat(chunk_list, ignore_index=True)
                df_bb.to_csv(bureau_balance_dest, index=False)
                logger.info(f"Saved filtered bureau_balance.csv (Rows: {len(df_bb)}) to {bureau_balance_dest}")
                
            logger.info("Raw Ingestion Stage completed successfully.")
            
        except Exception as e:
            logger.error(f"Error in raw ingestion stage: {e}")
            raise DataIngestionError(f"Failed to ingest raw data: {e}")

if __name__ == "__main__":
    stage = RawIngestionStage()
    stage.run()
