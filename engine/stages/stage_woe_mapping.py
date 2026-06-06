import os
import pickle
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from engine.logger.logger import get_logger
from engine.exception.exceptions import ScorecardException
from engine.settings.bin_optimization_rules import NUMERICAL_BINS, CATEGORICAL_GROUPS

logger = get_logger("woe_mapping")

class WoEMappingStage:
    def __init__(self, params_path: str = "params.yaml"):
        with open(params_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.input_file = self.config["quality_filtration"]["output_file"]
        self.output_dir = self.config["woe_mapping"]["output_dir"]
        self.test_size = self.config["woe_mapping"].get("test_size", 0.2)
        self.min_iv = self.config["woe_mapping"].get("min_iv", 0.02)
        self.random_state = self.config["data"].get("random_state", 42)

    def get_bin_edges(self, series: pd.Series, col_name: str) -> list:
        """Returns the bin edges for a numerical feature, prioritizing predefined rules."""
        if col_name in NUMERICAL_BINS:
            return NUMERICAL_BINS[col_name]
        
        # Automated binning using qcut
        try:
            non_null = series.dropna()
            if len(non_null.unique()) <= 2:
                # Binary/binary-like numerical
                edges = [-float('inf'), non_null.min() + 0.5, float('inf')]
                return edges
            
            _, bins = pd.qcut(non_null, q=5, retbins=True, duplicates='drop')
            edges = list(bins)
            edges[0] = -float('inf')
            edges[-1] = float('inf')
            return edges
        except Exception:
            try:
                _, bins = pd.cut(non_null, bins=5, retbins=True, duplicates='drop')
                edges = list(bins)
                edges[0] = -float('inf')
                edges[-1] = float('inf')
                return edges
            except Exception:
                return [-float('inf'), float('inf')]

    def run(self):
        try:
            logger.info("Starting WoE mapping stage...")
            os.makedirs(self.output_dir, exist_ok=True)
            
            if not os.path.exists(self.input_file):
                raise FileNotFoundError(f"Input file not found: {self.input_file}")
                
            df = pd.read_parquet(self.input_file)
            logger.info(f"Loaded filtered features. Shape: {df.shape}")
            
            # Split train and test
            df_train, df_test = train_test_split(
                df, test_size=self.test_size, random_state=self.random_state, stratify=df["TARGET"]
            )
            logger.info(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")
            
            # Save raw test set (without WoE transformations) for API / dashboard testing
            df_test_raw_path = os.path.join(self.output_dir, "test_raw.parquet")
            df_test.to_parquet(df_test_raw_path, index=False)
            logger.info(f"Saved raw test partition to {df_test_raw_path}")

            woe_bin_maps = {}
            iv_report = {}
            
            total_good = (df_train["TARGET"] == 0).sum()
            total_bad = (df_train["TARGET"] == 1).sum()
            
            if total_good == 0 or total_bad == 0:
                raise ScorecardException("Target variable is homogeneous in train set. Check data sampling.")
            
            # Predictors to transform (exclude SK_ID_CURR, TARGET)
            predictors = [c for c in df_train.columns if c not in ["SK_ID_CURR", "TARGET"]]
            selected_features = []
            
            for col in predictors:
                # Determine type
                is_num = pd.api.types.is_numeric_dtype(df_train[col])
                
                bin_woe_map = {}
                iv = 0.0
                
                if is_num:
                    edges = self.get_bin_edges(df_train[col], col)
                    # Create labels
                    labels = []
                    for i in range(len(edges) - 1):
                        labels.append(f"{edges[i]} to {edges[i+1]}")
                        
                    # Bin training series
                    binned = pd.cut(df_train[col], bins=edges, labels=labels, right=False)
                    
                    # Calculate counts for each bin
                    grouped = df_train.groupby(binned)["TARGET"].agg(["count", "sum"])
                    
                    # Calculate WoE and IV
                    for bin_label, row in grouped.iterrows():
                        n_bad = int(row["sum"])
                        n_good = int(row["count"] - n_bad)
                        
                        # Apply Laplace smoothing if 0
                        n_good_smooth = n_good + 0.5 if n_good == 0 or n_bad == 0 else n_good
                        n_bad_smooth = n_bad + 0.5 if n_good == 0 or n_bad == 0 else n_bad
                        
                        perc_good = n_good_smooth / total_good
                        perc_bad = n_bad_smooth / total_bad
                        
                        woe = np.log(perc_good / perc_bad)
                        bin_woe_map[bin_label] = float(woe)
                        
                        iv += (perc_good - perc_bad) * woe
                        
                    # Missing value bin
                    null_count = df_train[col].isnull().sum()
                    if null_count > 0:
                        n_bad_null = df_train[df_train[col].isnull()]["TARGET"].sum()
                        n_good_null = null_count - n_bad_null
                        
                        n_good_null = n_good_null + 0.5 if n_good_null == 0 or n_bad_null == 0 else n_good_null
                        n_bad_null = n_bad_null + 0.5 if n_good_null == 0 or n_bad_null == 0 else n_bad_null
                        
                        perc_good_null = n_good_null / total_good
                        perc_bad_null = n_bad_null / total_bad
                        woe_null = np.log(perc_good_null / perc_bad_null)
                        bin_woe_map["Missing"] = float(woe_null)
                        iv += (perc_good_null - perc_bad_null) * woe_null
                    else:
                        bin_woe_map["Missing"] = 0.0  # Default to 0 points if missing is encountered in production
                        
                    woe_bin_maps[col] = {
                        "type": "numerical",
                        "edges": edges,
                        "woe_map": bin_woe_map
                    }
                else:
                    # Categorical feature
                    raw_series = df_train[col].fillna("Missing").astype(str)
                    
                    # Apply manual categorical grouping if present
                    if col in CATEGORICAL_GROUPS:
                        group_map = CATEGORICAL_GROUPS[col]
                        mapped_series = raw_series.map(lambda x: group_map.get(x, "Other"))
                    else:
                        mapped_series = raw_series
                        
                    # Calculate counts for each group
                    grouped = df_train.groupby(mapped_series)["TARGET"].agg(["count", "sum"])
                    
                    for grp, row in grouped.iterrows():
                        n_bad = int(row["sum"])
                        n_good = int(row["count"] - n_bad)
                        
                        n_good_smooth = n_good + 0.5 if n_good == 0 or n_bad == 0 else n_good
                        n_bad_smooth = n_bad + 0.5 if n_good == 0 or n_bad == 0 else n_bad
                        
                        perc_good = n_good_smooth / total_good
                        perc_bad = n_bad_smooth / total_bad
                        
                        woe = np.log(perc_good / perc_bad)
                        bin_woe_map[grp] = float(woe)
                        
                        iv += (perc_good - perc_bad) * woe
                    
                    # Store mapping
                    woe_bin_maps[col] = {
                        "type": "categorical",
                        "group_map": CATEGORICAL_GROUPS.get(col, {}),
                        "woe_map": bin_woe_map
                    }
                
                iv_report[col] = iv
                if iv >= self.min_iv:
                    selected_features.append(col)
                    
            logger.info(f"Features with IV >= {self.min_iv}: {len(selected_features)}")
            
            # Map train and test to WoE values
            df_train_woe = pd.DataFrame()
            df_train_woe["SK_ID_CURR"] = df_train["SK_ID_CURR"]
            df_train_woe["TARGET"] = df_train["TARGET"]
            
            df_test_woe = pd.DataFrame()
            df_test_woe["SK_ID_CURR"] = df_test["SK_ID_CURR"]
            df_test_woe["TARGET"] = df_test["TARGET"]
            
            # Apply WoE transformations for selected features
            for col in selected_features:
                map_info = woe_bin_maps[col]
                
                # Transform train
                df_train_woe[col] = self.transform_to_woe(df_train[col], map_info)
                # Transform test
                df_test_woe[col] = self.transform_to_woe(df_test[col], map_info)
            
            # Save WoE datasets
            train_woe_path = os.path.join(self.output_dir, "train_woe.parquet")
            test_woe_path = os.path.join(self.output_dir, "test_woe.parquet")
            
            df_train_woe.to_parquet(train_woe_path, index=False)
            df_test_woe.to_parquet(test_woe_path, index=False)
            logger.info(f"Saved WoE-transformed train and test sets to {train_woe_path} and {test_woe_path}")
            
            # Save mapping definitions
            maps_pkl_path = os.path.join(self.output_dir, "woe_bin_maps.pkl")
            with open(maps_pkl_path, "wb") as f:
                pickle.dump(woe_bin_maps, f)
            logger.info(f"Saved WoE mapping dictionary to {maps_pkl_path}")
            
            # Save IV Report
            iv_report_path = os.path.join(self.output_dir, "iv_report.json")
            with open(iv_report_path, "w") as f:
                import json
                json.dump(iv_report, f, indent=2)
            logger.info(f"Saved IV report to {iv_report_path}")
            
        except Exception as e:
            logger.error(f"Error in WoE mapping: {e}")
            raise ScorecardException(f"Failed to run WoE mapping: {e}")

    def transform_to_woe(self, series: pd.Series, map_info: dict) -> pd.Series:
        """Transforms a series to WoE values using map_info."""
        woe_map = map_info["woe_map"]
        
        if map_info["type"] == "numerical":
            edges = map_info["edges"]
            labels = []
            for i in range(len(edges) - 1):
                labels.append(f"{edges[i]} to {edges[i+1]}")
                
            # Perform cut
            binned = pd.cut(series, bins=edges, labels=labels, right=False)
            # Map bin labels to WoE
            woe_series = binned.astype(object).map(woe_map)
            
            # Fill missing entries with the Missing WoE
            woe_series = woe_series.fillna(woe_map.get("Missing", 0.0))
            return woe_series.astype(np.float32)
        else:
            # Categorical mapping
            raw_series = series.fillna("Missing").astype(str)
            group_map = map_info["group_map"]
            
            # Map categories to groups
            if group_map:
                mapped_series = raw_series.map(lambda x: group_map.get(x, "Other"))
            else:
                mapped_series = raw_series
                
            # Map groups to WoE
            woe_series = mapped_series.map(woe_map)
            
            # Fallback for unseen categories
            fallback_woe = woe_map.get("Other", woe_map.get("Missing", 0.0))
            woe_series = woe_series.fillna(fallback_woe)
            return woe_series.astype(np.float32)

if __name__ == "__main__":
    stage = WoEMappingStage()
    stage.run()
