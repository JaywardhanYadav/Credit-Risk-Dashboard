import os
import yaml
import numpy as np
import pandas as pd
from engine.logger.logger import get_logger
from engine.exception.exceptions import FeatureEngineeringError

logger = get_logger("feature_engineering")

class FeatureEngineeringStage:
    def __init__(self, params_path: str = "params.yaml"):
        with open(params_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.raw_cache_dir = self.config["data"]["raw_cache_dir"]
        self.output_file = self.config["feature_engineering"]["output_file"]

    def downcast_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Downcasts numerical columns in a pandas DataFrame to save memory."""
        start_mem = df.memory_usage().sum() / 1024**2
        for col in df.columns:
            col_type = df[col].dtype
            if col_type != object and col_type.name != 'category' and not pd.api.types.is_datetime64_any_dtype(df[col]):
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type).startswith('int'):
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        end_mem = df.memory_usage().sum() / 1024**2
        logger.info(f"Memory decreased from {start_mem:.2f} MB to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")
        return df

    def run(self):
        try:
            logger.info("Starting feature engineering stage...")
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            
            # Load application_train.csv
            train_path = os.path.join(self.raw_cache_dir, "application_train.csv")
            if not os.path.exists(train_path):
                raise FileNotFoundError(f"application_train.csv not found at {train_path}")
            
            df = pd.read_csv(train_path)
            logger.info(f"Loaded train data. Shape: {df.shape}")

            # 1. Base Application Features
            df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / (df['AMT_CREDIT'] + 1e-5)
            df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1e-5)
            df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + 1e-5)
            df['CHILDREN_RATIO'] = df['CNT_CHILDREN'] / (df['CNT_FAM_MEMBERS'] + 1e-5)
            df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365.25
            df['EMPLOYED_YEARS'] = -df['DAYS_EMPLOYED'] / 365.25
            # Replace 365243 anomaly in DAYS_EMPLOYED
            df['EMPLOYED_YEARS'] = df['EMPLOYED_YEARS'].apply(lambda x: np.nan if x < 0 else x)

            # 2. Bureau & Bureau Balance Aggregations
            bureau_path = os.path.join(self.raw_cache_dir, "bureau.csv")
            if os.path.exists(bureau_path) and os.path.getsize(bureau_path) > 0:
                logger.info("Aggregating bureau.csv...")
                df_bureau = pd.read_csv(bureau_path)
                
                # Basic aggregations
                bureau_agg = df_bureau.groupby('SK_ID_CURR').agg(
                    BUREAU_LOANS_COUNT=('SK_ID_BUREAU', 'count'),
                    BUREAU_DAYS_CREDIT_MAX=('DAYS_CREDIT', 'max'),
                    BUREAU_DEBT_SUM=('AMT_CREDIT_SUM_DEBT', 'sum'),
                    BUREAU_CREDIT_SUM=('AMT_CREDIT_SUM', 'sum')
                )
                bureau_agg['BUREAU_DEBT_CREDIT_RATIO'] = bureau_agg['BUREAU_DEBT_SUM'] / (bureau_agg['BUREAU_CREDIT_SUM'] + 1e-5)
                
                # Active loans count
                active_counts = df_bureau[df_bureau['CREDIT_ACTIVE'] == 'Active'].groupby('SK_ID_CURR').size().rename('BUREAU_ACTIVE_LOANS')
                bureau_agg = bureau_agg.join(active_counts, how='left')
                bureau_agg['BUREAU_ACTIVE_LOANS'] = bureau_agg['BUREAU_ACTIVE_LOANS'].fillna(0)
                
                df = df.join(bureau_agg, on='SK_ID_CURR', how='left')
            else:
                logger.warning("bureau.csv not found or empty. Filling bureau features with NaN.")
                for col in ['BUREAU_LOANS_COUNT', 'BUREAU_DAYS_CREDIT_MAX', 'BUREAU_DEBT_SUM', 'BUREAU_CREDIT_SUM', 'BUREAU_DEBT_CREDIT_RATIO', 'BUREAU_ACTIVE_LOANS']:
                    df[col] = np.nan

            # 3. Previous Application Aggregations
            prev_path = os.path.join(self.raw_cache_dir, "previous_application.csv")
            if os.path.exists(prev_path) and os.path.getsize(prev_path) > 0:
                logger.info("Aggregating previous_application.csv...")
                df_prev = pd.read_csv(prev_path)
                
                prev_agg = df_prev.groupby('SK_ID_CURR').agg(
                    PREV_APP_COUNT=('SK_ID_PREV', 'count'),
                    PREV_ANNUITY_AVG=('AMT_ANNUITY', 'mean')
                )
                
                # Refusal/Rejection count and rate
                refused_counts = df_prev[df_prev['NAME_CONTRACT_STATUS'] == 'Refused'].groupby('SK_ID_CURR').size().rename('PREV_REJECT_COUNT')
                prev_agg = prev_agg.join(refused_counts, how='left')
                prev_agg['PREV_REJECT_COUNT'] = prev_agg['PREV_REJECT_COUNT'].fillna(0)
                prev_agg['PREV_REJECT_RATE'] = prev_agg['PREV_REJECT_COUNT'] / (prev_agg['PREV_APP_COUNT'] + 1e-5)
                
                df = df.join(prev_agg, on='SK_ID_CURR', how='left')
            else:
                logger.warning("previous_application.csv not found or empty. Filling prev application features with NaN.")
                for col in ['PREV_APP_COUNT', 'PREV_ANNUITY_AVG', 'PREV_REJECT_COUNT', 'PREV_REJECT_RATE']:
                    df[col] = np.nan

            # 4. Installments Payments Aggregations
            install_path = os.path.join(self.raw_cache_dir, "installments_payments.csv")
            if os.path.exists(install_path) and os.path.getsize(install_path) > 0:
                logger.info("Aggregating installments_payments.csv...")
                df_install = pd.read_csv(install_path)
                
                # Calculate delays and underpayments
                df_install['DPD_INSTALL'] = (df_install['DAYS_ENTRY_PAYMENT'] - df_install['DAYS_INSTALMENT']).clip(lower=0)
                df_install['UNDERPAY_INSTALL'] = (df_install['AMT_INSTALMENT'] - df_install['AMT_PAYMENT']).clip(lower=0)
                
                install_agg = df_install.groupby('SK_ID_CURR').agg(
                    DPD_MAX_INSTALL=('DPD_INSTALL', 'max'),
                    DPD_MEAN_INSTALL=('DPD_INSTALL', 'mean'),
                    UNDERPAY_MAX_INSTALL=('UNDERPAY_INSTALL', 'max'),
                    UNDERPAY_MEAN_INSTALL=('UNDERPAY_INSTALL', 'mean'),
                    INSTALL_PAYMENTS_COUNT=('SK_ID_PREV', 'count')
                )
                
                # Delinquency flag: fraction of payments made late
                late_pay_counts = df_install[df_install['DPD_INSTALL'] > 0].groupby('SK_ID_CURR').size().rename('INSTALL_LATE_PAY_COUNT')
                install_agg = install_agg.join(late_pay_counts, how='left')
                install_agg['INSTALL_LATE_PAY_COUNT'] = install_agg['INSTALL_LATE_PAY_COUNT'].fillna(0)
                install_agg['INSTALL_LATE_PAY_RATIO'] = install_agg['INSTALL_LATE_PAY_COUNT'] / (install_agg['INSTALL_PAYMENTS_COUNT'] + 1e-5)
                
                df = df.join(install_agg, on='SK_ID_CURR', how='left')
            else:
                logger.warning("installments_payments.csv not found or empty.")
                for col in ['DPD_MAX_INSTALL', 'DPD_MEAN_INSTALL', 'UNDERPAY_MAX_INSTALL', 'UNDERPAY_MEAN_INSTALL', 'INSTALL_PAYMENTS_COUNT', 'INSTALL_LATE_PAY_COUNT', 'INSTALL_LATE_PAY_RATIO']:
                    df[col] = np.nan

            # 5. Credit Card Balance Aggregations
            cc_path = os.path.join(self.raw_cache_dir, "credit_card_balance.csv")
            if os.path.exists(cc_path) and os.path.getsize(cc_path) > 0:
                logger.info("Aggregating credit_card_balance.csv...")
                df_cc = pd.read_csv(cc_path)
                
                # Utilization
                df_cc['CC_UTILIZATION'] = df_cc['AMT_BALANCE'] / (df_cc['AMT_CREDIT_LIMIT_ACTUAL'] + 1e-5)
                
                cc_agg = df_cc.groupby('SK_ID_CURR').agg(
                    DPD_MAX_CC=('SK_DPD', 'max'),
                    CC_UTILIZATION_AVG=('CC_UTILIZATION', 'mean'),
                    CC_BALANCE_AVG=('AMT_BALANCE', 'mean')
                )
                
                # Count and ratio of months with utilization > 90%
                df_cc['UTIL_OVER_90'] = (df_cc['CC_UTILIZATION'] > 0.9).astype(int)
                cc_util_counts = df_cc.groupby('SK_ID_CURR').agg(
                    CC_MONTHS_COUNT=('UTIL_OVER_90', 'count'),
                    CC_OVER_90_COUNT=('UTIL_OVER_90', 'sum')
                )
                cc_agg['CC_UTILIZATION_OVER_90_RATIO'] = cc_util_counts['CC_OVER_90_COUNT'] / (cc_util_counts['CC_MONTHS_COUNT'] + 1e-5)
                
                df = df.join(cc_agg, on='SK_ID_CURR', how='left')
            else:
                logger.warning("credit_card_balance.csv not found or empty.")
                for col in ['DPD_MAX_CC', 'CC_UTILIZATION_AVG', 'CC_BALANCE_AVG', 'CC_UTILIZATION_OVER_90_RATIO']:
                    df[col] = np.nan

            # 6. POS CASH Balance Aggregations
            pos_path = os.path.join(self.raw_cache_dir, "POS_CASH_balance.csv")
            if os.path.exists(pos_path) and os.path.getsize(pos_path) > 0:
                logger.info("Aggregating POS_CASH_balance.csv...")
                df_pos = pd.read_csv(pos_path)
                
                pos_agg = df_pos.groupby('SK_ID_CURR').agg(
                    DPD_MAX_POS=('SK_DPD', 'max'),
                    POS_FUTURE_INSTALMENTS_AVG=('CNT_INSTALMENT_FUTURE', 'mean')
                )
                df = df.join(pos_agg, on='SK_ID_CURR', how='left')
            else:
                logger.warning("POS_CASH_balance.csv not found or empty.")
                for col in ['DPD_MAX_POS', 'POS_FUTURE_INSTALMENTS_AVG']:
                    df[col] = np.nan

            # Replace infinite values with NaN to avoid scaling issues downstream
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Downcast to save system memory
            df = self.downcast_df(df)
            
            # Save final engineered dataset
            df.to_parquet(self.output_file, index=False)
            logger.info(f"Saved engineered features to Parquet: {self.output_file} (Shape: {df.shape})")
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            raise FeatureEngineeringError(f"Failed to run feature engineering: {e}")

if __name__ == "__main__":
    stage = FeatureEngineeringStage()
    stage.run()
