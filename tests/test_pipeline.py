import pandas as pd
import numpy as np
from engine.stages.stage_feature_engineering import FeatureEngineeringStage
from engine.stages.stage_quality_filtration import QualityFiltrationStage

def test_dataframe_downcasting():
    """Verifies that downcast_df successfully reduces memory size of floats and ints."""
    stage = FeatureEngineeringStage()
    
    # Create high precision floats and large ints
    df = pd.DataFrame({
        "int_col": [1, 2, 3, 4, 5],
        "float_col": [1.5, 2.5, 3.5, 4.5, 5.5]
    })
    
    assert df["int_col"].dtype == np.int64
    assert df["float_col"].dtype == np.float64
    
    df_down = stage.downcast_df(df)
    
    # Assert type change to smaller representations
    assert df_down["int_col"].dtype in [np.int8, np.int16, np.int32]
    assert df_down["float_col"].dtype == np.float32

def test_feature_engineering_ratios():
    """Verifies the base applicant ratio engineering formulas."""
    df_raw = pd.DataFrame({
        "AMT_INCOME_TOTAL": [100000.0, 200000.0],
        "AMT_CREDIT": [500000.0, 600000.0],
        "AMT_ANNUITY": [20000.0, 30000.0],
        "CNT_CHILDREN": [1, 0],
        "CNT_FAM_MEMBERS": [3.0, 2.0],
        "DAYS_BIRTH": [-10000, -20000],
        "DAYS_EMPLOYED": [-1000, 365243] # Include anomaly code
    })
    
    # Run inline feature derivations matching stage_feature_engineering
    df_raw['INCOME_CREDIT_PERC'] = df_raw['AMT_INCOME_TOTAL'] / (df_raw['AMT_CREDIT'] + 1e-5)
    df_raw['ANNUITY_INCOME_PERC'] = df_raw['AMT_ANNUITY'] / (df_raw['AMT_INCOME_TOTAL'] + 1e-5)
    df_raw['PAYMENT_RATE'] = df_raw['AMT_ANNUITY'] / (df_raw['AMT_CREDIT'] + 1e-5)
    df_raw['CHILDREN_RATIO'] = df_raw['CNT_CHILDREN'] / (df_raw['CNT_FAM_MEMBERS'] + 1e-5)
    df_raw['AGE_YEARS'] = -df_raw['DAYS_BIRTH'] / 365.25
    
    employed_yrs = -df_raw['DAYS_EMPLOYED'] / 365.25
    df_raw['EMPLOYED_YEARS'] = employed_yrs.apply(lambda x: np.nan if x < 0 else x)
    
    # Assert values
    assert np.isclose(df_raw.loc[0, "INCOME_CREDIT_PERC"], 0.2)
    assert np.isclose(df_raw.loc[0, "ANNUITY_INCOME_PERC"], 0.2)
    assert np.isclose(df_raw.loc[0, "PAYMENT_RATE"], 0.04)
    assert np.isclose(df_raw.loc[0, "CHILDREN_RATIO"], 1.0/3.0)
    assert np.isclose(df_raw.loc[0, "AGE_YEARS"], 10000.0/365.25)
    assert pd.isnull(df_raw.loc[1, "EMPLOYED_YEARS"]) # Verify anomaly replaced with NaN
