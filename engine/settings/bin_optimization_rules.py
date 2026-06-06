# Manual bin-merging plans to enforce monotonic trends and credit-risk logic

# Pre-defined numerical bin edges. Any value falling in [edges[i], edges[i+1]) belongs to bin i.
NUMERICAL_BINS = {
    'EXT_SOURCE_1': [-float('inf'), 0.2, 0.4, 0.6, 0.8, float('inf')],
    'EXT_SOURCE_2': [-float('inf'), 0.15, 0.3, 0.5, 0.7, float('inf')],
    'EXT_SOURCE_3': [-float('inf'), 0.2, 0.4, 0.6, 0.8, float('inf')],
    'DAYS_BIRTH': [-float('inf'), -20000, -15000, -12000, -9000, float('inf')],  # Age in days (negative)
    'DAYS_EMPLOYED': [-float('inf'), -3000, -1000, 0, 365243, float('inf')],       # 365243 is a common anomaly code
    'AMT_CREDIT': [-float('inf'), 250000, 500000, 800000, 1200000, float('inf')],
    'AMT_GOODS_PRICE': [-float('inf'), 250000, 500000, 750000, 1000000, float('inf')],
    'AMT_ANNUITY': [-float('inf'), 15000, 25000, 35000, 50000, float('inf')],
    'DAYS_LAST_PHONE_CHANGE': [-float('inf'), -2000, -1000, -500, 0, float('inf')],
    
    # Bureau features
    'BUREAU_DAYS_CREDIT_MAX': [-float('inf'), -2000, -1000, -500, 0, float('inf')],
    'BUREAU_DEBT_CREDIT_RATIO': [-float('inf'), 0.0, 0.2, 0.5, 0.8, float('inf')],
    
    # Delinquency features
    'DPD_MAX_CC': [-float('inf'), 0.0, 1.0, 30.0, float('inf')],
    'DPD_MAX_POS': [-float('inf'), 0.0, 1.0, 30.0, float('inf')],
    'DPD_MAX_INSTALL': [-float('inf'), 0.0, 1.0, 30.0, float('inf')],
}

# Categorical groupings to bin high-cardinality features or low-frequency groups
CATEGORICAL_GROUPS = {
    'NAME_EDUCATION_TYPE': {
        'Academic degree': 'Higher education',
        'Higher education': 'Higher education',
        'Incomplete higher': 'Secondary education',
        'Secondary / special': 'Secondary education',
        'Lower secondary': 'Lower secondary'
    },
    'NAME_INCOME_TYPE': {
        'State servant': 'Salaried',
        'Commercial associate': 'Salaried',
        'Working': 'Working',
        'Pensioner': 'Pensioner',
        # Group other rare income types
        'Unemployed': 'Other',
        'Businessman': 'Salaried',
        'Student': 'Other',
        'Maternity leave': 'Other'
    },
    'CODE_GENDER': {
        'M': 'M',
        'F': 'F',
        'XNA': 'F' # Group rare/missing gender with major
    },
    'FLAG_OWN_CAR': {
        'Y': 'Y',
        'N': 'N'
    },
    'FLAG_OWN_REALTY': {
        'Y': 'Y',
        'N': 'N'
    }
}
