from typing import Optional, Dict
from pydantic import BaseModel, Field

class ApplicantRequest(BaseModel):
    model_config = {
        "extra": "allow"
    }
    
    # Base application features
    SK_ID_CURR: Optional[int] = Field(default=999999, description="Unique applicant ID")
    AMT_INCOME_TOTAL: float = Field(..., description="Total income of the applicant", example=150000.0)
    AMT_CREDIT: float = Field(..., description="Credit amount of the loan", example=500000.0)
    AMT_ANNUITY: float = Field(..., description="Loan annuity", example=25000.0)
    CNT_CHILDREN: int = Field(default=0, description="Number of children", example=0)
    CNT_FAM_MEMBERS: float = Field(default=2.0, description="Family members count", example=2.0)
    DAYS_BIRTH: int = Field(..., description="Age in days (negative)", example=-15000)
    DAYS_EMPLOYED: int = Field(..., description="Days employed (negative)", example=-2000)
    DAYS_LAST_PHONE_CHANGE: float = Field(default=-500.0, description="Days since last phone change", example=-500.0)
    
    NAME_EDUCATION_TYPE: str = Field(..., description="Education level", example="Secondary / special")
    NAME_INCOME_TYPE: str = Field(..., description="Income source type", example="Working")
    CODE_GENDER: str = Field(..., description="Gender (M/F)", example="M")
    FLAG_OWN_CAR: str = Field(default="N", description="Car owner flag (Y/N)", example="N")
    FLAG_OWN_REALTY: str = Field(default="Y", description="Real estate owner flag (Y/N)", example="Y")
    
    # Optional bureau features
    BUREAU_LOANS_COUNT: Optional[float] = Field(default=None, description="Number of past loans in Bureau")
    BUREAU_DAYS_CREDIT_MAX: Optional[float] = Field(default=None, description="Max days since past Bureau credit")
    BUREAU_DEBT_SUM: Optional[float] = Field(default=None, description="Total debt sum from Bureau")
    BUREAU_CREDIT_SUM: Optional[float] = Field(default=None, description="Total credit sum from Bureau")
    BUREAU_ACTIVE_LOANS: Optional[float] = Field(default=None, description="Active Bureau loans count")
    
    # Optional previous application features
    PREV_APP_COUNT: Optional[float] = Field(default=None, description="Count of previous applications")
    PREV_ANNUITY_AVG: Optional[float] = Field(default=None, description="Average annuity of previous applications")
    PREV_REJECT_COUNT: Optional[float] = Field(default=None, description="Rejection count of previous applications")
    
    # Optional installment features
    DPD_MAX_INSTALL: Optional[float] = Field(default=None, description="Max delay days in installments payments")
    DPD_MEAN_INSTALL: Optional[float] = Field(default=None, description="Mean delay days in installments payments")
    UNDERPAY_MAX_INSTALL: Optional[float] = Field(default=None, description="Max underpaid amount in installments")
    UNDERPAY_MEAN_INSTALL: Optional[float] = Field(default=None, description="Mean underpaid amount in installments")
    INSTALL_PAYMENTS_COUNT: Optional[float] = Field(default=None, description="Number of installment payments made")
    INSTALL_LATE_PAY_RATIO: Optional[float] = Field(default=None, description="Fraction of late installment payments")
    
    # Optional credit card features
    DPD_MAX_CC: Optional[float] = Field(default=None, description="Max DPD for credit card accounts")
    CC_UTILIZATION_AVG: Optional[float] = Field(default=None, description="Average credit card utilization")
    CC_BALANCE_AVG: Optional[float] = Field(default=None, description="Average credit card balance")
    CC_UTILIZATION_OVER_90_RATIO: Optional[float] = Field(default=None, description="Fraction of months with utilization over 90%")
    
    # Optional POS features
    DPD_MAX_POS: Optional[float] = Field(default=None, description="Max DPD for POS CASH accounts")
    POS_FUTURE_INSTALMENTS_AVG: Optional[float] = Field(default=None, description="Average future installments for POS")
    
    # EAD estimation parameter
    loan_amount: float = Field(default=50000.0, description="Exposure at Default (EAD) represented by requested loan amount", example=50000.0)

class ScoringResponse(BaseModel):
    sk_id_curr: int
    score: int
    probability_of_default: float
    decision: str
    risk_tier: str
    score_breakdown: Dict[str, int]
