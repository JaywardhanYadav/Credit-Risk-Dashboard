class ScorecardException(Exception):
    """Base exception class for the Credit Risk Scorecard system."""
    pass

class DataIngestionError(ScorecardException):
    """Exception raised during raw data ingestion."""
    pass

class SchemaValidationError(ScorecardException):
    """Exception raised during schema validation checks."""
    pass

class FeatureEngineeringError(ScorecardException):
    """Exception raised during feature engineering and aggregation."""
    pass

class QualityFiltrationError(ScorecardException):
    """Exception raised during feature quality filtering."""
    pass

class ModelTrainingError(ScorecardException):
    """Exception raised during estimator training or calibration."""
    pass

class ScorecardScalingError(ScorecardException):
    """Exception raised during scorecard points scaling."""
    pass
