'''from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
#  combines infrequent categories, reducing dimensionality and improving generalization.
class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Groups infrequent categories in specified columns as 'Other'."""
    def __init__(self, threshold=100, columns=None):
        self.threshold = threshold
        self.columns = columns
        self.mapping_ = {}
    def fit(self, X, y=None):
        X_ = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for col in self.columns:
            freq = X_[col].value_counts()
            self.mapping_[col] = freq[freq >= self.threshold].index
        return self
    def transform(self, X, y=None):
        X_ = X.copy()
        for col in self.columns:
            mask = ~X_[col].isin(self.mapping_[col])
            X_.loc[mask, col] = "Other"
        return X_
# adds new columns with ratios/flags for predictive power.
class EngineeringFeatures(BaseEstimator, TransformerMixin):
    """Creates engineered features like ratios and flags."""
    def fit(self, X, y=None): return self
    def transform(self, X, y=None):
        X_ = X.copy()
        # Handle divide by zero and logics for ratio
        X_["VisitsPerPage"] = X_["TotalVisits"] / (X_["Page Views Per Visit"] + 1)
        X_["ActivityScoreRatio"] = (
            X_["Asymmetrique Activity Score"] / (X_["Asymmetrique Profile Score"] + 1)
        )
        X_["SpecializationMissing"] = X_["Specialization"].isna().astype(int)
        X_["Is_Lead_Profile_Unknown"] = X_["Lead Profile"].isna().astype(int)
        return X_
'''