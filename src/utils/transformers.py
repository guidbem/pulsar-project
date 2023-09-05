import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest

# Creates a transformer class that imputes missing values with the median
class Standardizer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = list(X.columns)
        
        self.scaler = StandardScaler()

        self.scaler.fit(X[self.columns])
        return self

    def transform(self, X):
        X_new = X.copy()
        X_temp = X_new[self.columns]
        X_new[self.columns] = self.scaler.transform(X_temp)
        return X_new
    
# Creates a transformer class that truncates outliers based on the IQR or standard deviation
class OutlierTruncator(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, method='iqr', threshold=1.5):
        self.columns = columns
        self.method = method
        self.threshold = threshold

    def fit(self, X, y=None):
        if self.columns is None:
            # Get the numeric columns
            self.columns = list(X.select_dtypes(include=['int64', 'float64']).columns)
            
        if self.method == 'iqr':
            self.q1 = X[self.columns].quantile(0.25)
            self.q3 = X[self.columns].quantile(0.75)
            self.iqr = self.q3 - self.q1
            self.lower_bound = self.q1 - self.threshold * self.iqr
            self.upper_bound = self.q3 + self.threshold * self.iqr

        elif self.method == 'std':
            self.mean = X[self.columns].mean()
            self.std = X[self.columns].std()
            self.lower_bound = self.mean - self.threshold * self.std
            self.upper_bound = self.mean + self.threshold * self.std
        else:
            raise ValueError('Truncation method must be either iqr or std')
        return self

    def transform(self, X):
        X_new = X.copy()
        X_new[self.columns] = X_new[self.columns].clip(lower=self.lower_bound, upper=self.upper_bound, axis=1)
        return X_new
    
# Creates a transformer class that flags outliers based on the Isolation Forest algorithm
class OutlierFlagger(BaseEstimator, TransformerMixin):
    def __init__(self, outlier_col_name, columns=None, contamination='auto', random_state=42):
        self.columns = columns
        self.contamination = contamination
        self.outlier_col_name = outlier_col_name
        self.random_state = random_state

    def fit(self, X, y=None):
        if self.columns is None:
            # Get the numeric columns
            self.columns = list(X.select_dtypes(include=['int64', 'float64']).columns)
            
        self.iforest = IsolationForest(contamination=self.contamination, random_state=self.random_state)
        self.iforest.fit(X[self.columns])
        return self

    def transform(self, X):
        X_new = X.copy()
        flags = self.iforest.predict(X_new[self.columns])
        flags = np.where(flags == -1, 1, 0)
        X_new[self.outlier_col_name] = flags
        return X_new


# Creates a transformers class that scales the data with the robust scaler
class RobustScaling(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = list(X.columns)
        
        self.scaler = RobustScaler()

        self.scaler.fit(X[self.columns])
        return self

    def transform(self, X):
        X_new = X.copy()
        X_temp = X_new[self.columns]
        X_new[self.columns] = self.scaler.transform(X_temp)
        return X_new
