"""ColumnTransformer over FEATURE_SPEC: impute, scale and encode inside the Pipeline.

Every statistic (medians, quantile fences, scaler parameters, one-hot categories) is
learnt by ``fit`` on the training rows only, so the same object is safe inside
GridSearchCV and on the held-out test set. The numeric block is cast to float32 at the
start so that the dense output of about 500 one-hot columns on 100k rows stays near
200 MB instead of 400 MB.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted, validate_data

from mlda.config import ONE_HOT_MIN_FREQUENCY
from mlda.features import FEATURE_SPEC

# Allowed values for build_preprocessor with a one-line rationale each (shown in the app).
OPTIONS: dict[str, dict[str | None, str]] = {
    "numeric_imputer": {
        "median": "SimpleImputer(median) on the training rows: the experience columns are "
        "right-skewed (skew 1.4 to 2.3) and the exposure columns more so (3.2 to 4.7), so the "
        "median is the safer centre.",
        "mean": "SimpleImputer(mean): the textbook default, pulled upwards by the long "
        "right tails of hours and experience.",
        "knn": "KNNImputer(5 neighbours) on the standardised numeric block (scaled for the "
        "neighbour search, scaled back afterwards): borrows values from similar training rows "
        "at a much higher cost on 100k rows.",
    },
    "scaler": {
        "standard": "StandardScaler to zero mean and unit variance: required by logistic "
        "regression, SVM, KNN and the MLP.",
        "minmax": "MinMaxScaler to [0, 1]: keeps the shape of each distribution and is "
        "sensitive to extreme training values.",
        "robust": "RobustScaler centres on the median and divides by the IQR, so a few "
        "extreme mines do not set the scale.",
        "none": "No scaling: tree ensembles and Gaussian naive Bayes do not depend on the "
        "scale of the inputs.",
    },
    "encoder": {
        "onehot": "OneHotEncoder with an infrequent bucket for levels rarer than "
        "min_frequency; unseen levels at inference fall into that bucket.",
        "ordinal": "OrdinalEncoder with -1 for unseen levels: compact for trees, but it "
        "imposes an arbitrary order that linear models would read as magnitude.",
    },
    "outlier_rule": {
        None: "No clipping: the error rules E1 to E30 already remove impossible values.",
        "iqr_clip": "Clip to Tukey fences Q1 - 1.5 IQR and Q3 + 1.5 IQR learnt on the "
        "training rows; a column whose IQR is zero (such as the victim count, 99 % ones) keeps "
        "open fences, because clipping it would flatten it to a constant.",
        "zscore_clip": "Clip to mean +/- 3 sample standard deviations learnt on the training rows.",
    },
}
DEFAULTS: dict[str, str | None] = {
    "numeric_imputer": "median",
    "scaler": "standard",
    "encoder": "onehot",
    "outlier_rule": None,
}
CLIPPER_DEFAULT_K: dict[str, float] = {"iqr": 1.5, "zscore": 3.0}
OUTLIER_RULE_TO_METHOD: dict[str, str] = {"iqr_clip": "iqr", "zscore_clip": "zscore"}


def _to_float32(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Dense float32 copy of the numeric block, NaN preserved."""
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=np.float32, na_value=np.nan)
    return np.asarray(X, dtype=np.float32)


class OutlierClipper(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Clip each numeric column to fences learnt on the training rows.

    ``method="iqr"`` uses Tukey's fences Q1 - k IQR and Q3 + k IQR (k = 1.5), and
    ``method="zscore"`` uses mean +/- k sample standard deviations (k = 3). NaN values
    pass through unchanged so that the imputer that follows still sees them. A column with
    zero IQR (or zero spread) keeps open fences: clipping it to its quartile would erase the
    few rows that differ, as happens to the capped victim count where 99 % of rows equal 1.
    """

    def __init__(self, method: str = "iqr", k: float | None = None) -> None:
        self.method = method
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> OutlierClipper:
        """Learn the lower and upper fence of every column, ignoring NaN."""
        if self.method not in CLIPPER_DEFAULT_K:
            raise ValueError(
                f"method must be one of {sorted(CLIPPER_DEFAULT_K)}, got {self.method!r}"
            )
        X = validate_data(self, X, dtype=FLOAT_DTYPES, ensure_all_finite="allow-nan")
        k = CLIPPER_DEFAULT_K[self.method] if self.k is None else float(self.k)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns are handled below
            if self.method == "iqr":
                q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
                lower, upper = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
                flat = q3 == q1  # fences at the quartile would turn the column into a constant
            else:
                centre, spread = np.nanmean(X, axis=0), np.nanstd(X, axis=0, ddof=1)
                lower, upper = centre - k * spread, centre + k * spread
                flat = spread == 0
        # A column with no finite training value (or a single one) or with zero spread gets open fences.
        self.lower_ = np.where(np.isfinite(lower) & ~flat, lower, -np.inf).astype(np.float64)
        self.upper_ = np.where(np.isfinite(upper) & ~flat, upper, np.inf).astype(np.float64)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Clip to the learnt fences; NaN stays NaN."""
        check_is_fitted(self)
        X = validate_data(
            self, X, reset=False, dtype=FLOAT_DTYPES, ensure_all_finite="allow-nan", copy=True
        )
        return np.clip(X, self.lower_.astype(X.dtype), self.upper_.astype(X.dtype))


class ScaledKNNImputer(TransformerMixin, BaseEstimator):
    """KNNImputer that measures distances on standardised columns and returns original units.

    ``prev_hours`` has a standard deviation of about a million while most columns sit between
    0 and 60, so on raw values the nan-Euclidean distance is decided by one column. The
    StandardScaler (NaN-aware) is fitted on the training rows, the neighbour search runs on the
    z-scores, and ``inverse_transform`` returns the imputed block in the original units so the
    scaler that follows in the pipeline behaves as before.
    """

    def __init__(self, n_neighbors: int = 5) -> None:
        self.n_neighbors = n_neighbors

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> ScaledKNNImputer:
        """Fit the scaler and the imputer on the training rows."""
        X = validate_data(self, X, dtype=FLOAT_DTYPES, ensure_all_finite="allow-nan")
        self.scaler_ = StandardScaler().fit(X)
        self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors, keep_empty_features=True)
        self.imputer_.fit(self.scaler_.transform(X))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Impute missing values from the nearest training rows; output is in original units."""
        check_is_fitted(self, ("scaler_", "imputer_"))
        X = validate_data(self, X, reset=False, dtype=FLOAT_DTYPES, ensure_all_finite="allow-nan")
        return self.scaler_.inverse_transform(self.imputer_.transform(self.scaler_.transform(X)))

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        """One output column per input column."""
        check_is_fitted(self, ("scaler_", "imputer_"))
        return self.imputer_.get_feature_names_out(input_features)


def _validate_option(name: str, value: str | None) -> str | None:
    allowed = OPTIONS[name]
    if name == "outlier_rule" and value == "none":
        value = None
    if value not in allowed:
        raise ValueError(f"{name} must be one of {list(allowed)}, got {value!r}")
    return value


def build_preprocessor(
    numeric_imputer: str = "median",
    scaler: str = "standard",
    encoder: str = "onehot",
    outlier_rule: str | None = None,
    min_frequency: float | int | None = ONE_HOT_MIN_FREQUENCY,
    knn_neighbors: int = 5,
) -> ColumnTransformer:
    """Build the unfitted ColumnTransformer for the columns in FEATURE_SPEC.

    numeric: float32 cast -> optional OutlierClipper -> SimpleImputer / KNNImputer ->
    scaler; categorical: SimpleImputer(constant "missing") -> OneHotEncoder with an
    infrequent bucket or OrdinalEncoder with -1 for unseen levels; binary: passthrough.
    The output is a dense float32 matrix because every branch emits float32 or int8.
    """
    numeric_imputer = _validate_option("numeric_imputer", numeric_imputer)
    scaler = _validate_option("scaler", scaler)
    encoder = _validate_option("encoder", encoder)
    outlier_rule = _validate_option("outlier_rule", outlier_rule)

    if numeric_imputer == "knn":
        imputer = ScaledKNNImputer(n_neighbors=knn_neighbors)
    else:
        imputer = SimpleImputer(strategy=numeric_imputer)
    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        "none": "passthrough",
    }
    clipper = (
        "passthrough"
        if outlier_rule is None
        else OutlierClipper(method=OUTLIER_RULE_TO_METHOD[outlier_rule])
    )
    numeric_pipeline = Pipeline(
        [
            ("cast", FunctionTransformer(_to_float32, feature_names_out="one-to-one")),
            ("clipper", clipper),
            ("imputer", imputer),
            ("scaler", scalers[scaler]),
        ]
    )

    if encoder == "onehot":
        categorical_encoder = OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=min_frequency,
            sparse_output=False,
            dtype=np.float32,
        )
    else:
        categorical_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32
        )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", categorical_encoder),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, FEATURE_SPEC["numeric"]),
            ("cat", categorical_pipeline, FEATURE_SPEC["categorical"]),
            ("bin", "passthrough", FEATURE_SPEC["binary"]),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Output column names of a fitted preprocessor (one-hot levels expanded)."""
    check_is_fitted(preprocessor)
    return [str(name) for name in preprocessor.get_feature_names_out()]
