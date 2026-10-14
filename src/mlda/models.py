"""Model zoo: the eight classifiers compared in the study and their search grids.

Each entry is a ``ModelSpec`` holding a factory for the estimator, the parameter grid that
``mlda.tuning.tune`` searches, the preprocessing options the model prefers (trees do not need
a scaler), an optional row cap for models whose cost grows faster than linearly with the
number of rows, and a short description in the vocabulary of the module.

Grid sizes are chosen so that a five-fold multi-metric GridSearchCV with training scores
finishes in roughly ten minutes per model on about 80,000 training rows and 500 one-hot
columns with eight parallel workers. Every stochastic estimator receives ``RANDOM_STATE``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from mlda.config import RANDOM_STATE

PREPROCESSOR_STEP = "pre"
ESTIMATOR_STEP = "clf"

# Row caps. SVC training is quadratic in the number of rows, and KNN scoring needs the
# distance from every scored row to every training row, so both are fitted on a stratified
# subsample (see mlda.tuning.tune for how out-of-fold probabilities still cover every row).
SVC_SAMPLE_CAP = 10_000
KNN_SAMPLE_CAP = 40_000
# Quick mode (tests, app, mlda.pipeline.train_all(quick=True)) caps every model at 10,000 rows,
# which leaves the SVC unchanged; its cost is quadratic in rows and the five inner Platt fits
# of CalibratedClassifierCV took about nine minutes at 10,000 rows, so quick mode uses fewer.
SVC_QUICK_SAMPLE_CAP = 4_000


@dataclass(frozen=True)
class ModelSpec:
    """Everything the tuner needs to know about one candidate model.

    ``param_grid`` keys are prefixed with ``clf__`` so they address the estimator step of the
    pipeline built by ``build_pipeline``. ``preprocess_overrides`` are keyword arguments for
    ``mlda.preprocessing.build_preprocessor`` (for example ``scaler="none"`` for tree models,
    whose splits are invariant to monotone rescaling). ``quick_sample_cap`` replaces the
    default 10,000-row cap of quick mode for models that are still too slow at that size.
    """

    name: str
    estimator: Callable[[], BaseEstimator]
    param_grid: dict[str, list[Any]]
    search: Literal["grid", "random"] = "grid"
    n_iter: int | None = None
    preprocess_overrides: dict[str, Any] = field(default_factory=dict)
    sample_cap: int | None = None
    description: str = ""
    taught_in_week: int = 0
    quick_sample_cap: int | None = None

    @property
    def n_candidates(self) -> int:
        """Number of parameter combinations the full (non-quick) search evaluates."""
        if self.search == "random" and self.n_iter is not None:
            return min(self.n_iter, len(ParameterGrid(self.param_grid)))
        return len(ParameterGrid(self.param_grid))


def _logreg() -> LogisticRegression:
    # The L2 penalty is the default (l1_ratio=0); the deprecated ``penalty`` keyword is not
    # passed so the factory stays valid on scikit-learn 1.8-1.10.
    return LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)


def _knn() -> KNeighborsClassifier:
    return KNeighborsClassifier(algorithm="brute")


def _gaussian_nb() -> GaussianNB:
    return GaussianNB()


def _decision_tree() -> DecisionTreeClassifier:
    return DecisionTreeClassifier(random_state=RANDOM_STATE)


def _random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)


def _hist_gb() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=RANDOM_STATE,
    )


def _svc_rbf() -> CalibratedClassifierCV:
    # ``SVC(probability=True)`` is deprecated in scikit-learn 1.9; its documented replacement
    # is Platt scaling through CalibratedClassifierCV with ensemble=False, which fits the same
    # sigmoid on five-fold decision values and then refits the SVC on all rows.
    svc = SVC(kernel="rbf", cache_size=500, random_state=RANDOM_STATE)
    return CalibratedClassifierCV(svc, method="sigmoid", cv=5, ensemble=False)


def _mlp() -> MLPClassifier:
    return MLPClassifier(
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        max_iter=200,
        random_state=RANDOM_STATE,
    )


MODEL_ZOO: dict[str, ModelSpec] = {
    "logreg": ModelSpec(
        name="logreg",
        estimator=_logreg,
        param_grid={
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__class_weight": [None, "balanced"],
        },
        preprocess_overrides={"scaler": "standard"},
        description=(
            "Logistic regression with an L2 penalty on standardised, one-hot encoded inputs; "
            "C controls the regularisation strength and class_weight='balanced' reweights the "
            "minority class instead of resampling."
        ),
        taught_in_week=4,
    ),
    "knn": ModelSpec(
        name="knn",
        estimator=_knn,
        param_grid={
            "clf__n_neighbors": [11, 25, 51],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [2],
        },
        preprocess_overrides={"scaler": "standard"},
        sample_cap=KNN_SAMPLE_CAP,
        description=(
            "k-nearest neighbours on StandardScaler output with Euclidean distance (p=2); an odd "
            "number of neighbours avoids ties and distance weighting lets close neighbours count "
            "more. Fitted on a 40,000-row stratified subsample because scoring cost grows with "
            "rows scored times rows stored."
        ),
        taught_in_week=5,
    ),
    "gaussian_nb": ModelSpec(
        name="gaussian_nb",
        estimator=_gaussian_nb,
        param_grid={"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1, 1.0]},
        preprocess_overrides={"scaler": "standard"},
        description=(
            "Gaussian naive Bayes assuming conditionally independent features; var_smoothing adds "
            "a share of the largest variance to every feature so that near-constant one-hot "
            "columns do not dominate the likelihood."
        ),
        taught_in_week=5,
    ),
    "decision_tree": ModelSpec(
        name="decision_tree",
        estimator=_decision_tree,
        param_grid={
            "clf__max_depth": [4, 8, 12, None],
            "clf__min_samples_leaf": [5, 20, 50],
            "clf__criterion": ["gini", "entropy"],
        },
        preprocess_overrides={"scaler": "none"},
        description=(
            "Single CART decision tree; max_depth and min_samples_leaf are pre-pruning controls "
            "against overfitting and the criterion compares Gini impurity with entropy. No scaler "
            "is needed because splits are invariant to monotone rescaling."
        ),
        taught_in_week=6,
    ),
    "random_forest": ModelSpec(
        name="random_forest",
        estimator=_random_forest,
        param_grid={
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [8, 12, None],
            "clf__min_samples_leaf": [5, 20],
            "clf__max_features": ["sqrt", "log2"],
        },
        search="random",
        n_iter=4,
        preprocess_overrides={"scaler": "none"},
        description=(
            "Bagged ensemble of decorrelated trees (bootstrap rows, random feature subsets at each "
            "split) whose averaged votes reduce the variance of a single tree; RandomizedSearchCV "
            "samples four points from the 24-point grid because each forest is costly to fit."
        ),
        taught_in_week=6,
    ),
    "hist_gb": ModelSpec(
        name="hist_gb",
        estimator=_hist_gb,
        param_grid={
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_iter": [200, 400],
            "clf__max_leaf_nodes": [15, 31],
            "clf__l2_regularization": [0.0, 1.0],
            "clf__early_stopping": [True],
        },
        preprocess_overrides={"scaler": "none"},
        description=(
            "Histogram-based gradient boosting: shallow trees fitted sequentially to the gradient "
            "of the log loss, with early stopping on a 10 % validation split; the learning rate "
            "and tree size trade bias against variance."
        ),
        taught_in_week=6,
    ),
    "svc_rbf": ModelSpec(
        name="svc_rbf",
        estimator=_svc_rbf,
        # Four candidates: one CV fit (6 inner SVC fits plus scoring 8,000 training rows)
        # takes about 130 s, so 20 fits over 8 workers stay near 7 minutes.
        param_grid={
            "clf__estimator__C": [1.0, 10.0],
            "clf__estimator__gamma": ["scale", 0.01],
        },
        preprocess_overrides={"scaler": "standard"},
        sample_cap=SVC_SAMPLE_CAP,
        quick_sample_cap=SVC_QUICK_SAMPLE_CAP,
        description=(
            "Support vector classifier with an RBF kernel on standardised inputs; C is the margin "
            "penalty and gamma the kernel width. Probabilities come from Platt scaling "
            "(CalibratedClassifierCV, sigmoid) and the model is fitted on a 10,000-row stratified "
            "subsample because training time is quadratic in the number of rows."
        ),
        taught_in_week=5,
    ),
    "mlp": ModelSpec(
        name="mlp",
        estimator=_mlp,
        param_grid={
            "clf__hidden_layer_sizes": [(64,), (128,), (128, 64)],
            "clf__alpha": [1e-4, 1e-2, 1e-1],
        },
        preprocess_overrides={"scaler": "standard"},
        description=(
            "Multi-layer perceptron (ReLU hidden layers, Adam optimiser) with early stopping on a "
            "10 % validation split; alpha is the L2 weight penalty and the hidden layer sizes set "
            "the capacity of the network."
        ),
        taught_in_week=10,
    ),
}


def build_pipeline(spec: ModelSpec, preprocessor: BaseEstimator) -> Pipeline:
    """Chain the fitted-inside-CV preprocessor with a fresh estimator from ``spec``.

    Keeping the preprocessor inside the Pipeline is what guarantees that imputers, scalers
    and encoders are fitted on the training folds only.
    """
    return Pipeline([(PREPROCESSOR_STEP, preprocessor), (ESTIMATOR_STEP, spec.estimator())])


def quick_grid(spec: ModelSpec) -> list[dict[str, list[Any]]]:
    """Shrink a grid to its first and last parameter combinations (at most two candidates).

    Used by the tests and by the app's live-training page, where a full search would be too
    slow. The two points are the first and last elements of ``ParameterGrid`` iteration, so
    every value still comes from the original grid.
    """
    points = list(ParameterGrid(spec.param_grid))
    chosen = [points[0]] if len(points) == 1 else [points[0], points[-1]]
    return [{key: [value] for key, value in point.items()} for point in chosen]


def get_preprocessor(pipeline: Pipeline) -> BaseEstimator:
    """Return the preprocessing step of a pipeline built by ``build_pipeline``."""
    return pipeline.named_steps[PREPROCESSOR_STEP]


def get_estimator(pipeline: Pipeline) -> BaseEstimator:
    """Return the final estimator of a pipeline built by ``build_pipeline``."""
    return pipeline.named_steps[ESTIMATOR_STEP]


def default_preprocessor(spec: ModelSpec) -> BaseEstimator:
    """Build the preprocessor the model prefers, importing ``mlda.preprocessing`` lazily.

    The lazy import keeps ``mlda.models`` importable in tests that supply their own
    ColumnTransformer and do not have the feature module available.
    """
    from mlda.preprocessing import build_preprocessor

    return build_preprocessor(**spec.preprocess_overrides)


def grid_summary() -> dict[str, int]:
    """Number of candidates per model in the full search (reported in the write-up)."""
    return {name: spec.n_candidates for name, spec in MODEL_ZOO.items()}
