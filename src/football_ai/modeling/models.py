from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Numeric and categorical feature sets used across models
# ---------------------------------------------------------------------------

XG_NUMERIC_FEATURES = ["x", "y", "shot_distance", "minute", "second"]
XG_CATEGORICAL_FEATURES = ["shot_body_part", "shot_technique", "shot_type"]

OUTCOME_NUMERIC_FEATURES = [
    "possession_pct",
    "passes_per_min",
    "pressures_per_min",
    "avg_shot_distance",
    "total_xg",
]


def _standard_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    transformers = [("num", StandardScaler(), numeric_features)]
    if categorical_features:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        )
    return ColumnTransformer(transformers=transformers)


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def build_xg_model(random_state: int = 42) -> Pipeline:
    """
    Gradient Boosting xG model pipeline (numeric + categorical features).
    """
    preprocessor = _standard_preprocessor(XG_NUMERIC_FEATURES, XG_CATEGORICAL_FEATURES)
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", GradientBoostingClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=random_state,
            )),
        ]
    )


def build_outcome_model_rf(random_state: int = 42) -> Pipeline:
    """
    Random Forest match outcome model pipeline.
    """
    preprocessor = _standard_preprocessor(OUTCOME_NUMERIC_FEATURES, [])
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                n_jobs=-1,
                random_state=random_state,
            )),
        ]
    )


def build_outcome_model_lr(random_state: int = 42) -> Pipeline:
    """
    Logistic Regression match outcome model (interpretable baseline).
    """
    preprocessor = _standard_preprocessor(OUTCOME_NUMERIC_FEATURES, [])
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", LogisticRegression(
                multi_class="multinomial",
                solver="lbfgs",
                max_iter=500,
                random_state=random_state,
            )),
        ]
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, callable] = {
    "xg_gbm": build_xg_model,
    "outcome_rf": build_outcome_model_rf,
    "outcome_lr": build_outcome_model_lr,
}


def get_model(name: str, **kwargs) -> Pipeline:
    """
    Retrieve a model pipeline by name from the registry.

    Available names: 'xg_gbm', 'outcome_rf', 'outcome_lr'
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
