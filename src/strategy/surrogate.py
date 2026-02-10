import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


def _create_bee_ml(model_name, poly_degree=2, random_state=None, **model_kwargs):
    """
    Create a regression model with the same 'available models + standard params'
    as your ML class: ridge, rf, xgb, nn; optional PolynomialFeatures.
    """
    name = str(model_name).lower()

    if random_state is not None and "random_state" not in model_kwargs:
        model_kwargs["random_state"] = random_state

    if name == "ridge":
        base = Ridge(**model_kwargs)
    elif name == "rf":
        base = RandomForestRegressor(**model_kwargs)
    elif name == "xgb":
        if XGBRegressor is None:
            raise ImportError("xgboost is not installed, cannot use model_name='xgb'")
        base = XGBRegressor(**model_kwargs)
    elif name == "nn":
        base = MLPRegressor(**model_kwargs)
    else:
        raise ValueError("model_name must be one of: ridge, rf, xgb, nn")

    if poly_degree is not None and poly_degree > 1:
        return Pipeline([
            ("poly", PolynomialFeatures(degree=poly_degree, include_bias=True)),
            ("est", base),
        ])

    return base


def _build_bee_model(
    bee,
    min_history_size,
    max_history_size,
    model_name,
    x_key,
    y_key,
    pipeline_for,
    model_for,
    poly_degree=2,
    random_state=None,
    model_kwargs=None,
):
    model_kwargs = model_kwargs or {}

    y_data = bee.memory.get(y_key, [])
    x_data = bee.memory.get(x_key, [])

    # Need enough history and aligned X/y
    n = min(len(y_data), len(x_data))
    if n < min_history_size:
        setattr(bee, model_for, None)
        return

    history_size = min(n, max_history_size)
    relevant_y = np.asarray(y_data[-history_size:]).ravel()
    relevant_X = np.asarray(x_data[-history_size:])

    # Reuse the same estimator object if already created
    if hasattr(bee, pipeline_for) and getattr(bee, pipeline_for) is not None:
        est = getattr(bee, pipeline_for)
    else:
        est = _create_bee_ml(
            model_name=model_name,
            poly_degree=poly_degree,
            random_state=random_state,
            **model_kwargs,
        )
        setattr(bee, pipeline_for, est)

    try:
        est.fit(relevant_X, relevant_y)
        setattr(bee, model_for, est)  # fitted estimator (same object)
    except Exception as e:
        print(f"Warning: Error fitting model for bee (y_key={y_key}): {e}")
        setattr(bee, model_for, None)


def build_ML_model_constraint(bee, min_ML_history_size, max_ML_history_size, model_name,
                             poly_degree=2, random_state=None, model_kwargs=None):
    _build_bee_model(
        bee=bee,
        min_history_size=min_ML_history_size,
        max_history_size=max_ML_history_size,
        model_name=model_name,
        x_key="x",
        y_key="g(x)",
        pipeline_for="constr_pipeline",
        model_for="constr_model",
        poly_degree=poly_degree,
        random_state=random_state,
        model_kwargs=model_kwargs,
    )


def build_ML_model_target(bee, min_ML_history_size, max_ML_history_size, model_name,
                          poly_degree=2, random_state=None, model_kwargs=None):
    _build_bee_model(
        bee=bee,
        min_history_size=min_ML_history_size,
        max_history_size=max_ML_history_size,
        model_name=model_name,
        x_key="x",
        y_key="f(x)",
        pipeline_for="target_pipeline",
        model_for="target_model",
        poly_degree=poly_degree,
        random_state=random_state,
        model_kwargs=model_kwargs,
    )


def _build_global_model(beehive, index, model_name, x_key, y_key, model_for,
                        poly_degree=2, random_state=None, model_kwargs=None):
    model_kwargs = model_kwargs or {}

    target_bee = beehive.population[index]
    history_X, history_y = [], []

    for bee in beehive.population:
        if x_key in bee.memory and y_key in bee.memory:
            xs = bee.memory[x_key]
            ys = bee.memory[y_key]
            n = min(len(xs), len(ys))
            if n > 0:
                history_X.extend(xs[:n])
                history_y.extend(ys[:n])

    if len(history_y) < beehive.min_ML_history_size:
        setattr(target_bee, model_for, None)
        return

    history_size = min(len(history_y), beehive.max_ML_history_size)
    relevant_X = np.asarray(history_X[-history_size:])
    relevant_y = np.asarray(history_y[-history_size:]).ravel()

    est = _create_bee_ml(
        model_name=model_name,
        poly_degree=poly_degree,
        random_state=random_state,
        **model_kwargs,
    )

    try:
        est.fit(relevant_X, relevant_y)
        setattr(target_bee, model_for, est)
    except Exception as e:
        print(f"Warning: Error fitting global model (y_key={y_key}): {e}")
        setattr(target_bee, model_for, None)


def build_ML_model_constraint_global(beehive, index, model_name,
                                    poly_degree=2, random_state=None, model_kwargs=None):
    _build_global_model(
        beehive=beehive,
        index=index,
        model_name=model_name,
        x_key="x",
        y_key="g(x)",
        model_for="constr_model",
        poly_degree=poly_degree,
        random_state=random_state,
        model_kwargs=model_kwargs,
    )


def build_ML_model_target_global(beehive, index, model_name,
                                poly_degree=2, random_state=None, model_kwargs=None):
    _build_global_model(
        beehive=beehive,
        index=index,
        model_name=model_name,
        x_key="x",
        y_key="f(x)",
        model_for="target_model",
        poly_degree=poly_degree,
        random_state=random_state,
        model_kwargs=model_kwargs,
    )
