from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

def _create_pipeline_ridge(poly_degree=2, ridge_alpha=0.5):
    """
    Helper function to create a standard Ridge regression pipeline.
    """
    return make_pipeline(
        PolynomialFeatures(poly_degree),
        Ridge(alpha=ridge_alpha)
    )

def _create_pipeline_rf(poly_degree=2):
    """
    Helper function to create a Random Forest regression pipeline.
    """
    return make_pipeline(
        PolynomialFeatures(poly_degree),
        RandomForestRegressor()
    )

def _create_pipeline_nn(max_iter=200):
    """
    Helper function to create a Neural Network regression pipeline.
    """
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(max_iter = max_iter)
    )

def _build_bee_model(bee, min_history_size, max_history_size, regressor_type, x, fg_x, pipeline_for, model_for):
    """
    Generic function to build an ML model for a single bee.

    Parameters:
        :param Bee bee                 : the Bee class instance
        :param int min_history_size    : minimum history size to build the model
        :param int max_history_size    : maximum history size to build the model
        :param str regressor_type      : type of regressor to use for the model
        :param str x                   : key to access data in bee.memory
        :param str fg_x                : key to access target data in bee.memory
        :param str pipeline_for        : attribute name where pipeline will be stored
        :param str model_for           : attribute name where model will be stored
    """
    fgx_data = bee.memory[fg_x]
    x_data = bee.memory[x]

    if not fgx_data or len(fgx_data) < min_history_size:
        setattr(bee, model_for, None)
        return

    history_size = min(len(fgx_data), max_history_size)
    relevant_data_fgx = fgx_data[-history_size:]
    relevant_data_x = x_data[-history_size:]

    if hasattr(bee, pipeline_for):
        pipeline = getattr(bee, pipeline_for)
    else:
        if regressor_type == "ridge":
            pipeline = _create_pipeline_ridge()
        elif regressor_type == "rf":
            pipeline = _create_pipeline_rf()
        elif regressor_type == "nn":
            pipeline = _create_pipeline_nn()
        else:
            raise ValueError("No supported regressor type found")
        setattr(bee, pipeline_for, pipeline)

    try:
        fit_model = pipeline.fit(relevant_data_x, relevant_data_fgx)
        setattr(bee, model_for, fit_model)

    except Exception as e:
        print(f"Warning: Error fitting model for bee (fg_x={fg_x}): {e}")
        setattr(bee, model_for, None)


def build_ML_model_constraint(bee, min_ML_history_size, max_ML_history_size, regressor_type):
    """
    Build ML model to predict constraint satisfaction and guide search towards feasible regions.

    Parameters:
        :param Bee bee                 : the Bee class instance
        :param int min_ML_history_size : minimum history size to build the model
        :param int max_ML_history_size : maximum history size to build the model
        :param str regressor_type      : type of regressor to use for the model
    """
    _build_bee_model(
        bee,
        min_ML_history_size,
        max_ML_history_size,
        regressor_type,
        "x",
        "g(x)",
        "constr_pipeline",
        "constr_model"
    )

def build_ML_model_target(bee, min_ML_history_size, max_ML_history_size, regressor_type):
    """
    Build ML model to predict objective function satisfaction and guide search towards feasible regions.

    Parameters:
        :param Bee bee                 : the Bee class instance
        :param int min_ML_history_size : minimum history size to build the model
        :param int max_ML_history_size : maximum history size to build the model
        :param str regressor_type      : type of regressor to use for the model
    """
    _build_bee_model(
        bee,
        min_ML_history_size,
        max_ML_history_size,
        regressor_type,
        "x",
        "f(x)",
        "target_pipeline",
        "target_model"
    )

def _build_global_model(beehive, index, regressor_type, x, fg_x, model_for):
    """
    Generic function to build an ML model for a single bee.

    Parameters:
        :param Beehive beehive    : the Beehive class instance
        :param int index          : index of the target bee to assign the model to
        :param str regressor_type : type of regressor to use for the model
        :param str x              : key to access data in bee.memory
        :param str fg_x           : key to access target data in bee.memory
        :param str model_for      : attribute name where model will be stored
    """

    target_bee = beehive.population[index]
    history_x = []
    history_fgx = []
    
    for bee in beehive.population:
        if x in bee.memory and fg_x in bee.memory:
            history_x.extend(bee.memory[x])
            history_fgx.extend(bee.memory[fg_x])

    if not history_fgx or len(history_fgx) < beehive.min_ML_history_size:
        setattr(target_bee, model_for, None)
        return

    history_size = min(len(history_fgx), beehive.max_ML_history_size)

    relevant_fgx_data = history_fgx[-history_size:]
    relevant_x_data = history_x[-history_size:]
    if regressor_type == "ridge":
        pipeline = _create_pipeline_ridge()
    elif regressor_type == "rf":
        pipeline = _create_pipeline_rf()
    elif regressor_type == "nn":
        pipeline = _create_pipeline_nn()
    else:
        raise ValueError("No supported regressor type found")
    
    try:
        fit_model = pipeline.fit(relevant_x_data, relevant_fgx_data)
        setattr(target_bee, model_for, fit_model)

    except Exception as e:
        print(f"Warning: Error fitting model for bee (fg_x={fg_x}): {e}")
        setattr(target_bee, model_for, None)
    

def build_ML_model_constraint_global(beehive, index, regressor_type):
    """
    Builds a global ML model for constraint prediction using data from all bees.

    Parameters:
        :param Beehive beehive    : the Beehive class instance
        :param int index          : index of the target bee to assign the model to
        :param str regressor_type : type of regressor to use for the model
    """
    _build_global_model(
        beehive,
        index,
        regressor_type,
        "x",
        "g(x)",
        "constr_model"
    )

def build_ML_model_target_global(beehive, index, regressor_type):
    """
    Builds a global ML model for target function prediction using data from all bees.

    Parameters:
        :param Beehive beehive    : the Beehive class instance
        :param int index          : index of the target bee to assign the model to
        :param str regressor_type : type of regressor to use for the model
    """
    _build_global_model(
        beehive,
        index,
        regressor_type,
        "x",
        "f(x)",
        "target_model"
    )
