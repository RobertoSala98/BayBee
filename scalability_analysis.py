import sys, os
import argparse
import warnings
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import simulate
import math
import json


# Load your fixed values
with open("discrete_values_rastrigin.json", "r") as f:
    values = json.load(f)


estimated_threshold_values = {
    4: 1.973502,
    8: 1.997782,
    16: 1.999430,
    32: 1.999996
}


optima = {
    4: 6.312566,
    8: 20.143131,
    16: 81.928674,
    32: 292.887380
}


def _evaluator(vectors):
    X = np.asarray(vectors, dtype=float)
    single = (X.ndim == 1)
    X = np.atleast_2d(X)
    n = X.shape[1]
    vals = 10 * n + np.sum(X*X - 10*np.cos(2*np.pi*X), axis=1)
    return vals[0] if single else vals


def _constraints(vectors):
    X = np.asarray(vectors, dtype=float)
    single = (X.ndim == 1)
    X = np.atleast_2d(X)
    vals = 2 - 5 * np.exp(np.sum(X, axis=1))
    return (vals[0] if single else vals) - estimated_threshold_values[X.shape[1]]


def _snap_function(vector, exclude_vectors=None, dimension=None):

    if exclude_vectors is None:
        exclude_vectors = {}

    if dimension is not None:
        
        temp_vector = vector.copy()
        abs_diff = abs(vector[dimension] - values[dimension])

        while True:

            if abs_diff.size <= 0:
                return exclude_vectors['actual']

            min_index = abs_diff.argmin()
            temp_vector[dimension] = values[dimension][min_index]
            
            if not any(np.array_equal(temp_vector, v) for v in exclude_vectors['memory']):
                return temp_vector
            
            else:
                abs_diff = np.delete(abs_diff, min_index)

    else:
        return np.array([values[dim][np.abs(values[dim] - vector[dim]).argmin()] for dim in range(len(vector))])


def run():
    args = parse_args()

    evaluator = _evaluator
    constraints = _constraints
    snap_function = _snap_function

    seeds = [361288763, 977158673, 248565991, 763729313, 736261687, 701927699, 818372591, 946799374, 433368984, 710895776, 220959875, 178717414, 499290437, 710177041, 789831346, 466765865, 664620228, 739167888, 825604850, 477438306, 531556250, 110346033, 334792365, 371288882, 172855517, 628572063, 440863561, 926975100, 288654213, 410630257]

    # Machine Learning treatment
    ML_approach_constr = args.ML_approach_constr
    ML_approach_target = args.ML_approach_target
    BO_approach = args.Bayesian_Optimization_approach
    BO_steps = args.Bayesian_Optimization_steps
    min_hist = args.min_hist
    max_hist = args.max_hist

    # automatically enable cache when using local memory
    args.cache = (args.memory == 'local')

    for ndim in [4, 8, 16, 32]:
        title = args.title + f"/{ndim}"

        lower_bound = np.array([-5.12] * ndim)
        upper_bound = np.array([5.12] * ndim)

        global_optima = optima[ndim]

        simulate.bee_algorithm(title=title,
                            num_runs=args.runs,
                            num_bees=args.bees,
                            max_itrs=args.max_itrs,
                            max_trials=args.max_trials,
                            global_optima=global_optima,
                            lower=lower_bound,
                            upper=upper_bound,
                            evaluator=evaluator,
                            constraints=constraints,
                            ML_approach_constr=ML_approach_constr,
                            ML_approach_target=ML_approach_target,
                            constr_regressor_type=args.constr_regressor_type,
                            target_regressor_type=args.target_regressor_type,
                            BO_approach=BO_approach,
                            min_hist=min_hist,
                            max_hist=max_hist,
                            snap_function=snap_function,
                            visualize_behaviour=args.visualize_behaviour,
                            BO_steps=BO_steps,
                            async_op=args.async_op,
                            std_ABC=args.standard_ABC,
                            seeds=seeds,
                            memory_type=args.memory,
                            cache=args.cache,
                            levy_step_size=args.levy_step_size,
                            resume=args.resume,
                            discrete_values=values[:ndim]
                            )

def parse_args():
    """
    Parses command-line arguments provided by the user.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-ln', '--linspace', action="store_true",
                        help='Will you be using linspace? If yes, specify the values in the config.json file')
    parser.add_argument('-t', '--title', type=str, metavar='', default="ABeeC",
                        help='What will be the name of the output folder the output data of this run will be saved to')
    parser.add_argument('-r', '--runs', type=int, metavar='', default=1,
                        help='How many times do you want to run the ABeeC algorithm')
    parser.add_argument('-b', '--bees', type=int, metavar='', default=30,
                        help='How many bees do you want to generate')
    parser.add_argument('-mi', '--max_itrs', type=int, metavar='', default=100,
                        help='With how many iterations do you want to run the algorithm')
    parser.add_argument('-mt', '--max_trials', type=int, metavar='', default=3,
                        help='How many times do you want to try to improve the solution before abandoning it')
    parser.add_argument('-levy', '--levy_step_size', type=float, metavar='', default=0.1,
                        help='What should be the Levy step size yuo want to consider')
    parser.add_argument('-c', '--constraint', action="store_true",
                        help='Do you want to add and consider any constraints?')
    parser.add_argument('-mlc', '--ML_approach_constr', type=str, metavar='', default=None,
                        help='Do you want to use Machine Learning for the function constraints? If yes, select either local or global. If no, then None')
    parser.add_argument('-mlt', '--ML_approach_target', type=str, metavar='', default=None,
                        help='Do you want to use Machine Learning for the target functions? If yes, select either local or global. If no, then None')
    parser.add_argument('-regc', '--constr_regressor_type', type=str, metavar='', default="ridge",
                        help='What regressor do you want to use for the constraint ML models? Default: ridge. Available: random forest (rr) and neural network (nn)')
    parser.add_argument('-regt', '--target_regressor_type', type=str, metavar='', default="ridge",
                        help='What regressor do you want to use for the target ML models? Default: ridge. Available: random forest (rr) and neural network (nn)')
    parser.add_argument('-BO', '--Bayesian_Optimization_approach', action='store_true',
                        help='Do you want to use Bayesian Optimization?')
    parser.add_argument('-BOsteps', '--Bayesian_Optimization_steps', type=int, metavar='', default=0,
                        help='How many steps do you want for BO?')
    parser.add_argument('-min_hist', '--min_hist', type=int, metavar='', default=10,
                        help='How many minimum samples do you want to use to train the ML models?')
    parser.add_argument('-max_hist', '--max_hist', type=int, metavar='', default=1000,
                        help='How many maximum samples do you want to use to train the ML models?')
    parser.add_argument('-vb', '--visualize_behaviour', action="store_true",
                        help='Do you want to visualize bees behaviour?')
    parser.add_argument('-a', '--async_op', action='store_true',
                        help='Run asynchronous event-driven simulation')
    parser.add_argument('-stdABC', '--standard_ABC', action="store_true",
                        help='Do you want to use the standard ABC algorithm?')
    parser.add_argument('-cf', '--config_file', type=str, metavar='', default="config.json",
                        help='Path to the config file to consider')
    parser.add_argument('-mem', '--memory', type=str, metavar='', default='local',
                        help='Do you want to use a local or a global memory for each bee? (cache will be enabled only in local mode)')
    parser.add_argument('-res', '--resume', action="store_true",
                        help='Do you want to resume the simulation from logs?')
    try:
        args = parser.parse_args()
        return args
    except argparse.ArgumentError:
        parser.print_help()
    exit()


if __name__ == "__main__":
    run()