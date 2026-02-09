import argparse
import warnings
import numpy as np

from src import simulate
from src.utils import config as cnfg, dataset


def run():
    args = parse_args()
    config = cnfg.load_config(args.config_file)
    global_optima = None
    evaluator = None
    constraints = None

    # automatically enable cache when using local memory
    args.cache = (args.memory == 'local')

    # if the user wants to work with a dataset
    if args.dataset:
        dataset_path = config["dataset"]["path"]
        # work with dataset from the given file
        dataset_obj = dataset.Dataset(dataset_path, args.config_file)
        lower_bound, upper_bound = dataset_obj.get_bounds()
        y_data = dataset_obj.get_y_column()

        # if the dataset has no objective function values, then the evaluator function should be provided
        if y_data is not None:
            evaluator = dataset_obj.evaluator
            global_optima = dataset_obj.get_global_optima()
        elif "evaluator" not in config or "global_optimum" not in config:
            warnings.warn("No objective function values found in the dataset! Please provide a 'config.json' file with an evaluator function and global optimum.")
        else:
            evaluator = eval(config["evaluator"])
            global_optima = config["global_optimum"]

        # if constraint flag is used, then the constraints should be extracted from the dataset
        if args.constraint:
            constraints = dataset_obj.get_constraint
        else:
            constraints = None

    # if the user wants to work with linspace
    elif args.linspace:
        values = config["linspace"]
        n_dim = config["n_dimension"]
        start_value, stop_value, samples_num = values

        values = np.linspace(start_value, stop_value, samples_num)

        lower_bound = np.array([values[0]] * n_dim)
        upper_bound = np.array([values[-1]] * n_dim)
        global_optima = np.min(values)

        func_evaluator = eval(config["evaluator"])
        evaluator = lambda snap_vector: func_evaluator(np.array([values[np.abs(values - x).argmin()] for x in snap_vector]))

    else:
        raise ValueError("No dataset or linspace values provided. Please specify either in the 'config.json' file!")

    # Machine Learning treatment
    ML_approach_constr = args.ML_approach_constr
    ML_approach_target = args.ML_approach_target
    BO_approach = args.Bayesian_Optimization_approach
    min_hist = args.min_hist
    max_hist = args.max_hist

    # If standard ABC is requested, override all non-standard features
    if args.standard_ABC:
        print("\nYou have selected to run the standard ABC algorithm. All non-standard features will be disabled.")
        ML_approach_constr = None
        ML_approach_target = None
        BO_approach = False
        min_hist = None
        max_hist = None
        args.memory = None
        args.cache = False
        args.levy_step_size = None
        args.visualize_behaviour = False
        args.async_op = False

    # run the bee optimization algorithm using the extracted information
    if len(config["seeds"]) != args.runs:
        seeds = [np.random.randint(1,1e9) for _ in range(args.runs)]
    else:
        seeds = config["seeds"]

    simulate.bee_algorithm(title=args.title,
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
                           regressor_type=args.regressor_type,
                           BO_approach=BO_approach,
                           min_hist=min_hist,
                           max_hist=max_hist,
                           dataset_obj=dataset_obj,
                           visualize_behaviour=args.visualize_behaviour,
                           BO_steps=config["BO_steps"],
                           async_op=args.async_op,
                           std_ABC=args.standard_ABC,
                           seeds=seeds,
                           memory_type=args.memory,
                           cache=args.cache,
                           levy_step_size=args.levy_step_size,
                           resume=args.resume
                        )


def parse_args():
    """
    Parses command-line arguments provided by the user.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', action="store_true",
                        help='Will you be using a dataset? If yes, specify the path to the dataset file in the config.json file')
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
    parser.add_argument('-levy', '--levy_step_size', type=int, metavar='', default=0.1,
                        help='What should be the Levy step size yuo want to consider')
    parser.add_argument('-c', '--constraint', action="store_true",
                        help='Do you want to add and consider any constraints?')
    parser.add_argument('-mlc', '--ML_approach_constr', type=str, metavar='', default=None,
                        help='Do you want to use Machine Learning for the function constraints? If yes, select either local or global. If no, then None')
    parser.add_argument('-mlt', '--ML_approach_target', type=str, metavar='', default=None,
                        help='Do you want to use Machine Learning for the target functions? If yes, select either local or global. If no, then None')
    parser.add_argument('-reg', '--regressor_type', type=str, metavar='', default="ridge",
                        help='What regressor do you want to use for the ML models? Default: ridge. Available: random forest (rr) and neural network (nn)')
    parser.add_argument('-BO', '--Bayesian_Optimization_approach', action='store_true',
                        help='Do you want to use Bayesian Optimization?')
    parser.add_argument('-min_hist', '--min_hist', type=int, metavar='', default=10,
                        help='How many minimum samples do you want to use to train the ML models?')
    parser.add_argument('-max_hist', '--max_hist', type=int, metavar='', default=1000,
                        help='How many maximum samples do you want to use to train the ML models?')
    parser.add_argument('-v', '--visualize_behaviour', action="store_true",
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
