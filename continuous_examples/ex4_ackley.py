import sys, os
import argparse
import warnings
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import simulate
import math


values = [[-30.749339264117793, -29.977398022268275, -22.223361791071024, -19.478770519169686, -12.808621455282367, -12.665191537643185, -9.757800421235604, -7.113944168749658, 4.929363179915541, 6.190570898870433, 12.681785766537587, 15.463468590135818, 18.735024323834025, 20.790066023759223, 21.588892163509385], 
          [-29.467433090841887, -25.986557087030384, -21.057301055465643, -15.792893814544296, -15.264209647136312, -15.0714123225796, -12.098022679682597, -8.971039301889068, -8.64969476164611, 5.181300237500949, 7.3467978478267995, 9.734974803972797, 15.191547977767016, 23.44858509312239, 27.547476191659285], 
          [-28.322716541283576, -27.544122407676348, -25.153548729763404, -22.867653367964557, -14.352718925281216, -12.892241534224812, -6.992103583235142, -1.758578882129978, 5.703424803962292, 12.597540824204444, 13.719861560818686, 15.968499180287303, 16.270413760483194, 22.85359177267842, 28.59541989373809], 
          [-31.117689799560402, -27.16677679141081, -21.38728184452902, -4.283757968595005, -3.3897267504634883, 2.1627514752108397, 3.231691461422905, 5.491946581498304, 7.543146298946624, 10.590899064384061, 12.08822876669764, 17.129594009056923, 24.331878016472757, 31.055720510050534, 32.63786627082466], 
          [-32.1920088321971, -27.719314284986293, -26.211438778999945, -25.3315836915392, -18.487510622403097, -17.201539219047955, -14.762120421437395, -9.748534730954162, -6.761081877544406, 10.664373630929155, 16.988493633541026, 17.92569984005813, 24.901654603952174, 26.154587979476396, 29.542194208426174], 
          [-32.51275557869885, -22.821512744244007, -19.535028592147256, -19.34587046513043, -10.639601087253634, -9.065427931726283, -8.90894607418462, -6.562883507552279, -4.417755600028105, 16.163356651400015, 16.973426844522336, 18.10520451744827, 18.27398982674245, 20.563975433433036, 25.4398049823049], 
          [-32.11337505850667, -32.02241443877436, -25.88387570070126, -25.491475534404323, -11.249159162875433, -10.515329437319227, -8.395478152977915, -7.33423729562432, 4.336064492156211, 8.98819127355494, 12.291731802511087, 13.697652144930515, 14.837726286565207, 21.020641433594093, 24.003624630943868], 
          [-31.65546574989186, -25.57653673682635, -24.762936101426398, -24.222019280345748, -20.66239617465625, -18.005604310762422, -12.711720490587979, 5.453876827713977, 6.937919810996938, 12.614247592497918, 13.054768903764078, 18.53904667193897, 26.759713733833898, 27.4071110827782, 31.25165767139218]]
ndim = 8
global_optima = 13.867005853882118


def _evaluator(vectors):
    """
    An n-dimensional Ackley function is defined as:
    f(x) = -a * exp(-b * sqrt{(1/n) * sum_{i=1}^n {x_i^2} }) - exp((1/n) * sum_{i=1}^n {cos(c * x_i)}) + a + exp(1)
    recommended values are a = 20, b = 0.2, and c = 2*PI
    where  -32.768 <= x_i <= 32.768

    The global minima of the function being f(x) = 0 at all x_i = 0
    """
    X = np.asarray(vectors, dtype=float)
    single = (X.ndim == 1)
    X = np.atleast_2d(X)

    a = 20.0
    b = 0.2
    c = 2 * np.pi
    n = X.shape[1]

    sum_sq = np.sum(X**2, axis=1)
    sum_cos = np.sum(np.cos(c * X), axis=1)

    part1 = -a * np.exp(-b * np.sqrt(sum_sq / n))
    part2 = -np.exp(sum_cos / n)
    vals = part1 + part2 + a + np.exp(1.0)

    return vals[0] if single else vals


def _constraints(vectors):
    X = np.asarray(vectors, dtype=float)
    single = (X.ndim == 1)
    X = np.atleast_2d(X)

    if X.shape[1] < 8:
        raise ValueError(f"Constraint expects at least 8 dims, got {X.shape[1]}")

    vals = ((X[:, 0] - 3*X[:, 3])**3
            - 2*np.sqrt(X[:, 7]**2 + np.exp(X[:, 6]))
            + 5*X[:, 4]*X[:, 5]
            + np.sin(X[:, 1]**2 + np.cos(X[:, 2]))
            - 2)

    return vals[0] if single else vals


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

    lower_bound = np.array([-32.768] * ndim)
    upper_bound = np.array([32.768] * ndim)

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
                           snap_function=snap_function,
                           visualize_behaviour=args.visualize_behaviour,
                           BO_steps=BO_steps,
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