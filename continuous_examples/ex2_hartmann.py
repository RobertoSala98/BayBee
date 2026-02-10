import sys, os
import argparse
import warnings
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import simulate
import math


values = [[0.05053739836308857, 0.10333867740437064, 0.10728094041940994, 0.18204674642045227, 0.18645228833623484, 0.19871255073375094, 0.22337637366974217, 0.23451999531366796, 0.236412873068787, 0.23901401046404425, 0.29804472388524705, 0.35991236906553015, 0.40430540130663384, 0.4513982239294564, 0.4910558824540314, 0.5294076866049742, 0.5897462678513798, 0.6055832839910167, 0.611214143331521, 0.6222085756919103, 0.659850539254682, 0.6792755292795539, 0.7940777724107138, 0.8521091469524942, 0.8575552800550001, 0.8609757895513726, 0.8610726159335803, 0.8662541700154157, 0.9495294126880695, 0.9941590465133928], 
          [6.071471734592215e-06, 0.030024600422091097, 0.04459722110277098, 0.1428629152781522, 0.1528340326130968, 0.15979291611468138, 0.1643802741512519, 0.18346744091019662, 0.21690112365925995, 0.29333298229462246, 0.3796152949767201, 0.39712473378343205, 0.41468106664214877, 0.42546466739169087, 0.48206145976353487, 0.5487295554584174, 0.5804880331876636, 0.6133121654173096, 0.6849695219935442, 0.7412193552152007, 0.7503949001299803, 0.8035685623034213, 0.8194490831760519, 0.8535159876195894, 0.8923802802324431, 0.9246099381734738, 0.9559970089800331, 0.9700590604046107, 0.9859373396099906, 0.9959430015765135], 
          [0.004478943060226559, 0.01634017814475419, 0.023232082238245022, 0.10705592829066202, 0.11537597848162695, 0.12169657747825402, 0.13515886023050583, 0.14181886785198328, 0.20824240924945414, 0.21451148377059392, 0.2196094804890134, 0.28555874997484, 0.2975154121834157, 0.3594278508281451, 0.4433208364450053, 0.4731569832323975, 0.551007860839997, 0.57125099764655, 0.670523042930109, 0.6954811400011713, 0.7266873887629604, 0.7685693663859626, 0.7780521961027121, 0.8095801989306989, 0.8207939150993205, 0.8238065750876028, 0.8645223427580611, 0.8690861633695999, 0.8941414702554904, 0.9904622683606165], 
          [0.0020425796122335305, 0.029389443447512265, 0.034236133282891545, 0.12739637540742188, 0.15624705041367637, 0.2110490241069729, 0.228320277234783, 0.3320417772212887, 0.3806985740193176, 0.4121065698573486, 0.4759053176965884, 0.5300417119273639, 0.531638444756365, 0.6227996849420024, 0.6389238495263752, 0.6943962626394442, 0.7214221765131326, 0.7540003630136686, 0.7819538984429113, 0.7898085134102034, 0.7935562085742808, 0.8165060353455081, 0.8219619400736126, 0.8703899841992605, 0.9189929976524063, 0.9346288946569021, 0.9566861556705828, 0.9637833862468366, 0.9914750126366936, 0.9948796764604972], 
          [0.055139980675629974, 0.05561850416754743, 0.1096156364914399, 0.2053396266816555, 0.2334802233783727, 0.2339702119057031, 0.27453250369624616, 0.31009168347263294, 0.3622545557705451, 0.40176363071523535, 0.4177060560418645, 0.45761218169438045, 0.4605029429851417, 0.47400549150664184, 0.4882502067530168, 0.5006555052686719, 0.5731526799564297, 0.5764776357123829, 0.5864095472346662, 0.6221697427513527, 0.6754591510934168, 0.6848043529841172, 0.7118832944891249, 0.750145432176584, 0.7973281944556664, 0.8184133239805863, 0.8575173554965344, 0.8742696783031395, 0.9164272866459805, 0.9288386435792234], 
          [0.02788974717338244, 0.06579761161321218, 0.17242433482617292, 0.2074841028152975, 0.271668876254335, 0.32223366663752395, 0.33150040275518555, 0.34733185069741834, 0.43888430217766883, 0.45204802564470536, 0.5300875611784167, 0.5624061118757375, 0.5803823944345422, 0.6004654820199715, 0.6142331257937306, 0.6401066360606366, 0.6921037135674261, 0.7117728584075088, 0.7171387049139946, 0.7522456026969083, 0.7603341076882156, 0.8020348929507494, 0.8112056705999194, 0.8571307338311696, 0.8883453261060202, 0.8915875408873446, 0.9172772998650482, 0.9202087996070702, 0.9368908323161482, 0.9923256929229647]]
ndim = 6
global_optima = -2.94765433808579


def _evaluator(vectors):
    """
    An n-dimensional Hartmann function is defined as:
    f(x) = -sum_{i=1}^4 {alpha_i * exp( -sum_{j=1}^6 {A_{ij} * (x_j - P_{ij})^2} )}
    where  0 < x_i < 1

    The global minima of the function being f(x) = -3.32237 at (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)
    """
    X = np.asarray(vectors, dtype=float)
    single = (X.ndim == 1)
    X = np.atleast_2d(X)

    if X.shape[1] != 6:
        raise ValueError(f"Hartmann6 expects 6D inputs, got {X.shape[1]}")

    alpha = np.array([1.0, 1.2, 3.0, 3.2]) 
    A = np.array([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14]
    ], dtype=float)
    P = 1.0e-4 * np.array([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381]
    ], dtype=float)

    d = X[:, None, :] - P[None, :, :]
    inner = -np.sum(A[None, :, :] * d * d, axis=2)
    external = np.sum(alpha[None, :] * np.exp(inner), axis=1)

    results = -(external + 2.58) / 1.94
    return results[0] if single else results


def _constraints(vectors):
    X = np.asarray(vectors, dtype=float)
    single = (X.ndim == 1)
    X = np.atleast_2d(X)

    if X.shape[1] < 5:
        raise ValueError(f"Constraint expects at least 5 dims, got {X.shape[1]}")

    num = 4 * np.power(X[:, 0]**2 + X[:, 3]**3, 1/8)
    den = (X[:, 1] * X[:, 2]) - np.sinh(X[:, 4])
    vals = num / den - 1

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

    lower_bound = np.array([0.0] * ndim)
    upper_bound = np.array([1.0] * ndim)

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