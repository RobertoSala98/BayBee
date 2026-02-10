import sys, os
import argparse
import warnings
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import simulate
import math


values = [[-412.45588524172706, -336.3323515201625, -334.9074401818383, -257.5916559124198, -203.6819988621823, -75.08771159359952, -21.443931907456772, -8.39491084872401, 9.221521150793421, 11.407938860456113, 49.23386599268849, 70.61359802605125, 108.23858808053149, 159.68972535913736, 170.07660929457336, 206.50141156853306, 221.47483535600543, 269.01058778013817, 278.4105272099896, 358.83758246934974], 
          [-363.24854767071565, -297.6281685587142, -295.82360511630424, -277.8094141567889, -231.94544367219493, -229.33667270901958, -222.46219825220436, -184.40303367564138, -144.53949521894538, -51.20643227507702, 37.891341531814305, 111.94872026375356, 166.00961065022148, 247.19410399053072, 264.8291887788099, 295.60244501684133, 383.6985862862705, 424.435034944706, 427.10590113532135, 490.1556405091734], 
          [-471.78226549613447, -448.95208491145956, -440.94424522990283, -428.61321971224686, -427.7987843882121, -388.4442569171488, -143.66334573179608, -135.09092174636936, -115.43664607135815, -56.66105480594808, -2.171310720879603, 40.17477603601537, 85.97359994612032, 89.94752030742086, 192.81131005743453, 216.1791326203063, 240.75392858180828, 264.92105489667097, 343.03898026701404, 343.5697221853228], 
          [-475.8791049426883, -449.82785077416645, -423.8021424267606, -112.16223333295807, -59.99671727560974, -29.041855813281643, -23.69873517521205, -22.279866825228396, 30.409950322035456, 54.229175203283035, 120.64775628988014, 174.91091053782895, 210.201031453582, 219.688613410916, 255.91929629594017, 282.988443634807, 318.9426391500091, 393.49741154614776, 407.9457696018603, 456.5100264785036], 
          [-497.8390477810296, -456.7882234336763, -427.8573160203114, -412.180939371279, -336.08826667432334, -305.1506278098983, -13.483816670943781, 17.47285365442849, 111.02121623542075, 141.47645770536724, 194.32859852199624, 221.92271298841672, 257.8446019728011, 307.0584189029073, 338.1431268567528, 339.5691987374387, 368.3372857667483, 373.60228882535466, 376.3730490226858, 464.826965661798], 
          [-453.95046823732264, -453.52110210032106, -450.80431557171505, -443.91721579723753, -398.89428811893754, -371.2115022243877, -348.2331746377745, -270.17208273918607, -245.36841400238785, -202.89621863225938, -41.953258723031695, -29.472869971735463, -0.35998752916890453, 62.24887913350915, 105.83515207796665, 192.5251339484687, 236.8691139868214, 310.8285928391541, 316.64774016066883, 458.2949265920977], 
          [-469.4832946927785, -420.834680115889, -382.7276042807018, -360.1985040170629, -356.9522903612442, -208.84054614566196, -126.97274554342397, -87.07094066232179, -78.03168115468083, -50.631657136607544, -23.39768616539851, 15.96695423869403, 17.50861456296343, 27.761047453775177, 46.09383677679068, 134.29302208289835, 302.6920123189901, 379.8777918613888, 416.7440302492047, 473.2837772413958]]
ndim = 7
global_optima = 1055.8186278882022


def _evaluator(vectors):
    """
    An n-dimensional Schwefel function is defined as:
    f(x) = 418.9828*n - sum_{i=1}^n { x_i * sin(sqrt{abs(x_i)})}
    where  -500 <= x_i <= 500

    The global minima of the function being f(x) = 0 at all x_i = 420.9687
    """
    X = np.asarray(vectors, dtype=float)
    single = (X.ndim == 1)
    X = np.atleast_2d(X)

    n = X.shape[1]
    vals = 418.9829 * n - np.sum(X * np.sin(np.sqrt(np.abs(X))), axis=1)
    return vals[0] if single else vals


def _constraints(vectors):
    X = np.asarray(vectors, dtype=float)
    single = (X.ndim == 1)
    X = np.atleast_2d(X)

    if X.shape[1] < 7:
        raise ValueError(f"Constraint expects at least 7 dims, got {X.shape[1]}")

    expr1 = 400**2 - (X[:, 0] - 350)**2 - (X[:, 3] + 150)**2 - (X[:, 6] - 200)**2
    expr2 = 300**2 - (X[:, 1] + 100)**2 - (X[:, 4] - 50)**2 - (X[:, 5] + 35)**2
    expr3 = -0.001 * X[:, 2]**2 + 90

    vals = np.minimum(np.maximum(expr1, expr2), expr3)
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

    lower_bound = np.array([-500.0] * ndim)
    upper_bound = np.array([500.0] * ndim)

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