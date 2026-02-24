import sys, os
import argparse
import warnings
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import simulate
import math


values = [[-9.816108766090439, -9.574372495194348, -9.212141951433692, -9.121942926870483, -9.024397431899025, -8.110078377848325, -8.080435223926925, -7.91450573919705, -7.785521383091312, -7.401539042305254, -7.246063890462324, -7.148286173047758, -6.644348849262201, -6.553448217119726, -5.8098805686390005, -5.624587129485656, -5.501680916117233, -4.98904861116749, -4.736358695558898, -4.700551151598134, -4.682379209901968, -4.644444226341939, -4.44094542451781, -4.3293568102156215, -4.215167901606229, -4.038599581449778, -4.0282314910152355, -3.8711495647143, -3.6756365039710737, -3.6668276407505456, -3.5576909481674157, -3.3228648817100197, -2.896983408788147, -2.7675083864286414, -2.7504630013348352, -2.59529933887843, -2.4252706718132107, -2.4114879766051462, -2.094302113691784, -2.0300210619384673, -1.6832119117273, -1.436858758825995, -1.3717338481589163, -1.3062227947790817, -0.3803576961126858, -0.25641131755698865, 0.055889959882399864, 0.25945532786593795, 0.28125551989419506, 0.33692435694406875, 0.36665223009998016, 0.6472931496595535, 0.650086871266593, 0.6998797444306462, 0.8401308311539033, 0.8791215922893265, 0.98712532552506, 1.0023878056692972, 1.0958982603945024, 1.2414366326720039, 1.616654707240187, 2.0186846260769507, 2.0365039889630907, 2.3416060641187553, 2.8025266850081785, 4.082310509047431, 4.145699700829217, 4.182471512676676, 4.261992962632629, 4.5179503997882, 4.620171320859427, 5.003539596032859, 5.154558924521002, 5.163862724946505, 5.263306812072599, 5.677411738284636, 5.9574747613175045, 6.018724386274769, 6.746182178441639, 6.98732393186344, 7.079828139441162, 7.238161112075979, 7.283377286787001, 7.414104961825483, 7.461549408384567, 7.630188457167751, 7.732488675544861, 7.909964122397913, 7.97022201756554, 8.0581095775587, 8.522196590744262, 8.607064715528189, 8.778086113311133, 8.845492695552462, 8.913464626551978, 8.935829396665486, 9.071338247655422, 9.780941265892022, 9.823490942543074, 9.972181512243367],
          [-9.324748552642037, -9.033952295770892, -8.873615154552958, -8.637952680456229, -7.661064317099598, -7.51706442816948, -6.7503577571466895, -6.64195071882224, -6.4516337734503555, -6.24476052324175, -6.227867537968277, -5.961907170279233, -5.424896819827398, -5.408021424835101, -5.125327782332533, -5.0578879115055475, -4.887742096510839, -4.864816098768898, -4.863176290620849, -4.225376972144506, -4.147273824483899, -3.9839927619108746, -3.9440423205370507, -3.0608859938988386, -2.921218132248729, -2.8514228065658216, -2.6866545442631367, -2.437668530190593, -2.431263985203074, -2.136468840549817, -2.0391174762913122, -1.9550726423324463, -1.7116555623479197, -1.4158101316469072, -1.33461097847173, -0.7549937429478355, -0.634694894611048, -0.39933646916691146, -0.3089777211090201, -0.28581034453549314, -0.252431771268979, -0.0916879461220752, -0.08544632452082013, 0.05335516555168773, 0.17995930291251305, 0.2965409928755953, 0.46598363387350084, 0.596426705584328, 0.6450080144814745, 1.1280110772722622, 1.2348694988684805, 1.5907038445541755, 1.7085591090101637, 2.163010623624597, 2.269712211506885, 2.639705922722243, 2.652751287575798, 2.930126741840322, 3.023179110017594, 3.1771216890426057, 3.198060162303488, 3.2122819298645293, 3.48746656219099, 3.9548151884052345, 4.169347022271523, 4.228335084279681, 4.790816247928829, 5.00002379254169, 5.170534498966733, 5.686243468336777, 5.870343283262558, 5.99123629788936, 6.067284642003752, 6.346393660275854, 6.562787116185827, 6.588756149387848, 6.705669003815068, 6.850288024675621, 6.857858695773913, 6.874409378814697, 7.2468254099234315, 7.3101517080789975, 7.645101742622828, 7.965230575200561, 7.974517841047039, 8.004934635775612, 8.182056217470205, 8.212160579287044, 8.340798409785112, 8.39650022679298, 8.428813060497152, 8.606636384141158, 9.212873412593595, 9.239604273531082, 9.266605092409268, 9.464198524926644, 9.525274279101147, 9.600170310525925, 9.628396718409913, 9.875117679604703],
          [-9.9953533046307, -9.99138564105911, -9.982837750688278, -9.845294247641696, -9.813123220487576, -9.45892834025747, -9.360068356294269, -8.976192029727743, -8.450863716991865, -8.40091818714046, -8.350088566872703, -8.283545849665899, -8.113621480054615, -8.094250629998857, -7.828354827522759, -7.445733988952492, -7.2819970372571134, -7.042750690488171, -6.671965291405577, -6.653961350141124, -6.6285950164578615, -6.362134800868886, -6.31721907809731, -6.217354239629113, -6.1527099640231775, -5.9997058287234895, -5.992114077161199, -5.548741368000806, -5.49631664257952, -5.450830851059518, -5.207298388868368, -5.0204856301530265, -4.996214754173116, -4.801797319356577, -4.7570836120213205, -4.346086244171447, -3.94278220590059, -3.7849275799115967, -3.7031227905552777, -3.596605916597147, -3.561085842696361, -2.0168024594189244, -1.7568566527224565, -1.5217744101137303, -1.4700689088106849, -1.362673403067049, -1.2788176205033466, -1.0317043383055946, -0.6078195558274206, -0.26091148000332964, 0.01182678949908933, 0.3330150818453532, 0.754204522297881, 0.8139093485023103, 1.3957355446282964, 1.429172757877673, 1.5454668028614975, 1.5563756975422898, 1.664751198410773, 1.6930581519370218, 1.7633077479553059, 2.4064522719284547, 2.5562944169230235, 2.7708446952825216, 3.234606648390459, 3.4154339718972437, 3.816248273689098, 3.8850334929424157, 3.978579800752824, 4.12923653970765, 4.930388780942071, 5.0191103201055665, 5.11166835520722, 5.1460726172247835, 5.553111945551031, 5.701471962598543, 5.847663854551577, 6.300840528365331, 6.390714125652568, 6.410489071239223, 6.593732035735737, 6.60285676153417, 6.641051408263966, 6.786930022891703, 6.9263467029646435, 7.027867689747801, 7.646388243570954, 7.91466542724627, 8.101110894259115, 8.227642212825074, 8.434139019129688, 8.719956915430089, 9.224218333404927, 9.348079648911241, 9.416246227930827, 9.609548269355365, 9.611745522979632, 9.822162261843118, 9.85641717464246, 9.99416192351266],
          [-9.88667756549486, -9.777842880726695, -9.401029241026572, -9.257226962228312, -8.988170930972046, -8.96428570286468, -8.82506461972566, -7.388452757069935, -7.362347572392474, -7.179717476542596, -7.091728166765028, -6.984015169019511, -6.760925297120648, -6.548911294984066, -5.811922529939255, -5.797776760192274, -5.571244481948225, -5.359633034787801, -5.301324691485996, -4.241414415093201, -3.7780539125249284, -3.6499944662232515, -3.5606116096925895, -3.508941042667577, -3.162926545428279, -2.973456681211206, -2.972497630657842, -2.865253760225956, -2.809329385669603, -2.5977750079986723, -2.461180520054742, -2.329100148859622, -2.1747391443584174, -1.9460350592365376, -1.4699953176557496, -1.4284401419310218, -1.2322425120542864, -1.1737621678209091, -1.0806958712442913, -1.0460241207405137, -0.3963041905419491, 0.35566303572599445, 0.4122428821721513, 0.7236207067933531, 1.1271353832656814, 1.1764276745020084, 1.2037128928289995, 1.2318816921372466, 1.2528563510324275, 1.2837671059637206, 1.9241144521166653, 2.121981866913403, 2.156456606053359, 2.200208944874751, 2.2087694321220646, 2.43945078820542, 2.4740353920799834, 2.5544510589591773, 2.598936260529019, 2.9733051594165705, 2.980333984766082, 3.4934905809453376, 3.623497599299011, 3.9359830566401435, 3.9728278985838656, 4.059937994608214, 4.251205135804039, 4.313524386445749, 4.533771879643986, 4.624707430057287, 4.711450505039611, 4.73368665926408, 4.79804386738256, 5.062697756141585, 5.312969149719249, 5.817998306207857, 6.046318455577804, 6.059002450319699, 6.298483315207978, 6.342470872105185, 6.348327608442034, 6.60555294874564, 6.821999953646806, 7.00536274666775, 7.0173453465816, 7.037996686750574, 7.259560610836559, 7.312320429351249, 7.9077228086494635, 8.195877462324656, 8.37449508952561, 8.559337363990537, 8.822073174633324, 8.890896535641236, 8.912163460057556, 8.972974373059618, 9.128146384806083, 9.40336361055791, 9.888007640844851, 9.938164501532857]]
ndim = 4
global_optima = 0.46560597365889334


def _evaluator(vectors):
    """
    An n-dimensional Levy function is defined as:
    f(x) = sin^2(PI * w_1) + sum_{i=1}^{n-1} {(w_i - 1)^2 * [1 + 10 * sin^2(PI * w_i + 1)]} + (w_n - 1)^2 * [1 + sin^2(2*PI * w_n)]
    where  -10 <= x_i <= 10

    The global minima of the function being f(x) = 0 at all x_i = 1
    """
    X = np.asarray(vectors, dtype=float)
    single = (X.ndim == 1)
    X = np.atleast_2d(X)

    w = 1 + (X - 1) / 4

    part1 = (np.sin(np.pi * w[:, 0]) ** 2
             + (w[:, -1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[:, -1]) ** 2))

    part2 = np.sum((w[:, :-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:, :-1] + 1) ** 2),
                   axis=1)

    vals = part1 + part2
    return vals[0] if single else vals


def _constraints(vectors):
    X = np.asarray(vectors, dtype=float)
    single = (X.ndim == 1)
    X = np.atleast_2d(X)

    if X.shape[1] < 4:
        raise ValueError(f"Constraint expects at least 4 dims, got {X.shape[1]}")

    arg = X[:, 0]**2 + X[:, 2]**2
    # Optional: guard log domain (arg must be > 0 to avoid -inf/nan)
    if np.any(arg <= 0):
        raise ValueError("log argument x0^2 + x2^2 must be > 0")

    vals = np.log(arg) - (X[:, 1] + X[:, 3])**3 - 4
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

    lower_bound = np.array([-10.0] * ndim)
    upper_bound = np.array([10.0] * ndim)

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