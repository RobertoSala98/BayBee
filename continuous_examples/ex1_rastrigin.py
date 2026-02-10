import sys, os
import argparse
import warnings
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import simulate
import math


values = [[-4.872378693083864, -4.3280438143327675, -4.227027245920693, -4.201338889701047, -3.9220949141581682, -3.90852765139255, -3.828599099236892, -3.8194387386790516, -3.2806982278634247, -3.113974716127316, -2.9963939063985867, -2.984048601280556, -2.8938758351173592, -2.8772818303251677, -2.834813106580696, -2.7412757997764423, -2.7212280230949832, -2.582352544867249, -2.497110338726518, -2.0388455399641816, -1.88412756481496, -1.8790449184955151, -1.8015992078723775, -1.7102789039135144, -1.5234784101147465, -1.4260731937411641, -1.3700378378894595, -1.242968236396056, -1.2221984720793717, -1.0515094243806429, -1.030331052867007, -0.74254822403649, -0.7284504468069395, -0.624426404734213, -0.4198692886215829, -0.2379098251666143, -0.17984710975125395, 0.09761131700312031, 0.2886506482278506, 0.5382120507301495, 0.6472257604841856, 0.705463918265667, 0.8100159590220697, 0.9945406090682534, 1.1346918948781024, 1.2157764089950618, 1.3382926293234876, 1.5604071865228653, 1.5774152559974395, 1.766584878714391, 2.1879927882426102, 2.189028787419268, 2.505848408085237, 3.206993565421132, 3.523263218081916, 3.711866240300501, 4.841137942269673, 4.951444359840564, 4.974503956241162, 5.002636364851562],
          [-5.056463395670901, -4.7615607792273265, -4.636470960751633, -4.625377799058645, -4.485594855058846, -4.467121502856424, -4.424924173825898, -4.09603104166392, -4.026984993399862, -3.8130155270451045, -3.5361390239872015, -3.491067136742112, -3.266557681424515, -3.1084374893146687, -2.9095240112878105, -2.65969460206686, -2.3916145590263977, -2.122909942766902, -2.0460502352706316, -1.8480665883720646, -1.816400051597518, -1.685204492501009, -1.6204940699113446, -1.5309684768942078, -1.0741412929804195, -0.9162153802432167, -0.906742327667982, -0.8955749044870425, -0.47016727237834033, -0.4561274573110756, -0.4228044332595857, -0.3026030887827291, -0.1049828919829654, 0.13344266670003524, 0.39084472174957874, 0.43837980688473976, 0.5051769246897315, 0.6049789940256494, 0.8212899790676103, 0.9242154487896768, 0.958265378781558, 1.2845425295569477, 1.3406617484091985, 1.6630313444078606, 1.6781981386359863, 1.9458212633773586, 2.456113860505557, 2.785352791280598, 2.8125314279202804, 3.0085623251238944, 3.0330098287305516, 3.468606742938996, 3.5114208981823625, 3.5795249197899244, 3.7761993403242533, 3.8548014312125067, 4.421216561411977, 4.6261885774638545, 4.777376956148873, 4.900844516112614],
          [-5.117042402439771, -5.046628883266594, -4.915324077152397, -4.783538231701681, -4.762080853815542, -4.338782591832311, -4.26062145629707, -3.882014028596898, -3.8581871938101076, -3.652904671688913, -3.5635247663652923, -3.5512486765242963, -3.548857400763759, -3.5182775221683857, -3.272199170786279, -3.2432445100568303, -3.21537149511786, -3.1830728532449326, -2.2061876888552185, -2.149653266193526, -2.0508971408054117, -2.0458344297121096, -1.8313795495497756, -1.6918344794629285, -1.4078189033779402, -1.35646686437559, -1.307046867067843, -1.2301605416986159, -1.1668185467299201, -1.1324051641448638, -1.0846291995889645, -1.0115898897726767, -0.5305817241415802, -0.28682820196010184, -0.14411873684395538, -0.03186826058545211, 0.21361520429095116, 0.2404436141265931, 0.253254271562537, 0.390614380135772, 0.43960773629776995, 0.5974260141017327, 1.185861586153961, 1.2164552432223061, 1.6127568266194308, 1.6949089199581406, 2.4734298739367055, 2.5772111192832776, 3.297960409818951, 3.353502512867345, 3.462474332643395, 3.5674930689773516, 3.7136305331611466, 3.7932025062868293, 3.9421712604482932, 4.282236221709474, 4.45655325326135, 4.464977263997802, 5.053970433852213, 5.054661496565196],
          [-5.016522147315945, -4.899121532578645, -4.168143143366031, -4.015399041344509, -3.76419585550957, -2.937662185754914, -2.913498720964825, -2.805884256684958, -2.698451937821236, -2.672314459537769, -2.4695975748758223, -2.3117378686932932, -2.2443779116987335, -2.087487896281748, -1.8482255968109405, -1.7459222516964017, -1.5035384225359576, -1.458763462363728, -1.204656148855312, -1.1562327442992002, -0.9777934588320054, -0.6698295938131, -0.6477995973859594, -0.42539164256275885, -0.35658684776842087, -0.33845580946985176, -0.13238380515140236, 0.37243556475643924, 0.5800179672119903, 0.7421040202352165, 0.8300643998715937, 1.0184736754154633, 1.4758051480916823, 1.4794277194745336, 1.541814715819112, 1.614229718257933, 1.8413280392114357, 2.0220933508841323, 2.0432617571471186, 2.7134326907611186, 2.8003003468944963, 2.9234622454942896, 2.9375157023490877, 3.052103293455983, 3.1124618406728457, 3.156183260213381, 3.179880692890227, 3.873731824814473, 3.914666773990633, 3.967579233140035, 3.9687154283243578, 4.030997831521234, 4.104724602393023, 4.133632594475684, 4.152852581694844, 4.315655225267192, 4.347865440404559, 4.615792829147707, 4.723355944137825, 5.094514648303254],
          [-5.061887256058969, -4.976765036360426, -4.796466392373344, -4.718194283269481, -4.380464596625036, -3.194590608568798, -3.1031114143504794, -2.8332037574403306, -2.686540100926427, -2.4278997261368445, -2.3868867593317864, -2.376174484120253, -2.2379141814164667, -2.180757774064483, -2.1786609577557043, -2.0788703477192234, -2.0639476862012986, -1.798066938142, -1.7211535320415576, -1.6699796446983033, -1.6239469755855862, -1.6000180895970835, -1.5341395256684258, -0.6816918027841616, -0.5343386481453614, -0.480603020149986, -0.4314905805351126, 0.16760926592725234, 0.2269163085685868, 0.6677351485466545, 0.7779012033548351, 0.8275709971784151, 0.8780955629393201, 0.8858853631285681, 0.9637496863584234, 1.1698512739145803, 1.2227950756697359, 1.4733621290369783, 1.8088101313711924, 1.8621160814529683, 1.953297501128823, 2.040005398653128, 2.147184692599539, 2.152300507137345, 2.2754379676881875, 2.3487128652106994, 3.202184862428399, 3.3274423445563324, 3.4615737871649452, 3.5176367612427315, 3.5211502837828705, 3.541559249016209, 3.6221631400504526, 3.9216136905544827, 3.9863623890328794, 4.238505622640715, 4.459060099613015, 4.678434195577256, 4.69110085256322, 4.695496144966001]]
ndim = 5
global_optima = 5.792602630145936


def _evaluator(vectors):
    """
    A n-dimensional Rastrigin's function is defined as:
    f(x) = 10*n + sum_{i=1}^n { x_i^2 - 10*cos(2*PI*x_i) }
    where  -5.12 <= x_i <= 5.12

    The global minima of the function being f(x) = 0 at all x_i = 0
    """
    X = np.asarray(vectors, dtype=float)
    single = (X.ndim == 1)
    X = np.atleast_2d(X)   # (m, n) even if user passed (n,) [vector]
    
    n = X.shape[1]
    vals = 10 * n + np.sum(X*X - 10*np.cos(2*np.pi*X), axis=1)

    return vals[0] if single else vals


def _constraints(vectors):
    X = np.asarray(vectors, dtype=float)
    single = (X.ndim == 1)
    X = np.atleast_2d(X)

    if X.shape[1] < 5:
        raise ValueError(f"Need at least 5 dimensions, got {X.shape[1]}")

    vals = 2 - 5*np.exp(np.sum(X[:, :5], axis=1))
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

    lower_bound = np.array([-5.12] * ndim)
    upper_bound = np.array([5.12] * ndim)

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