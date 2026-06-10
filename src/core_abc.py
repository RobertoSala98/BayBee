import os
import sys
import copy
import numpy as np
from tqdm import tqdm
from time import time
from collections import deque
from scipy.optimize import minimize_scalar

from src.utils.cache import BloomFilter
from src.utils import logging
from src.strategy import bo, levy as levy_dist, memory, surrogate


class Bee(object):
    """
    Represents a candidate solution (bee) in the ABC algorithm.
    """
    __slots__ = (
        "vector", "constraint", "value", "fitness", "bee_id", "counter", "role", "memory", "constr_model",
        "target_model", "BO_to_do", "exclude_exploration_dim", "skip_exploration", "target_pipeline", "constr_pipeline",
        'std_ABC', 'counter_idx'
    )

    bee_counter = 0

    def __init__(self, lower, upper, objective_function, constraint_function=None, init_model=None, snap_function=None,
                 act_iter=-1, std_ABC=False, prev_memory={}, avoid_memory=[]):
        """
        Instantiates a Bee object with a random position within the given bounds and evaluates its fitness.

        Parameters:
            :param list lower               : lower bound of solution vector
            :param list upper               : upper bound of solution vector
            :param def  objective_function  : objective function that evaluates the quality of the solution vector
            :param def  constraint_function : constraints function
            :param def  init_model          : model used to initialize the vector
            :param def  snap_function       : snap function
            :param int  act_iter            : current iteration
            :param bool std_ABC             : flag to use standard ABC
            :param dict prev_memory         : previous memory to recover for the bee
            :param set avoid_memory         : set of vectors to avoid during initialization
        """
        self.std_ABC = std_ABC

        # generates a random solution vector within the given bounds
        self._random(lower, upper, init_model)

        # adjust the vector using snap_function
        if snap_function is not None:
            self.vector[:] = snap_function(self.vector)

        while any(np.array_equal(self.vector, mem) for mem in avoid_memory):
            self._random(lower, upper, init_model)
            if snap_function is not None:
                self.vector[:] = snap_function(self.vector)

        # initialize problem constraint(s) if provided
        if constraint_function is None:
            self.constraint = 0
        else:
            # checks if the problem constraint(s) are satisfied
            self.constraint = constraint_function(self.vector)

        # computes objective function value
        if objective_function is not None:
            self.value = objective_function(self.vector)
        else:
            # if no objective func is provided, set a high value (worst case)
            self.value = np.float64(sys.float_info.max)

        # compute fitness based on constraint and objective value
        self._fitness()

        # assign unique id to bee
        self.bee_id = Bee.bee_counter
        Bee.bee_counter += 1

        # initialize trial limit counter
        self.counter = 0

        # initialize bee role
        self.role = "Initialization"

        # initialize the memory of each bee
        self.memory = {"x": [], "f(x)": [], "g(x)": [], "role": [], "iter": []}
        memory.register_value(self, self.vector, self.value, self.constraint, self.role, act_iter)

        # initialize ML models and store BO data
        self.constr_model = None
        self.target_model = None
        self.BO_to_do = 0

        # dimensions to exclude from exploration
        self.exclude_exploration_dim = set()
        self.skip_exploration = False

        # Recover old memory
        if prev_memory != {}:
            for key, value in self.memory.items(): prev_memory[key].append(value[0])
            self.memory = prev_memory

    def _random(self, lower, upper, init_model=None):
        """
        Initialises a solution vector randomly within the given bounds.

        Parameters:
            :param list lower       : lower bound of solution vector
            :param list upper       : upper bound of solution vector
            :param def  init_model  : initialization ML model
        """
        self.vector = np.zeros(len(lower))

        # if no init model is provided, use uniform random sampling
        if init_model is None:
            self.vector[:] = np.random.uniform(low=lower, high=upper)

        else:
            counter_init = 0
            # loop until feasible vector is found or max iter is reached
            while True:
                self.vector[:] = np.random.uniform(low=lower, high=upper)
                if init_model.predict([self.vector]) >= 0 or counter_init >= 1e5:
                    break
                else:
                    counter_init += 1

    def _fitness(self):
        """
        Evaluates the fitness of a solution vector. Lower objective values correspond to higher fitness.
        """
        if self.constraint < 0:
            self.fitness = 0.0
        elif self.value >= 0:
            self.fitness = 1 / (1 + self.value)
        else:
            self.fitness = 1 + abs(self.value)


class Beehive(object):
    """
    Implementation of the Artificial Bee Colony (ABC) algorithm.
    """
    def __init__(self,
                 lower, upper,
                 objective_function=None,
                 constraint_function=None,
                 numb_bees=30,
                 max_itrs=100,
                 max_trials=None,
                 selection_function=None,
                 extra_params=None,
                 log_filename="bee_log.csv",
                 global_optima=None,
                 levy_step_size=0.1,
                 ML_approach_constr=None,
                 ML_approach_target=None,
                 constr_regressor_type='ridge',
                 target_regressor_type='ridge',
                 BO_approach=False,
                 min_ML_history_size=10, max_ML_history_size=100,
                 dataset_obj=None,
                 snap_function=None,
                 BO_steps=1,
                 std_ABC=False,
                 seed=-1,
                 memory_type='local',
                 cache=False,
                 resume_sim=False,
                 discrete_values=[]):
        """
        Initializes the Beehive with a set of bees and algorithm parameters.

        Parameters:
            :param list lower                   : lower bound of solution vector
            :param list upper                   : upper bound of solution vector
            :param def objective_function       : evaluation function of the optimal problem
            :param def constraint_function      : evaluation function of the constraints
            :param def numb_bees                : number of active bees within the hive
            :param int max_itrs                 : maximum number of iterations
            :param int max_trials               : max number of trials without any improvement
            :param def selection_function       : custom selection function
            :param dict extra_params            : optional extra arguments for selection
            :param str log_filename             : name of the log file to store the bee's information
            :param float global_optima          : known global optimum value of the objective function
            :param float levy_step_size         : step size for Lévy flight
            :param str ML_approach_constr       : machine learning approach for black-box constraints
            :param str ML_approach_target       : machine learning approach for black-box objective function
            :param str self.regressor_type           : regression technique to use for ML models
            :param bool BO_approach             : flag to use Bayesian Optimization
            :param int min_ML_history_size      : minimum history size to build the model
            :param int max_ML_history_size      : maximum history size to build the model
            :param Dataset dataset_obj          : dataset object if using a dataset
            :param def snap_function            : snap function
            :param int BO_steps                 : number of Bayesian Optimization steps
            :param bool std_ABC                 : flag to use standard ABC
            :param int seed                     : seed for random number generator
            :param str memory_type              : memory type to use during simulation ('local', 'global' or 'none')
            :param bool cache                   : use cache or not
            :param bool resume_sim              : resume from a previous simulation
        """
        # ensure that upper and lower bounds match in length
        assert (len(upper) == len(lower)), "'lower' and 'upper' must be a list of the same length."

        # generate or set seed for reproducibility
        self.seed = seed
        np.random.seed(self.seed)
        
        # ensure even number of bees
        self.size = numb_bees + (numb_bees % 2)

        # assigns properties of algorithm
        self.dim = len(lower)
        self.max_itrs = max_itrs
        self.max_itrs_async = 2 * max_itrs * self.size

        # define max trials
        if max_trials is not None:
            self.max_trials = max_trials
        else:
            self.max_trials = 0.6 * self.size * self.dim

        # assigns properties of the optimization
        self.selection_function = selection_function
        self.extra_params = extra_params
        self.evaluate = objective_function
        self.constraints = constraint_function
        self.dataset_obj = dataset_obj

        # convert and assigns bounds
        self.lower = np.array(lower, dtype='float64')
        self.upper = np.array(upper, dtype='float64')
        
        # initialize current best and its solution vector
        self.best = np.float64(sys.float_info.max)
        self.solution = None

        # decide snap function based on dataset object or provided one
        if self.dataset_obj is not None:
            self.snap_function = self.dataset_obj.get_snapped_point
        elif snap_function is not None:
            self.snap_function = snap_function
        else:
            self.snap_function = None

        # initialize population of bees
        self.iter = 0
        Bee.bee_counter = 0
        self.population = [Bee(lower, upper, self.evaluate, self.constraints, snap_function=self.snap_function, act_iter=-1, std_ABC=std_ABC) for _ in range(self.size)]
        self.mutated_population = copy.deepcopy(self.population)

        # find the best solution vector
        self.find_best()
        self.global_optima = global_optima
        self.levy_step_size = levy_step_size

        # initialize the CSV log file to record algorithm's process
        self.log_filename = log_filename
        self.path_to_log = "output/" + self.log_filename
        self.log_file = None
        self.csv_writer = None
        logging.initialize_log_file(self, resume_sim)

        # initialize parameters for ML models
        self.ML_approach_constr = ML_approach_constr
        self.ML_approach_target = ML_approach_target
        self.min_ML_history_size = min_ML_history_size
        self.max_ML_history_size = max_ML_history_size
        self.constr_regressor_type = constr_regressor_type
        self.target_regressor_type = target_regressor_type

        # initialize parameters for BO
        self.BO_approach = BO_approach
        self.BO_steps = BO_steps

        # prepare for async
        self.std_ABC = std_ABC
        self.global_time = 0.0
        self.scheduler = []

        self.memory_type = memory_type
        if self.memory_type == 'global':
            self.global_mem = [bee.memory['x'][0][:] for bee in self.population]

        self.use_cache = cache
        if self.use_cache:
            capacity = self.max_itrs_async
            self.cache = BloomFilter(capacity, 1e-6)
            for bee in self.population:
                self.cache.add(np.round(bee.vector, 8).tobytes())

        self.discrete_values = discrete_values

    def _mutate(self, vector, current_bee, indices):
        """
        Mutates a single dimension of the solution vector by differential-like operation.

        Parameters:
            :param np.array vector : Solution vector to be mutated (will be modified in place).
            :param int current_bee : Index of the current bee being mutated.
            :param list indices    : List of indices of other valid bees to use for mutation.
        """
        recent_memory = memory.retrieve_recent_memory(self, current_bee)

        while True:
            other_bee = np.random.choice(indices)
            dim = np.random.choice(
                [_ for _ in range(self.dim) if _ not in self.population[current_bee].exclude_exploration_dim])

            current_val = self.population[current_bee].vector[dim]
            other_val = self.population[other_bee].vector[dim]

            if self.ML_approach_target in ['local', 'global'] and self.population[current_bee].target_model is not None:
                temp_vector = np.array(vector).copy()

                def f(x):
                    temp_vector[dim] = x
                    # temp_vector[:] = self.snap_function(temp_vector, recent_memory, dim)
                    return self.population[current_bee].target_model.predict([temp_vector])

                result_minimization = minimize_scalar(f, bounds=(min(other_val, 2 * current_val - other_val),
                                                                 max(other_val, 2 * current_val - other_val)),
                                                      method='bounded')
                vector[dim] = result_minimization.x
                """
                # Comment: this code selects the next point minimizing the AF and moving along all the dimensions, but it is very slow
                def f(x):
                    return self.population[current_bee].target_model.predict([x])

                bounds = []
                for idx in range(len(vector)):
                    current_val = self.population[current_bee].vector[idx]
                    other_val = self.population[other_bee].vector[idx]
                    bounds.append((min(other_val, 2*current_val - other_val), max(other_val, 2*current_val - other_val)))
                result_minimization = differential_evolution(f, bounds)
                vector = result_minimization.x
                """

            else:
                phi = np.random.uniform(-1, 1)
                vector[dim] = current_val + phi * (current_val - other_val)

            if not self.std_ABC:
                vector[:] = self.snap_function(vector, recent_memory, dim)

                if np.all(vector == self.population[current_bee].vector):
                    self.population[current_bee].exclude_exploration_dim.add(int(dim))
                    if len(self.population[current_bee].exclude_exploration_dim) == self.dim:
                        self.population[current_bee].skip_exploration = True
                        break
                else:
                    break

            else:
                vector[:] = self.snap_function(vector)
                break

    def _check(self, vector):
        """
        Checks that a solution vector is contained within the pre-determined lower and upper bounds of the problem.
        If any dimension exceeds the bound, it is clipped to the boundary value.

        Parameters:
            :param np.array vector : the solution vector to check and clip.
        """
        for i in range(self.dim):
            # checks lower bound
            if vector[i] < self.lower[i]:
                vector[i] = self.lower[i]
            # checks upper bound
            elif vector[i] > self.upper[i]:
                vector[i] = self.upper[i]

        return vector

    def _process_candidate(self, index, is_scout):
        """
        Helper function to handle post generation steps for candidate bee.

        Parameters:
            :param int index     : index of candidate bee
            :param bool is_scout : is passed candidate a scout
        """
        candidate_bee = self.mutated_population[index]

        if not is_scout:
            if self.dataset_obj is None:
                candidate_bee.vector[:] = self._check(candidate_bee.vector)

            if self.constraints is not None:
                candidate_bee.constraint = self.constraints(candidate_bee.vector)
            else:
                candidate_bee.constraint = 0

            if self.evaluate is not None:
                candidate_bee.value = self.evaluate(candidate_bee.vector)
            else:
                candidate_bee.value = np.float64(sys.float_info.max)
            candidate_bee._fitness()

        if self.memory_type == 'global':
            self.global_mem.append(candidate_bee.vector[:])

        if self.use_cache:
            cache_key = np.round(candidate_bee.vector, 8).tobytes()
            if cache_key in self.cache:
                return 0.0

        if self.dataset_obj:
            return self.dataset_obj.get_execution_time(candidate_bee.vector)
        else:
            return 0.0

    def run(self, start_itr=0, checkpoint_dir=None):
        """
        Runs the ABC algorithm synchronously through all iterations.

        Parameters:
            :param int start_itr       : iteration to start from
            :param str checkpoint_dir  : directory to save the checkpoint
        """
        # stores convergence information
        cost = {"best": [], "mean": [], "feasible": [], "abs_regret": []}
        self._async = False
        self.global_simulation_time = 0.0

        try:
            print("\n")

            self.find_best()
            absolute_regret = abs(self.best - self.global_optima)
            self.iter = start_itr - 1

            if start_itr == 0:
                # log initial state for each bee and set counter index
                for bee_id in range(self.size):
                    logging.log_bee(self, self.population[bee_id], bee_id, absolute_regret)
                    self.mutated_population[bee_id].counter_idx = bee_id

            for itr in tqdm(range(start_itr, self.max_itrs)):
                self.iter = itr

                # execute worker
                worker_exec_time = []
                for index in range(self.size):
                    
                    update_time_start = time()

                    if self.population[index].BO_to_do > 0:
                        exec_time = bo.apply_BO(self, index, 'Worker')
                    else:
                        exec_time = self.worker_step(index)
                    
                    update_time = time() - update_time_start
                    worker_exec_time.append(exec_time + update_time)

                    cur_absolute_regret = abs(self.best - self.global_optima)
                    logging.log_bee(self, self.mutated_population[index], index, cur_absolute_regret,
                                    self.global_simulation_time, update_time, exec_time)

                max_worker_exec_time = max(worker_exec_time) if worker_exec_time else 0.0
                self.global_simulation_time += max_worker_exec_time

                for i in range(self.size):
                    self.update_bee_status(i, cache_value=True)

                self.find_best()

                # execute onlooker
                onlooker_exec_time = []
                for index in range(self.size):

                    update_time_start = time()

                    if self.population[index].BO_to_do > 0:
                        exec_time = bo.apply_BO(self, index, 'Onlooker')
                    else:
                        exec_time = self.onlooker_step(self.compute_probability(), index)

                    update_time = time() - update_time_start
                    onlooker_exec_time.append(exec_time + update_time)

                    cur_absolute_regret = abs(self.best - self.global_optima)
                    logging.log_bee(self, self.mutated_population[index], index, cur_absolute_regret,
                                    self.global_simulation_time, update_time, exec_time)

                max_onlooker_exec_time = max(onlooker_exec_time) if onlooker_exec_time else 0.0
                self.global_simulation_time += max_onlooker_exec_time

                for i in range(self.size):
                    self.update_bee_status(i, cache_value=True)

                self.find_best()

                # execute scout
                scout_index = []
                scout_exec_time = []
                for index in list(range(self.size)):

                    if self.population[index].counter > self.max_trials and self.population[index].BO_to_do == 0:
                        
                        update_time_start = time()

                        exec_time = self.send_scout(index)
                        update_time = time() - update_time_start
                        scout_exec_time.append(exec_time + update_time)
                        scout_index.append(index)

                        cur_absolute_regret = abs(self.best - self.global_optima)
                        logging.log_bee(self, self.mutated_population[index], index, cur_absolute_regret,
                                        self.global_simulation_time, update_time, exec_time)

                max_scout_exec_time = max(scout_exec_time) if scout_exec_time else 0.0
                self.global_simulation_time += max_scout_exec_time

                for i in scout_index:
                    self.update_bee_status(i, cache_value=True)

                # find best
                self.find_best()

                # compute absolute regret
                absolute_regret = abs(self.best - self.global_optima)

                # stores convergence information
                cost["best"].append(self.best)
                cost["feasible"].append(sum([1 for bee in self.population if bee.constraint >= 0]))
                cost["mean"].append \
                    (sum([bee.value for bee in self.population if bee.constraint >= 0]) / cost["feasible"][-1] if
                     cost["feasible"][-1] > 0 else np.inf)
                cost["abs_regret"].append(absolute_regret)

                checkpoint_rate = 1000
                if checkpoint_dir and (itr + 1) % checkpoint_rate == 0:
                    checkpoint_path = os.path.join("output", checkpoint_dir, "checkpoint.pkl")
                    memory.save_checkpoint(self, checkpoint_path)

            non_repeated_points, all_points_ = logging.log_observed_points(self)

        finally:
            self.log_file.close()
            print(f"Final global time: {round(self.global_simulation_time, 2)} s")

        return cost, non_repeated_points, all_points_, self.global_simulation_time

    def run_async(self, checkpoint_dir=None):
        """
        Runs the ABC algorithm asynchronously with an event-driven manner.

        Parameters:
            :param str checkpoint_dir  : directory to save the checkpoint
        """
        # stores convergence information
        cost = {"best": [], "mean": [], "feasible": [], "abs_regret": []}
        self._async = True

        if self.iter == 0 and not self.scheduler:
            self.find_best()
            absolute_regret = abs(self.best - self.global_optima)
            self.iter = -1

            # log initial state for each bee and set counter index
            for bee_id in range(self.size):
                logging.log_bee(self, self.population[bee_id], bee_id, absolute_regret)
                self.mutated_population[bee_id].counter_idx = bee_id

            # initialize scheduler: [bee_id, finish_time, actual_role, update_bee, cache_value]
            self.scheduler = [[bee_id, 0.0, "Initialization", False, False] for bee_id in range(self.size)]
            self.iter = 0

        # Async algorithm
        self.max_itrs = self.max_itrs_async
        max_itrs = self.max_itrs_async
        print()
        pbar = tqdm(total=max_itrs, initial=self.iter)

        try:
            while self.iter < max_itrs:
                next_event = min(self.scheduler, key=lambda x: x[1])
                bee_id = next_event[0]
                self.global_time = next_event[1]
                prev_role = next_event[2]
                store_data = True

                update_time_start = time()
                if next_event[3]:
                    self.update_bee_status(bee_id, next_event[4])
                    self.find_best()

                if prev_role in ['Initialization', 'Scout']:
                    execution_time = self.worker_step(bee_id)
                    update_time = time() - update_time_start
                    if execution_time == 0.0:
                        store_data = False
                    self.scheduler[bee_id] = [bee_id, self.global_time + update_time + execution_time, "Worker", True, store_data]

                elif prev_role == "Worker":
                    if self.population[bee_id].BO_to_do > 0:
                        execution_time = bo.apply_BO(self, bee_id, "Worker")
                        update_time = time() - update_time_start
                        if execution_time == 0.0:
                            store_data = False
                        self.scheduler[bee_id] = [bee_id, self.global_time + update_time + execution_time, "Worker", True,
                                            store_data]
                    else:
                        execution_time = self.onlooker_step(self.compute_probability(), bee_id)
                        update_time = time() - update_time_start
                        if execution_time == 0.0:
                            store_data = False
                        self.scheduler[bee_id] = [bee_id, self.global_time + update_time + execution_time, "Onlooker", True,
                                            store_data]

                elif prev_role == "Onlooker":
                    if self.population[bee_id].counter > self.max_trials:
                        execution_time = self.send_scout(bee_id)
                        update_next_event = True
                    else:
                        execution_time = 0.0
                        update_next_event = False
                    update_time = time() - update_time_start
                    if execution_time == 0.0:
                        store_data = False
                    self.scheduler[bee_id] = [bee_id, self.global_time + update_time + execution_time, "Scout", update_next_event, store_data]

                else:
                    assert ("There is a mismatch with bee's role")

                if store_data:
                    self.find_best()
                    absolute_regret = abs(self.best - self.global_optima)
                    logging.log_bee(self, self.mutated_population[bee_id], bee_id, absolute_regret, self.global_time,
                                 update_time, execution_time)

                    cost["best"].append(self.best)
                    cost["feasible"].append(sum([1 for bee in self.population if bee.constraint >= 0]))
                    cost["mean"].append(
                        sum([bee.value for bee in self.population if bee.constraint >= 0]) / cost["feasible"][-1] if
                        cost["feasible"][-1] > 0 else np.inf)
                    cost["abs_regret"].append(absolute_regret)

                    self.iter += 1
                    pbar.update(1)

                    checkpoint_rate = 1000
                    if checkpoint_dir and ((self.iter / (self.size * 2)) % checkpoint_rate == 0):
                        checkpoint_path = os.path.join("output", checkpoint_dir, f"checkpoint.pkl")
                        memory.save_checkpoint(self, checkpoint_path)

            pbar.close()

            while len(self.scheduler) > 0:
                next_event = min(self.scheduler, key=lambda x: x[1])
                bee_id = next_event[0]
                self.global_time = next_event[1]

                if next_event[3]:
                    self.update_bee_status(bee_id, next_event[4])

                self.scheduler.remove(next_event)

            non_repeated_points, all_points_ = logging.log_observed_points_async(self)

        finally:
            self.log_file.close()
            print(f"Final global time: {round(self.global_time, 2)} s")

        return cost, non_repeated_points, all_points_, self.global_time

    def compute_probability(self):
        """
        Computes the selection probability for each bee based on fitness.
        """
        # retrieves fitness of bees within the hive
        values = np.zeros(self.size)
        for bee_idx in range(self.size):
            if not self.population[bee_idx].skip_exploration:
                values[bee_idx] = self.population[bee_idx].fitness

        max_value = values.max()

        # computes probabilities
        probas = np.where(values > 0, 0.9 * values / max_value + 0.1, 0.0)

        # Always use standard ABC probability computation if std_ABC is True
        if self.std_ABC:
            self.probas = probas.tolist()
        else:
            # use custom selection function if provided
            if self.selection_function is not None:
                if self.extra_params is not None:
                    self.probas = self.selection_function(values.tolist(), **self.extra_params)
                else:
                    self.probas = self.selection_function(values.tolist())
            else:
                self.probas = probas.tolist()

        return np.cumsum(self.probas).tolist()

    def find_best(self):
        """
        Finds current best bee candidate within the population (lowest objective value).
        """
        # extract the objective function values of all bees
        values = []
        for bee in self.population:
            if bee.constraint >= 0:
                values.append(bee.value)
            else:
                values.append(np.float64(sys.float_info.max))
        index = values.index(min(values))

        # update the best solution if a better one is found
        if values[index] < self.best:
            self.best = values[index]
            self.solution = self.population[index].vector[:]

    def select(self, probas, beta):
        """
        Selects a bee based on a roulette wheel selection mechanism based on cumulative probabilities.

        Parameter(s):
            :param list probas : list of cumulative probabilities for each bee.
            :param float beta  : the "roulette wheel selection" parameter (0 <= beta <= max(cumulative probas))
        """
        # selects a new potential "onlooker" bee
        for index in range(self.size):
            if beta < probas[index]:
                return index
        return None

    def worker_step(self, index):
        """
        Perform a single worker update step for the bee at the given index.

        Parameters:
            :param int index : index of the bee
        """
        if self.population[index].constraint >= 0:
            execution_time = self.send_worker(index, 'Worker')
        else:
            execution_time = self.send_scout(index)

        return execution_time

    def send_workers(self):
        """
        Deploys worker bees to generate new candidate solutions.
        """
        for index in range(self.size):
            self.worker_step(index)

    def send_worker(self, index, role):
        """
        Generates a new candidate solution for a bee as a worker or onlooker.

        Parameters:
            :param int index : index of the bee
            :param str role  : the role of the bee
        """
        # apply BO starting from the actual position
        if self.BO_approach:
            if np.random.uniform(0, 1) < 0.3 * (self.iter / self.max_itrs):
                self.population[index].BO_to_do = self.BO_steps
                execution_time = bo.apply_BO(self, index, role)
                return execution_time

        if self.population[index].skip_exploration:
            return self.send_scout(index)

        self.population[index].role = role

        # prepare candidate bee
        self.mutated_population[index] = copy.deepcopy(self.population[index])
        self.mutated_population[index].role = role
        self.mutated_population[index].counter_idx = index

        if self.ML_approach_target == 'local':
            bee_to_train = self.population[index]
            surrogate.build_ML_model_target(bee_to_train, self.min_ML_history_size, self.max_ML_history_size, self.target_regressor_type)
        elif self.ML_approach_target == 'global':
            surrogate.build_ML_model_target_global(self, index, self.target_regressor_type)

        # selects another bee
        indices = [bee_idx for bee_idx in range(self.size) if
                   bee_idx != index and self.population[bee_idx].constraint >= 0]
        if len(indices) == 0:  # if bee index is the only feasible one, try to move with respect to an unfeasible one
            indices = [_ for _ in range(self.size) if _ != index]

        # mutate the chosen dimension to create a new candidate solution
        self._mutate(self.mutated_population[index].vector, index, indices)

        if self.population[index].skip_exploration:
            return self.send_scout(index)

        return self._process_candidate(index, is_scout=False)

    def onlooker_step(self, cum_prob, current_index):
        """
        Perform only one onlooker step for the bee at the given index.

        Parameters:
            :param list cum_prob      : cumulative probabilities for each bee
            :param int  current_index : index of the bee
        """
        # if sum of prob is 0, replace onlooker with scout
        if sum(cum_prob) == 0:
            execution_time = self.send_scout(current_index)

        else:
            # roulette wheel selection
            beta_onlooker = np.random.rand() * cum_prob[-1]
            index = self.select(cum_prob, beta_onlooker)

            # probability of using a Lévy flight for mutation
            if not self.std_ABC:
                if self.max_trials > 1:
                    p_levy = 0.1 + 0.4 * (self.population[index].counter / (self.max_trials - 1))
                else:
                    p_levy = 0.1
            else:
                p_levy = 0.0

            if np.random.rand() < p_levy:
                execution_time = self.send_onlooker(current_index, index, "Onlooker", levy=True)
            else:
                execution_time = self.send_onlooker(current_index, index, "Onlooker")

        return execution_time

    def send_onlookers(self):
        """
        Deploys onlooker bees to further exploit promising solutions found by worker bees.
        """
        # sends onlookers
        probas = self.compute_probability()

        for numb_onlookers in range(self.size):
            self.onlooker_step(probas, numb_onlookers)

    def send_onlooker(self, current_index, index, role, levy=False):
        """
        Generates a new candidate solution for a bee as an onlooker.

        Parameters:
            :param int current_index  : index of the bee doing onlooker step
            :param int index          : index of the bee being evaluated
            :param str role           : the role of the bee ('Worker', 'Onlooker')
            :param bool levy          : if True, use Lévy flight for mutation
        """ 
        if levy: role += "_Levy"
        self.population[current_index].role = role

        # make a copy of the current bee's solution
        self.mutated_population[current_index] = copy.deepcopy(self.population[index])
        self.mutated_population[current_index].bee_id = self.population[current_index].bee_id
        self.mutated_population[current_index].memory = copy.copy(self.population[current_index].memory)

        self.mutated_population[current_index].role = role
        self.mutated_population[current_index].counter_idx = index

        if levy:
            levy_dist.levy_flight(self, self.mutated_population[current_index].vector, index)
        else:
            if self.ML_approach_target == 'local':
                bee_to_train = self.population[index]
                surrogate.build_ML_model_target(bee_to_train, self.min_ML_history_size, self.max_ML_history_size, self.target_regressor_type)
            elif self.ML_approach_target == 'global':
                surrogate.build_ML_model_target_global(self, index, self.target_regressor_type)

            # selects another bee
            indices = [bee_idx for bee_idx in range(self.size) if bee_idx != index and self.population[bee_idx].constraint >= 0]
            if len(indices) == 0:   # if bee index is the only feasible one, try to move with respect to an unfeasible one
                indices = [_ for _ in range(self.size) if _ != index]

            # mutate the chosen dimension to create a new candidate solution
            self._mutate(self.mutated_population[current_index].vector, index, indices)

        if self.population[index].skip_exploration:
            return self.send_scout(current_index)

        return self._process_candidate(current_index, is_scout=False)

    def send_scouts(self):
        """
        Identifies worker bees whose solutions have not improved for a specified number of trials (max_trials).
        """
        # retrieves the number of trials for all bees
        trials = [self.population[i].counter for i in range(self.size)]

        # identifies the bee with the greatest number of trials
        indices = [_ for _ in range(len(trials)) if trials[_] > self.max_trials]

        # checks if its number of trials exceeds the pre-set maximum number of trials
        for index in indices:
            self.send_scout(index)

    def send_scout(self, index):
        """
        Replace a specific bee with a new scout bee, use the ML model to guide the search.

        Parameters:
            :param int index : index of the bee to be replaced
        """
        id_ = self.population[index].bee_id
        memory_ = self.population[index].memory
        self.population[index].role = 'Scout'

        avoid_memory = []
        if self.memory_type == 'local':
            if 'x' in memory_:
                avoid_memory = list(memory_['x'])

            if self.use_cache:
                for bee in self.mutated_population:
                    actual_running = bee.vector[:]
                    if not any(np.array_equal(actual_running, mem) for mem in avoid_memory):
                        avoid_memory.append(actual_running)

        elif self.memory_type == 'global':
            avoid_memory = [vec[:] for vec in self.global_mem]

        # ML model on constraints
        constr_scout_model = None
        if self.ML_approach_constr == 'local':
            bee_to_train = self.population[index]
            surrogate.build_ML_model_constraint(bee_to_train, self.min_ML_history_size, self.max_ML_history_size, self.constr_regressor_type)
            constr_scout_model = bee_to_train.constr_model
        elif self.ML_approach_constr == 'global':
            surrogate.build_ML_model_constraint_global(self, index, self.constr_regressor_type)
            constr_scout_model = self.population[index].constr_model
        
        # creates a new scout bee randomly
        self.mutated_population[index] = Bee(self.lower, self.upper, self.evaluate, self.constraints, init_model=constr_scout_model,
                                             snap_function=self.snap_function, act_iter=self.iter, std_ABC=self.std_ABC,
                                             prev_memory=memory_, avoid_memory=avoid_memory)

        self.mutated_population[index].bee_id = id_
        self.mutated_population[index].role = 'Scout'
        self.mutated_population[index].memory['role'][-1] = 'Scout'
        self.mutated_population[index].counter_idx = index

        return self._process_candidate(index, is_scout=True)

    def update_bee_status(self, index, cache_value=False):
        """
        Update the state of a bee based on the candidate solution.

        Parameters:
            :param int index        : index of the bee in the population
            :param bool cache_value : flag to see if cache is valid
        """
        if self.mutated_population[index].role == "Scout":
            self.population[index] = copy.deepcopy(self.mutated_population[index])
            if self.use_cache and cache_value:
                self.cache.add(np.round(self.population[index].vector, 8).tobytes())

        else:
            if self.mutated_population[index].fitness > self.population[index].fitness:
                self.population[index] = copy.deepcopy(self.mutated_population[index])
                self.population[index].counter = 0
                self.population[index].exclude_exploration_dim = set()
                self.population[index].skip_exploration = False
                if self.use_cache and cache_value:
                    self.cache.add(np.round(self.population[index].vector, 8).tobytes())
            else:
                counter_idx = self.mutated_population[index].counter_idx
                self.population[counter_idx].counter += 1

            memory.register_value(self.population[index],
                                  self.mutated_population[index].vector[:],
                                  self.mutated_population[index].value,
                                  self.mutated_population[index].constraint,
                                  self.mutated_population[index].role,
                                  self.iter)
