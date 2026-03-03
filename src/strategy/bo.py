import os
import copy
import numpy as np
from src.bo_manager import BOmanager as BO
from src.strategy import memory


def apply_BO(beehive, index, role):
    """
    Applies Bayesian Optimization (BO) to refine the solution of a specific bee.
    
    Parameters:
        :param Beehive beehive : the Beehive class instance
        :param int index       : index of the bee in the population
        :param str role        : the current role of the bee
    """

    _init_points = np.asarray(beehive.population[index].memory['x'])
    log_dir = os.path.dirname(beehive.path_to_log) if beehive.path_to_log else "."

    if beehive.memory_type == 'global':
        _memory = memory.retrieve_recent_memory(beehive, index)
    else:
        _memory = None

    if beehive.dataset_obj is None:
        BO_model = BO(  f=beehive.evaluate,
                        g=beehive.constraints,
                        discrete_values = beehive.discrete_values,
                        bounds=np.column_stack([beehive.lower, beehive.upper]),
                        dataset=None,
                        constraints_bounds=[(0, np.inf)],
                        seed=beehive.seed,
                        output_path=log_dir,
                        init_points=_init_points,
                        n_iter=1,
                        log_name=f'BO_Bee_{index}_iter_{beehive.iter}_{role}',
                        memory=_memory,
                        constr_regressor_type=beehive.constr_regressor_type,
                        target_regressor_type=beehive.target_regressor_type)

    else:
        def dummy_function(_X):
            raise RuntimeError("Objective should not be called in dataset mode.")

        BO_model = BO(  f=dummy_function,
                        dataset=beehive.dataset_obj.filename,
                        x_columns=beehive.dataset_obj.columns,
                        target_column=beehive.dataset_obj.target_column,
                        constraint_column=beehive.dataset_obj.constraint_column,
                        constraints_bounds=[(0, beehive.dataset_obj.constraint_value)],
                        seed=beehive.seed,
                        output_path=log_dir,
                        init_points=_init_points,
                        n_iter=1,
                        log_name=f'BO_Bee_{index}_iter_{beehive.iter}_{role}',
                        memory=_memory,
                        constr_regressor_type=beehive.constr_regressor_type,
                        target_regressor_type=beehive.target_regressor_type)

    beehive.population[index].BO_to_do -= 1

    beehive.mutated_population[index] = copy.deepcopy(beehive.population[index])
    beehive.mutated_population[index].vector[:] = BO_model.points
    beehive.mutated_population[index].value = float(BO_model.target)
    beehive.mutated_population[index].constraint = float(BO_model.constr)
    beehive.mutated_population[index]._fitness()
    beehive.mutated_population[index].role = 'Worker_BO'
    beehive.mutated_population[index].counter_idx = index

    if beehive.memory_type == 'global':
        beehive.global_mem.append(beehive.mutated_population[index].vector[:])

    if beehive.use_cache and (beehive.mutated_population[index].vector.tobytes() in beehive.cache):
        return 0.0

    if beehive.dataset_obj is not None:
        return beehive.dataset_obj.get_execution_time(beehive.mutated_population[index].vector)
    else:
        return 0.0