from time import time
import numpy as np
import os
import json

import matplotlib.pyplot as plt

from src import core_abc
from src.strategy import memory
from src.utils import plot, printing, config, logging

plt.ioff() # Turn off interactive mode for plotting to avoid displaying figures


def bee_algorithm(title, num_runs, num_bees, max_itrs, max_trials, global_optima, lower, upper, evaluator, constraints,
                  ML_approach_constr, ML_approach_target, regressor_type, BO_approach, min_hist, max_hist, levy_step_size, dataset_obj=None, snap_function=None,
                  visualize_behaviour=False, BO_steps=1, async_op=False, std_ABC=False, seeds=[], memory_type='local', cache=False, resume=False):
    """
    Sets up and runs the Artificial Bee Colony (ABC) optimization algorithm for a specified number of runs.
    It handles the initialization of the Beehive, execution of the optimization, logging of results,
    and generation of various performance plots.

    Parameters:
        :param str title                    : Base name for the output folder and log files
        :param int num_runs                 : Number of independent times the ABC algorithm will be run
        :param int num_bees                 : Number of bees in the hive (population size)
        :param int max_itrs                 : Maximum number of iterations for each run
        :param int max_trials               : Maximum number of trials a bee can attempt to improve its solution before becoming a scout
        :param float global_optima          : The known global optimum value of the objective function, used for regret calculation
        :param list lower                   : A list or numpy array representing the lower bounds for each dimension of the search space
        :param list upper                   : A list or numpy array representing the upper bounds for each dimension of the search space
        :param function evaluator           : The objective function to be minimized. It takes a solution vector and returns a scalar value
        :param function constraints         : A function that evaluates constraint satisfaction for a solution vector
        :param str ML_approach_constr       : Specifies the machine learning approach for constraint handling
        :param str ML_approach_target       : Specifies the machine learning approach for guiding the target function search
        :param bool BO_approach             : If True, Bayesian Optimization will be used to refine solutions
        :param regressor_type               : Type of regression to use for ML models
        :param int min_hist                 : Minimum history size required to build ML models
        :param int max_hist                 : Maximum history size to use for building ML models
        :param float levy_step_size         : Size of the step to be used in levy distribution
        :param Dataset dataset_obj          : An instance of the Dataset class, used if the optimization is performed on a discrete dataset
        :param function snap_function       : Function for snapping solutions. Defaults to None
        :param int BO_steps                 : Number of Bayesian Optimization steps to perform if BO_approach is True
        :param bool async_op                : If True, run the asynchronous version of the Beehive algorithm
        :param bool std_ABC                 : Use the standard ABC algorithm (non-asynchronous). Defaults to False
        :param list seeds                   : List of seeds for each run. Defaults to []
        :param str memory_type              : Define which memory (local or global) each bee should use during simulation
        :param bool cache                   : Automatically set cache to 'True' if used memory is local
        :param bool resume                  : If True, the simulation will resume from the last saved checkpoint
    """
    # create or overwrite the folder to store results into
    config.create_folder(title)

    # initialize arrays to store the results
    num_entries = (2 * num_bees * max_itrs) if async_op else max_itrs
    best_costs = np.zeros((num_runs, num_entries))
    mean_costs = np.zeros((num_runs, num_entries))
    abs_regrets = np.zeros((num_runs, num_entries))

    best_solution = []
    best = []

    # execute multiple runs of the ABeeC optimization algorithm
    for run_id in range(num_runs):
        # create directory for each run
        run_folder = f"abc_run{run_id + 1}"
        run_dir = os.path.join(title, run_folder)
        config.create_folder(run_dir)

        # define log filename
        log_filename = f"{run_dir}/bee_logrun_{run_id + 1}.csv"

        start_itr = 0
        resume_run = False
        model_run = None

        if resume:
            last_itr_log = os.path.join("output", log_filename)
            last_itr = logging.last_iteration(last_itr_log)
            if last_itr >= 0:
                to_checkpoint = os.path.join("output", run_dir, "checkpoint.pkl")
                checkpoint_model = memory.load_checkpoint(to_checkpoint)

                resume_itr = checkpoint_model.iter
                logging.trim_log(last_itr_log, resume_itr)

                model_run = checkpoint_model
                model_run.log_file = log_filename
                model_run.path_to_log = last_itr_log

                logging.initialize_log_file(model_run, resume=True)

                start_itr = model_run.iter + 1
                resume_run = True

        if not resume_run:
            model_run = core_abc.Beehive(
                lower=lower,
                upper=upper,
                objective_function=evaluator,
                constraint_function=constraints,
                numb_bees=num_bees,
                max_itrs=max_itrs,
                max_trials=max_trials,
                log_filename=log_filename,
                global_optima=global_optima,
                ML_approach_constr=ML_approach_constr,
                ML_approach_target=ML_approach_target,
                regressor_type = regressor_type,
                BO_approach=BO_approach,
                min_ML_history_size=min_hist,
                max_ML_history_size=max_hist,
                dataset_obj=dataset_obj,
                snap_function=snap_function,
                BO_steps=BO_steps,
                std_ABC=std_ABC,
                seed=seeds[run_id],
                memory_type=memory_type,
                cache=cache,
                levy_step_size=levy_step_size,
                resume_sim=resume
                )

        model = model_run

        # run the optimization and save results
        start_time = time()
        try:
            if async_op:
                cost, non_repeated_points, all_points_, global_time = model.run_async(run_dir)
            else:
                cost, non_repeated_points, all_points_, global_time = model.run(start_itr, run_dir)
            plot.plot_utilization(run_dir, log_filename)

            print(f"Run: {run_id + 1}, Execution time: {round(time() - start_time, 2)} seconds, Seed: {seeds[run_id]}")

        except Exception as e:
            print(f"Exception: Error occurred during model execution for run {run_id + 1}: {e}")
            continue

        try:
            best_costs[run_id, :] = cost["best"]
            mean_costs[run_id, :] = cost["mean"]
            abs_regrets[run_id, :] = cost["abs_regret"]
        
        except ValueError as e:
            print(f"Error: Shape mismatch for cost data in run {run_id + 1}: {e}. Skipping result storage.")
            continue

        best_solution.append(model.solution)
        cur_mapr = round((model.best - global_optima) / abs(global_optima) * 100, 2)
        print(f"Run {run_id + 1}/{num_runs}, Best Fitness Value ABC: {model.best}, MAPR: {cur_mapr}%")

        best.append(model.best)

        data_to_store = {
            "MAPR": (model.best - global_optima) / abs(global_optima),
            "Evaluated points": all_points_,
            "Non-repeated points": non_repeated_points,
            "Global time": global_time
        }

        with open(f"output/{run_dir}/summary.json", "w", encoding="utf-8") as f:
            json.dump(data_to_store, f, indent=4)

    # after completing all runs, produce plots and statistics for each run
    avg_best_fitness = np.mean(best)
    avg_mapr = round((avg_best_fitness - global_optima)/abs(global_optima)*100, 2)
    print(f"\nAverage best costs: {avg_best_fitness}, MAPR: {avg_mapr}%")

    # get best and worst run
    if len(best) < num_runs:
        print(f"Warning: Some runs may have failed. Only {len(best)} successful runs, out of {num_runs}.")
        if best:
            printing.return_best(np.array(best), best_solution)
            printing.return_worst(np.array(best), best_solution)
    else:
        final_best_fitness = best_costs[:, -1]
        printing.return_best(final_best_fitness, best_solution)
        printing.return_worst(final_best_fitness, best_solution)

    # get average fitness values
    mean_costs = printing.return_avg(best_costs, abs_regrets, mean_costs)
    # plot the avg convergence of the algorithm
    plot.convergence_plot(mean_costs, title, len(best))

    if visualize_behaviour:
        for run_id in range(len(best)):
            # define the output directory for each run
            output_dir = f"output/{title}/abc_run{run_id + 1}"
            log_filepath = str(f"{output_dir}/bee_logrun_{run_id + 1}.csv")

            if not os.path.exists(log_filepath):
                print(f"Warning: Log file not found for visualization: {log_filepath}. Skipping run {run_id + 1}.")
                continue

            # parse log file
            try:
                position_data = logging.from_log(log_filepath)
            except Exception as e:
                print(f"Error parsing log file {log_filepath}: {e}. Skipping visualization for this run.")
                continue

            # generate plots to visualize the distribution and behavior of bees
            try:
                plot.bee_plot(position_data, title, output_dir)
                plot.bee_plot_detail(position_data, title, output_dir)
                plot.bee_plot_behaviour(position_data, title, output_dir)
            except Exception as e:
                print(f"Error generating plots for run {run_id + 1}: {e}")
