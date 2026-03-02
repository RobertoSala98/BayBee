import numpy as np

def return_best(final_best_fitness, best_solution):
    """
    Return the best fitness value and solution values.

    Parameters:
        :param final_best_fitness   : best fitness values
        :param best_solution        : best solution values
    """
    best_run_id = np.argmin(final_best_fitness)

    allbest_fitness = final_best_fitness[best_run_id]
    allbest_solution = best_solution[best_run_id]

    print(f"\nOverall Best Fitness Value ABC, at run {best_run_id + 1}: {allbest_fitness}")
    print(f"Overall Best Solution Values ABC, at run {best_run_id + 1}: {allbest_solution}")

def return_worst(final_best_fitness, best_solution):
    """
    Return the worst fitness value and solution values.

    Parameters:
        :param final_best_fitness   : best fitness values
        :param best_solution        : best solution values
    """
    worst_run_id = np.argmax(final_best_fitness)

    allworst_fitness = final_best_fitness[worst_run_id]
    allworst_solution = best_solution[worst_run_id]

    print(f"\nOverall Worst Fitness Value ABC, at run {worst_run_id + 1}: {allworst_fitness}")
    print(f"Overall Worst Solution Values ABC, at run {worst_run_id + 1}: {allworst_solution}")

def return_avg(best_costs, abs_regret, mean_costs):
    """
    Return the average fitness value, absolute regret, and mean costs.

    Parameters:
        :param best_costs: best costs
        :param abs_regret: absolute regret
        :param mean_costs: mean costs
    """
    avg_best_cost = np.mean(best_costs, axis=0)
    avg_mean_cost = np.mean(mean_costs, axis=0)
    avg_abs_regret = np.mean(abs_regret, axis=0)
    avg_cost = {"best": avg_best_cost, "mean": avg_mean_cost, "abs_regret": avg_abs_regret}

    overall_average_fitness = np.mean(best_costs)
    overall_std_fitness = np.std(best_costs)

    print(f"\nOverall Average Fitness over all runs and iterations: {overall_average_fitness}")
    print(f"Overall Standard Deviation of Fitness: {overall_std_fitness}\n")

    return avg_cost
