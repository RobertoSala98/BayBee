from src.dMALIBOO.Dataset import Dataset
from src.dMALIBOO.GaussianProcess import GP
from src.dMALIBOO.AcquisitionFunction import AF
from src.dMALIBOO.BayesianOptimization import BO
from src.dMALIBOO.Logger import Logger

import numpy as np
import os

def dummy_function(_X):
    raise RuntimeError("Objective should not be called in dataset mode.")


class BOmanager(object):
    """
    Manager class for running Bayesian Optimization using the D-MALIBOO approach.
    """

    def __init__(self, f, dataset, constraints_bounds, seed, output_path, g=dummy_function, bounds={}, discrete_values=[], x_columns=[], target_column="", constraint_column="", init_points=None, n_iter=1, log_name='results'):
        """
        Initialize the Bayesian Optimization manager.

        Parameters:
            :param function f               : function to optimize
            :param dict bounds             : dictionary of parameter bounds
            :param str dataset              : path to dataset
            :param str target_column        : target column to optimize
            :param str constraint_column    : constraint column to constrain
            :param int seed                 : random seed for reproducibility
            :param str output_path          : path to output directory
            :param iterable init_points     : initial points for optimization
            :param int n_iter               : number of iterations for optimization
            :param str log_name             : name of log file
        """
        output_path = output_path + '/BO_logs/'

        if os.path.exists(f"{output_path}/{log_name}"):
            os.remove(f"{output_path}/{log_name}")

        os.makedirs(output_path, exist_ok=True)

        if dataset is not None:
            ds = Dataset.from_file(dataset, x_cols=x_columns, y_col=target_column, g_cols=[constraint_column])
            bounds = np.column_stack([ds.X.min(axis=0), ds.X.max(axis=0)])

            gp = GP(kernel_name="Matern", random_state=seed, length_scale=1.0, nu=2.5)()

            af = AF(kind="ei", kappa=1.0,
                    ml_on_bounds=True, ml_on_bounds_parameters={"name": "ridge", "task": "regression", "constraint_bounds": constraints_bounds},
                    ml_on_target=True, ml_on_target_parameters={"name": "ridge", "task": "regression"},
                    bounds=bounds,
                    random_state=seed,
                )

            logger = Logger(dim=ds.dim)
            
            bo = BO(objective_function = f, domain_bounds = bounds, gp = gp, af = af, constraint_functions = None, initial_points = 5, random_state = seed, logger = logger, dataset = ds)

        else:
            gp = GP(kernel_name="Matern", random_state=seed, length_scale=1.0, nu=2.5)()

            af = AF(kind="ei", kappa=1.0,
                    ml_on_bounds=True, ml_on_bounds_parameters={"name": "ridge", "task": "regression", "constraint_bounds": constraints_bounds},
                    ml_on_target=True, ml_on_target_parameters={"name": "ridge", "task": "regression"},
                    bounds=bounds,
                    random_state=seed,
                )

            logger = Logger(dim=len(bounds))

            bo = BO(objective_function = f, domain_bounds = bounds, gp = gp, af = af, constraint_functions = [g], initial_points = 5, random_state = seed, logger = logger)
        
        bo.initialize(X0=init_points)
        bo.run(n_iterations=n_iter, verbose=False)
        bo.logger.to_csv(os.path.join(output_path, log_name + ".csv"), "mape", "accuracy")

        self.all_points = bo.X_train
        self.points = bo.X_train[-len(init_points):]
        self.target = bo.y_train[-len(init_points):]
        if dataset is not None:
            self.constr = constraints_bounds[0][1] - bo.G_train[-len(init_points):]
        else:
            self.constr = bo.G_train[-len(init_points):]
        self.feasibility = self.constr > 0

        best_idx = bo.best_idx

        if best_idx is not None:
            self.best_solution = bo.X_train[best_idx]
            self.best_target = bo.y_train[best_idx]
            self.best_feasible = True
            self.best_constraint = constraints_bounds[0][1] - bo.G_train[best_idx]
        else:
            self.best_solution = None
            self.best_target = None
            self.best_feasible = None
            self.best_constraint = None


if __name__ == "__main__":
    # For testing purposes
    X_init = np.array([
        [ 8,  8, 1, 128, 1024, 256, 1, 20],
        [32, 12, 2, 192,  256,  10, 4, 10],
        [72, 12, 1, 256,  256,  30, 4, 50],
        [72, 48, 3, 224,  256,  30, 2,  2],
        [20, 48, 1,  64,  256,  10, 1, 20],
        [16, 32, 2, 128, 1024,  10, 3,  1],
        [12,  8, 2, 160,  256,  30, 4,  2],
        [16, 24, 1, 128, 1024, 256, 1, 20],
        [32, 20, 1, 224,  256,  50, 1,  2],
        [20, 72, 3, 256, 1024,  30, 3, 20],
    ], dtype=float)

    BOmanager(
        f = None,
        dataset = "resources/ligen.csv",
        x_columns = ['ALIGN_SPLIT', 'OPTIMIZE_SPLIT', 'OPTIMIZE_REPS', 'CUDA_THREADS', 'N_RESTART', 'CLIPPING', 'SIM_THRESH', 'BUFFER_SIZE'],
        target_column = 'RMSD^3*TIME',
        constraint_column = 'RMSD_0.75',
        constraints_bounds = (0, 2.1),
        seed = 1,
        output_path = "outputs/test",
        init_points = X_init
    )