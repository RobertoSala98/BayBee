import pandas as pd
import numpy as np
from scipy.spatial import KDTree

from src.utils.config import load_config


class Dataset:
    def __init__(self, filename, config_file, x_data=None, y_data=None):
        """
        Initialize the dataset object by reading data from a file. KDTree is used for efficient neighbor search.

        Parameters:
            :param str filename     : path to the dataset file.
            :param np.array x_data  : dataset coordinates values.
            :param np.array y_data  : dataset objective values.
        """
        # initialize the dataset object
        self.filename = filename
        self.x_data = x_data
        self.y_data = y_data
        self.columns = None
        self.target_column = None
        self.constraint_column = None
        self.constraint_data = None
        self.execution_time_column = None
        self.execution_time_data = None

        # read the dataset file
        self.read(config_file)

        # create a KDTree
        self.kdtree = KDTree(self.x_data)

        # find lower and upper bounds of the dataset
        self.lower_bound = np.min(self.x_data, axis=0)
        self.upper_bound = np.max(self.x_data, axis=0)

        # find the global optimum value
        if self.y_data is not None:
            filtered_y_data = self.y_data[self.constraint_data >= 0]
            self.global_optima = np.min(filtered_y_data)
        else:
            self.global_optima = None

    def read(self, config_file):
        """
        Reads the dataset from the CSV file and extracts the coordinate values (x_i) and objective function values (y).
        """
        data = pd.read_csv(self.filename, sep=',', quotechar="'")

        # Extract x_data if it exists in the dataset
        x_columns = load_config(config_file)["dataset"]["x_column"]
        self.columns = x_columns
        self.x_data = data[x_columns].values

        # Extract y_data if it exists in the dataset
        y_column = load_config(config_file)["dataset"]["y_column"]
        self.target_column = y_column
        self.y_data = data[y_column].values if y_column in data.columns else None

        # Extract constraint column if it exists
        constraint_column = load_config(config_file)["dataset"]["constraint_column"]
        self.constraint_value = load_config(config_file)["dataset"]["constraint_value"]
        self.constraint_column = constraint_column
        self.constraint_data = self.constraint_value - data[constraint_column].values if constraint_column in data.columns else None

        # Extract execution time if it exists
        execution_time_column = load_config(config_file)["dataset"]["execution_time_column"]
        self.execution_time_column = execution_time_column
        self.execution_time_data = data[execution_time_column].values if execution_time_column in data.columns else None

    def evaluator(self, vector):
        """
        Finds the closest data point in the dataset to a given vector and returns the corresponding objective value.

        Parameters:
            :param np.array vector: input vector to evaluate.
        """
        # find the index of the closest point in the dataset
        _, index = self.kdtree.query(vector, k=1)
        return self.y_data[index]

    def get_snapped_point(self, vector, exclude_vectors=None, dimension=None):
        """
        Finds the closest data point in the dataset to a given vector, excluding a specified vector if provided.

        Parameters:
            :param np.array vector          : input vector to snap.
            :param np.array exclude_vectors : vector to exclude (optional).
            :param int dimension            : dimension to snap (optional).
        """
        if exclude_vectors is None:
            exclude_vectors = {}

        if dimension is not None:
            other_dims = [i for i in range(self.x_data.shape[1]) if i != dimension]
            mask = np.all(self.x_data[:, other_dims] == vector[other_dims], axis=1)
            valid_indices = np.nonzero(mask)[0]

            if valid_indices.size > 0:
                if 'memory' in exclude_vectors and len(exclude_vectors['memory']) > 0:
                    candidate = self.x_data[valid_indices]
                    exclude_mask = np.any(np.all(candidate[:, None, :] == exclude_vectors['memory'], axis=2), axis=1)
                    valid_indices = valid_indices[~exclude_mask]

            if valid_indices.size == 0:
                return exclude_vectors['actual']

            diffs = np.abs(self.x_data[valid_indices, dimension] - vector[dimension])
            best_index = valid_indices[np.argmin(diffs)]
            return self.x_data[best_index]

        else:
            _, index  = self.kdtree.query(vector, k=1)
            return self.x_data[index]

    def get_constraint(self, vector):
        """
        Return the constraint value for the closest data point in the dataset to a given vector.

        Parameters:
            :param np.array vector: input vector to get constraint.
        """
        _, index = self.kdtree.query(vector, k=1)
        return self.constraint_data[index]

    def get_execution_time(self, vector):
        """
        Return the execution time value for the closest data point in the dataset to a given vector.

        Parameters:
            :param np.array vector: input vector to get constraint.
        """
        _, index = self.kdtree.query(vector, k=1)
        return self.execution_time_data[index]

    def get_bounds(self):
        """
        Return the lower and upper bounds for the dataset.
        """
        return self.lower_bound, self.upper_bound

    def get_global_optima(self):
        """
        Return the global optimum value.
        """
        return self.global_optima

    def get_y_column(self):
        """
        Return the objective function values.
        """
        return self.y_data
