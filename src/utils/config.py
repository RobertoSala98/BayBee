import json
import os

def create_folder(title):
    """
    Create an output folder if it does not exist, used to store generated plots and results.

    Parameters:
        :param str title : name of the folder
    """
    # define the output directory path
    output_dir = os.path.join("output", title)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def load_config(config_file):
    """
    Load the configuration file containing the evaluator function and global optima value.

    Parameters:
        :param str config_file : path to the configuration file.
    """
    try:
        with open(config_file, "r") as file:
            config = json.load(file)
            if "evaluator" not in config or "global_optimum" not in config:
                raise ValueError("Config file must contain 'evaluator' and 'global_optimum'!")
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{config_file}' not found!")
