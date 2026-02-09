import pandas as pd
import csv
import os


def initialize_log_file(beehive, resume=False):
    """
    Initializes the CSV log file to store bee-related information throughout the optimization process.

    Parameters:
        :param Beehive beehive : beehive object
        :param bool resume     : flag to resume from a previous run
    """
    try:
        write_header = False
        if not resume:
            mode = "w"
            write_header = True
        else:
            mode = "a"
            if not os.path.exists(beehive.path_to_log):
                write_header = True

        # define the log file location
        beehive.log_file = open(beehive.path_to_log, mode=mode, newline="")

        # define log file headers
        fieldnames = [
            "index", "bee_ID", "bee_index", "bee_Role", "position", "fitness", "objective_function", "constraints",
            "valid", "trial_counter", "regret", "start_clock_time", "update_time", "execution_time"
        ]

        # initialize the CSV writer
        beehive.csv_writer = csv.DictWriter(beehive.log_file, fieldnames=fieldnames)

        if write_header:
            beehive.csv_writer.writeheader()

    except IOError as e:
        print(f"Error: Cannot set-up logger at {beehive.path_to_log}: {e}")
        raise


def trim_log(log_path, last_itr):
    """
    Trim log file removing all entries from given iteration onwards.

    Parameters:
        :param log_path     : path to the CSV log file
        :param last_itr     : last simulation iteration index to keep
    """
    df = pd.read_csv(log_path)
    df['index'] = pd.to_numeric(df['index'])
    df.dropna(subset=['index'])

    df['index'] = df['index'].astype(int)
    df_filtered = df[df['index'] <= last_itr]

    df_filtered.to_csv(log_path, index=False, header=True)


def log_bee(beehive, bee, bee_index, absolute_regret, actual_time=0.0, update_time=0.0, execution_time=0.0):
    """
    Logs information about a specific bee at a given iteration to the CSV log file.

    Parameters:
        :param Beehive beehive       : beehive object
        :param Bee bee               : the bee object to log
        :param int bee_index         : index of the bee in the population
        :param float absolute_regret : absolute regret of the bee's solution
        :param float actual_time     : start time
        :param float update_time     : update time
        :param float execution_time  : execution time of bee step
    """
    if beehive.csv_writer is None:
        print("Warning: CSV writer not initialized. Cannot log bee event.")
        return

    beehive.csv_writer.writerow({
        "index": beehive.iter,
        "bee_ID": bee.bee_id,
        "bee_index": bee_index,
        "bee_Role": bee.role,
        "position": bee.vector.tolist(),
        "fitness": bee.fitness,
        "objective_function": bee.value,
        "constraints": bee.constraint,
        "valid": float(bee.constraint >= 0),
        "trial_counter": bee.counter,
        "regret": absolute_regret,
        "start_clock_time": actual_time,
        "update_time": update_time,
        "execution_time": execution_time
    })


def from_log(logfile):
    """
    Read and parse a CSV log file containing bee optimization run data.

    Parameters:
        :param str logfile : path to log file.
    """
    try:
        # save the log file into a DataFrame
        data = pd.read_csv(logfile)

        # drop rows that have missing position information
        data = data.dropna(subset=["position"])

        # convert "position" column from "[x, y, z]" into separate numeric columns
        position_str = data["position"].str.strip('[]')
        positions = position_str.str.split(",", expand=True)
        roles = data["bee_Role"].str.split(" ", expand=True)
        index = data["index"]

        # convert positions into numeric values and create dimensional columns
        pos_vectors = positions.apply(pd.to_numeric)
        pos_data = pd.DataFrame(pos_vectors.values.tolist(), columns=[f"dim_{i+1}" for i in range(pos_vectors.shape[1])])
        pos_data["bee_Role"] = roles
        pos_data["index"] = index

        return pos_data

    except FileNotFoundError:
        print(f"Error: Log file not found at {logfile}")
        return pd.DataFrame()

    except Exception as e:
        print(f"Warning: Cannot read logfile {logfile}: {e}")
        return pd.DataFrame()


def log_observed_points(beehive):
    """
    Logs all observed points (solutions) from all bees' memories to a separate CSV file.

    Parameters:
        :param Beehive beehive : beehive object
    """
    all_points = {}
    counter = 0
    repetitions = [0] * beehive.max_itrs
    evaluated = [0] * beehive.max_itrs
    all_points_to_log = [
        ['bee_ID', 'iter', 'role', 'position', 'OF', 'constr_delta', 'valid', 'regret[%]', 'exec_time']]

    for bee in beehive.population:
        for idx in range(len(bee.memory['x'])):
            point_ = tuple(bee.memory['x'][idx].tolist())
            evaluated[bee.memory['iter'][idx]] += 1
            exec_time_ = beehive.dataset_obj.get_execution_time(
                bee.memory['x'][idx]) if beehive.dataset_obj is not None else 0.0
            all_points_to_log.append([bee.bee_id,
                                      bee.memory['iter'][idx],
                                      bee.memory['role'][idx],
                                      bee.memory['x'][idx].tolist(),
                                      bee.memory['f(x)'][idx],
                                      bee.memory['g(x)'][idx],
                                      bool(bee.memory['g(x)'][idx] >= 0),
                                      (bee.memory['f(x)'][idx] - beehive.global_optima) / beehive.global_optima * 100,
                                      exec_time_])

            if point_ not in all_points:
                all_points[point_] = 1
            else:
                all_points[point_] += 1
                repetitions[bee.memory['iter'][idx]] += 1

            counter += 1

    with open('output/' + beehive.log_filename[:-4] + '_all_points.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(all_points_to_log)
    # percentage_unique_points = [round((1 - repetitions[idx] / evaluated[idx])*100, 2) for idx in range(len(evaluated))]

    print("\n")
    print(f"Number of evalutaion of the OF: {counter}")
    print(f"Number of non-repeated points: {len(all_points)}")
    print(f"{round((1 - len(all_points) / counter) * 100, 2)}% of the evaluated points are repeated")
    # print(f"Percentage of non-repeated points across iterations: {percentage_unique_points}")

    return len(all_points), counter


def log_observed_points_async(beehive):
    """
    Logs observed solution points asynchronously.

    Parameters:
        :param Beehive beehive : beehive object
    """
    all_points = {}
    counter = 0

    for bee in beehive.population:
        for idx in range(len(bee.memory['x'])):

            point_ = tuple(bee.memory['x'][idx].tolist())

            if point_ not in all_points:
                all_points[point_] = 1
            else:
                all_points[point_] += 1

            counter += 1

    print("\n")
    print(f"Number of evalutaion of the OF: {counter}")
    print(f"Number of non-repeated points: {len(all_points)}")
    print(f"{round((1 - len(all_points) / counter) * 100, 2)}% of the evaluated points are repeated")

    return len(all_points), counter


def last_iteration(path):
    """
    Retrieve the index of the last completed iteration from log file.

     Parameters:
        :param str path : path to log file
    """
    if not os.path.exists(path):
        return -1

    df = pd.read_csv(path, usecols=['index'])
    if df.empty:
        return -1

    numeric_indices = pd.to_numeric(df['index']).dropna()
    if numeric_indices.empty:
        return -1
    return int(numeric_indices.max())