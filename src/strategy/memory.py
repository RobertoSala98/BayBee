import pickle

def register_value(bee, x, obj_fun, constr_value, role, act_iter):
    """
    Registers current provided values in the bee's memory to track optimization history.

    Parameters:
        :param Bee bee              : the Bee class instance
        :param list x               : current solution vector
        :param float obj_fun        : objective function value
        :param float constr_value   : constraint function value
        :param str role             : current role of the bee
        :param int act_iter         : current iteration index
    """
    bee.memory["x"].append(x.copy())
    bee.memory["f(x)"].append(obj_fun)
    bee.memory["g(x)"].append(constr_value)
    bee.memory["role"].append(role)
    bee.memory["iter"].append(act_iter)


def retrieve_recent_memory(bee, index):
    """
    Retrieves the recent memory of a bee, up to the last 'Initialization' or 'Scout' role.
    This memory is used to exclude recently visited points during exploration.

    Parameters:
        :param Bee bee    : the Bee class instance
        :param int index  : index of the bee in the population.
    """
    bee_instance = bee.population[index]
    exclude_vectors = {"actual": bee_instance.vector[:].copy(), "memory": []}

    if bee.memory_type == 'local':
        memory_vector, cache_vector = [], []
        seen_vector = set()

        for mem_vector in reversed(bee_instance.memory['x']):
            mem_vector_tuple = tuple(mem_vector.tolist())
            if mem_vector_tuple not in seen_vector:
                memory_vector.append(mem_vector)
                seen_vector.add(mem_vector_tuple)

        if bee.use_cache:
            cache_vector = [mutant.vector for mutant in bee.mutated_population]
        else:
            cache_vector = [bee.mutated_population[index].vector]

        for cur in cache_vector:
            cur_tuple = tuple(cur.tolist())
            if cur_tuple not in seen_vector:
                memory_vector.append(cur)
                seen_vector.add(cur_tuple)

        exclude_vectors["memory"] = memory_vector

    elif bee.memory_type == 'global':
        exclude_vectors["memory"] = [vec.copy() for vec in bee.global_mem]

    elif bee.memory_type is None:
        exclude_vectors["memory"] = []

    else:
        raise ValueError("Invalid memory_type. You have to select either a 'local' or 'global' approach for memory retrieval.")

    return exclude_vectors


def save_checkpoint(beehive, path):
    """
    Save current state of the beehive without considering logging attributes.

    Parameters:
        :param Beehive beehive  : the Beehive class instance
        :param str path         : path to save the checkpoint
    """
    log_file = beehive.log_file
    csv_writer = beehive.csv_writer
    beehive.log_file = None
    beehive.csv_writer = None

    try:
        with open(path, 'wb') as f:
            pickle.dump(beehive, f)

    except Exception as e:
        print(f"Warning: Failed to save checkpoint {path}: {e}")

    finally:
        beehive.log_file = log_file
        beehive.csv_writer = csv_writer


def load_checkpoint(path):
    """
    Load a previously saved beehive checkpoint from a file.

    Parameters:
        :param str path  : path to save the checkpoint
    """
    try:
        with open(path, 'rb') as f:
            beehive = pickle.load(f)
        return beehive

    except Exception as e:
        raise IOError(f"Exception: Cannot load checkpoint {path}: {e}")
