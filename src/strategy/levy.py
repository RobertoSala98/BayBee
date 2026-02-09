import numpy as np
import math

from src.strategy import memory

def levy_flight(beehive, vector, index, lambda_=1.5):
    """
    Performs a Lévy flight mutation on a single dimension of the vector (Mantegna's algorithm).

    Parameters:
        :param Beehive beehive : beehive object
        :param np.array vector : candidate solution vector
        :param int index       : index of the bee in the population
        :param float lambda_   : Lévy distribution parameter (defaults to 1.5)
    """
    current_bee = beehive.population[index]
    
    # compute standard deviation of the step distribution
    x = math.gamma(1 + lambda_)
    y = np.sin(np.pi * lambda_ / 2)
    z = math.gamma((1 + lambda_) / 2)
    w = 2 ** ((lambda_ - 1) / 2)
    
    sigma = (x * y / (z * w)) ** (1 / lambda_)

    recent_memory = memory.retrieve_recent_memory(beehive, index)

    while True:
        # generate random step using normal distribution
        # (alpha, beta) = (skewness, characteristic exponent)
        u = np.random.normal(0, sigma)
        v = np.random.normal(0, 1)
        
        step = u / abs(v) ** (1 / lambda_)

        # choose a random not excluded dimension
        available_dim = [_ for _ in range(beehive.dim) if _ not in current_bee.exclude_exploration_dim]
        if not available_dim:
            current_bee.skip_exploration = True
            break

        d = np.random.choice(available_dim)
        
        vector[d] = beehive.solution[d] + step * beehive.levy_step_size * (vector[d] - beehive.solution[d])

        # snap vector based on recent memory
        if beehive.snap_function:
            vector[:] = beehive.snap_function(np.copy(vector), recent_memory, d)

        # if no change, exclude this dimension from exploration
        if np.array_equal(vector, current_bee.vector):
            current_bee.exclude_exploration_dim.add(int(d))
            if len(current_bee.exclude_exploration_dim) == beehive.dim:
                current_bee.skip_exploration = True
                break
        else:
            break