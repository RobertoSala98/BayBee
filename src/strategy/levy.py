import numpy as np
import math

from src.strategy import memory

def levy_flight(beehive, vector, index, beta=1.5, eps=1e-12):
    """
    Lévy flight mutation on a single randomly chosen dimension (Mantegna's algorithm).

    Parameters:
        beehive: beehive object
        vector (np.ndarray): candidate solution vector (mutated in place)
        index (int): index of the bee in the population
        beta (float): Lévy stability parameter (a.k.a. exponent), typically in (1, 2]
        eps (float): guard for division by ~0 in v
        atol_nochange (float): tolerance used to decide if mutation had no effect
    """
    current_bee = beehive.population[index]

    # --- Mantegna sigma_u for symmetric alpha-stable (beta) ---
    # sigma_u = [ Γ(1+beta) sin(pi*beta/2) / ( Γ((1+beta)/2) * beta * 2^((beta-1)/2) ) ]^(1/beta)
    x = math.gamma(1.0 + beta)
    y = math.sin(math.pi * beta / 2.0)
    z = math.gamma((1.0 + beta) / 2.0)
    w = 2.0 ** ((beta - 1.0) / 2.0)
    sigma_u = (x * y / (z * beta * w)) ** (1.0 / beta)

    recent_memory = memory.retrieve_recent_memory(beehive, index)

    # Try until a dimension actually changes, or until all dims are excluded
    while True:
        available_dim = [d for d in range(beehive.dim)
                         if d not in current_bee.exclude_exploration_dim]
        if not available_dim:
            current_bee.skip_exploration = True
            return vector

        d = int(np.random.choice(available_dim))

        # Sample Lévy step s = u / |v|^(1/beta)
        u = np.random.normal(0.0, sigma_u)
        v = np.random.normal(0.0, 1.0)
        while abs(v) < eps:
            v = np.random.normal(0.0, 1.0)
        step = u / (abs(v) ** (1.0 / beta))

        best = beehive.solution

        if best is None:
            # randomly move by step * beehive.levy_step_size * random(lb, ub)
            vector[d] = vector[d] + step * beehive.levy_step_size * np.random.uniform(-1, 1)
        else:
            vector[d] = best[d] + step * beehive.levy_step_size * (vector[d] - best[d])

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