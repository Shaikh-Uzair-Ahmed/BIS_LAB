import numpy as np
import math

def knapsack_fitness(solution, values, weights, capacity):
    """Calculate fitness: total value if weight within capacity, else zero."""
    total_weight = np.sum(solution * weights)
    if total_weight > capacity:
        return 0  # Penalize overweight solutions
    return np.sum(solution * values)

def levy_flight(Lambda, size):
    """Generate Levy flight steps."""
    sigma = (math.gamma(1 + Lambda) * math.sin(math.pi * Lambda / 2) /
             (math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
    u = np.random.normal(0, sigma, size)
    v = np.random.normal(0, 1, size)
    step = u / (np.abs(v) ** (1 / Lambda))
    return step

def sigmoid(x):
    """Sigmoid function for mapping continuous to probability."""
    return 1 / (1 + np.exp(-x))

def cuckoo_search_knapsack(values, weights, capacity, n_nests=25, max_iter=100, pa=0.25):
    """
    Cuckoo Search for 0/1 Knapsack Problem.

    Args:
        values: numpy array of item values
        weights: numpy array of item weights
        capacity: max capacity of knapsack
        n_nests: number of nests (population size)
        max_iter: max iterations
        pa: probability of abandoning nests

    Returns:
        best_solution: binary numpy array with item selection
        best_fitness: total value of best_solution
    """
    n_items = len(values)
    nests = np.random.randint(0, 2, size=(n_nests, n_items))
    fitness = np.array([knapsack_fitness(n, values, weights, capacity) for n in nests])

    best_idx = np.argmax(fitness)
    best_solution = nests[best_idx].copy()
    best_fitness = fitness[best_idx]

    Lambda = 1.5  # Levy flight exponent

    for iteration in range(max_iter):
        for i in range(n_nests):
            step = levy_flight(Lambda, n_items)
            current = nests[i].astype(float)
            new_solution_cont = current + step
            probs = sigmoid(new_solution_cont)
            new_solution_bin = (probs > 0.5).astype(int)

            new_fitness = knapsack_fitness(new_solution_bin, values, weights, capacity)

            # Greedy selection
            if new_fitness > fitness[i]:
                nests[i] = new_solution_bin
                fitness[i] = new_fitness

                if new_fitness > best_fitness:
                    best_fitness = new_fitness
                    best_solution = new_solution_bin.copy()

        # Abandon worst nests with probability pa
        n_abandon = int(pa * n_nests)
        if n_abandon > 0:
            abandon_indices = np.random.choice(n_nests, n_abandon, replace=False)
            for idx in abandon_indices:
                nests[idx] = np.random.randint(0, 2, n_items)
                fitness[idx] = knapsack_fitness(nests[idx], values, weights, capacity)

        # Update global best after abandonment
        current_best_idx = np.argmax(fitness)
        if fitness[current_best_idx] > best_fitness:
            best_fitness = fitness[current_best_idx]
            best_solution = nests[current_best_idx].copy()

        # Print progress: every 10 iterations and first iteration
        if iteration == 0 or (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}/{max_iter}, Best Fitness: {best_fitness}")

    return best_solution, best_fitness

if __name__ == "__main__":
    # Example knapsack problem
    values = np.array([60, 100, 120, 80, 30])
    weights = np.array([10, 20, 30, 40, 50])
    capacity = 100

    best_sol, best_val = cuckoo_search_knapsack(values, weights, capacity, n_nests=30, max_iter=100, pa=0.25)

    print("\nBest solution found:")
    print(best_sol)
    print("Total value:", best_val)
    print("Total weight:", np.sum(best_sol * weights))
