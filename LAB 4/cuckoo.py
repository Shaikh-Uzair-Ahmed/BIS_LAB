import numpy as np

def knapsack_fitness(solution, values, weights, capacity):
    total_weight = np.sum(solution * weights)
    total_value = np.sum(solution * values)
    if total_weight > capacity:
        return 0  # Penalize overweight solutions
    else:
        return total_value

def levy_flight(Lambda, size):
    # Generate step sizes using Levy distribution for exploration
    sigma = (np.math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
            (np.math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
    u = np.random.normal(0, sigma, size)
    v = np.random.normal(0, 1, size)
    step = u / (np.abs(v) ** (1 / Lambda))
    return step

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def cuckoo_search_knapsack(values, weights, capacity, n_nests=25, max_iter=100, pa=0.25):
    n_items = len(values)
    # Initialize nests randomly (binary solutions)
    nests = np.random.randint(0, 2, (n_nests, n_items))

    fitness = np.array([knapsack_fitness(nest, values, weights, capacity) for nest in nests])
    best_idx = np.argmax(fitness)
    best_nest = nests[best_idx].copy()
    best_fitness = fitness[best_idx]

    Lambda = 1.5  # Levy flight parameter

    for iter in range(max_iter):
        # Generate new solutions by Levy flights
        for i in range(n_nests):
            step = levy_flight(Lambda, n_items)
            # Create a new solution by adding Levy step to the current solution
            # Convert current binary solution to continuous for Levy flight
            current = nests[i].astype(float)
            new_solution = current + step

            # Convert continuous values back to binary using sigmoid + threshold
            probs = sigmoid(new_solution)
            new_solution_bin = (probs > 0.5).astype(int)

            new_fitness = knapsack_fitness(new_solution_bin, values, weights, capacity)

            # If new solution is better, replace
            if new_fitness > fitness[i]:
                nests[i] = new_solution_bin
                fitness[i] = new_fitness

                # Update global best
                if new_fitness > best_fitness:
                    best_fitness = new_fitness
                    best_nest = new_solution_bin.copy()

        # Abandon some nests and create new ones (with probability pa)
        n_abandon = int(pa * n_nests)
        abandon_indices = np.random.choice(n_nests, n_abandon, replace=False)
        for idx in abandon_indices:
            nests[idx] = np.random.randint(0, 2, n_items)
            fitness[idx] = knapsack_fitness(nests[idx], values, weights, capacity)

        # Update global best after abandonment
        current_best_idx = np.argmax(fitness)
        if fitness[current_best_idx] > best_fitness:
            best_fitness = fitness[current_best_idx]
            best_nest = nests[current_best_idx].copy()

        # Optional: print progress
        # print(f"Iteration {iter+1}, Best Fitness: {best_fitness}")

    return best_nest, best_fitness

# Example usage:
if __name__ == "__main__":
    values = np.array([60, 100, 120])
    weights = np.array([10, 20, 30])
    capacity = 50

    best_solution, best_value = cuckoo_search_knapsack(values, weights, capacity, n_nests=30, max_iter=200)
    print("Best solution:", best_solution)
    print("Total value:", best_value)
    print("Total weight:", np.sum(best_solution * weights))
