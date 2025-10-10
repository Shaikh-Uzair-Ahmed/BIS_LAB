import numpy as np
import random

def initialize_pheromone(num_cities, initial_pheromone=1.0):
    return np.ones((num_cities, num_cities)) * initial_pheromone

def calculate_probabilities(pheromone, distances, visited, alpha=1, beta=2):
    pheromone = np.copy(pheromone)
    pheromone[list(visited)] = 0  # zero out visited cities

    heuristic = 1 / (distances + 1e-10)  # inverse of distance
    heuristic[list(visited)] = 0

    prob = (pheromone ** alpha) * (heuristic ** beta)
    total = np.sum(prob)
    if total == 0:
        # If no options (all visited), choose randomly among unvisited
        choices = [i for i in range(len(distances)) if i not in visited]
        return choices, None
    prob = prob / total
    return range(len(distances)), prob

def select_next_city(probabilities, cities):
    if probabilities is None:
        return random.choice(cities)
    return np.random.choice(cities, p=probabilities)

def path_length(path, distances):
    length = 0
    for i in range(len(path)):
        length += distances[path[i-1]][path[i]]
    return length

def ant_colony_optimization(distances, n_ants=5, n_iterations=50, decay=0.5, alpha=1, beta=2):
    num_cities = len(distances)
    pheromone = initialize_pheromone(num_cities)
    best_path = None
    best_length = float('inf')

    for iteration in range(n_iterations):
        all_paths = []
        for _ in range(n_ants):
            path = [0]  # start at city 0
            visited = set(path)

            for _ in range(num_cities - 1):
                current_city = path[-1]
                cities, probabilities = calculate_probabilities(pheromone[current_city], distances[current_city], visited, alpha, beta)
                next_city = select_next_city(probabilities, cities)
                path.append(next_city)
                visited.add(next_city)

            length = path_length(path, distances)
            all_paths.append((path, length))

            if length < best_length:
                best_length = length
                best_path = path

        # Evaporate pheromone
        pheromone *= (1 - decay)

        # Deposit pheromone proportional to path quality
        for path, length in all_paths:
            deposit = 1 / length
            for i in range(len(path)):
                pheromone[path[i-1]][path[i]] += deposit

    return best_path, best_length

# Example usage
if __name__ == "__main__":
    distances = np.array([
        [np.inf, 2, 2, 5, 7],
        [2, np.inf, 4, 8, 2],
        [2, 4, np.inf, 1, 3],
        [5, 8, 1, np.inf, 2],
        [7, 2, 3, 2, np.inf]
    ])

    best_path, best_length = ant_colony_optimization(distances)
    print(f"Best path: {[int(city) for city in best_path]} with length: {best_length:.2f}")
