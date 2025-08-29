import numpy as np
import matplotlib.pyplot as plt

# 1. Define the function to optimize
def func(x):
    return x * x  # Adjusted for new range

# 2. Parameters
POP_SIZE = 50
MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.8
GENERATIONS = 10
CHROMOSOME_LENGTH = 10  # Now allows x in [0, 1023]

# 3. Decode chromosome to integer
def decode(chromosome):
    return int("".join(str(bit) for bit in chromosome), 2)

# 4. Create initial population
def create_population():
    return np.random.randint(2, size=(POP_SIZE, CHROMOSOME_LENGTH))

# 5. Evaluate fitness
def evaluate_fitness(population):
    decoded = np.array([decode(chrom) for chrom in population])
    fitness = func(decoded)
    return fitness

# 6. Selection (Roulette Wheel)
def select(population, fitness):
    min_fitness = np.min(fitness)
    if min_fitness < 0:
        fitness = fitness - min_fitness + 1e-6
    total_fitness = np.sum(fitness)
    probabilities = fitness / total_fitness
    indices = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, p=probabilities)
    return population[indices]

# 7. Crossover (Single-point)
def crossover(population):
    new_population = []
    for i in range(0, POP_SIZE, 2):
        parent1 = population[i]
        parent2 = population[(i + 1) % POP_SIZE]
        if np.random.rand() < CROSSOVER_RATE:
            point = np.random.randint(1, CHROMOSOME_LENGTH - 1)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            new_population.extend([child1, child2])
        else:
            new_population.extend([parent1, parent2])
    return np.array(new_population)

# 8. Mutation
def mutate(population):
    for i in range(POP_SIZE):
        for j in range(CHROMOSOME_LENGTH):
            if np.random.rand() < MUTATION_RATE:
                population[i, j] = 1 - population[i, j]
    return population

# 9. Main GA loop
def genetic_algorithm():
    population = create_population()
    best_solution = None
    best_fitness = -np.inf
    best_fitness_list = []

    for generation in range(GENERATIONS):
        fitness = evaluate_fitness(population)
        max_idx = np.argmax(fitness)
        current_best_fitness = fitness[max_idx]
        current_best_solution = decode(population[max_idx])

        # Update global best
        print(f"Generation {generation + 1}:  x = {current_best_solution}, f(x) = {current_best_fitness:.4f}")

        best_fitness_list.append(current_best_fitness)

        # Elitism
        elite = population[max_idx].copy()

        # GA steps
        population = select(population, fitness)
        population = crossover(population)
        population = mutate(population)

        # Preserve elite
        population[np.random.randint(POP_SIZE)] = elite

    # Plot fitness over generations
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, GENERATIONS + 1), best_fitness_list, label='Best Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Best Fitness Over Generations')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return current_best_solution, current_best_fitness

# Run the GA
best_x, best_val = genetic_algorithm()
print(f"\n🔍 Final Best Solution: x = {best_x}, f(x) = {best_val:.4f}")
