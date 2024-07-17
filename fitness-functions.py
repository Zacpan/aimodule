import random
import matplotlib.pyplot as plt

# Class for individual candidate solutions, features genes and a fitness score
class Individual:
    def __init__(self, gene_length, bounds):
        self.genes = [0.0] * gene_length
        self.bounds = bounds
        self.fitness_score = float('inf')

    def calculate_fitness(self, fitness_function):
        self.fitness_score = fitness_function(self.genes)

# Constants
GENE_LENGTH_1 = 20
GENE_LENGTH_2 = 20
POPULATION_SIZE = 50
MUTATION_RATE = 0.01
GENERATIONS = 50

# Creates the initial population of individuals with random genes
def create_initial_population(pop_size, gene_length, bounds):
    start_pop = []
    for _ in range(pop_size):
        random_genes = [random.uniform(bounds[0], bounds[1]) for _ in range(gene_length)]
        individual = Individual(gene_length, bounds)
        individual.genes = random_genes.copy()
        start_pop.append(individual)
    return start_pop

# Tournament selection function that chooses next generation's parents
def tournament_selection(population, tournament_size):
    offspring = []
    for _ in range(len(population)):
        contestants = random.sample(population, tournament_size)
        best_contestant = min(contestants, key=lambda ind: ind.fitness_score)
        offspring.append(best_contestant)
    return offspring

# Single-point crossover function
def single_point_crossover(offspring, gene_length):
    for i in range(0, len(offspring), 2):
        if i + 1 < len(offspring):
            parenta = offspring[i]
            parentb = offspring[i + 1]
            crosspoint = random.randint(1, gene_length - 1)
            child1_genes = parenta.genes[:crosspoint] + parentb.genes[crosspoint:]
            child2_genes = parentb.genes[:crosspoint] + parenta.genes[crosspoint:]
            offspring[i].genes = child1_genes
            offspring[i + 1].genes = child2_genes
    return offspring

# Bit-wise mutation function
def bitwise_mutation(offspring, mutation_rate, bounds):
    for individual in offspring:
        for i in range(len(individual.genes)):
            if random.random() < mutation_rate:
                individual.genes[i] = random.uniform(bounds[0], bounds[1])
    return offspring

# Prints population fitness
def print_population_fitness(population, label):
    total_fitness = sum(ind.fitness_score for ind in population)
    print(f"{label} Total Fitness: {total_fitness}")

# Tracks and prints fitness statistics
def track_fitness_statistics(population):
    best_fitness = min(ind.fitness_score for ind in population)
    mean_fitness = sum(ind.fitness_score for ind in population) / len(population)
    return best_fitness, mean_fitness

# Utilize matplotlib to show best fitness against mean fitness
def plot_fitness_statistics(best_fitness_over_gens, mean_fitness_over_gens):
    generations = range(len(best_fitness_over_gens))
    plt.plot(generations, best_fitness_over_gens, label='Best Fitness')
    plt.plot(generations, mean_fitness_over_gens, label='Mean Fitness')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('Fitness Progress Over Generations')
    plt.legend()
    plt.show()

# Fitness functions
def fitness_function_1(genes):
    return sum(100 * (genes[i + 1] - genes[i]**2)**2 + (1 - genes[i])**2 for i in range(len(genes) - 1))

def fitness_function_2(genes):
    return 0.5 * sum(genes[i]**4 - 16*genes[i]**2 + 5*genes[i] for i in range(len(genes)))

# Main function for optimization
def optimize(fitness_function, bounds, gene_length, pop_size=50, generations=50, mutation_rate=0.01, tournament_size=2):
    population = create_initial_population(pop_size, gene_length, bounds)
    for individual in population:
        individual.calculate_fitness(fitness_function)
    print_population_fitness(population, "Initial Population")

    best_fitness_over_gens = []
    mean_fitness_over_gens = []

    for generation in range(generations):
        offspring_population = tournament_selection(population, tournament_size)
        offspring_population = single_point_crossover(offspring_population, gene_length)
        offspring_population = bitwise_mutation(offspring_population, mutation_rate, bounds)
        for individual in offspring_population:
            individual.calculate_fitness(fitness_function)
        population = offspring_population.copy()
        best_fitness, mean_fitness = track_fitness_statistics(population)
        best_fitness_over_gens.append(best_fitness)
        mean_fitness_over_gens.append(mean_fitness)
        print_population_fitness(population, f"Generation {generation + 1}")

    plot_fitness_statistics(best_fitness_over_gens, mean_fitness_over_gens)

if __name__ == "__main__":
    # Optimize for the first fitness function
    BOUNDS_1 = (-100, 100)
    optimize(fitness_function_1, BOUNDS_1, GENE_LENGTH_1, pop_size=POPULATION_SIZE, generations=GENERATIONS, mutation_rate=MUTATION_RATE)

    # Optimize for the second fitness function
    BOUNDS_2 = (-5, 5)
    optimize(fitness_function_2, BOUNDS_2, GENE_LENGTH_2, pop_size=POPULATION_SIZE, generations=GENERATIONS, mutation_rate=MUTATION_RATE)





