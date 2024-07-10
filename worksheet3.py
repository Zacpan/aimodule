import random
import matplotlib.pyplot as plt

#Class for individual candidate solutions, features genes and a fitness score
class Individual:
    def __init__(self, gene_length):
        self.genes = [0] * gene_length
        self.fitness_score = 0

    def calculate_fitness(self):
        self.fitness_score = sum(self.genes)

#Constants
GENE_LENGTH = 50
POPULATION_SIZE = 50
MUTATION_RATE = 0.01
GENERATIONS = 50

#Creates the initial population of individuals with random genes
def create_initial_population(pop_size, gene_length):
    start_pop = []
    for _ in range(pop_size):
        random_genes = [random.randint(0, 1) for _ in range(gene_length)]
        individual = Individual(gene_length)
        individual.genes = random_genes.copy()
        individual.calculate_fitness()
        start_pop.append(individual)
    return start_pop

#Tournament selection function that chooses next generation's parents
def tournament_selection(population, tournament_size):
    offspring = []
    for _ in range(len(population)):
        contestants = random.sample(population, tournament_size)
        best_contestant = max(contestants, key=lambda ind: ind.fitness_score)
        offspring.append(best_contestant)
    return offspring

#Single-point crossover function
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

#Bit-wise mutation function
def bitwise_mutation(offspring, mutation_rate, gene_length):
    for individual in offspring:
        for i in range(gene_length):
            if random.random() < mutation_rate:
                individual.genes[i] = 1 if individual.genes[i] == 0 else 0
    return offspring

#Prints population fitness
def print_population_fitness(population, label):
    total_fitness = sum(ind.fitness_score for ind in population)
    print(f"{label} Total Fitness: {total_fitness}")

#Tracks and prints fitness statistics
def track_fitness_statistics(population):
    best_fitness = max(ind.fitness_score for ind in population)
    mean_fitness = sum(ind.fitness_score for ind in population) / len(population)
    return best_fitness, mean_fitness



#Utilize matplotlib to show best fitness against mean fitness
def plot_fitness_statistics(best_fitness_over_gens, mean_fitness_over_gens):
    generations = range(len(best_fitness_over_gens))
    plt.plot(generations, best_fitness_over_gens, label='Best Fitness')
    plt.plot(generations, mean_fitness_over_gens, label='Mean Fitness')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('Fitness Progress Over Generations')
    plt.legend()
    plt.show()


#Main function
if __name__ == "__main__":
    population = create_initial_population(POPULATION_SIZE, GENE_LENGTH)
    print_population_fitness(population, "Initial Population")

    TOURNAMENT_SIZE = 2
    best_fitness_over_gens = []
    mean_fitness_over_gens = []




    for generation in range(GENERATIONS):
        offspring_population = tournament_selection(population, TOURNAMENT_SIZE)
        offspring_population = single_point_crossover(offspring_population, GENE_LENGTH)
        offspring_population = bitwise_mutation(offspring_population, MUTATION_RATE, GENE_LENGTH)
        
        for individual in offspring_population:
            individual.calculate_fitness()
        
        population = offspring_population.copy()
        
        
        best_fitness, mean_fitness = track_fitness_statistics(population)
        best_fitness_over_gens.append(best_fitness)
        mean_fitness_over_gens.append(mean_fitness)
        
        print_population_fitness(population, f"Generation {generation + 1}")

    plot_fitness_statistics(best_fitness_over_gens, mean_fitness_over_gens)
