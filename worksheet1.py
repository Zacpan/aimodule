import random



#Class for individual candidate solutions, features genes and a fitness score
class Individual:
    #Constructor that sets up genes
    def __init__(self, gene_length):
        self.genes = [0] * gene_length
        self.fitness_score = 0

    #Checks if fitness has increased
    def calculate_fitness(self):
        self.fitness_score = sum(self.genes)

# Constants
GENE_LENGTH = 10
POPULATION_SIZE = 50


#Creates the initial populations of individuals with random genes
def create_initial_population(pop_size, gene_length):
    start_pop = []
    for _ in range(pop_size):
        random_genes = [random.randint(0, 1) for _ in range(gene_length)]
        individual = Individual(gene_length)
        individual.genes = random_genes.copy()
        individual.calculate_fitness()
        start_pop.append(individual)
    return start_pop



#Tournement selection function that chooses next generations parents
def tournament_selection(population, tournament_size):
    offspring = []

    for _ in range(len(population)):
        contestants = random.sample(population, tournament_size)
        best_contestant = max(contestants, key=lambda ind: ind.fitness_score)
        offspring.append(best_contestant)
    return offspring

#Prints populationfitness
def print_population_fitness(population, label):


    total_fitness = sum(ind.fitness_score for ind in population)
    print(f"{label} Total Fitness: {total_fitness}")

if __name__ == "__main__":
    
    population = create_initial_population(POPULATION_SIZE, GENE_LENGTH)
    print_population_fitness(population, "Initial Population")

    TOURNAMENT_SIZE = 2
    offspring_population = tournament_selection(population, TOURNAMENT_SIZE)
    print_population_fitness(offspring_population, "Offspring Population")