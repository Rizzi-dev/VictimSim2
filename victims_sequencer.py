import numpy as np
import random
from sklearn.cluster import KMeans
from itertools import permutations

class Sequencer:
    def __init__(self):
        self.name = "Sequencer"

    def euclidean_distance(self, a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


    def total_distance(self, route):
        return sum(self.euclidean_distance(route[i], route[i+1])
                for i in range(len(route)-1))

    def genetic_algorithm(self, victims, population_size=50, generations=100,
                        mutation_rate=0.1, tournament_size=3):
        
        # --- precompute origin distances ---
        dist_to_origin = { pt: self.euclidean_distance(pt, (0,0)) for pt in victims }
        max_dist       = max(dist_to_origin.values())
        
        # ============ Inicialização ============
        def create_individual():
            individual = victims[:]
            random.shuffle(individual)
            return individual

        population = [create_individual() for _ in range(population_size)]

        # ============ Avaliação ============
        def fitness(individual):
            tour_length = self.total_distance(individual)

            order_penalty = sum(
                i * (max_dist - dist_to_origin[pt])
                for i, pt in enumerate(individual)
            )

            return tour_length + 2 * order_penalty

        # ============ Seleção (Torneio) ============
        def tournament_selection():
            competitors = random.sample(population, tournament_size)
            return min(competitors, key=fitness)

        # ============ Crossover (Order Crossover) ============
        def crossover(parent1, parent2):
            size = len(parent1)
            start, end = sorted(random.sample(range(size), 2))
            child = [None]*size
            child[start:end+1] = parent1[start:end+1]
            ptr = 0
            for gene in parent2:
                if gene not in child:
                    while child[ptr] is not None:
                        ptr += 1
                    child[ptr] = gene
            return child

        # ============ Mutação (Swap) ============
        def mutate(individual):
            if random.random() < mutation_rate:
                i, j = random.sample(range(len(individual)), 2)
                individual[i], individual[j] = individual[j], individual[i]

        # ============ Loop do AG ============
        best = min(population, key=fitness)

        for _ in range(generations):
            new_population = []

            for _ in range(population_size):
                parent1 = tournament_selection()
                parent2 = tournament_selection()
                child = crossover(parent1, parent2)
                mutate(child)
                new_population.append(child)

            population = new_population
            current_best = min(population, key=fitness)
            if fitness(current_best) < fitness(best):
                best = current_best

        return best, fitness(best)
