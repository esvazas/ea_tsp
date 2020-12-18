import os
import datetime
import time
import psutil
import random
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import kendalltau
from math import isinf, isnan

import Reporter


# a constant of the big value we use instead of inf
BIG_VALUE = 1.7976931348623157e+300

# Modify the class name to match your student number.
class r0769473:
    n = 10
    # iteration when k-opt and inversion displacement mutation should turn on
    local_iteration_num = 5
    selection_iteration_num = 5

    # Convergence parameters
    conv_tolerance = 1e-5  # convergence tolerance
    conv_el = 7  # elements to consider for checking
    max_iterations = 100  # max iterations until stop


    def __init__(self, mu=260, lam=60, report_memory=False, create_convergence_plot=False,
                 create_div_plot=False):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.mem_recording = report_memory
        self.convergency_plot = create_convergence_plot
        self.diversity_plot = create_div_plot
        self.population_size = mu
        self.offspring_size = lam


    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")  # probably a n x n matrix (n number of cities)
        file.close()

        self.n = distanceMatrix.shape[0]  # number of cities
        i = 0
        obj_values = [0] * self.conv_el

        # Create init population and mutation/recombination probabilities
        population = self.initialization()

        # Evaluation of initial population
        population_fitnesses = [self.fitness(distanceMatrix, i) for i in population]
        meanObjective = np.mean(population_fitnesses)
        bestObjective = np.min(population_fitnesses)
        print(population.shape)
        print("Mean objective: " + str(meanObjective))
        print("Best objective: " + str(bestObjective))

        start_time = time.time()
        pop_idx = np.random.choice(population.shape[0], min(self.population_size, 20), replace=False)
        population[pop_idx] = np.array([self.two_opt_v2(distanceMatrix, i) for i in tqdm(population[pop_idx])], dtype=np.uint16)
        print("2-opt required time: ", time.time() - start_time, population.shape)

        # Evaluation of population after k-opt local search
        population_fitnesses = [self.fitness(distanceMatrix, i) for i in population]
        meanObjective = np.mean(population_fitnesses)
        bestObjective = np.min(population_fitnesses)
        print("Mean objective: " + str(meanObjective))
        print("Best objective: " + str(bestObjective))

        if self.convergency_plot:
            mean_objectives = []
            best_objectives = []
            iterations_conv = []
        max_memory = 0
        if self.diversity_plot:
            iterations_div = []
            all_individuals = []

        if self.mem_recording:
            pid = os.getpid()
        start_time = time.time()

        two_opt_executed = False
        while self.should_continue(i, obj_values, meanObjective, two_opt_executed):
            print(i)

            # Your code here.
            # Initialize offspring population
            offspring = np.empty([self.offspring_size, self.n], dtype=np.uint16)

            # Compute adaptive recombination and mutation parameters
            rec_list = []
            mut_list = []

            if i > 0 and i <= self.selection_iteration_num:
                print("Fitness sharing ON")

            for j in range(0, self.offspring_size - 1, 2):
                for rec_i in range(2):
                    if i > 0 and i <= self.selection_iteration_num:
                        parent1 = self.selection(distanceMatrix, population, self.get_k(i), use_shared=True)
                        parent2 = self.selection(distanceMatrix, population, self.get_k(i), use_shared=True)
                    else:
                        parent1 = self.selection(distanceMatrix, population, self.get_k(i), use_shared=False)
                        parent2 = self.selection(distanceMatrix, population, self.get_k(i), use_shared=False)
                    rec_prob = self.get_recombination_prob(
                        meanObjective, bestObjective, min(self.fitness(distanceMatrix, parent1), self.fitness(distanceMatrix, parent2)))
                    rec_list.append(rec_prob)
                    if random.random() < rec_prob:
                        offspring[j+rec_i] = self.recombination_hgrex(parent1, parent2, distanceMatrix)
                    else:
                        offspring[j+rec_i] = parent1 if self.fitness(distanceMatrix, parent1) <= self.fitness(distanceMatrix, parent2) else parent2


            for k in range(len(offspring)):
                mut_prob = self.get_mutation_prob(i)
                mut_list.append(mut_prob)
                if random.random() < mut_prob:
                    #if i >= self.local_iteration_num:
                    offspring[k] = self.inverted_displacement_mutation(distanceMatrix, offspring[k])
                    #offspring[k] = self.mutation(distanceMatrix, offspring[k])
                    #else:
                    #    offspring[k] = self.mutation(distanceMatrix, offspring[k])


            if i >= self.local_iteration_num:
                #two_opt_executed = True
                #for el in tqdm(range(5)):
                #    individual, idx = self.selection(distanceMatrix, offspring, 2, return_index=True)
                #    offspring[idx] = self.two_opt_v2(distanceMatrix, individual)
                off_idx = np.random.choice(range(offspring.shape[0]), 20, replace=False)
                offspring[off_idx] = np.array([self.two_opt_v2(distanceMatrix, i) for i in tqdm(offspring[off_idx])], dtype=np.uint16)
                two_opt_executed = True

            print("Recombination probability (average): ", np.mean(rec_list))
            print("Mutation probability (average): ", np.mean(mut_list))


            # (lambda,mu)
            population = self.elimination(distanceMatrix, offspring)

            population_fitnesses = [self.fitness(distanceMatrix, i) for i in population]
            meanObjective = np.mean(population_fitnesses)
            bestObjective = np.min(population_fitnesses)
            idx_min = np.argmin(population_fitnesses)
            bestSolution = population[idx_min]

            if i % 1 == 0:
                print("Mean objective: " + str(meanObjective))
                print("Best objective: " + str(bestObjective))
            # print("Best solution: " + str(bestSolution))

            # Add objective function to check for convergence condition
            obj_values.insert(0, bestObjective)
            if len(obj_values) > self.conv_el:
                obj_values.pop()
            print(obj_values)


            if self.convergency_plot:
                mean_objectives.append(meanObjective)
                best_objectives.append(bestObjective)
                iterations_conv.append(i)

            if self.mem_recording:
                py = psutil.Process(pid)
                memory_use = py.memory_info()[0] / (2. ** 20)  # memory use in MB...I think
                max_memory = max(max_memory, memory_use)

            if self.diversity_plot:
                for individual in population:
                    all_individuals.append(self.fitness(distanceMatrix, individual))
                    iterations_div.append(i)

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                print("\n No time left !!!\n")
                break

            i += 1

        # Plot the diversity plot if needed
        if self.diversity_plot:
            plt.figure()
            df = pd.DataFrame(data={"iterations": iterations_div, "fitness": all_individuals})
            df.plot(kind="scatter", x="iterations", y="fitness", alpha=0.1)
            plt.savefig("diversity_plot.png")

        # plot the convergence plot if needed
        if self.convergency_plot:
            plt.figure()
            plt.plot(iterations_conv, best_objectives, 'r')
            plt.plot(iterations_conv, mean_objectives, 'b')
            plt.legend(["best objective value", "mean objective value"])
            plt.xlabel("iterations")
            plt.ylabel("fitness")
            plt.savefig("convergence_plot.png")

        # Your code here.
        return meanObjective, bestObjective, bestSolution, i, time.time() - start_time, max_memory

    def distance(self, x, population):
        return [(np.abs(kendalltau(x, pop)[0])) for pop in population] # Reverse correlation metric

    def manhattan_distance(self, a, population):
        return [sum(abs(a-pop)) for pop in population]

    def sharedFitness(self, distance_matrix, individual, population):
        alpha, sigma = 0.3, (individual.size**2)/3
        rand_idx = np.random.choice(range(population.shape[0]), int(population.shape[0]/3), replace=False)
        dist = self.manhattan_distance(individual.astype(np.int32), population[rand_idx].astype(np.int32))
        onePlusBeta = np.sum([1 - (d / sigma) ** alpha for d in dist if d <= sigma])
        fval = self.fitness(distance_matrix, individual)
        return max(fval * onePlusBeta**np.sign(fval), fval)

    def fitness(self, dist_matrix, individual):
        '''
        Compute fitness: length
        :param dist_matrix: (np.ndarray) distance matrix of population
        :param individual: (np.ndarray) of length n ([1 2 3 4])
        :return: float
        '''
        # In 2D: individual subset [4,7] (from 4rth city to 7th city): dist_matrix[4,7]
        dist = [dist_matrix[i, individual[(idx + 1) % individual.size]] for idx, i in enumerate(
            individual)]
        a = sum(dist)
        if isinf(a):
            return BIG_VALUE
        return sum(dist)

    def get_k(self, t):
        return 15
        #if t < 10:
        #    return 15
        #else:
        #    return 5

    def get_mutation_prob(self, t):
        return max(0.9-0.8*t/20, 0.1)
        #return 0.5

    #def get_mutation_prob(self, f_avg, f_min, f, mut1=0.75, mut2=0.01):
    #    if f <= f_avg:
    #        mut_prob = mut1 - (mut1-mut2) * (f - f_min) / (f_avg - f_min)
    #    else:
    #        mut_prob = mut2
    #    return mut_prob

    def get_recombination_prob(self, f_avg, f_min, f, rec1=0.9, rec2=0.6):
        if isinf(f) or f >= np.sqrt(BIG_VALUE):
            return 0.75
        if f <= f_avg:
            rec_prob = rec1 - ((rec1-rec2) * (f-f_min) / (f_avg-f_min))
        else:
            rec_prob = rec2
        return rec_prob

    #def get_mutation_prob(self, f_avg, f_max, f, mut1=0.1, mut2=0.001):
    #    if f <= f_avg:
    #        mut_prob = mut1 - ((mut1 - mut2) * (f - f_max) / (f_avg-f_max))
    #    else:
    #        mut_prob = mut1
    #    return mut_prob

    #def get_recombination_prob(self, f_avg, f_max, f, rec1=0.7, rec2=0.2):
    #    if f <= f_avg:
    #        rec_prob = rec1 - ((rec1 - rec2) * (f_avg-f) / (f_avg-f_max))
    #    else:
    #        rec_prob = rec1
    #    return rec_prob

    def initialization(self):
        '''
        Initialize population of random individuals
        :return: (np.ndarray) all population
        '''
        population = np.empty([self.population_size, self.n], dtype=np.uint16)
        for i in range(self.population_size):
            # create random individual
            individual = np.random.permutation(self.n)
            population[i] = individual
        return population

    def selection(self, dist_matrix, population, k, use_shared=False, return_index=False):
        '''
        k-tournament selection.
        :param dist_matrix: (np.ndarray)
        :param population: (np.ndarray)
        :param k: (int) the tournament size
        :return: selected individual
        '''
        # print(k)
        # print(population.shape[0])
        selected_idx = np.random.choice(population.shape[0], k, replace=False)
        selected = population[selected_idx]
        if use_shared:
            fitness_values = [self.sharedFitness(dist_matrix, s, population) for s in selected]
            # Check if fitness sharing truly works: average should be larger
            #print("---------")
            #print("Average sharedFitness: ", np.mean(fitness_values))
            #fval = [self.fitness(dist_matrix, s) for s in selected]
            #print("Average normalFitness: ", np.mean(fval))
            #print("***--------***")
        else:
            fitness_values = [self.fitness(dist_matrix, s) for s in selected]
        min_idx = np.argmin(fitness_values)
        if return_index:
            return selected[min_idx], selected_idx[min_idx]
        else:
            return selected[min_idx]


    def insertion_mutation(self, individual):
        '''
        Insertion mutation of a subset:
        Insert randomly selected subset to randomly chosen index.
        '''
        idxs = np.random.randint(0, self.n + 1, 2)
        if (idxs[0] <= idxs[1]):
            pick_idxs = list(range(idxs[0], idxs[1], 1))
        else:
            pick_idxs = list(range(idxs[1], idxs[0], 1))

        idx = np.random.randint(0, self.n + 1, 1)[0]
        individual = self.change_position(individual, pick_idxs, idx)
        return individual

    # Only to offspring, inversion mutation
    # returns the same individual but mutated
    def mutation(self, distMatrix, individual):
        '''
        Inversion mutation.
        If the same index is twice randomly chosen, it will inverse the subarray of length one
        and will have no real effect.
        np.flip turns [1 2 3 4] into [4 3 2 1]
        :param individual: (np.ndarray) the individual to be mutated
        :return: the mutated individual
        '''
        individual_orig = individual.copy()
        idxs = np.random.randint(0, self.n + 1, 2)
        if (idxs[0] <= idxs[1]):
            individual[idxs[0]:idxs[1]] = np.flip(individual[idxs[0]:idxs[1]])
        else:
            individual[idxs[1]:idxs[0]] = np.flip(individual[idxs[1]:idxs[0]])
        if self.fitness(distMatrix, individual) <= np.sqrt(BIG_VALUE):
            return individual
        else:
            return individual_orig
        return individual

    def change_position(self, individual, pick_idx, put_idx):
        put_idxs = list(range(put_idx, put_idx + len(pick_idx)))
        orig_data = individual.copy()
        data = np.insert(individual, put_idx, individual[pick_idx])
        possible_idx = [np.where(data == orig_data[i])[0] for i in pick_idx]
        delete_idx = [i[0] if i[0] not in put_idxs else i[1] for i in possible_idx]
        return np.delete(data, delete_idx)

    def inverted_displacement_mutation(self, distMatrix, individual):
        individual_orig = individual.copy()
        idxs = np.random.randint(0, self.n + 1, 2)
        if (idxs[0] <= idxs[1]):
            pick_idxs = list(range(idxs[0], idxs[1], 1))
            individual[idxs[0]:idxs[1]] = np.flip(individual[idxs[0]:idxs[1]])
        else:
            pick_idxs = list(range(idxs[1], idxs[0], 1))
            individual[idxs[1]:idxs[0]] = np.flip(individual[idxs[1]:idxs[0]])

        idx = np.random.randint(0, self.n + 1, 1)[0]
        individual = self.change_position(individual, pick_idxs, idx)
        if self.fitness(distMatrix, individual) <= np.sqrt(BIG_VALUE):
            return individual
        else:
            return individual_orig


    # Order recombination
    # returns one child or child1 and child2 (as a tuple)
    def recombination(self, parent1, parent2):
        left_index = np.random.randint(0, self.n)
        right_index = np.random.randint(left_index, self.n)

        off1 = np.empty(self.n, int)
        off2 = np.empty(self.n, int)

        j = right_index - left_index
        off1[0: j] = parent1[left_index: right_index]
        off2[0: j] = parent2[left_index: right_index]
        j1 = j2 = j

        # compare with itertools? (possibly more efficient than casting)
        for i in list(range(right_index, self.n)) + list(range(0, right_index)):
            if not (parent2[i] in parent1[left_index: right_index]):
                off1[j1] = parent2[i]
                j1 = j1 + 1
            if not (parent1[i] in parent2[left_index: right_index]):
                off2[j2] = parent1[i]
                j2 = j2 + 1

        return off1, off2

    def recombination_scx(self, parent1, parent2, cost_matrix):
        chromo = list()
        chromo.append(parent1[0])  # Add first gene to the chromosome
        chromo_seq = list(set(list(range(self.n))) - set(chromo))
        for i in range(self.n - 1):
            gene1 = chromo[i]
            # Find gene1 indices in parents
            idx1 = np.where(parent1 == gene1)[0][0]
            idx2 = np.where(parent2 == gene1)[0][0]

            # Extract elements to compare with
            # First condition: out of size
            # Second condition: following element is already selected
            if idx1 >= parent1.size - 1 or parent1[idx1 + 1] in chromo:
                gene21 = list(set(chromo_seq) - set(chromo))[0]
            else:
                gene21 = parent1[idx1 + 1]

            if idx2 >= parent2.size - 1 or parent2[idx2 + 1] in chromo:
                gene22 = list(set(chromo_seq) - set(chromo))[0]
            else:
                gene22 = parent2[idx2 + 1]

            # Simply append if both genes are the same
            if gene21 == gene22:
                chromo.append(gene21)
                continue

            # Compare cost for (gene1,gene21) and (gene1,gene22)
            cost1 = cost_matrix[gene1, gene21]
            cost2 = cost_matrix[gene1, gene22]
            if cost1 <= cost2:
                chromo.append(gene21)
            elif cost1 > cost2:
                chromo.append(gene22)
        return np.array(chromo)

    def recombination_hgrex(self, parent1, parent2, cost_matrix):
        chromo = list()
        rand_idx = np.random.randint(0, self.n - 1)
        chromo.append(parent1[rand_idx])
        chromo_seq = list(set(list(range(self.n))) - set(chromo))
        for i in range(self.n - 1):
            gene1 = chromo[i]
            # Find gene1 indices in parents
            idx1 = np.where(parent1 == gene1)[0][0]
            idx2 = np.where(parent2 == gene1)[0][0]

            if idx1 >= parent1.size - 1:
                gene21 = parent1[0]
            else:
                gene21 = parent1[idx1 + 1]

            if idx2 >= parent2.size - 1:
                gene22 = parent2[0]
            else:
                gene22 = parent2[idx2 + 1]

            if gene21 == gene22 and gene21 not in chromo:
                chromo.append(gene21)
            elif gene21 in chromo and gene22 in chromo:
                gene2_diff = list(set(chromo_seq) - set(chromo))
                cost2 = [cost_matrix[gene1, g2] for g2 in gene2_diff]
                chromo.append(gene2_diff[cost2.index(min(cost2))])
                continue
            elif gene21 in chromo and gene22 not in chromo:
                chromo.append(gene22)
                continue
            elif gene21 not in chromo and gene22 in chromo:
                chromo.append(gene21)
                continue
            else: # Compare cost for (gene1,gene21) and (gene1,gene22)
                cost1 = cost_matrix[gene1, gene21]
                cost2 = cost_matrix[gene1, gene22]
                if cost1 <= cost2:
                    chromo.append(gene21)
                elif cost1 > cost2:
                    chromo.append(gene22)
        individual = np.array(chromo)
        if self.fitness(cost_matrix, individual) < np.sqrt(BIG_VALUE):
            return individual
        else:
            return parent1 if self.fitness(cost_matrix, parent1) < self.fitness(cost_matrix, parent2) else parent2

    def elimination(self, distanceMatrix, population, share_fitness=False, survive_num=None):
        '''
        (lambda+mu)-elimination.
        :param population: (ndarray) the population of which individuals should be eliminated
        :param distanceMatrix: (ndarray) the distance matrix
        :return: the surviving population
        '''
        if share_fitness:
            idxs_sorted = np.argsort([self.sharedFitness(distanceMatrix, i, population) for i in population])
        else:
            idxs_sorted = np.argsort([self.fitness(distanceMatrix, i) for i in population])

        if survive_num is not None:
            surviving_idxs = idxs_sorted[:survive_num]
        else:
            surviving_idxs = idxs_sorted[:(self.offspring_size)]

        return population[surviving_idxs]

    def cost_change(self, dist_matrix, X1, X2, Y1, Y2):
        dx_1, dx_2 = dist_matrix[X1, Y1], dist_matrix[X2, Y2]
        dy_1, dy_2 = dist_matrix[X1, X2], dist_matrix[Y1, Y2]
        if isinf(dx_1) or isinf(dx_2) or isinf(dy_1) or isinf(dy_2):
            return BIG_VALUE
        else:
            return dx_1 + dx_2 - dy_1 - dy_2

    def two_opt(self, dist_matrix, individual):
        best, improved, max_iter = individual, True, 0
        while improved and max_iter < 10:
            improved = False
            max_iter += 1
            for i in range(1, len(individual) - 2):
                for j in range(i + 1, len(individual)):
                    if j - i == 1: continue
                    if self.cost_change(dist_matrix, best[i-1], best[i], best[j-1], best[j]) < 0:
                        best[i:j] = best[j - 1:i - 1:-1]
                        improved = True
            individual = best
        return best

    def two_opt_v2(self, cost_mat, route):
        best = route
        improved, max_iter = True, 0

        dontLook_mask = np.ones(len(route), dtype=bool)
        dontLook_mask[:] = False

        while improved and max_iter < 10:
            improved = False
            max_iter += 1

            for counter_1 in range(1, len(route) - 2):

                if dontLook_mask[best[counter_1 - 1]]:
                    continue

                for mode in ['backward', 'forward']:
                    if mode == 'backward':
                        i = counter_1
                    else:
                        i = (counter_1 - len(route) + 1) % len(route)

                    X1 = best[i - 1]
                    X2 = best[i]
                    radius = self.fitness(cost_mat, np.array([X1, X2]))

                    for j in range(i + 1, len(route)):
                        if j - i == 1: continue

                        Y1 = best[j - 1]
                        Y2 = best[j]

                        if cost_mat[X2, Y2] > radius:
                            if cost_mat[X1, Y1] > radius:
                                continue

                        if self.cost_change(cost_mat, X1, X2, Y1, Y2) < 0:
                            best_cp = best.copy()
                            best_cp[i:j] = best_cp[j - 1:i - 1:-1]
                            if self.fitness(cost_mat, best_cp) < np.sqrt(BIG_VALUE):
                                best[i:j] = best[j - 1:i - 1:-1]
                                dontLook_mask[X1] = False
                                dontLook_mask[X2] = False
                                dontLook_mask[Y1] = False
                                dontLook_mask[Y2] = False
                                improved = True

                    dontLook_mask[best[counter_1]] = True
                route = best
        return best


    def should_continue(self, iterations, obj_values, meanObjective, two_opt_executed):
        '''
        Define algorithm stop conditions based on:
        1. Exceeded number of iterations
        2. Convergence condition for multiple iterations: |f(x_i) - f(x_i+1)| < tol*f(x_i)
        # new value = obj_values[0]
        # all previous values = obj_values[1:]
        :param iterations: i (int)
        :param obj_values: (list) list of fitness values for the previous self.conv_el iterations
        :return: (bool)
        '''
        previous_mean = np.mean(obj_values[1:])
        if previous_mean == BIG_VALUE:
            still_converging = True
        else:
            still_converging = np.abs(obj_values[0] - previous_mean) >= self.conv_tolerance * previous_mean
        mean_still_too_large = np.abs(obj_values[0] - meanObjective) >= meanObjective * 0.01
        return ((iterations <= self.max_iterations and still_converging) and mean_still_too_large) or not two_opt_executed


# Parameters you can change to
nb_runs = 3
get_avg_iterations = True
get_avg_time = True
get_max_memory = True
plot_variance = False
tour_file = "../tour100.csv"
create_convergence_plot = False
create_div_plot = False


# Needed structures to collect the needed information
max_memory = [0 for i in range(nb_runs)]
all_times = [0 for i in range(nb_runs)]
best_objective = [0 for i in range(nb_runs)]
best_solution = [0 for i in range(nb_runs)]
mean_objective = [0 for i in range(nb_runs)]
iterations = [0 for i in range(nb_runs)]
runs = [i for i in range(nb_runs)]

# Run the algorithm the needed number of times
for i in range(nb_runs):
    instance = r0769473(create_convergence_plot=create_convergence_plot, report_memory=get_max_memory,
                        create_div_plot=create_div_plot)
    mean_objective[i], best_objective[i], best_solution[i], iterations[i], all_times[i], max_memory[
        i] = instance.optimize(filename=tour_file)
    print("Mean objective: " + str(mean_objective))
    print("Best objective: " + str(best_objective))
    print("Best solution: " + str(best_solution))

print(all_times)

# Reports the average values over the different runs to the results file if needed
result_file = open("results.txt", "a")
result_file.writelines("______________" + str(datetime.datetime.now()) + " " + tour_file + "________________\n")
if get_avg_iterations:
    result_file.write("AVG ITERATIONS: " + str(sum(iterations) / len(iterations)) + "\n")
if get_max_memory:
    result_file.write("AVG MAX MEMORY: " + str(sum(max_memory) / len(max_memory)) + "\n")
if get_avg_time:
    result_file.write("AVG TIME: " + str(sum(all_times) / len(all_times)) + "\n")
result_file.close()

# plot the variance of the best and mean objective over the different runs if needed and save in variance_plot.png
if plot_variance:
    plt.figure()
    plt.plot([i for i in range(10)], [29397.588832904523, 37575.46542469684, 36626.87945307234, 19782.022462996574, 18452.348158944424, 20436.89896739982, 20872.489638196774, 20024.801366432475, 18941.51304846782, 51084.26926587962], 'r')
    # plt.plot(runs, mean_objective, 'b')
    # plt.legend(["best objective value", "mean objective value"])
    plt.xlabel("runs")
    plt.ylabel("best objective value")
    plt.savefig("variance_plot.png")
