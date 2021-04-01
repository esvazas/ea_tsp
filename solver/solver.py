import random
import numpy as np
from math import isinf
from . import Reporter

# a constant of the big value we use instead of inf
BIG_VALUE = 1.7976931348623157e+300
BIG_VALUE_sqrt = np.sqrt(BIG_VALUE)

class TSP_solver:

    # Convergence parameters
    conv_tolerance = 1e-3  # convergence tolerance
    conv_tolerance2 = 1e-5  # convergence tolerance
    conv_el = 3  # elements to consider for checking
    conv_el2 = 5  # elements to consider for checking

    max_iterations_conv1 = 9 # max iterations until swapping starts
    max_iterations = 100  # max iterations until stop
    execute_swapping = False

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.obj_values_min = [[0] * self.conv_el for idx in range(3)]
        self.obj_values_min2 = [[0] * self.conv_el2 for idx in range(3)]
        self.obj_values_avg = [[0] * self.conv_el2 for idx in range(3)]
        self.conv_tol = self.conv_tolerance

    def optimize(self, filename):
        '''
        Main Evolutionary Algorithm loop to solve TSP
        :param filename: str - path to TSP input matrix location
        :return: bool: run was successful
        '''

        # Read distance matrix from file.
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")  # probably a n x n matrix (n number of cities)
        file.close()
        self.n = distanceMatrix.shape[0]  # number of cities
        i = 0

        # Define parameters for:
        # k-opt, population size, offspring size
        lam, mu, k_opt_number, islands_to_k_opt, move_num, mutation_num = self.get_params(self.n)

        # Create init population and mutation/recombination probabilities
        populations = [self.initialization(distanceMatrix, pop_size) for pop_size in self.population_sizes]

        # Evaluate populations:
        mean_objectives = []
        best_objectives = []
        for idx, pop in enumerate(populations):
            mb, bo, _ = self.evaluate_population(distanceMatrix, pop, str(idx))
            mean_objectives.append(mb)
            best_objectives.append(bo)

        if np.min(mean_objectives) <= BIG_VALUE_sqrt:
            inf_edges = False
        else:
            inf_edges = True

        i_swapping = 0
        while self.should_continue(i, inf_edges):
            # Move elements within 2/3 populations
            if self.execute_swapping:
                i_swapping += 1
                rand_idx = np.random.choice(range(3), 2, replace=False)
                seq_from = np.random.permutation(3)[rand_idx]
                seq_to = np.roll(seq_from, -1)
                for idx_from, idx_to in zip(seq_from, seq_to):
                    # Select elements to move from pop1 to pop2
                    selected_individuals, selected_idx = [], []
                    moved_elements = 0
                    while moved_elements < move_num:
                        ind, index = self.selection(distanceMatrix, populations[idx_from], 3, return_index=True)
                        if index in selected_idx:
                            continue
                        selected_individuals.append(ind)
                        selected_idx.append(index)
                        moved_elements += 1

                    # Remove selected elements from pop1
                    populations[idx_from] = np.delete(populations[idx_from], selected_idx, axis=0)
                    # Move elements to pop2
                    populations[idx_to] = np.concatenate((populations[idx_to], selected_individuals), axis=0)

            # Initialize offspring populations
            offsprings = [np.empty([off_size, self.n], dtype=np.uint16) for off_size in self.offspring_sizes]

            # Fill all offsprings by selection and recombination
            for idx in range(len(populations)):
                # Selection and Recombination for 1st population
                if idx == 0:
                    # Selection and Recombination for 1st population
                    for j in range(0, self.offspring_sizes[idx] - 1, 2):
                        parent1 = self.selection(distanceMatrix, populations[idx], self.k)
                        parent2 = self.selection(distanceMatrix, populations[idx], self.k)
                        rec_prob = self.get_recombination_prob(mean_objectives[idx], best_objectives[idx], min(
                            self.fitness(distanceMatrix, parent1), self.fitness(distanceMatrix, parent2)))
                        if random.random() < rec_prob:
                            child1, child2 = self.recombination(distanceMatrix, parent1, parent2)
                            offsprings[idx][j] = child1
                            offsprings[idx][j+1] = child2
                        else:
                            offsprings[idx][j] = parent1
                            offsprings[idx][j+1] = parent2

                # Selection and Recombination for 2 population
                elif idx == 1:
                    for j in range(0, self.offspring_sizes[idx] - 1, 2):
                        for rec_i in range(2):
                            parent1 = self.selection(distanceMatrix, populations[idx], self.k)
                            parent2 = self.selection(distanceMatrix, populations[idx], self.k)
                            rec_prob = self.get_recombination_prob(mean_objectives[idx], best_objectives[idx], min(
                                self.fitness(distanceMatrix, parent1), self.fitness(distanceMatrix, parent2)))
                            if random.random() < rec_prob:
                                offsprings[idx][j+rec_i] = self.recombination_scx(parent1, parent2, distanceMatrix)
                            else:
                                offsprings[idx][j+rec_i] = parent1 if self.fitness(distanceMatrix, parent1) <= self.fitness(distanceMatrix, parent2) else parent2

                # Selection and Recombination for 3 population
                elif idx == 2:
                    for j in range(0, self.offspring_sizes[idx] - 1, 2):
                        for rec_i in range(2):
                            parent1 = self.selection(distanceMatrix, populations[idx], self.k)
                            parent2 = self.selection(distanceMatrix, populations[idx], self.k)
                            rec_prob = self.get_recombination_prob(mean_objectives[idx], best_objectives[idx], min(
                                self.fitness(distanceMatrix, parent1), self.fitness(distanceMatrix, parent2)))
                            if random.random() < rec_prob:
                                offsprings[idx][j+rec_i] = self.recombination_hgrex(parent1, parent2, distanceMatrix)
                            else:
                                offsprings[idx][j+rec_i] = parent1 if self.fitness(distanceMatrix, parent1) <= self.fitness(distanceMatrix, parent2) else parent2
                else:
                    raise ValueError("Unknown offspring and population!")

            # Mutation
            for idx in range(len(offsprings)):
                if idx == 0:
                    for k in range(len(offsprings[0])):
                        mut_prob = self.get_mutation_prob(i, idx, mutation_num)
                        if random.random() < mut_prob:
                            offsprings[idx][k] = self.inverted_displacement_mutation(offsprings[idx][k])
                elif idx == 1:
                    for k in range(len(offsprings[1])):
                        mut_prob = self.get_mutation_prob(i, idx, mutation_num)
                        if random.random() < mut_prob:
                            offsprings[idx][k] = self.mutation(offsprings[idx][k])
                elif idx == 2:
                    for k in range(len(offsprings[2])):
                        mut_prob = self.get_mutation_prob(i, idx, mutation_num)
                        if random.random() < mut_prob:
                            offsprings[idx][k] = self.inverted_displacement_mutation(offsprings[idx][k])
            # 2-opt
            if i_swapping >= 3:
                rand_idx = np.random.choice(range(3), islands_to_k_opt, replace=False)
                for idx in rand_idx:
                    off_idx = np.random.choice(offsprings[idx].shape[0], k_opt_number, replace=False)
                    offsprings[idx][off_idx] = np.array([self.two_opt_v2(distanceMatrix, i, inf_edges) for i in offsprings[idx][off_idx]], dtype=np.uint16)

            # (lambda,mu)
            populations = []
            for idx in range(len(offsprings)):
                pop = self.elimination(distanceMatrix, offsprings[idx], self.offspring_sizes[idx])
                populations.append(pop)

            # Evaluate populations:
            mean_objectives = []
            best_objectives = []
            obj_indices = []
            for idx in range(len(populations)):
                mo, bo, index = self.evaluate_population(distanceMatrix, populations[idx], str(idx))
                best_objectives.append(bo)
                mean_objectives.append(mo)
                obj_indices.append(index)
            meanObjective = np.mean(mean_objectives)
            bestObjective = np.min(best_objectives)
            best_objective_index = np.argmin(best_objectives)
            bestSolution = populations[best_objective_index][obj_indices[best_objective_index]]

            # Add objective function to check for convergence condition
            for idx in range(len(populations)):
                self.obj_values_min[idx].insert(0, best_objectives[idx])
                self.obj_values_min2[idx].insert(0, best_objectives[idx])
                self.obj_values_avg[idx].insert(0, mean_objectives[idx])
                if len(self.obj_values_min[idx]) > self.conv_el:
                    self.obj_values_min[idx].pop()
                if len(self.obj_values_min2[idx]) > self.conv_el2:
                    self.obj_values_min2[idx].pop()
                if len(self.obj_values_avg[idx]) > self.conv_el2:
                    self.obj_values_avg[idx].pop()

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                break

            i += 1
        return 0

    def get_params(self, n):
        '''
        Define EA parameters for a problem depending on the problem size
        :param n: size of a problem matrix
        :return: lam - offspring size
        :return: mu - population size
        :return: k_opt_number - number for the k-opt
        :return: islands_to_k_opt - number of islands for k-opt
        :return: move_num - number of elements to move inbetween islands
        :return: mutation_num - number of iterations when mutation prob significantly decreases for every island
        '''
        if n < 225:
            lam = 90
            islands_to_k_opt = 2
            move_num = 5
            mutation_num = [15, 15, 30]
            self.k = 5
        elif 225 <= n < 330:
            lam = 48
            islands_to_k_opt = 2
            move_num = 4
            mutation_num = [14, 14, 22]
            self.k = 3
        elif 330 <= n < 400:
            lam = 42
            islands_to_k_opt = 2
            move_num = 4
            mutation_num = [12, 12, 18]
            self.k = 3
        elif 400 <= n < 500:
            lam = 48
            islands_to_k_opt = 1
            move_num = 4
            mutation_num = [12, 12, 20]
            self.k = 4
        elif 500 <= n < 650:
            lam = 36
            islands_to_k_opt = 1
            move_num = 4
            mutation_num = [12, 12, 20]
            self.k = 3
        elif 650 <= n < 800:
            lam = 30
            islands_to_k_opt = 1
            move_num = 4
            mutation_num = [12, 12, 20]
            self.k = 3
        else:
            lam = 24
            islands_to_k_opt = 1
            move_num = 3  # how many elements to move
            mutation_num = [10, 10, 13]
            self.k = 3

        mu, k_opt_number = 5*lam, int(lam/3)
        self.population_sizes = [int(mu/3), int(mu/3), int(mu/3)]
        self.offspring_sizes = [int(lam/3), int(lam/3), int(lam/3)]
        return lam, mu, k_opt_number, islands_to_k_opt, move_num, mutation_num

    def evaluate_population(self, distanceMatrix, population, string):
        '''
        Compute objective function for current population
        :param distanceMatrix: input matrix
        :param population: list containing current population
        :param string:
        :return: computed evaluations
        '''

        population_fitnesses = [self.fitness(distanceMatrix, i) for i in population]
        idx_min = np.argmin(population_fitnesses)
        meanObjective = np.mean(population_fitnesses)
        bestObjective = np.min(population_fitnesses)
        return meanObjective, bestObjective, idx_min

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

    def get_mutation_prob(self, iter_num, pop_idx, mutation_num):
        '''
        Compute mutation probability based on the current iteration number and population index
        :param iter_num: int - iteration number
        :param pop_idx:  int - island of interest number
        :param mutation_num: np.array - number of iterations required till mutation prob becomes insignificant
        :return: current mutation probability
        '''
        if pop_idx == 0:
            return max(1 - (0.9 * iter_num / mutation_num[0]), 0.05)
        elif pop_idx == 1:
            return max(1 - (0.9 * iter_num / mutation_num[1]), 0.05)
        elif pop_idx == 2:
            return 0.7 if iter_num < mutation_num[2] else 0.15

    def get_recombination_prob(self, f_avg, f_min, f, rec1=0.9, rec2=0.6):
        ''' Function to iteratively update recombination probability based on the current objective. '''
        if f_avg == f_min:
            return rec2
        elif f < f_avg:
            rec_prob = rec1 - ((rec1-rec2) * (f-f_min)/ (f_avg-f_min))
        else:
            rec_prob = rec2
        return rec_prob

    def initialization(self, distMatrix, population_size):
        '''
        Initialize population of random individuals
        :param distMatrix: input matrix
        :param population_size: size of a population
        :return: (np.ndarray) all population
        '''
        population = np.empty([population_size, self.n], dtype=np.uint16)
        for i in range(population_size):
            individual = np.random.permutation(self.n)
            if np.bool(i%10):
                # create random individual
                individual = individual
            else:
                # create nn individual
                individual = self.nearest_neighbour(distMatrix, individual)
            population[i] = individual
        return population

    def nearest_neighbour(self, distMatrix, individual):
        """Nearest neighbor algorithm.
        :param distMatrix: NxN input array indicating distance between N locations
        :param individual: individual of interest for NN computations
        :return: NN path for the individual
        """
        start_idx = individual[0]
        path = [start_idx]
        N = individual.shape[0]
        mask = np.ones(N, dtype=bool)  # boolean values indicating which
        # locations have not been visited
        mask[start_idx] = False

        for i in range(N - 1):
            last = path[-1]
            next_ind = np.argmin(distMatrix[last][mask])  # find minimum of remaining locations
            next_loc = np.arange(N)[mask][next_ind]  # convert to original location
            path.append(next_loc)
            mask[next_loc] = False
        return path

    def selection(self, dist_matrix, population, k, return_index=False):
        '''
        k-tournament selection.
        :param dist_matrix: (np.ndarray)
        :param population: (np.ndarray)
        :param k: (int) the tournament size
        :return: selected individual
        '''
        selected_idx = np.random.choice(population.shape[0], k, replace=False)
        selected = population[selected_idx]
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
    def mutation(self, individual):
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
        return individual

    def change_position(self, individual, pick_idx, put_idx):
        ''' Change position of an element in a list. '''
        put_idxs = list(range(put_idx, put_idx + len(pick_idx)))
        orig_data = individual.copy()
        data = np.insert(individual, put_idx, individual[pick_idx])
        possible_idx = [np.where(data == orig_data[i])[0] for i in pick_idx]
        delete_idx = [i[0] if i[0] not in put_idxs else i[1] for i in possible_idx]
        return np.delete(data, delete_idx)

    def inverted_displacement_mutation(self, individual):
        ''' Inverted displacement mutation applied for an individual. '''
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
        return individual

    def recombination(self, cost_matrix, parent1, parent2):
        '''
        Order recombination
        :param cost_matrix: input distance matrix
        :param parent1: parent to participate in the recombination
        :param parent2: parent to participate in the recombination
        :return: returns one child or child1 and child2 (as a tuple)
        '''
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

        # Avoid inf edges
        return off1, off2

    def recombination_scx(self, parent1, parent2, cost_matrix):
        '''
        SCX recombination
        :param cost_matrix: input distance matrix
        :param parent1: parent to participate in the recombination
        :param parent2: parent to participate in the recombination
        :return: returns one child
        '''
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
        '''
        HGREX recombination
        :param cost_matrix: input distance matrix
        :param parent1: parent to participate in the recombination
        :param parent2: parent to participate in the recombination
        :return: returns one child
        '''
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
        return np.array(chromo)


    def elimination(self, distanceMatrix, population, offspring_size):
        '''
        (lambda+mu)-elimination.
        :param population: (ndarray) the population of which individuals should be eliminated
        :param distanceMatrix: (ndarray) the distance matrix
        :return: the surviving population
        '''
        idxs_sorted = np.argsort([self.fitness(distanceMatrix, i) for i in population])
        surviving_idxs = idxs_sorted[:offspring_size]
        return population[surviving_idxs]

    def cost_change(self, dist_matrix, X1, X2, Y1, Y2):
        dx_1, dx_2 = dist_matrix[X1, Y1], dist_matrix[X2, Y2]
        dy_1, dy_2 = dist_matrix[X1, X2], dist_matrix[Y1, Y2]
        if isinf(dx_1) or isinf(dx_2) or isinf(dy_1) or isinf(dy_2):
            return BIG_VALUE
        else:
            return dx_1 + dx_2 - dy_1 - dy_2

    def two_opt_v2(self, cost_mat, route, inf_edges):
        ''' Implemtation of 2-opt algorithm. '''
        best = route
        improved, max_iter = True, 0

        dontLook_mask = np.ones(len(route), dtype=bool)
        dontLook_mask[:] = False

        while improved and max_iter < 3:
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
                            if inf_edges:
                                best_cp = best.copy()
                                best_cp[i:j] = best_cp[j - 1:i - 1:-1]
                                if self.fitness(cost_mat, best_cp) < np.sqrt(BIG_VALUE):
                                    best[i:j] = best[j - 1:i - 1:-1]
                                    dontLook_mask[X1] = False
                                    dontLook_mask[X2] = False
                                    dontLook_mask[Y1] = False
                                    dontLook_mask[Y2] = False
                                    improved = True
                            else:
                                best[i:j] = best[j - 1:i - 1:-1]
                                dontLook_mask[X1] = False
                                dontLook_mask[X2] = False
                                dontLook_mask[Y1] = False
                                dontLook_mask[Y2] = False
                                improved = True

                    dontLook_mask[best[counter_1]] = True
                route = best
        return best

    def is_still_converging(self, obj_values):
        '''
        Check convergence criterion based on objective value.
        True if relative improvement is larger than condition
        '''
        previous_mean = np.mean(obj_values[1:])
        if previous_mean == BIG_VALUE:
            still_converging = True
        else:
            still_converging = np.abs(obj_values[0] - previous_mean) >= self.conv_tol * previous_mean
        return still_converging

    def should_continue(self, iterations, inf_edges):
        ''' Main function to control convergence. '''

        # Still converging is an individual case of every population
        if not self.execute_swapping:
            obj_v = self.obj_values_min
            all_converging = []
            for idx in range(len(obj_v)):
                all_converging.append(self.is_still_converging(obj_v[idx]))
            if np.sum(all_converging) != len(obj_v):
                still_converging = False
            else:
                still_converging = True
        # Still converging is defined by minimum best values among all populations together
        else:
            obj_v = np.min(np.array(self.obj_values_min2), axis=0)
            still_converging = self.is_still_converging(obj_v)

        cond1 = iterations <= self.max_iterations and still_converging
        if not self.execute_swapping:
            # Convergence condition before swapping:
            # Simple convergence according to best fitness average
            if not cond1 or iterations >= self.max_iterations_conv1: # Catch when swapping starts
                self.execute_swapping = True
                self.obj_values_min2 = [[0] * self.conv_el2 for idx in range(3)]
                self.obj_values_avg = [[0] * self.conv_el2 for idx in range(3)]
                self.conv_tol = self.conv_tolerance2
                return True
            else:
                return cond1
        else:
            # Add condition for mean fitness by checking mean fitness for every population
            last_meanObjectives = np.array(self.obj_values_avg)[:,0]
            last_bestObjectives = np.array(self.obj_values_min2)[:,0]
            means_too_large = np.abs(last_bestObjectives - last_meanObjectives) >= last_meanObjectives * 0.05
            num_means_to_converge = 0 if inf_edges else 0
            if np.sum(means_too_large) <= num_means_to_converge:
                mean_too_large = False
            else:
                mean_too_large = True
            return (cond1 or mean_too_large) and iterations <= self.max_iterations