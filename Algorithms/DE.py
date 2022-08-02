import random
import numpy
import time



def DE(func, n_dim, size_pop, max_iter, lb, ub):

    mutation_factor = 0.5
    crossover_ratio = 0.7
    stopping_func = None

    


    # initialize population
    population = []

    population_fitness = numpy.array([float("inf") for _ in range(size_pop)])

    for p in range(size_pop):
        sol = []
        for d in range(n_dim):
            d_val = random.uniform(lb[d], ub[d])
            sol.append(d_val)

        population.append(sol)

    population = numpy.random.uniform(0, 1, size=(size_pop, n_dim)) * (ub-lb) + lb
    
    x_best = population[0, :]
    y_best = float("inf")
    x_best_hist = []
    
    # calculate fitness for all the population
    for i in range(size_pop):
        fitness = func(population[i, :])
        population_fitness[p] = fitness
        # s.func_evals += 1

        # is leader ?
        if fitness < y_best:
            y_best = fitness
            x_best = population[i, :]

    convergence_curve = numpy.zeros(max_iter)



    t = 0
    while t < max_iter:
        # should i stop


        # loop through population
        for i in range(size_pop):
            # 1. Mutation

            # select 3 random solution except current solution
            ids_except_current = [_ for _ in range(size_pop) if _ != i]
            id_1, id_2, id_3 = random.sample(ids_except_current, 3)

            mutant_sol = []
            for d in range(n_dim):
                d_val = population[id_1, d] + mutation_factor * (
                    population[id_2, d] - population[id_3, d]
                )

                # 2. Recombination
                rn = random.uniform(0, 1)
                if rn > crossover_ratio:
                    d_val = population[i, d]

                # add dimension value to the mutant solution
                mutant_sol.append(d_val)

            # 3. Replacement / Evaluation

            # clip new solution (mutant)
            mutant_sol = numpy.clip(mutant_sol, lb, ub)

            # calc fitness
            mutant_fitness = func(mutant_sol)
            # s.func_evals += 1

            # replace if mutant_fitness is better
            if mutant_fitness < population_fitness[i]:
                population[i, :] = mutant_sol
                population_fitness[i] = mutant_fitness

                # update leader
                if mutant_fitness < y_best:
                    y_best = mutant_fitness
                    x_best = mutant_sol

        convergence_curve[t] = y_best
        x_best_hist.append(x_best)
        t+=1

    # return solution
    return numpy.array(x_best_hist), convergence_curve
