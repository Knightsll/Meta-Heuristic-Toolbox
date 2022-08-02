import random
import numpy
import math
import time


def SSA(func, n_dim, size_pop, max_iter, lb, ub):

    # max_iter=1000
    # lb=-100
    # ub=100
    # n_dim=30

    Convergence_curve = numpy.zeros(max_iter)
    x_best_hist = []
    # Initialize the positions of salps
    SalpPositions = numpy.random.uniform(0, 1, size=(size_pop, n_dim)) * (ub-lb) + lb
    SalpFitness = numpy.full(size_pop, float("inf"))

    FoodPosition = SalpPositions[0, :]
    FoodFitness = func(FoodPosition)
    # Moth_fitness=numpy.fell(float("inf"))

    



    for i in range(0, size_pop):
        # evaluate moths
        SalpFitness[i] = func(SalpPositions[i, :])

    sorted_salps_fitness = numpy.sort(SalpFitness)
    I = numpy.argsort(SalpFitness)

    Sorted_salps = numpy.copy(SalpPositions[I, :])

    FoodPosition = numpy.copy(Sorted_salps[0, :])
    FoodFitness = sorted_salps_fitness[0]

    Iteration = 0

    # Main loop
    while Iteration < max_iter:

        # Number of flames Eq. (3.14) in the paper
        # Flame_no=round(N-Iteration*((N-1)/max_iter));

        c1 = 2 * math.exp(-((4 * Iteration / max_iter) ** 2))
        # Eq. (3.2) in the paper

        for i in range(0, size_pop):

            SalpPositions = numpy.transpose(SalpPositions)

            if i < size_pop / 2:
                for j in range(0, n_dim):
                    c2 = random.random()
                    c3 = random.random()
                    # Eq. (3.1) in the paper
                    if c3 < 0.5:
                        SalpPositions[j, i] = FoodPosition[j] + c1 * (
                            (ub[j] - lb[j]) * c2 + lb[j]
                        )
                    else:
                        SalpPositions[j, i] = FoodPosition[j] - c1 * (
                            (ub[j] - lb[j]) * c2 + lb[j]
                        )

                    ####################

            elif i >= size_pop / 2 and i < size_pop + 1:
                point1 = SalpPositions[:, i - 1]
                point2 = SalpPositions[:, i]

                SalpPositions[:, i] = (point2 + point1) / 2
                # Eq. (3.4) in the paper

            SalpPositions = numpy.transpose(SalpPositions)

        for i in range(0, size_pop):

            # Check if salps go out of the search spaceand bring it back
            for j in range(n_dim):
                SalpPositions[i, j] = numpy.clip(SalpPositions[i, j], lb[j], ub[j])

            SalpFitness[i] = func(SalpPositions[i, :])

            if SalpFitness[i] < FoodFitness:
                FoodPosition = numpy.copy(SalpPositions[i, :])
                FoodFitness = SalpFitness[i]



        Convergence_curve[Iteration] = FoodFitness
        x_best_hist.append(FoodPosition)
        Iteration = Iteration + 1



    return numpy.array(x_best_hist), Convergence_curve
