# -*- coding: utf-8 -*-
"""
Created on Mon May 16 00:27:50 2016

@author: Hossam Faris
"""

import random
import numpy
import math
import time


def GWO(func, n_dim, size_pop, max_iter, lb, ub):

    # max_iter=1000
    # lb=-100
    # ub=100
    # n_dim=30
    # SearchAgents_no=5

    # initialize alpha, beta, and delta_pos
    Alpha_pos = numpy.zeros(n_dim)
    Alpha_score = float("inf")

    Beta_pos = numpy.zeros(n_dim)
    Beta_score = float("inf")

    Delta_pos = numpy.zeros(n_dim)
    Delta_score = float("inf")



    # Initialize the positions of search agents
    Positions = numpy.random.uniform(0, 1, size=(size_pop, n_dim)) * (ub-lb) + lb


    Convergence_curve = numpy.zeros(max_iter)
    x_best_history = []


    # Main loop
    for l in range(0, max_iter):
        for i in range(0, size_pop):

            # Return back the search agents that go beyond the boundaries of the search space
            for j in range(n_dim):
                Positions[i, j] = numpy.clip(Positions[i, j], lb[j], ub[j])

            # Calculate objective function for each search agent
            fitness = func(Positions[i, :])

            # Update Alpha, Beta, and Delta
            if fitness < Alpha_score:
                Delta_score = Beta_score  # Update delte
                Delta_pos = Beta_pos.copy()
                Beta_score = Alpha_score  # Update beta
                Beta_pos = Alpha_pos.copy()
                Alpha_score = fitness
                # Update alpha
                Alpha_pos = Positions[i, :].copy()

            if fitness > Alpha_score and fitness < Beta_score:
                Delta_score = Beta_score  # Update delte
                Delta_pos = Beta_pos.copy()
                Beta_score = fitness  # Update beta
                Beta_pos = Positions[i, :].copy()

            if fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score:
                Delta_score = fitness  # Update delta
                Delta_pos = Positions[i, :].copy()

        a = 2 - l * ((2) / max_iter)
        # a decreases linearly fron 2 to 0

        # Update the Position of search agents including omegas
        for i in range(0, size_pop):
            for j in range(0, n_dim):

                r1 = random.random()  # r1 is a random number in [0,1]
                r2 = random.random()  # r2 is a random number in [0,1]

                A1 = 2 * a * r1 - a
                # Equation (3.3)
                C1 = 2 * r2
                # Equation (3.4)

                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])
                # Equation (3.5)-part 1
                X1 = Alpha_pos[j] - A1 * D_alpha
                # Equation (3.6)-part 1

                r1 = random.random()
                r2 = random.random()

                A2 = 2 * a * r1 - a
                # Equation (3.3)
                C2 = 2 * r2
                # Equation (3.4)

                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])
                # Equation (3.5)-part 2
                X2 = Beta_pos[j] - A2 * D_beta
                # Equation (3.6)-part 2

                r1 = random.random()
                r2 = random.random()

                A3 = 2 * a * r1 - a
                # Equation (3.3)
                C3 = 2 * r2
                # Equation (3.4)

                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])
                # Equation (3.5)-part 3
                X3 = Delta_pos[j] - A3 * D_delta
                # Equation (3.5)-part 3

                Positions[i, j] = (X1 + X2 + X3) / 3  # Equation (3.7)

        Convergence_curve[l] = Alpha_score
        x_best_history.append(Alpha_pos)




    return numpy.array(x_best_history), Convergence_curve
