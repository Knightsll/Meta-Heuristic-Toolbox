# -*- coding: utf-8 -*-


import random
import numpy

import time


def PSO(func, n_dim, size_pop, max_iter, lb, ub):

    # PSO parameters

    Vmax = 6
    wMax = 0.9
    wMin = 0.2
    c1 = 2
    c2 = 2


    ######################## Initializations

    vel = numpy.zeros((size_pop, n_dim))

    pBestScore = numpy.zeros(size_pop)
    pBestScore.fill(float("inf"))

    pBest = numpy.zeros((size_pop, n_dim))
    gBest = numpy.zeros(n_dim)

    gBestScore = float("inf")

    pos = numpy.zeros((size_pop, n_dim))
    for i in range(n_dim):
        pos[:, i] = numpy.random.uniform(0, 1, size_pop) * (ub[i] - lb[i]) + lb[i]

    convergence_curve = numpy.zeros(max_iter)
    x_best_hist = []




    for l in range(0, max_iter):
        for i in range(0, size_pop):
            # pos[i,:]=checkBounds(pos[i,:],lb,ub)
            for j in range(n_dim):
                pos[i, j] = numpy.clip(pos[i, j], lb[j], ub[j])
            # Calculate objective function for each particle
            fitness = func(pos[i, :])

            if pBestScore[i] > fitness:
                pBestScore[i] = fitness
                pBest[i, :] = pos[i, :].copy()

            if gBestScore > fitness:
                gBestScore = fitness
                gBest = pos[i, :].copy()

        # Update the W of PSO
        w = wMax - l * ((wMax - wMin) / max_iter)

        for i in range(0, size_pop):
            for j in range(0, n_dim):
                r1 = random.random()
                r2 = random.random()
                vel[i, j] = (
                    w * vel[i, j]
                    + c1 * r1 * (pBest[i, j] - pos[i, j])
                    + c2 * r2 * (gBest[j] - pos[i, j])
                )

                if vel[i, j] > Vmax:
                    vel[i, j] = Vmax

                if vel[i, j] < -Vmax:
                    vel[i, j] = -Vmax

                pos[i, j] = pos[i, j] + vel[i, j]

        convergence_curve[l] = gBestScore
        x_best_hist.append(gBest)


    return x_best_hist, convergence_curve
