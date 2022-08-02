# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 16:22:40 2022

@author: 山抹微云
"""

import random
import numpy
import math

import time


def HHO(func, n_dim, size_pop, max_iter, lb, ub):

    # n_dim=30
    # size_pop=50
    # lb=-100
    # ub=100
    # max_iter=500

    # initialize the location and Energy of the rabbit
    


    # Initialize the locations of Harris' hawks
    X = numpy.random.uniform(0, 1, size=(size_pop, n_dim)) * (ub-lb) + lb
    Rabbit_Location = X[0, :]
    Rabbit_Energy = float("inf")  # change this to -inf for maximization problems
    # Initialize convergence
    convergence_curve = numpy.zeros(max_iter)
    x_best_hist = []
    ############################



    t = 0  # Loop counter

    # Main loop
    while t < max_iter:
        
        for i in range(0, size_pop):

            # Check boundries

            X[i, :] = numpy.clip(X[i, :], lb, ub)

            # fitness of locations
            fitness = func(X[i, :])

            # Update the location of Rabbit
            if fitness < Rabbit_Energy:  # Change this to > for maximization problem
                Rabbit_Energy = fitness
                Rabbit_Location = X[i, :].copy()

        E1 = 2 * (1 - (t / max_iter))  # factor to show the decreaing energy of rabbit

        # Update the location of Harris' hawks
        for i in range(0, size_pop):

            E0 = 2 * random.random() - 1  # -1<E0<1
            Escaping_Energy = E1 * (
                E0
            )  # escaping energy of rabbit Eq. (3) in the paper

            # -------- Exploration phase Eq. (1) in paper -------------------

            if abs(Escaping_Energy) >= 1:
                # Harris' hawks perch randomly based on 2 strategy:
                q = random.random()
                rand_Hawk_index = math.floor(size_pop * random.random())
                X_rand = X[rand_Hawk_index, :]
                if q < 0.5:
                    # perch based on other family members
                    X[i, :] = X_rand - random.random() * abs(
                        X_rand - 2 * random.random() * X[i, :]
                    )

                elif q >= 0.5:
                    # perch on a random tall tree (random site inside group's home range)
                    X[i, :] = (Rabbit_Location - X.mean(0)) - random.random() * (
                        (ub - lb) * random.random() + lb
                    )

            # -------- Exploitation phase -------------------
            elif abs(Escaping_Energy) < 1:
                # Attacking the rabbit using 4 strategies regarding the behavior of the rabbit

                # phase 1: ----- surprise pounce (seven kills) ----------
                # surprise pounce (seven kills): multiple, short rapid dives by different hawks

                r = random.random()  # probablity of each event

                if (
                    r >= 0.5 and abs(Escaping_Energy) < 0.5
                ):  # Hard besiege Eq. (6) in paper
                    X[i, :] = (Rabbit_Location) - Escaping_Energy * abs(
                        Rabbit_Location - X[i, :]
                    )

                if (
                    r >= 0.5 and abs(Escaping_Energy) >= 0.5
                ):  # Soft besiege Eq. (4) in paper
                    Jump_strength = 2 * (
                        1 - random.random()
                    )  # random jump strength of the rabbit
                    X[i, :] = (Rabbit_Location - X[i, :]) - Escaping_Energy * abs(
                        Jump_strength * Rabbit_Location - X[i, :]
                    )

                # phase 2: --------performing team rapid dives (leapfrog movements)----------

                if (
                    r < 0.5 and abs(Escaping_Energy) >= 0.5
                ):  # Soft besiege Eq. (10) in paper
                    # rabbit try to escape by many zigzag deceptive motions
                    Jump_strength = 2 * (1 - random.random())
                    X1 = Rabbit_Location - Escaping_Energy * abs(
                        Jump_strength * Rabbit_Location - X[i, :]
                    )
                    X1 = numpy.clip(X1, lb, ub)

                    if func(X1) < fitness:  # improved move?
                        X[i, :] = X1.copy()
                    else:  # hawks perform levy-based short rapid dives around the rabbit
                        X2 = (
                            Rabbit_Location
                            - Escaping_Energy
                            * abs(Jump_strength * Rabbit_Location - X[i, :])
                            + numpy.multiply(numpy.random.randn(n_dim), Levy(n_dim))
                        )
                        X2 = numpy.clip(X2, lb, ub)
                        if func(X2) < fitness:
                            X[i, :] = X2.copy()
                if (
                    r < 0.5 and abs(Escaping_Energy) < 0.5
                ):  # Hard besiege Eq. (11) in paper
                    Jump_strength = 2 * (1 - random.random())
                    X1 = Rabbit_Location - Escaping_Energy * abs(
                        Jump_strength * Rabbit_Location - X.mean(0)
                    )
                    X1 = numpy.clip(X1, lb, ub)

                    if func(X1) < fitness:  # improved move?
                        X[i, :] = X1.copy()
                    else:  # Perform levy-based short rapid dives around the rabbit
                        X2 = (
                            Rabbit_Location
                            - Escaping_Energy
                            * abs(Jump_strength * Rabbit_Location - X.mean(0))
                            + numpy.multiply(numpy.random.randn(n_dim), Levy(n_dim))
                        )
                        X2 = numpy.clip(X2, lb, ub)
                        if func(X2) < fitness:
                            X[i, :] = X2.copy()

        convergence_curve[t] = Rabbit_Energy
        x_best_hist.append(Rabbit_Location)
        t+=1

    return numpy.array(x_best_hist), convergence_curve


def Levy(n_dim):
    beta = 1.5
    sigma = (
        math.gamma(1 + beta)
        * math.sin(math.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = 0.01 * numpy.random.randn(n_dim) * sigma
    v = numpy.random.randn(n_dim)
    zz = numpy.power(numpy.absolute(v), (1 / beta))
    step = numpy.divide(u, zz)
    return step
