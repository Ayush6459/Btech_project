import numpy as np
import pandas as pd
import random as rd
from random import randint
import matplotlib.pyplot as plt
import time
import timeit

def kanpsackDP(n,W,v,w,values,weights):
    n = n
    W = W
    v = v
    w = w

    values = values
    wieghts = weights
    # initialize the memoize matrix
    t = [[-1 for i in range(W+1)] for j in range(n+1)]
    def knapsack(wieghts,values,W,n):
        if n==0 or W==0:
            return 0
        if t[n][W] !=-1:
            return t[n][W]
        if wieghts[n-1]<=W:
            t[n][W]=max(values[n-1]+knapsack(wieghts,values,W-wieghts[n-1],n-1),knapsack(wieghts,values,W,n-1))
            return t[n][W]
        elif wieghts[n-1]>W:
            t[n][W]=knapsack(wieghts,values,W,n-1)
            return t[n][W]
    
    print(values)
    print('\n')
    print(wieghts)
    print('\n')
    print(knapsack(wieghts, values, W, n))
    print('\n')




def knapsack_genetic(n,W,v,w,values,weights):

    n = n
    W = W
    v = v
    w = w
    num_generations = 50

    item_number = np.arange(1, n+1)
    weight = weights
    value = values
    knapsack_threshold = W
    print("The list as follows :")

    print('item no.    Weight    Value')

    for i in range(item_number.shape[0]):
        print('{0}           {1}            {2}\n'.format(
            item_number[i], weight[i], value[i]))

    solution_per_pop = 5
    pop_size = (solution_per_pop, item_number.shape[0])
    print('population size = {}'.format(pop_size))
    initial_population = np.random.randint(2, size=pop_size)
    initial_population = initial_population.astype(int)

    print('Initial population: \n{}'.format(initial_population))

    def cal_fitness(weight, value, population, threshold):
        fitness = np.empty(population.shape[0])
        for i in range(population.shape[0]):
            S1 = np.sum(population[i] * value)
            S2 = np.sum(population[i] * weight)
            if S2 <= threshold:
                fitness[i] = S1
            else:
                fitness[i] = 0
        return fitness.astype(int)

    def selection(fitness, num_parents, population):
        fitness = list(fitness)
        parents = np.empty((num_parents, population.shape[1]))
        for i in range(num_parents):
            max_fitness_idx = np.where(fitness == np.max(fitness))
            parents[i, :] = population[max_fitness_idx[0][0], :]
            fitness[max_fitness_idx[0][0]] = -999999
        return parents

    def crossover(parents, num_offsprings):
        offsprings = np.empty((num_offsprings, parents.shape[1]))
        crossover_point = int(parents.shape[1]/2)
        crossover_rate = 0.8
        i = 0
        while (parents.shape[0] < num_offsprings):
            parent1_index = i % parents.shape[0]
            parent2_index = (i+1) % parents.shape[0]
            x = rd.random()
            if x > crossover_rate:
                continue
            parent1_index = i % parents.shape[0]
            parent2_index = (i+1) % parents.shape[0]
            offsprings[i, 0:crossover_point] = parents[parent1_index,
                                                       0:crossover_point]
            offsprings[i, crossover_point:] = parents[parent2_index,
                                                      crossover_point:]
            i = +1
            num_offsprings -= 1
        return offsprings

    def mutation(offsprings):
        mutants = np.empty((offsprings.shape))
        mutation_rate = 0.4
        for i in range(mutants.shape[0]):
            random_value = rd.random()
            mutants[i, :] = offsprings[i, :]
            if random_value > mutation_rate:
                continue
            int_random_value = randint(0, offsprings.shape[1]-1)
            if mutants[i, int_random_value] == 0:
                mutants[i, int_random_value] = 1
            else:
                mutants[i, int_random_value] = 0
        return mutants

    def optimize(weight, value, population, pop_size, num_generations, threshold):
        parameters, fitness_history = [], []
        num_parents = int(pop_size[0]/2)
        num_offsprings = pop_size[0] - num_parents
        for i in range(num_generations):
            fitness = cal_fitness(weight, value, population, threshold)
            fitness_history.append(fitness)
            parents = selection(fitness, num_parents, population)
            offsprings = crossover(parents, num_offsprings)
            mutants = mutation(offsprings)
            population[0:parents.shape[0], :] = parents
            population[parents.shape[0]:, :] = mutants

        print('Last generation: \n{}\n'.format(population))
        fitness_last_gen = cal_fitness(weight, value, population, threshold)
        print('max fitness : {}'.format(max(fitness_last_gen)))
        print('Fitness of the last generation: \n{}\n'.format(fitness_last_gen))
        max_fitness = np.where(fitness_last_gen == np.max(fitness_last_gen))
        parameters.append(population[max_fitness[0][0], :])
        return parameters, fitness_history

    parameters, fitness_history = optimize(
        weight, value, initial_population, pop_size, num_generations, knapsack_threshold)
    print('The optimized parameters for the given inputs are: \n{}'.format(parameters))
    selected_items = item_number * parameters
    print('\nSelected items that will maximize the knapsack without breaking it:')
    for i in range(selected_items.shape[1]):
        if selected_items[0][i] != 0:
            print('{}\n'.format(selected_items[0][i]))

    '''fitness_history_mean = [np.mean(fitness) for fitness in fitness_history]
    fitness_history_max = [np.max(fitness) for fitness in fitness_history]
    plt.plot(list(range(num_generations)),
            fitness_history_mean, label='Mean Fitness')
    plt.plot(list(range(num_generations)),
            fitness_history_max, label='Max Fitness')
    plt.legend()
    plt.title('Fitness through the generations')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.show()
    print(np.asarray(fitness_history).shape)
    '''



if __name__ == '__main__':
    loop=int(input("enter the loop value : "))
    time_taken_DP=[]
    time_taken_genetic=[]
    value_given=[]
    for i in range(loop):
        n = int(input('enter the no. of itmes you have : \n'))
        W = int(input('enter the maximum capacity of knapsack : \n'))
        v = int(input('enter the maximum range of value to store : \n'))
        w = int(input('enter the maximum range of the weight to store : \n'))
        value_given.append(n)
        values = np.random.randint(1, v, size=n)
        wieghts = np.random.randint(1, w, size=n)
        start_time=timeit.default_timer()
        kanpsackDP(n,W,v,w,values,wieghts)
        end_time=timeit.default_timer()
        time_taken_DP.append(end_time-start_time)
        print('\n')
        print('\n')
        start_time1 = timeit.default_timer()
        knapsack_genetic(n,W,v,w,values,wieghts)
        end_time1 = timeit.default_timer()
        time_taken_genetic.append(end_time1-start_time1)

    print(time_taken_DP)
    print(value_given)
    print(time_taken_genetic)
    plt.plot(value_given,time_taken_DP,label='time_vs_value_DP')
    plt.plot(value_given,time_taken_genetic,label='time_vs_value_genetic')
    plt.xlabel('value')
    plt.ylabel('time')
    plt.show()