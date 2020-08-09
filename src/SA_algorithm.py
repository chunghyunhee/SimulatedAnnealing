import time
import random
import math
import numpy as np

## custom section
initial_temperature = 100
cooling = 0.8 # cooling coef.
number_variables = 2
upper_bounds = [3,3]
lower_bounds = [-3, -3]
computing_time = 1 # seconds

## custom objective function
def objective_function(X):
    x = X[0]
    y = X[1]
    value = 3*(1-x)**2*math.exp(-x**2 - (y+1)**2) - 10*(x/5 - x**3 - y**5)*math.exp(-x**2 - y**2) -1/3*math.exp(-(x+1)**2 - y**2)
    return value

## simulated Annealing algorithm

## 1. Genertate an initial solution randomly
initial_solution = np.zeros((number_variables))
for v in range(number_variables):
    initial_solution[v] = random.uniform(lower_bounds[v], upper_bounds[v])
current_solution = initial_solution
best_solution = initial_solution
n=1 # no of solutions accepted
best_fitness = objective_function(best_solution)
current_temperature = initial_temperature # current temperature
start = time.time()
no_attemps = 100 # number of attemps in each level of temperature
record_best_fitness = []

for i in range(9999999):
    for j in range(no_attemps):
        for k in range(number_variables):
            ## 2. generate a candidate solution y randomly based on solution x
            current_solution[k] = best_solution[k] + 0.1*(random.uniform(lower_bounds[k], upper_bounds[k]))
            current_solution[k] = max(min(current_solution[k], upper_bounds[k]), lower_bounds[k]) # repaire the solution respecting the bounds

        ## 3. check if y is better than x
        current_fitness = objective_function(current_solution)
        E = abs(current_fitness - best_solution)
        if i==0 and j==0:
            EA = E
        if current_fitness < best_fitness:
            p = math.exp(-E/(EA*current_temperature))
            # make a decision to accept the worse solution or not
            ## 4. make a decision whether r < p
            if random.random()<p:
                accept = True # this worse solution is not accepted
            else:
                accept = False # this worse solution is not accepted
        else:
            accept = True # accept better solution
        ## 5. make a decision whether step comdition of inner loop is met
        if accept == True:
            best_solution = current_solution # update the best solution
            best_fitness = objective_function(best_solution)
            n = n + 1 #count the solutions accepted
            EA = (EA*(n-1)+E)/n # accept EA
    print('interation : {}, best_solution:{}, best_fitness:{}'. format(i, best_solution, best_fitness))
    record_best_fitness.append(best_fitness)
    ## 6. decrease the temperature
    current_temperature = current_temperature * cooling
    ## 7. stop condition of outer loop is met
    end = time.time()
    if end-start >= computing_time:
        break
