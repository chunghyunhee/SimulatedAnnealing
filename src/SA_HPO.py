import pandas as pd
import numpy as np
import collections import OrderedDict
from random import random

def choose_params(param_dict, curr_params = None):
    """
    function to choose parameters for next iteration
    param param_dict: ordered dictionary of hyperparmeter search space
    param curr_params: Dict of current hyperparamters
    return: Dictionary of parameters
    """
    if curr_params:
        next_params = curr_params.copy()
        param_to_update = np.random.choice(list(param_dict.keys()))
        param_vals = param_dict[param_to_update]
        curr_index = param_vals.index(curr_params[param_to_update])
        if curr_index == 0:
            next_params[param_to_update] = param_vals[1]
        elif curr_index == len(param_vals) - 1:
            next_params[param_to_update] = param_vals[curr_index-1]
        else :
            next_params[param_to_update] = param_vals[curr_index + np.random.choice([-1, 1])]
    else :
        next_params = dict()
        for k, v in param_dict.items():
            next_params[k] = np.random.choice(v)

    return next_params

def Simulated_Annealing(param_dict, const_param, X_train, X_valid, Y_train, Y_valid,
                        fn_train, maxiters  = 100, alpha = 0.85, beta = 1.3, T_0 = 0.40, update_iters = 5):
    """
    function to perform hyperparamter search using SA
    param param_dict: ordered dictionary of hyperparameter seatch space
    param const_param: static parameters of the model
    param alpha: factor to reduce temperature
    param beta: constant to reduce temperature
    param T_0: initial temperature
    param update_iters: number of iterations required to update temperature

    return: dataframe of the parameters explored and correspoding metric
    """
    columns = [*param_dict.keys()] + ['Metric', 'Best Metric']
    results = pd.DataFrame(index = range(maxiters), columns = columns)
    best_metric = -1.
    prev_metric = -1.
    prev_param = None
    best_parma = dict()
    weights = list(map(lambda x: 10**x, list(range(len(param_dict)))))
    hash_values = set()
    T = T_0

    for i in range(maxiters):
        print('Starting iteration {}'. format(i))
        while(True):
            curr_params = choose_params(param_dict, prev_dict)
            indices = [param_dict[k].index(v) for k,v in current_params.items()]
            hash_val = sum([i*j for (i,j) in zip(weights, indices)])
            if hash_val in hash_values:
                print('Combination revisited')
            else:
                hash_values.add(hash_val)
                break
        model, metric = fn_train(curr_params, const_param,X_train,
                                 X_valid, Y_train, Y_valid)
        if metric > prev_metric :
            print('Local improvement in metric from {:8.4f} to {:8.4f}'.foramt(prev_metric, metric)+'-parameters accepted')
            prev_params = curr_params.copy()
            prev_metric = metric

            if metric > best_metric:
                print('Golobal improvement in metric from {:8.4f} to {:8.4f)'
                      .format(best_metric, metric) +
                      '- best parameters updated')
                best_metric = metric
                best_params = curr_params.copy()
                best_model = model

            else:
                rnd = np.random.uniform()
                diff = metric - prev_metric
                threshold = np.exp(beta*diff / T)
                if rnd < threshold :
                    print('No improvement but parameters accepted. Metric change'+
                          ': {:8.4f} threshold: {:6.4f} random number: {:6.4f}'
                          .format(diff, threshold, rnd)
                          )
                    prev_metric = metric
                    prev_params = curr_params
                else:
                    print('No Improvement and parameters rejected. Metric change' +
                      ': {:8.4f} threshold: {:6.4f} random number: {:6.4f}'
                      .format(diff, threshold, rnd))

            results.loc[i, list(curr_params.keys())] = list(curr_params.values())
            results.loc[i, 'Metric'] = metric
            results.loc[i, 'Best Metric'] = best_metric

            if i % update_iters == 0:
                T = alpha * T

        return results, best_model