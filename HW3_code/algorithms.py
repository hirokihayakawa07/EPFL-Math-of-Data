import time
import numpy as np
from random import randint

from utils import print_end_message, print_start_message, print_progress

def ista(fx, gx, gradf, proxg, params):
    method_name = 'ISTA'
    print_start_message(method_name)

    tic = time.time()

    lmbd = params['lambda']
    
    run_details = {'X_final': None, 'conv': np.zeros(params['maxit'] + 1)}
    run_details['conv'][0] = fx(params['x0']) + lmbd * gx(params['x0'])
    
    ############## YOUR CODES HERE - parameter setup##############
    alpha = 1/ params['Lips']
    x = params['x0']
    
    for k in range(0, params['maxit']):
#         print('F',fx(x) + lmbd * gx(x))
#         print('f',fx(x))
#         print('g',lmbd * gx(x))
        x = proxg(x - alpha * gradf(x), lmbd*alpha)
        run_details['conv'][k] = fx(x) + lmbd * gx(x)
        ############## YOUR CODES HERE ##############
        if k % params['iter_print'] == 0:
            print_progress(k, params['maxit'], run_details['conv'][k], fx(x), gx(x))

    run_details['X_final'] = x
    ############## YOUR CODES HERE ##############

    print_end_message(method_name, time.time() - tic)
    return run_details




def fista(fx, gx, gradf, proxg, params):
    if params['restart_fista']:
        method_name = 'FISTA-RESTART'
    else:
        method_name = 'FISTA'
    print_start_message(method_name)
    tic = time.time()
    lmbd = params['lambda']
    
    run_details = {'X_final': None, 'conv': np.zeros(params['maxit'] + 1)}
    run_details['conv'][0] = fx(params['x0']) + lmbd * gx(params['x0'])

    ############## YOUR CODES HERE - parameter setup############## 
    alpha = 1/ params['Lips']
    x = params['x0']
    x_pre = x
    theta = 1
    theta_pre = theta
    
    t = 1
    y = params['x0']
    
    for k in range(0, params['maxit']):
        if method_name == 'FISTA-RESTART':
            y = x + theta * (1/theta_pre - 1) * (x - x_pre)
#             x = x_next
            x_next = proxg( y - alpha * gradf(y), lmbd*alpha)
            if gradient_scheme_restart_condition(x, x_next, y) > 0:
                theta_pre = 1
                theta = 1
                y = x
                x_next = proxg(y - alpha * gradf(y), lmbd*alpha)
            theta_next = (np.sqrt(theta**4 + 4*theta**2) - theta**2) / 2
            x_pre = x
            x = x_next
            theta_pre = theta
            theta = theta_next
            
        if method_name == 'FISTA':
            x_next = proxg(y - alpha * gradf(y), lmbd*alpha)
            t_next = (1 + np.sqrt(4*t**2 + 1))/2
            y = x_next + (t - 1)/t_next * (x_next - x)
            x = x_next
            t = t_next
            ############## YOUR CODES HERE##############
        
        # record convergence
        run_details['conv'][k] = fx(x_next) + lmbd * gx(x_next)

        if k % params['iter_print'] == 0:
            print_progress(k, params['maxit'], run_details['conv'][k], fx(y), gx(y))

    run_details['X_final'] = x_next
    ############## YOUR CODES HERE##############

    print_end_message(method_name, time.time() - tic)
    return run_details




def gradient_scheme_restart_condition(X_k, X_k_next, Y_k):
    return np.trace(np.transpose(Y_k - X_k_next) @ (X_k_next - X_k))
    ############## YOUR CODES HERE ##############
        
#     raise NotImplemented('Implement the method!')





def prox_sg(fx, gx, stocgradfx, proxg, params):
    method_name = 'PROX-SG'
    print_start_message(method_name)

    tic = time.time()

    lmbd = params['lambda']
    
    run_details = {'X_final': None, 'conv': np.zeros(params['maxit'] + 1)}
    run_details['conv'][0] = fx(params['x0']) + lmbd * gx(params['x0'])

    ############## YOUR CODES HERE - parameter setup##############
    ms = params['minib_size']
    x = params['x0']
    sum_gamma = 0
    sum_gamma_x = 0
    for k in range(0, params['maxit']):
        gamma = params['stoch_rate_regime'](k)
        sum_gamma = sum_gamma + gamma
        sum_gamma_x = sum_gamma_x + gamma*x
        x_hat = sum_gamma_x / sum_gamma # ergodic
        x = proxg(x - gamma*stocgradfx(x, ms), lmbd*gamma)
            ############## YOUR CODES HERE ##############

        run_details['conv'][k] = fx(x_hat) + lmbd * gx(x_hat) # record ergodic convergence

        if k % params['iter_print'] == 0:
            print_progress(k, params['maxit'], run_details['conv'][k], fx(x_hat), gx(x_hat))

    run_details['X_final'] = x
    ############## YOUR CODES HERE ##############

    print_end_message(method_name, time.time() - tic)
    return run_details
