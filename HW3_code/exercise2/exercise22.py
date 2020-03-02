import time
import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle
from skimage.measure import compare_ssim as ssim

from utils import apply_random_mask, psnr, load_image
from operators import TV_norm, RepresentationOperator, p_omega, p_omega_t, l1_prox

RAND_SEED = 666013

def ISTA(fx, gx, gradf, proxg, params):
## TO BE FILLED ##
    method_name = 'ISTA'
    print(method_name, 'start')
    alpha = 1/ params['Lips']
    x = params['x0']
    lmbd = params['lambda']
    run_details = {'iter': np.zeros(params['maxit']),'fx': np.zeros(params['maxit']), 'gx': np.zeros(params['maxit']),'Fx': np.zeros(params['maxit']),'rerr': np.zeros(params['maxit'])}
    err_0 = 0
    Fx = 0
    Fx_next = 0
    
    for k in range(0, params['maxit']):
        x_next = proxg(x - alpha * gradf(x), alpha)
        Fx = fx(x) + gx(x)
        Fx_next = fx(x_next) + gx(x_next)
        err = np.linalg.norm(x_next - x)
        if k % 100 == 0:
            print('iter = {:f}, F(X) = {:f}, f(X) = {:f}, g(X) = {:f}, rerr = {:f}.'.format(k, Fx, fx(x), gx(x), np.log10(err/err_0)))
        if k == 1:
            err_0 = err
            print('err_0 = ', err_0)
        x = x_next
            
        run_details['iter'][k] = k
        run_details['fx'][k] = fx(x_next)
        run_details['gx'][k] = gx(x_next)
        run_details['Fx'][k] = fx(x_next) + gx(x_next)
        run_details['rerr'][k] = err/err_0
    print('iter = {:f}, F(X) = {:f}, f(X) = {:f}, g(X) = {:f},rerr = {:f}.'.format(k, Fx, fx(x_next), gx(x_next), err/err_0))    
#     plt.semilogy(run_details['iter'], run_details['rerr'])                      
    return x, run_details

def FISTA(fx, gx, gradf, proxg, params, verbose=False):
## TO BE FILLED ##
    print(params['method'], 'start')

    alpha = 1/ params['Lips']
    lmbd = params['lambda']
    x = params['x0']
    x_pre = params['x0']
    theta = 1
    theta_pre = 1
    
    t = 1
    t_next = 1
    y_pre = params['x0']
    y = params['x0']
    err_0 = 0
    run_details = {'iter': np.zeros(params['maxit']),'fx': np.zeros(params['maxit']), 'gx': np.zeros(params['maxit']),'Fx': np.zeros(params['maxit']),'rerr': np.zeros(params['maxit'])}
    
    for k in range(0, params['maxit']):
        if  params['method'] == 'FISTA-RESTART':
            y = x + theta * (1/theta_pre - 1) * (x - x_pre)
            x_next = proxg( y - alpha * gradf(y), alpha)
            restart_condition = np.trace(np.transpose(y - x_next) @ (x_next - x))
            if restart_condition > params['restart_criterion']:
                theta_pre = 1
                theta = 1
                y = x
                x_next = proxg(y - alpha * gradf(y), alpha)
                print('restarted')
            theta_next = (np.sqrt(theta**4 + 4*theta**2) - theta**2) / 2
            x_pre = x
            x = x_next
            theta_pre = theta
            theta = theta_next
            
            Fx = fx(x_next) + gx(x_next)
            err = np.linalg.norm(y - y_pre)
            if k == 1:
                err_0 = err
                print('err_0 = ', err_0)
            if err / err_0 < params['stopping_criterion']:
                print('iter = {:f}, F*(X) = {:f}, f*(X) = {:f}, g*(X) = {:f},rerr = {:f}. '.format(k, Fx, fx(x_next), gx(x_next), np.log10(err/err_0)))
                break
            if k % 100 == 0:
                print('iter = {:f}, F(X) = {:f}, f(X) = {:f}, g(X) = {:f},log10(rerr) = {:f}.'.format(k, Fx, fx(x_next), gx(x_next), np.log10(err/err_0)))
            y_pre = y
    
            
        if params['method'] == 'FISTA':
            x_next = proxg(y - alpha * gradf(y), alpha)
            t_next = (1 + np.sqrt(4*t**2 + 1))/2
            y_next = x_next + (t - 1)/t_next * (x_next - x)
            Fx = fx(x_next) + gx(x_next)
            err = np.linalg.norm(y_next - y)
            if k == 1:
                err_0 = err           
#             if err / err_0 < params['stopping_criterion']:
#                 print('iter = {:f}, F*(X) = {:f}, f*(X) = {:f}, g*(X) = {:f},rerr = {:f}. '.format(k, Fx, fx(x_next), gx(x_next), err/err_0))
#                 break
            if k % 100 == 0:
                print('iter = {:f}, F(X) = {:f}, f(X) = {:f}, g(X) = {:f},rerr = {:f}.'.format(k, Fx, fx(x_next), gx(x_next), err/err_0))
            x = x_next
            y = y_next
            t = t_next

        run_details['iter'][k] = k
        run_details['fx'][k] = fx(x_next)
        run_details['gx'][k] = gx(x_next)
        run_details['Fx'][k] = fx(x_next) + gx(x_next)
        run_details['rerr'][k] = err/err_0
    print('iter = {:f}, F(X) = {:f}, f(X) = {:f}, g(X) = {:f},rerr = {:f}.'.format(k, Fx, fx(x_next), gx(x_next), err/err_0))
#     plt.semilogy(run_details['iter'], run_details['rerr'])            
    return x, run_details


def reconstruct_l1(image, indices, optimizer, params):
    # Wavelet operator
    r = RepresentationOperator(m=params["m"])
    # Define the overall operator
    forward_operator = lambda x: p_omega(r.WT(x), indices) ## TO BE FILLED ##  # P_Omega.W^T
    adjoint_operator = lambda x: r.W(p_omega_t(x, indices, params['m'])) ## TO BE FILLED ##  # W. P_Omega^T

    # Generate measurements
    b = p_omega(image, indices) ## TO BE FILLED ##

    fx = lambda x: (1.0 / 2) * np.linalg.norm(b - forward_operator(x))**2 ## TO BE FILLED ##
    gx = lambda x: params['lambda'] * np.linalg.norm(x, 1) ## TO BE FILLED ##
    proxg = lambda x, y: l1_prox(x, params['lambda'] * y)
    gradf = lambda x: adjoint_operator(forward_operator(x) - b) ## TO BE FILLED ##

    x, info = optimizer(fx, gx, gradf, proxg, params)
    return r.WT(x).reshape((params['m'], params['m'])), info


def reconstruct_TV(image, indices, optimizer, params):
    """
        image: undersampled image (mxm) to be reconstructed
        indices: indices of the undersampled locations
        optimizer: method of reconstruction (FISTA/ISTA function handle)
        params:
    """
    
    # Define the overall operator
    forward_operator = lambda x: p_omega(x, indices)  ## TO BE FILLED ##  # P_Omega.W^T
    adjoint_operator = lambda x: p_omega_t(x, indices, params['m']).reshape(params['N'],1) ## TO BE FILLED ##  # W. P_Omega^T

    # Generate measurements
    b = p_omega(image,indices) ## TO BE FILLED ##

    fx = lambda x: (1.0 / 2) * np.linalg.norm(b - forward_operator(x))**2 ## TO BE FILLED ##
    gx = lambda x: params['lambda'] * TV_norm(x, opt='iso') ## TO BE FILLED ##
    proxg = lambda x, y: denoise_tv_chambolle(x.reshape((params['m'], params['m'])),
                                              weight=params["lambda"] * y, eps=1e-5,
                                              n_iter_max=50).reshape((params['N'], 1))
    gradf = lambda x: adjoint_operator(forward_operator(x) - b) ## TO BE FILLED ##

    x, info = optimizer(fx, gx, gradf, proxg, params)
    return x.reshape((params['m'], params['m'])), info


# %%

if __name__ == "__main__":
    np.random.seed(RAND_SEED)

    ##############################
    # Load image and sample mask #
    ##############################
    shape = (256, 256)
    params = {
        'maxit': 200,
        'tol': 10e-15,
        'Lips': 1,  ## TO BE FILLED ##,
        'lambda': 0.01, ## TO BE FILLED ##,
        'x0': np.zeros((shape[0] * shape[1], 1)),
        'restart_criterion': 0, ## TO BE FILLED ##, gradient_scheme,
        'stopping_criterion': 10**(-15), ## TO BE FILLED ##,
        'iter_print': 50,
        'shape': shape,
        'restart_param': 50,
        'verbose': True,
        'm': shape[0],
        'rate': 0.4,
        'N': shape[0] * shape[1],
        'method': 'FISTA'
    }
    PATH = 'data/me.jpg' ## TO BE FILLED ##
    image = load_image(PATH, params['shape'])

    im_us, mask = apply_random_mask(image, params['rate'])
    indices = np.nonzero(mask.flatten(order='F'))[0]
    params['indices'] = indices
    # Choose optimization parameters


    #######################################
    # Reconstruction with L1 and TV norms #
    #######################################
    t_start = time.time()
    reconstruction_l1 = reconstruct_l1(image, indices, FISTA, params)[0]
    t_l1 = time.time() - t_start

    psnr_l1 = psnr(image, reconstruction_l1)
    ssim_l1 = ssim(image, reconstruction_l1)

    t_start = time.time()
    reconstruction_tv = reconstruct_TV(image, indices, FISTA, params)[0]
    t_tv = time.time() - t_start

    psnr_tv = psnr(image, reconstruction_tv)
    ssim_tv = ssim(image, reconstruction_tv)

    # Plot the reconstructed image alongside the original image and PSNR
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original')
    ax[1].imshow(im_us, cmap='gray')
    ax[1].set_title('Original with missing pixels')
    ax[2].imshow(reconstruction_l1, cmap="gray")
    ax[2].set_title('L1 - PSNR = {:.2f}\n SSIM  = {:.2f} - Time: {:.2f}s'.format(psnr_l1, ssim_l1, t_l1))
    ax[3].imshow(reconstruction_tv, cmap="gray")
    ax[3].set_title('TV - PSNR = {:.2f}\n SSIM  = {:.2f}  - Time: {:.2f}s'.format(psnr_tv, ssim_tv, t_tv))
    [axi.set_axis_off() for axi in ax.flatten()]
    plt.tight_layout()
    plt.show()
