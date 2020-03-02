%*******************  EE556 - Mathematics of Data  ************************
% Function:  [x, info] = SVR(fx, gradf, gradfsto, parameter)       
% Purpose:   Implementation of the stochastic gradient descent algorithm with variance reduction.     
% Parameter: parameter.x0         - Initial estimate.
%            parameter.maxit      - Maximum number of iterations.
%            parameter.L          - Maximum of Lipschitz constant for gradients.
%            parameter.strcnvx    - Strong convexity parameter of f(x).
%            parameter.no0functions - number of functions
%            fx                 - objective function
%            gradfsto           - stochastic gradient mapping of objective function
%            gradf              - gradient mapping of objective function
%*************************** LIONS@EPFL ***********************************
function [x, info] = SVR(fx, gradf, gradfsto, parameter)
 
    fprintf('%s\n', repmat('*', 1, 68));
    fprintf('Stochastic Gradient Descent with variance reduction\n')
        
        
    % Initialize.
    %%%% YOUR CODES HERE
	x = parameter.x0;
    n = parameter.no0functions;
    Lmax = parameter.Lmax; 
    gamma = 0.01 / Lmax;
    q = fix(1000 * Lmax);

    % Main loop.
    for iter    = 1:parameter.maxit
            
        % Start timer
        tic;
       
        % Update the next iteration. (main algorithmic steps here!)
        % Use the notation xb_next for x_{k+1}, and xb for x_{k}, and similar for other variables.
        
         %%%% YOUR CODES HERE
        % implement SVR algorithm
        x_tilde = x;
        v = gradf(x);
        x_tilde_l = zeros(10,q+1);
        x_tilde_l(:,1) = x_tilde;
        
        for l = 0 : q-1
            il = randi(n); % pick il uniformly at random
            vl = gradfsto(x_tilde_l(:,l+1),il) - gradfsto(x_tilde,il) + v;
            x_tilde_l(:,l+2) = x_tilde_l(:,l+1) - gamma * vl;
        end
        x_next = mean(x_tilde_l(:,2:q+1),2); % update x
                       
        % Compute error and save data to be plotted later on.
        info.itertime(iter ,1)      = toc;
        info.fx(iter, 1)            = fx(x);
        
         % Print the information.
        if mod(iter, 5) == 0 || iter == 1
        fprintf('Iter = %4d,  f(x) = %0.9f\n', ...
                iter,  info.fx(iter, 1));
        end
        
        % Prepare the next iteration
        x = x_next;
 
    end
 
    % Finalization.
    info.iter           = iter;
    
end
%**************************************************************************
% END OF THE IMPLEMENTATION.
%**************************************************************************?
