%*******************  EE556 - Mathematics of Data  ************************
% Function:  [x, info] = SGD(fx, gradf, parameter)       
% Purpose:   Implementation of the stochastic gradient descent algorithm.     
% Parameter: parameter.x0         - Initial estimate.
%            parameter.maxit      - Maximum number of iterations.
%            parameter.L          - Maximum of Lipschitz constant for gradients.
%            parameter.strcnvx    - Strong convexity parameter of f(x).
%            parameter.no0functions - number of functions
%            fx                 - objective function
%            gradfsto           - stochastic gradient mapping of objective function
%*************************** LIONS@EPFL ***********************************
function [x, info] = SGD(fx, gradfsto, parameter)
 
    fprintf('%s\n', repmat('*', 1, 68));
    fprintf('Stochastic Gradient Descent\n')
        
        
    % Initialize x and alpha.
    %%%% YOUR CODES HERE
	x = parameter.x0;
    n = parameter.no0functions; % to be used to pick up ik in {1,...,n}
    k = 1; % alpha = 1/k, k starts from 1
    
    % Main loop.
    for iter    = 1:parameter.maxit
            
        % Start timer
        tic;
       
        % Update the next iteration. (main algorithmic steps here!)
        % Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        
        
         %%%% YOUR CODES HERE
        % update the next iterations
        x_next = x - 1/k * gradfsto(x,randi(n));
        k = k + 1;
        
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