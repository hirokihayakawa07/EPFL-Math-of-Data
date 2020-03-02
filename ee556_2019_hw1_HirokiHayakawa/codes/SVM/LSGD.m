%*******************  EE556 - Mathematics of Data  ************************
% Function:  [x, info] = LSGD(fx, gradf, parameter)       
% Purpose:   Implementation of the gradient descent with line-search.     
% Parameter: parameter.x0         - Initial estimate.
%            parameter.maxit      - Maximum number of iterations.
%            parameter.Lips       - Lipschitz constant for gradient.
%			 parameter.strcnvx	- strong convexity parameter
%            fx                 - objective function
%            gradf              - gradient mapping of objective function
%*************************** LIONS@EPFL ***********************************
function [x, info] = LSGD(fx, gradf, parameter)

    fprintf('%s\n', repmat('*', 1, 68));
    fprintf('Gradient Descent with line search\n')

    % Initialize x and L0.
   
     %%%% YOUR CODES HERE
	x = parameter.x0;
    L = parameter.Lips; % set L_0 = L
    
    % Main loop.
    for iter = 1:parameter.maxit
               
        % Start the clock.
        tic;
        
        % Update the next iteration. (main algorithmic steps here!)
        % Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        
         %%%% YOUR CODES HERE
        d = - gradf(x);
        L_k0 = 1/2 * L; % set L_k,0 = 1/2 L_k-1 for each iteration
        
        % find the minimum i that satisfies the condition of Line-search
        for i = 0:10^6
            if fx(x + 1/((2^i) * L_k0) * d) <= fx(x) - 1/((2^(i+1)) * L_k0) * norm(d)^2
                break
            end
        end
        
        % set stepsize which refrects the result of Line search and update x
        L = 2^i * L_k0;
        x_next = x + 1 / L * d;
        
        % Compute error and save data to be plotted later on.
        info.itertime(iter ,1)      = toc;
        info.fx(iter, 1)            = fx(x);
        
        % Print the information.
        if mod(iter, 5) == 0 || iter == 1
        fprintf('Iter = %4d,  f(x) = %0.9f\n', ...
                iter,  info.fx(iter, 1));
        end
                
        % Prepare the next iteration
        x   =   x_next;
        
    end
    % Finalization.
    info.iter           = iter;
    
end
%**************************************************************************
% END OF THE IMPLEMENTATION.
%**************************************************************************
