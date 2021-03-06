%*******************  EE556 - Mathematics of Data  ************************
% Function:  [x, info] = GDstr(fx, gradf, parameter)       
% Purpose:   Implementation of the gradient descent algorithm.     
% Parameter: parameter.x0         - Initial estimate.
%            parameter.maxit      - Maximum number of iterations.
%            parameter.Lips       - Lipschitz constant for gradient.
%			 parameter.strcnvx	- strong convexity parameter
%            fx                 - objective function
%            gradf              - gradient mapping of objective function
%*************************** LIONS@EPFL ***********************************
function [x, info] = GDstr(fx, gradf, parameter)

    fprintf('%s\n', repmat('*', 1, 68));
    fprintf('Gradient Descent with strong convexity\n')
        
    % Initialize x and alpha.
     %%%% YOUR CODES HERE
	x =  parameter.x0; % initial condition
    alpha = 2 / ( parameter.Lips + parameter.strcnvx); % stepsize
    
    % Main loop.
    for iter    = 1:parameter.maxit
            
        % Start timer
        tic;
       
        % Update the next iteration. (main algorithmic steps here!)
        % Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        
         %%%% YOUR CODES HERE
         x_next = x - alpha * eye(10) * gradf(x); % update x
        % Compute error and save data to be plotted later on.
        info.itertime(iter ,1)      = toc;
        info.fx(iter, 1)            = fx(x);
        
         % Print the information.
        if mod(iter, 5) == 0 || iter == 1
        fprintf('Iter = %4d,  f(x) = %0.9f\n', ...
                iter,  info.fx(iter, 1));
        end
        
        % Prepare the next iteration
        x   = x_next;

    end

    % Finalization.
    info.iter           = iter;
    
end
%**************************************************************************
% END OF THE IMPLEMENTATION.
%**************************************************************************
