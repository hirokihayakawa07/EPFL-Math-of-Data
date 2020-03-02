%*******************  EE556 - Mathematics of Data  ************************
% Function:  [x, info] = SAG(fx, gradf, parameter)       
% Purpose:   Implementation of the stochastic averaging gradient algorithm.     
% Parameter: parameter.x0         - Initial estimate.
%            parameter.maxit      - Maximum number of iterations.
%            parameter.L          - Maximum of Lipschitz constant for gradients.
%            parameter.strcnvx    - Strong convexity parameter of f(x).
%            parameter.no0functions - number of functions
%            fx                 - objective function
%            gradfsto           - stochastic gradient mapping of objective function
%*************************** LIONS@EPFL ***********************************
function [x, info] = SAG(fx, gradfsto, parameter)

    fprintf('%s\n', repmat('*', 1, 68));
    fprintf('Stochastic Gradient Descent with averaging\n')
        
        
    % Initialize.
    %%%% YOUR CODES HERE
	x = parameter.x0;
    n = parameter.no0functions;
    Lmax = parameter.Lmax;
    p = length(x);
    v = zeros(p,n); % prepare v to store v^k_i, v^0 = zeros
    	    
    % Main loop.    
    for iter    = 1:parameter.maxit
            
        % Start timer
        tic;
       
        % Update the next iteration. (main algorithmic steps here!)
        % Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        
         %%%% YOUR CODES HERE
        ik = randi(n); % pick ik uniformly at random
        x_next = x - 1/16/Lmax * mean(v,2); % update x, sum v^k_i /n = average v^k_i for i in 1,...,n
        v(:,ik) = gradfsto(x,ik); % update v
               
        % Compute error and save data to be plotted later on.
        info.itertime(iter ,1)      = toc;
        info.fx(iter, 1)            = fx(x);
        
         % Print the information.
        if mod(iter, 5) == 0 || iter == 1
        fprintf('Iter = %4d,  f(x) = %0.9f\n', ...
                iter,  info.fx(iter, 1));
        end
        
        % prepare next iteration
        x = x_next;

    end

    % Finalization.
    info.iter           = iter;
    
end
%**************************************************************************
% END OF THE IMPLEMENTATION.
%**************************************************************************
