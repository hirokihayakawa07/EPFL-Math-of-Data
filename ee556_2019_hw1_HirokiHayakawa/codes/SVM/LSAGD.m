%*******************  EE556 - Mathematics of Data  ************************
% Function:  [x, info] = LSAGD (fx, gradf, parameter)
% Purpose:   Implementation of AGD with line search.     
% Parameter: parameter.x0         - Initial estimate.
%            parameter.maxit      - Maximum number of iterations.
%            parameter.Lips       - Lipschitz constant for gradient.
%			 parameter.strcnvx	- strong convexity parameter
%            fx                 - objective function
%            gradf              - gradient mapping of objective function
%*************************** LIONS@EPFL ***********************************
function [x, info] = LSAGD(fx, gradf, parameter)

    fprintf('%s\n', repmat('*', 1, 68));
    fprintf('Accelerated Gradient with line search\n')
    
    % Initialize x, y, t and L0.
     %%%% YOUR CODES HERE
    x = parameter.x0;
    y = parameter.x0;
    t = 1;
    L = parameter.Lips;
    
    % Main loop.
    for iter = 1:parameter.maxit
                           
        % Start the clock.
        tic;
        
        % Update the next iteration. (main algorithmic steps here!)
        % Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        
         %%%% YOUR CODES HERE
        
        d = - gradf(y);
        L_k0 = 1/2 * L; % set L_k,0 = 1/2 L_k-1 for each iteration
        
        % find the minimum i that satisfies the condition of Line-search
        for i = 0:10^6
            if fx(y + 1/((2^i) * L_k0) * d) <= fx(y) - 1/((2^(i+1)) * L_k0) * norm(d)^2
                break
            end
        end
        
        % set stepsize which refrects the result of Line search and update
        % the next iterations
        L = 2^i * L_k0;
        x_next = y + 1 / L * d;
        t_next = 1/2 * (1 + sqrt(1 + 4 * (L / (2*L_k0)) * t^2));
        y = x_next + ((t - 1) / (t_next) ) * (x_next - x);
        
        % Compute error and save data to be plotted later on.
        info.itertime(iter ,1)  = toc;
        info.fx(iter, 1)        = fx(x);
        
        % Print the information.
        if mod(iter, 5) == 0 || iter == 1
        fprintf('Iter = %4d, f(x) = %0.9f\n', ...
                iter, info.fx(iter, 1));
        end

        % Prepare the next iteration
        x     = x_next;
        t     = t_next;

    end

    % Finalization.
    info.iter           = iter;

end
%**************************************************************************
% END OF THE IMPLEMENTATION.
%**************************************************************************
