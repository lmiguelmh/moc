function [theta, J_history] = myGradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

theta0 = theta(1,1);
theta1 = theta(1,2);
x = X(:,2);

__a0 = theta0;
__a1 = theta1;

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    hx = theta0 + theta1 * x;
    _a0 = theta0 - alpha * 1 / m * sum(hx - y) * 1;
    _a1 = theta1 - alpha * 1 / m * sum((hx - y) .* x);
    theta0 = _a0;
    theta1 = _a1;

    %{
    hx = X*theta';
    %hx-y
    %delta = 1/length(X) * sum((hx-y)* X);
    %delta = 1/m * sum((hx-y).*X(:,2));
    delta = 1/m * sum((hx-y).*X(:,2));
    theta = theta - alpha * delta;
    %}
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

theta0 
theta1 

end
