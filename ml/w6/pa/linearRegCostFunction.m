function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% h = theta0*x0 + theta1*x1
%print(size(X)); % 12 x 2 (incluye los 1s)
%print(size(y)); % 12 x 1
%print(size(theta)); % 2 x 1
%print(size(lambda)); % 1 x 1


% a different approach from the other weeks
% h are the predictions
h = theta'*X'; % remember X no x (in the given formula)
h = h';
sqrErrors = (h-y).^2;
J = 1/(2*m) * sum(sqrErrors);
J_reg = lambda / (2*m) * sum(theta(2:end).^2);
J = J + J_reg;

grad = 1/m .* (h - y)' * X; % the matrix multiplication calcs the sum too!
grad_reg = [0; lambda ./ m .* theta(2:end)]';
grad = grad + grad_reg;


% =========================================================================

grad = grad(:);

end
