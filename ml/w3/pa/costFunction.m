function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%


% LOGISTIC REGRESSION COST FUNCTION 
% vectorized form 
%predictions = 1 ./ (1 .+ sigmoid(X*theta)); %ERROR!
predictions = sigmoid(X*theta);
cost = -y .* log(predictions) - (1 .- y) .* log(1 .- predictions);
J = 1/m * sum(cost);
%J

% LOGISTIC REGRESSION COST FUNCTION
% iterative form 
%sum_cost = 0;
%for i=1:m,
%%  prediction = 1 / (1 + sigmoid(X(i,:)*theta)); %ERROR!
%  prediction = sigmoid(X(i,:)*theta);
%  sum_cost += -y(i)*log(prediction) - (1-y(i))*log(1-prediction);
%end;
%J = 1/m * sum_cost;
%J

% LOGISTIC REGRESSION GRADIENT (SEE ml.w3.v5-last image!)
% vectorized form
%grad = (1/m * sum( (predictions - y) .* X))';
grad = 1/m * sum((predictions - y) .* X);

% =============================================================

end
