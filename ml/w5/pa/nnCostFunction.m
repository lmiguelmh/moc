function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%


%size(Theta1); % 25x401
%size(Theta2); % 10x26
%size(X) % 5000x400 

a1 = X; 
a1 = [ones(size(X, 1), 1) a1];
z2 = Theta1 * a1';
a2 = sigmoid(z2);

%size(a2) % 25x5000
a2 = [ones(1, size(a2, 2)); a2];
z3 = Theta2 * a2;
a3 = sigmoid(z3);

%size(a3) % 10x5000 
%[_, p] = max(a3, [], 1);
%p = p'; % esto obtiene el máximo valor 

predictions = a3; % 10 predictions o h(x) (10x5000) 
%round(predictions(:,1)) % ahora son 1s y 0s (No es necesario para el cálculo del error!)

% convertir Y a un vector de 1 y 0s
%size(Y); % 5000x1
yvector = zeros(size(predictions)); % (10x5000)
%fprintf("generando matriz y\n");
% todo this could be replaced by a logical array?
for i=1:size(y)
  yvector(y(i), i) = 1;
end
%fprintf("finalizando matriz y\n");
%yvector(:, size(y))

cost = -yvector .* log(predictions) - (1 .- yvector) .* log(1 .- predictions);
%J = 1/m * sum(sum(cost));
J = 1/m * sum(sum(cost)) + lambda / (2*m) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%


%size(X) % 5000x400 
%yvector % 10x5000
%m=5000

%disp("Theta1");disp(size(Theta1)); % 25x401
%disp("Theta1_grad");disp(size(Theta1_grad)); % 25x401

%disp("Theta2");disp(size(Theta2)); % 10x26
%disp("Theta2_grad");disp(size(Theta2_grad)); %10x26

%for t = 1:m
Delta1 = zeros(size(Theta1_grad));
Delta2 = zeros(size(Theta2_grad));
for t = 1:m

  yt = yvector(:,t); 
  %disp("yt");disp(size(yt)); % 10x1
  a1t = a1(t,:);
  %disp("a1t");disp(size(a1t)); % 1x401
  a2t = a2(:,t); 
  %disp("a2t");disp(size(a2t)); % 26x1
  a3t = a3(:,t); 
  %disp("a3t");disp(size(a3t)); % 10x1
  z2t = z2(:,t); 
  %disp("z2t");disp(size(z2t)); % 25x1
  
  delta3t = a3t - yt; 
  %disp("delta2t");disp(size(delta2t)); % 10x1
  
  delta2t = (Theta2(:,2:end)' * delta3t) .* sigmoidGradient(z2t);
  %disp("delta3t");disp(size(delta3t)); % 25x1
  
  Delta2 = Delta2 + delta3t * a2t'; 
  Delta1 = Delta1 + delta2t * a1t; % a1t no tiene transpuesta por la implementación de arriba
end

%Delta1 = Delta1 ./ m;
%disp("Delta1");disp(size(Delta1)); % 25x401
%Delta2 = Delta2 ./ m;
%disp("Delta2");disp(size(Delta2)); % 10x26

% without regularization
%Theta1_grad = Delta1 ./ m;
%Theta2_grad = Delta2 ./ m;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


%disp("Delta1");disp(size(Delta1)); % 25x401
%disp("Delta2");disp(size(Delta2)); % 10x26

%for i=size(Theta1_grad, 1)
%  for j=size(Theta1_grad,2)
%    
%  end
%end

Theta1_grad = Delta1 ./ m;
Theta2_grad = Delta2 ./ m;
%regularizing layer1 l=1
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda ./ m .* Theta1(:,2:end);
%regularizing layer2 l=2
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda ./ m .* Theta2(:,2:end);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
