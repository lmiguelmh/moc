
%data = load('ex1data1.txt'); % read comma separated data
% disp(data);
% population  profit
% 6.11010   17.59200
% 5.52770    9.13020
% 8.51860   13.66200
% 7.00320   11.85400

%{
data = [1 -890; 2 -1411; 2 -1560; 3 -2220; 3 -2091; 4 -2878; 5 -3537; 6 -3268; 6 -3920; 6 -4163; 8 -5471; 10 -5157];
X = data; 
y = data(:, 2);
%plot(X, y, 'rx', 'MarkerSize', 10); % Plot the data
%theta = zeros(1, 2); % THETA ESTÁ DE FORMA DIFERENTE EN EL EX1.M
theta = [0; 0];
%disp(theta);
J = computeCost(X, y, theta);
output_precision(8);
disp(J);
theta = [-569; -530];
J = computeCost(X, y, theta);
disp(J);
%sprintf("%5.8f", J);
%}

%{
X = [1 1; 1 2; 1 3];
y = [1; 2; 3];
theta = [0;1];
J = computeCost(X,y,theta);
%}

data = [1 -890; 2 -1411; 2 -1560; 3 -2220; 3 -2091; 4 -2878; 5 -3537; 6 -3268; 6 -3920; 6 -4163; 8 -5471; 10 -5157];
X = data(:, 1);
y = data(:, 2);
m = length(y);
X = [ones(m, 1), data(:,1)]; 

theta = [-570; -531];
J = computeCost(X, y, theta);

theta = [-569.601597962; -530.906795758]; %result
J = computeCost(X, y, theta);

theta = [-569; -530];
J = computeCost(X, y, theta);

alpha = 0.005;
num_iters = 10000;
theta = [-560; -520];
%myGradientDescent(X, y, theta, alpha, num_iters);
gradientDescent(X, y, theta, alpha, num_iters);

featureNormalize(data);

X = [ones(m, 1), data(:,1)]; 
y = data(:, 2);
normalEqn(X, y);

%{
predictions = X*theta';
delta0 = 1/m * sum((predictions-y) .* X(:,1))
delta1 = 1/m * sum((predictions-y) .* X(:,2))

predictions = X*theta';
delta = [];
for j = 1:size(X,2)
  dj = 1/m * sum((predictions-y) .* X(:,j))
  delta = [delta dj];
end
delta
%}

%{
delta = [delta1];
delta = [delta0 delta];
delta = [delta1 delta];
size(X,2)
%}