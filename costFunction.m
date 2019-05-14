function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

disp(m);

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

a = X * theta;
disp(size(a));

b = sigmoid(a);
disp(b);
%b is the hypothesis

c = log(b);
disp(c);

c_1 = log(1 - b);
disp(c_1);

d = y.*c;
disp(d);

e = (1 - y).*c_1;
disp(e);

J = (sum(d) + sum(e)) * (-1/m);

%calculating partial derivatives below

f = b - y;
disp(f);

f_t = f.';

g = f_t * X;

grad = (g.') * (1/m);
disp(grad);
% =============================================================

end
