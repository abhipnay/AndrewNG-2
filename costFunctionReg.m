function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

a = X * theta;
%disp(size(a));

b = sigmoid(a);
%disp(b);
%b is the hypothesis

c = log(b);
%disp(c);

c_1 = log(1 - b);
%disp(c_1);

d = y.*c;
%disp(d);

e = (1 - y).*c_1;
%disp(e);

J_1 = (sum(d) + sum(e)) * (-1/m);

const = lambda/(2 * m);

theta_squared = theta .* theta;

theta_squared(1,:) = [];

J_2 = (sum(theta_squared)) * const;

J = J_1 + J_2;

%Calculating the Gradient Descent

f = b - y;

f_t = f.';

g = f_t * X;

grad_1 = (g.') * (1/m);

const_2 = (lambda)/(m);

theta_1 = theta;
theta_1(1,1) = 0;

grad_2 = const_2 * theta_1;

grad = grad_1 + grad_2;

% =============================================================

end
