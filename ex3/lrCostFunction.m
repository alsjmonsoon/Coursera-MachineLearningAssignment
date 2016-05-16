function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations.
%
% Hint: When computing the gradient of the regularized cost function,
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta;
%           temp(1) = 0;   % because we don't add anything for j = 0
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
% make hypothesis vector, formed from the sigmoid() of the products of theta and X
h=sigmoid(X*theta);

% Make the costFunction
d=(-(y')*log(h))-((1-y)'*log(1-h));

% Scale the result by 1/m.
unreg=(1/m)*d;

% modify theta(1)=0 because theta1 is the biased feautre
theta(1)=0;

% calculate the sum of the squares of theta using vector multiplication
regTerm=theta'*theta;

%  scale the cost regularization term by (lambda / (2 * m))
reg=(lambda/(2*m))*regTerm;

% add your unregularized and regularized cost terms together
J=unreg+reg;

% set unregularized gradient (note: not gradient descent)
unreg_grad=(1/m)*(X'*(h-y));

% calculate the regularized gradient term
reg_grad=(lambda/m)*theta;

% sum up the unregularized and regularized terms
grad=unreg_grad+reg_grad;










% =============================================================

grad = grad(:);

end
