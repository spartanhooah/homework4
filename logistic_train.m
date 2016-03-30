function [weights] = logistic_train(data, labels, varargin)
%
% code to train a logistic regression classifier
%
% INPUTS:
%   data    = n*(d+1) matrix withn samples and d features, where
%             column d+1 is all ones (corresponding to the intercept term)
%   labels  = n*1 vector of class labels (taking values 0 or 1)
%   epsilon = optional argument specifying the convergence
%             criterion - if the change in the absolute difference in
%             predictions, from one iteration to the next, averaged across
%             input features, is less than epsilon, then halt
%             (if unspecified, use a default value of 1e-5)
%   maxiter = optional argument that specifies the maximum number of
%             iterations to execute (useful when debugging in case your
%             code is not converging correctly!)
%             (if unspecified can be set to 1000)
%
% OUTPUT:
%    weights = (d+1)*1 vector of weights where the weights correspond to
%              the columns of "data"

% Parse arguments, assign defaults as needed
num_args = length(varargin);
if num_args == 0
    epsilon = 1e-5;
    maxiter = 1000;
elseif num_args == 1
    epsilon = varargin{1};
    maxiter = 1000;
elseif num_args == 2
    epsilon = varargin{1};
    maxiter = varargin{2};
else
    disp('Too many arguments given.')
    return
end

weights = zeros(size(data, 2), 1);
y_new = zeros(size(data, 1), 1);

for i = 1:maxiter
    % Build the R matrix by calculating the y_n entries, then the entries
    % along the diagonal, and finally by turning the vector into the
    % matrix's diagonal.
    y_old = sigmf(data * weights, [1 0]);
    R_vec = y_old .* (1 - y_old);
    R = diag(R_vec);
    z = data * weights - inv(R + 0.2*eye(size(R, 1))) * (y_old - labels);

    weights = weights - inv(data' * R * data) * data' * R * z;
    y_new = sigmf(data * weights, [1 0]);
    if sum(abs(y_old - y_new)) / size(y_new, 1) < epsilon
        break
    end
    y_old = y_new;
end