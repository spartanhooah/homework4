function [w, c] = logistic_l1_train(data, labels, par)
% OUTPUT w is equivalent to the first d dimension of weights in logistic train
% c is the bias term, equivalent to the last dimension in weights in logistic train.

data = load('data.txt');
labels = load('labels.txt');
data = [ones(size(data, 1), 1), data];

train_data = data(1:50, :);
train_labels = labels(1:50, :);
test_data = data(2001:4601, :);
test_labels = labels(2001:4601);

% Specify the options (use without modification).
opts.rFlag = 1; % range of par within [0, 1].
opts.tol = 1e-6; % optimization precision
opts.tFlag = 4; % termination options.
opts.maxIter = 5000; % maximum iterations.

[w, c] = LogisticR(train_data, train_labels