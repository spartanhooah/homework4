clear
data = load('data.txt');
labels = load('labels.txt');
data = [ones(size(data, 1), 1), data];

train_data = data(1:50, :);
train_labels = labels(1:50, :);
test_data = data(2001:4601, :);
test_labels = labels(2001:4601);

pars = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5];

for par = pars
    [weights, bias] = logistic_l1_train(train_data, train_labels, par);
    predictions = test_data * weights;
    disp(nnz(weights))
end