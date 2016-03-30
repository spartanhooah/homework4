clear
load('ad_data.mat');

% Add bias
X_train = [ones(size(X_train, 1), 1) X_train];
X_test = [ones(size(X_test, 1), 1) X_test];

pars = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];
num_selected = zeros(size(pars));
AUCs = zeros(size(pars));
i = 1;
for par = pars
    [weights, bias] = logistic_l1_train(X_train, y_train, par);
    num_selected(i) = nnz(weights);
    predictions = X_test * weights;
    [X, Y, T, AUC] = perfcurve(y_test, predictions, 1);
    AUCs(i) = AUC;
    i = i + 1;
end