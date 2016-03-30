clear

data = load('data.txt');
data = [ones(size(data, 1), 1), data];
labels = load('labels.txt');

train_labels = labels(1:2000);
test_data = data(2001:4601, :);
test_labels = labels(2001:4601);
amounts = [200, 500, 800, 1000, 1500, 2000];
accuracies = [];

for amount = amounts
    weights = logistic_train(data(1:amount, :), labels(1:amount));
    predictions = sigmf(test_data * weights, [1 0]) >= 0.5;
    
    errors = test_labels ~= predictions;
    accuracy = sum(errors) / numel(errors);
    accuracies = [accuracies, accuracy];
end

plot(amounts, accuracies)
%axis([200, 2000, 0.98, 1])