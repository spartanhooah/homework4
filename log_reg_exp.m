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
    m_weights = glmfit(data(1:amount, 2:end), labels(1:amount), 'binomial');
    predictions = test_data * weights;
    m_predictions = test_data * m_weights;
    errors = sum(abs(test_labels - predictions));
    test_error = errors / size(test_labels, 1);
    accuracies = [accuracies, test_error];
end

plot(amounts, accuracies)
axis([200, 2000, 0.98, 1])