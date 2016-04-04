clear

% Load data, add bias.
data = load('data.txt');
data = [ones(size(data, 1), 1), data];
labels = load('labels.txt');

% Split up data into training and testing
train_labels = labels(1:2000);
test_data = data(2001:4601, :);
test_labels = labels(2001:4601);

% amounts determines how much of the training data to use
amounts = [200, 500, 800, 1000, 1500, 2000];

accuracies = [];

for amount = amounts
    weights = logistic_train(data(1:amount, :), labels(1:amount));
    predictions = test_data * weights;
    
    correct = test_labels == predictions;
    accuracy = sum(correct) / numel(correct);
    accuracies = [accuracies, accuracy];
end

plot(amounts, accuracies, '-d')
title('{\bf Testing Accuracy vs. Training Set Size}')
xlabel('Training set size')
ylabel('Testing accuracy')