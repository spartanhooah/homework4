clear

data = dlmread('data.txt', ' ');
data = [ones(size(data, 1), 1), data];
labels = dlmread('labels.txt', ' ', 0, 1);
train_data = data(1:2000, :);
train_labels = labels(1:2000);
test_data = data(2001:4601, :);
test_labels = labels(2001:4601);
amounts = [200, 500, 800, 1000, 1500, 2000];

 for amount= amounts
     weights = logistic_train(train_data(1:amount, :), labels(1:amount));
     predictions = test_data * weights;
     errors = abs(test_labels - predictions');
     test_error = errors / size(test_labels, 1);
     disp(test_error);
 end