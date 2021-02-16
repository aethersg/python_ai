# Standalone simple linear regression example
from math import sqrt
import matplotlib.pyplot as plt
import sklearn.metrics as sm
 
# Calculate root mean squared error
# gets a measure of how close two numbers are.
def rmse_metric(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)
	mean_error = sum_error / float(len(actual))
	return sqrt(mean_error)
 
# Evaluate regression algorithm on training dataset
def evaluate_algorithm(dataset, algorithm):
	test_set = list()
	for row in dataset:
		row_copy = list(row)
		row_copy[-1] = None
		test_set.append(row_copy)
	y_test_pred = algorithm(dataset, test_set)
	print(y_test_pred)
	y_test = [row[-1] for row in dataset]
	x_test = [row[0] for row in dataset]
	
	# Plot outputs
	plt.scatter(x_test, y_test, color='green')
	plt.plot(x_test, y_test_pred, color='black', linewidth=4)
	plt.xticks(())
	plt.yticks(())
	plt.show()

	# Compute performance metrics
	print("Linear regressor performance:")
	print("Mean absolute error =", round(sm.mean_absolute_error(y_test,
																y_test_pred), 2))
	print("Mean squared error =", round(sm.mean_squared_error(y_test,
															y_test_pred), 2))
	print("Median absolute error =", round(sm.median_absolute_error(y_test,
																	y_test_pred), 2))
	print("Explain variance score =", round(sm.explained_variance_score(y_test,
																		y_test_pred), 2))
	print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

	rmse = rmse_metric(y_test, y_test_pred)
	print('root mean squared error: %.3f' % (rmse))
	return y_test_pred
 
# Calculate the mean value of a list of numbers
def mean(values):
	return sum(values) / float(len(values))
 
# Calculate covariance between x and y
# how much do the variables change together
def covariance(x, mean_x, y, mean_y):
	covar = 0.0
	for i in range(len(x)):
		covar += (x[i] - mean_x) * (y[i] - mean_y)
	return covar
 
# Calculate the variance of a list of numbers
# How volatile is the variable 
def variance(values, mean):
	return sum([(x-mean)**2 for x in values])
 
# Calculate coefficients
# how important are the variables
def coefficients(dataset):
	x = [row[0] for row in dataset]
	y = [row[1] for row in dataset]
	x_mean, y_mean = mean(x), mean(y)
	#calculates the 
	b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
	#calculates the starting point
	b0 = y_mean - b1 * x_mean
	return [b0, b1]
 
# Simple linear regression algorithm
def simple_linear_regression(train, test):
	predictions = list()
	b0, b1 = coefficients(train)
	for row in test:
		yhat = b0 + b1 * row[0]
		predictions.append(yhat)
	return predictions
 
# Test simple linear regression
dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
evaluate_algorithm(dataset, simple_linear_regression)
