import pickle
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt

# Input file containing data
input_file = 'data.txt'

# Read data
data = np.loadtxt(input_file, delimiter=',')
x, y = data[:, :-1], data[:, -1]

# Train and test split
num_training = int(0.8 * len(x))
num_test = len(x) - num_training
# Training data
x_train, y_train = x[:num_training], y[:num_training]
# Test data
x_test, y_test = x[num_training:], y[num_training:]

# Create linear regressor object
regressor = linear_model.LinearRegression()
# Train the model using the training sets
regressor.fit(x_train, y_train)

# Predict the output
y_test_pred = regressor.predict(x_test)

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

# Model persistence
output_model_file = 'model.pkl'
# Save the model
with open(output_model_file, 'wb') as f:
    pickle.dump(regressor, f)

# Load the model
with open(output_model_file, 'rb') as f:
    regressor_model = pickle.load(f)
# Perform prediction on test data
y_test_pred_new = regressor_model.predict(x_test)
print("\nNew mean absolute error =", round(sm.mean_absolute_error(y_test,
                                                                  y_test_pred_new), 2))
