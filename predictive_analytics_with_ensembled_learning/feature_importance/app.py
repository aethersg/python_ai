import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split
import sklearn.utils

# Load housing data
housing_data = datasets.load_boston()

# Shuffle the data
x, y = sklearn.utils.shuffle(housing_data.data, housing_data.target, random_state=7)

# Split data into training and testing datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

# AdaBoost Regressor model
regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=400, random_state=7)
regressor.fit(x_train, y_train)

# Evaluate performance of AdaBoost regressor
y_pred = regressor.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
print("\nADABOOST REGRESSOR")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))

# Extract feature importance
feature_importance = regressor.feature_importances_
feature_names = housing_data.feature_names

# Normalize the importance values
feature_importance = 100.0 * (feature_importance / max(feature_importance))

# Sort the values and flip them
index_sorted = np.flipud(np.argsort(feature_importance))

# Arrange the x ticks
pos = np.arange(index_sorted.shape[0]) + 0.5

# Plot the bar graph
plt.figure()
plt.bar(pos, feature_importance[index_sorted], align='center')
plt.xticks(pos, feature_names[index_sorted])
plt.ylabel('Relative Importance')
plt.title('Feature importance using AdaBoost regressor')
plt.show()
