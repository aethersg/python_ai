import numpy as np
from sklearn import linear_model
from logistic_regression_classifier.utilities import visualize_classifier

# Define sample input data
x = np.array([[3.1, 7.2], [4, 6.7], [2.9, 8], [5.1, 4.5], [6, 5], [5.6, 5],
              [3.3, 0.4], [3.9, 0.9], [2.8, 1], [0.5, 3.4], [1, 4], [0.6, 4.9]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

# Create the logistic regression classifier
# Change the C value from 1 to 100 and you will see the boundaries become more accurate:
# The reason is that C imposes a certain penalty on mis-classification, so the algorithm
# customizes more to the training data. You should be careful with this parameter, because if
# you increase it by a lot, it will over-fit to the training data and it won't generalize well.

classifier = linear_model.LogisticRegression(solver='liblinear', C=100)

# Train the classifier
classifier.fit(x, y)

# Visualize the performance of the classifier
visualize_classifier(classifier, x, y)
