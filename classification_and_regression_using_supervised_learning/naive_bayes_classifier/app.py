import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from classification_and_regression_using_supervised_learning.naive_bayes_classifier.utilities import \
    visualize_classifier

# Input file containing data
input_file = 'data_multi_var.txt'

# Load data from input file
data = np.loadtxt(input_file, delimiter=',')
x, y = data[:, :-1], data[:, -1]

# Create Naive Bayes classifier
classifier = GaussianNB()

# Train the classifier
classifier.fit(x, y)

# Predict the values for training data
y_pred = classifier.predict(x)

# Compute accuracy
accuracy = 100.0 * (y == y_pred).sum() / x.shape[0]
print("Accuracy of Naive Bayes classifier =", round(accuracy, 2), "%")
# Visualize the performance of the classifier
visualize_classifier(classifier, x, y)

# Split data into training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
classifier_new = GaussianNB()
classifier_new.fit(x_train, y_train)
y_test_pred = classifier_new.predict(x_test)

# compute accuracy of the classifier
accuracy = 100.0 * (y_test == y_test_pred).sum() / x_test.shape[0]
print("Accuracy of the new classifier =", round(accuracy, 2), "%")
# Visualize the performance of the classifier
visualize_classifier(classifier_new, x_test, y_test)

num_folds = 3
accuracy_values = cross_val_score(classifier, x, y, scoring='accuracy', cv=num_folds)
print("Accuracy: " + str(round(100 * accuracy_values.mean(), 2)) + "%")
precision_values = cross_val_score(classifier, x, y, scoring='precision_weighted', cv=num_folds)
print("Precision: " + str(round(100 * precision_values.mean(), 2)) + "%")
recall_values = cross_val_score(classifier, x, y, scoring='recall_weighted', cv=num_folds)
print("Recall: " + str(round(100 * recall_values.mean(), 2)) + "%")
f1_values = cross_val_score(classifier, x, y, scoring='f1_weighted', cv=num_folds)
print("F1: " + str(round(100 * f1_values.mean(), 2)) + "%")
