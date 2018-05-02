import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

# Load input data
input_file = 'data_random_forests.txt'
data = np.loadtxt(input_file, delimiter=',')
x, y = data[:, :-1], data[:, -1]

# Separate input data into three classes based on labels
class_0 = np.array(x[y == 0])
class_1 = np.array(x[y == 1])
class_2 = np.array(x[y == 2])

# Split the data into training and testing datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=5)

# Define the parameter grid
parameter_grid = [{'n_estimators': [100], 'max_depth': [2, 4, 7, 12, 16]},
                  {'max_depth': [4], 'n_estimators': [25, 50, 100, 250]}]

metrics = ['precision_weighted', 'recall_weighted']

for metric in metrics:
    print("\n##### Searching optimal parameters for", metric)
    classifier = GridSearchCV(ExtraTreesClassifier(random_state=0), parameter_grid, cv=5, scoring=metric)
    classifier.fit(x_train, y_train)
    print("\nGrid scores for the parameter grid:")
    for params, avg_score, _ in classifier.grid_scores_:
        print(params, '-->', round(avg_score, 3))
    print("\nBest parameters:", classifier.best_params_)
    y_pred = classifier.predict(x_test)
    print("\nPerformance report:\n")
    print(classification_report(y_test, y_pred))
