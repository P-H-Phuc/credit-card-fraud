# =============================================================================
# Credit card fraud project
# =============================================================================

# import library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import data
data = pd.read_csv('C:/Users/Administrator/Documents/projectPy/creditcard.csv')
    
    # split data training
data = data.sample(frac = 0.2, random_state = 1)

# plot histogram each parameter
data.hist(figsize = (20, 20))
plt.show()

# determine the number of fraud cases
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
outlier_fraction = len(fraud) / float(len(valid))

# Correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize=(12, 9))
sns.heatmap(data = corrmat, vmax = .8, square=True)
plt.show()

# X comes in all columns except Class column in data
X = data.loc[:, data.columns != 'Class']
# Y comes in all class label for each sample
Y = data['Class']

# Applying Algorithms
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# define the outlier detection method
classifiers = {
    # contamination is the number of outliers
    "Isolation Forest" : IsolationForest(max_samples = len(X),
                                         contamination = outlier_fraction,
                                         random_state = 1),
    "Local Outlier Factor" : LocalOutlierFactor(n_neighbors = 20,
                                                contamination = outlier_fraction)
    }

# Fit model
# The number of outliers
n_outliers = len(fraud)
for i, (clf_name, clf) in enumerate(classifiers.items()):
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
        
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    # Calculate number of errors
    n_errors = (y_pred != Y).sum()
    # Classification matrix
    print('{}: {}'.format(clf_name, n_errors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))
















 