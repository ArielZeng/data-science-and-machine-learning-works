import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# R interface
from rpy2 import robjects
from rpy2.robjects import FloatVector, numpy2ri
from rpy2.robjects.packages import importr
import warnings
import os

# Enable conversion between numpy arrays and R vectors
numpy2ri.activate()

# Suppressing R warnings
def custom_rpy2_warn_handler(message, category, filename, lineno, file=None, line=None):
    pass  # Suppress all R warnings

# Setting the warning handling function
warnings.showwarning = custom_rpy2_warn_handler

# Setting the environment variable LANG to ensure R output encoding is UTF-8
os.environ["LANG"] = "en_US.UTF-8"

def fit_calling_R(y, method_num = 1):
    y_R = FloatVector(y)
    # Adding our vector to R's global environment
    robjects.globalenv["x"] = y_R
    # R code to call the stable.fit function
    r_code = '''
    library(stable)
    ret <- stable.fit(x, method = my_method, param = 0)
    '''
    r_code = r_code.replace("my_method", f"{method_num}")
    # Executing the R code within Python
    robjects.r(r_code)
    # Retrieving the results
    alpha = robjects.r('ret["alpha"]')
    beta = robjects.r('ret["beta"]')
    gamma = robjects.r('ret["gamma"]')
    delta = robjects.r('ret["delta"]')
    print(f"Fit using Stable in R: - alpha: {alpha[0]:.2f}, beta: {beta[0]:.2f}, gamma: {gamma[0]:.2f}, delta: {delta[0]:.2f}")

    return float(alpha[0]), float(beta[0]), float(gamma[0]), float(delta[0])

def predict_calling_R(y, alpha, beta, gamma, delta):
    robjects.globalenv["x"] = y
    robjects.globalenv["alpha"] = alpha
    robjects.globalenv["beta"] = beta
    robjects.globalenv["gamma"] = gamma
    robjects.globalenv["delta"] = delta
    # R code to call the stable.fit function
    r_code = '''
    library(stable)
    prob <- dstable(x, alpha=alpha, beta = beta, gamma = gamma, delta = delta)
'''
    # Executing the R code within Python
    robjects.r(r_code)
    prob = robjects.r('prob')
    prob = np.array(prob)
    return prob

class NormalNaiveBayes:
    def __init__(self):
        self.params = {}
        self.class_prior = {}

    def fit(self, X, Y):
        self.classes = np.unique(Y)
        for cls in self.classes:
            X_cls = X[Y == cls]
            self.params[cls] = [norm.fit(X_cls[:, i]) for i in range(X.shape[1])]
            self.class_prior[cls] = float(len(X_cls)) / len(X)

    def predict(self, X):
        preds = []
        for x in X:
            probs = []
            for cls in self.classes:
                prob = np.log(self.class_prior[cls])  # Use log probabilities
                for i, val in enumerate(x):
                    loc, scale = self.params[cls][i]
                    prob += np.log(norm.pdf(val, loc, scale))  # Sum log probabilities
                probs.append(prob)
            preds.append(self.classes[np.argmax(probs)])

        return np.array(preds)


class StableNaiveBayes:
    def __init__(self):
        self.params = {}
        self.class_prior = {}

    def fit(self, X, Y):
        self.classes = np.unique(Y)
        for cls in self.classes:
            X_cls = X[Y == cls]
            param_list = [fit_calling_R(X_cls[:, i]) for i in range(X.shape[1])]
            self.params[cls] = param_list
            self.class_prior[cls] = float(len(X_cls)) / len(X)

    def predict(self, X):
        preds = []
        for x in X:
            probs = []
            for cls in self.classes:
                prob = np.log(self.class_prior[cls])  # Use log probabilities
                for i, val in enumerate(x):
                    alpha, beta, gamma, delta = self.params[cls][i]
                    stable_prob = predict_calling_R(val, alpha, beta, gamma, delta)
                    prob += np.log(stable_prob)  # Sum log probabilities
                probs.append(prob)

            if probs[0][0] > probs[1][0]:
                preds.append(0)
            else:
                preds.append(1)
        return np.array(preds)


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
data = pd.read_csv(url, header=None, names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class'])
class_labels = ['Iris-versicolor', 'Iris-virginica']
data = data[data['Class'].isin(class_labels)]

X = data[['sepal-width']].values
Y = data['Class'].values

le = LabelEncoder()
Y = le.fit_transform(Y)

# Set up k-fold cross-validation
k = 10
random_seed = 0
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_seed)

stable_model_accuracies = []
gaussian_nb_accuracies = []

for train_index, test_index in kf.split(X, Y):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    stable_classifier = StableNaiveBayes()
    stable_classifier.fit(X_train, Y_train)
    predictions = stable_classifier.predict(X_test)
    stable_model_accuracy = accuracy_score(Y_test, predictions)
    stable_model_accuracies.append(stable_model_accuracy)


    gaussian_nb_classifier = NormalNaiveBayes()
    gaussian_nb_classifier.fit(X_train, Y_train)
    predictions = gaussian_nb_classifier.predict(X_test)
    gaussian_nb_accuracy = accuracy_score(Y_test, predictions)
    gaussian_nb_accuracies.append(gaussian_nb_accuracy)


# Calculate the average accuracy across all folds
average_stable_accuracy = np.mean(stable_model_accuracies)
average_gaussian_nb_accuracy = np.mean(gaussian_nb_accuracies)

print(f"Average accuracy for StableNaiveBayes: {average_stable_accuracy}")
print(f"Average accuracy for GaussianNaiveBayes: {average_gaussian_nb_accuracy}")





