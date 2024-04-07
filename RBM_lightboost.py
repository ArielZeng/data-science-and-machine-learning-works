import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
import lightgbm as lgb
import numpy as np
import time
from sklearn.metrics import classification_report

# Load Fashion-MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train_full, y_train_full), (_, _) = fashion_mnist.load_data()

# Data preprocessing: reshape and scale pixel values to 0-1
X_train_full = X_train_full.reshape((-1, 28*28)) / 255.0

# Split the dataset into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=0)

# Initialize RBM
rbm = BernoulliRBM(
    random_state=0,
    verbose=True,
    learning_rate=0.01,
    n_iter=20,
    n_components=200
)

# LightGBM Classifier Wrapper
class LGBMClassifierWrapper(BaseEstimator):
    def __init__(self, params=None, num_boost_round=100):
        self.params = params
        self.num_boost_round = num_boost_round
        self.model = None

    def fit(self, X, y):
        dtrain = lgb.Dataset(X, label=y)
        self.model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round
        )
        return self

    def predict(self, X):
        return np.argmax(self.model.predict(X, num_iteration=self.model.best_iteration), axis=1)

# Set LightGBM parameters
lgbm_params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 10,
    'metric': 'multi_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

# Initialize LightGBM Classifier
lgbm = LGBMClassifierWrapper(params=lgbm_params, num_boost_round=20)

# Create a pipeline with RBM and LightGBM
rbm_features_classifier = Pipeline([
    ("rbm", rbm),
    ("lightboost", lgbm)
])

# Measure training time
start_train = time.time()
rbm_features_classifier.fit(X_train, Y_train)
training_time = time.time() - start_train
print(f"Training time: {training_time} seconds")

# Measure prediction time
start_predict = time.time()
Y_pred = rbm_features_classifier.predict(X_val)
prediction_time = time.time() - start_predict
print(f"Prediction time: {prediction_time} seconds")

# Print classification report
print("Classification report using RBM and LightGBM features:\n")
print(classification_report(Y_val, Y_pred))

# Training time: 821.4624063968658 seconds
# Prediction time: 0.1526174545288086 seconds
# Classification report using RBM and LightGBM features:
#
#               precision    recall  f1-score   support
#
#            0       0.78      0.85      0.82      1184
#            1       0.99      0.95      0.97      1187
#            2       0.74      0.79      0.77      1206
#            3       0.86      0.88      0.87      1225
#            4       0.74      0.77      0.76      1217
#            5       0.93      0.92      0.93      1220
#            6       0.69      0.56      0.62      1186
#            7       0.91      0.91      0.91      1156
#            8       0.96      0.96      0.96      1208
#            9       0.93      0.94      0.94      1211
#
#     accuracy                           0.85     12000
#    macro avg       0.85      0.85      0.85     12000
# weighted avg       0.85      0.85      0.85     12000

