import tensorflow as tf
from sklearn.model_selection import train_test_split
from minisom import MiniSom
import lightgbm as lgb
import numpy as np
import time
from sklearn.metrics import classification_report

# Load Fashion-MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train_full, y_train_full), (_, _) = fashion_mnist.load_data()

# Reshape and scale pixel values to 0-1
X_train_full = X_train_full.reshape((-1, 28*28)) / 255.0

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=0)

# Initialize and train Self-Organizing Map (SOM)
som_size = 10  # Size of the SOM grid
som = MiniSom(som_size, som_size, 784, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X_train)
som.train_random(X_train, 1000)

# Function to transform data using the SOM
def transform_with_som(som, data):
    return np.array([som.winner(d) for d in data])

# Transform training and testing data
X_train_transformed = transform_with_som(som, X_train)
X_test_transformed = transform_with_som(som, X_test)

# LightGBM model parameters and training
lgb_train = lgb.Dataset(X_train_transformed, Y_train)
params = {
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

# Measure the training time
start_train = time.time()
gbm = lgb.train(params, lgb_train, num_boost_round=20)
training_time = time.time() - start_train

# Measure the prediction time
start_predict = time.time()
y_pred = gbm.predict(X_test_transformed, num_iteration=gbm.best_iteration)
prediction_time = time.time() - start_predict
y_pred_class = np.argmax(y_pred, axis=1)

# Classification report
print(f"Training time: {training_time} seconds")
print(f"Prediction time: {prediction_time} seconds")
print("Classification report using LightGBM model:\n")
print(classification_report(Y_test, y_pred_class))

# Training time: 0.5602054595947266 seconds
# Prediction time: 0.007051706314086914 seconds
# Classification report using LightGBM model:
#
#               precision    recall  f1-score   support
#
#            0       0.67      0.77      0.72      1184
#            1       0.96      0.88      0.92      1187
#            2       0.53      0.47      0.50      1206
#            3       0.70      0.80      0.75      1225
#            4       0.51      0.56      0.53      1217
#            5       0.80      0.78      0.79      1220
#            6       0.42      0.31      0.36      1186
#            7       0.80      0.77      0.79      1156
#            8       0.85      0.91      0.88      1208
#            9       0.85      0.90      0.87      1211
#
#     accuracy                           0.71     12000
#    macro avg       0.71      0.71      0.71     12000
# weighted avg       0.71      0.71      0.71     12000
