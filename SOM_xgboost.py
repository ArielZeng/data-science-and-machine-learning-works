import tensorflow as tf
from sklearn.model_selection import train_test_split
from minisom import MiniSom
from xgboost import XGBClassifier
import numpy as np
import time
from sklearn.metrics import classification_report

# Load the Fashion-MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train_full, y_train_full), (_, _) = fashion_mnist.load_data()

# Reshape and scale pixel values to 0-1
X_train_full = X_train_full.reshape((-1, 28*28)) / 255.0

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=0)

# Initialize and train the Self-Organizing Map (SOM)
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

# Initialize XGBoost model
xgb_model = XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05, use_label_encoder=False, eval_metric='mlogloss')

# Measure the training time
start_train = time.time()
xgb_model.fit(X_train_transformed, Y_train)
training_time = time.time() - start_train

# Measure the prediction time
start_predict = time.time()
y_pred = xgb_model.predict(X_test_transformed)
prediction_time = time.time() - start_predict

# Classification report
print(f"Training time: {training_time} seconds")
print(f"Prediction time: {prediction_time} seconds")
print("Classification report using XGBoost model:\n")
print(classification_report(Y_test, y_pred))

# Training time: 3.506807804107666 seconds
# Prediction time: 0.01715373992919922 seconds
# Classification report using XGBoost model:
#
#               precision    recall  f1-score   support
#
#            0       0.60      0.85      0.71      1184
#            1       0.97      0.84      0.90      1187
#            2       0.49      0.48      0.48      1206
#            3       0.72      0.76      0.74      1225
#            4       0.47      0.59      0.52      1217
#            5       0.73      0.81      0.77      1220
#            6       0.40      0.16      0.23      1186
#            7       0.84      0.73      0.78      1156
#            8       0.94      0.87      0.90      1208
#            9       0.82      0.92      0.87      1211
#
#     accuracy                           0.70     12000
#    macro avg       0.70      0.70      0.69     12000
# weighted avg       0.70      0.70      0.69     12000