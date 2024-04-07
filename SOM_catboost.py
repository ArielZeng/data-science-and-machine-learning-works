import tensorflow as tf
from sklearn.model_selection import train_test_split
from minisom import MiniSom
from catboost import CatBoostClassifier
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

# Parameters for the Self-Organizing Map (SOM)
som_size = 10  # Size of the SOM grid
som = MiniSom(som_size, som_size, 784, sigma=1.0, learning_rate=0.5)

# Initialize and train the SOM
som.random_weights_init(X_train)
som.train_random(X_train, 1000)

# Define a function to transform data using the SOM
def transform_with_som(som, data):
    return np.array([som.winner(d) for d in data])

# Transform both training and testing data
X_train_transformed = transform_with_som(som, X_train)
X_test_transformed = transform_with_som(som, X_test)

# Initialize CatBoost Classifier
cat_model = CatBoostClassifier(
    iterations=50,
    learning_rate=0.19,
    loss_function='MultiClass',
    depth=5,
    verbose=0
)

# Measure training time
start_train = time.time()
cat_model.fit(X_train_transformed, Y_train)
training_time = time.time() - start_train
print(f"Training time: {training_time} seconds")

# Measure prediction time
start_predict = time.time()
y_pred = cat_model.predict(X_test_transformed)
prediction_time = time.time() - start_predict
print(f"Prediction time: {prediction_time} seconds")

# Print classification report
print("Classification report:\n")
print(classification_report(Y_test, y_pred))

# Training time: 0.5524411201477051 seconds
# Prediction time: 0.00805354118347168 seconds
# Classification report:
#
#               precision    recall  f1-score   support
#
#            0       0.67      0.76      0.72      1184
#            1       0.97      0.81      0.89      1187
#            2       0.53      0.45      0.49      1206
#            3       0.67      0.85      0.75      1225
#            4       0.52      0.47      0.49      1217
#            5       0.80      0.72      0.76      1220
#            6       0.40      0.40      0.40      1186
#            7       0.75      0.82      0.79      1156
#            8       0.91      0.90      0.90      1208
#            9       0.84      0.86      0.85      1211
#
#     accuracy                           0.70     12000
#    macro avg       0.71      0.70      0.70     12000
# weighted avg       0.71      0.70      0.70     12000