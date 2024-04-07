import tensorflow as tf
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report
import time

# Load Fashion-MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# Reshape the data and scale pixel values to 0-1
X_train_full = X_train_full.reshape((-1, 28*28)) / 255.0

# Split the dataset into training and testing sets
X_train, X_val, Y_train, Y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=0)

# Initialize CatBoostClassifier
cat_model = CatBoostClassifier(
    iterations=50,
    learning_rate=0.19,
    loss_function='MultiClass',
    depth=5,
    verbose=0
)

# Measure training time
start_train = time.time()
cat_model.fit(X_train, Y_train)
training_time = time.time() - start_train
print(f"Training time: {training_time} seconds")

# Measure prediction time
start_predict = time.time()
Y_pred = cat_model.predict(X_val)
prediction_time = time.time() - start_predict
print(f"Prediction time: {prediction_time} seconds")

# Print the classification report
print("Classification report using CatBoost features:\n")
print(classification_report(Y_val, Y_pred))
# Training time: 14.994281768798828 seconds
# Prediction time: 2.8789756298065186 seconds
# Classification report using CatBoost features:
#
#               precision    recall  f1-score   support
#
#            0       0.79      0.80      0.80      1184
#            1       0.98      0.94      0.96      1187
#            2       0.74      0.77      0.75      1206
#            3       0.80      0.88      0.84      1225
#            4       0.72      0.76      0.74      1217
#            5       0.94      0.88      0.91      1220
#            6       0.67      0.56      0.61      1186
#            7       0.85      0.89      0.87      1156
#            8       0.95      0.95      0.95      1208
#            9       0.90      0.93      0.92      1211
#
#     accuracy                           0.84     12000
#    macro avg       0.84      0.84      0.84     12000
# weighted avg       0.84      0.84      0.84     12000


