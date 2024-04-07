import tensorflow as tf
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import time

# Load Fashion-MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train_full, y_train_full), (_, _) = fashion_mnist.load_data()

# Reshape the data and scale pixel values to 0-1
X_train_full = X_train_full.reshape((-1, 28*28)) / 255.0

# Split the dataset into training and testing sets
X_train, X_val, Y_train, Y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=0)

# Initialize XGBClassifier
xgb_model = XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05, use_label_encoder=False, eval_metric='mlogloss')

# Measure training time
start_train = time.time()
xgb_model.fit(X_train, Y_train)
training_time = time.time() - start_train
print(f"Training time: {training_time} seconds")

# Measure prediction time
start_predict = time.time()
Y_pred = xgb_model.predict(X_val)
prediction_time = time.time() - start_predict
print(f"Prediction time: {prediction_time} seconds")

# Print the classification report
print("Classification report using XGBoost features:\n")
print(classification_report(Y_val, Y_pred))

# Training time: 120.01682901382446 seconds
# Prediction time: 0.037512779235839844 seconds
# Classification report using XGBoost features:
#
#               precision    recall  f1-score   support
#
#            0       0.82      0.86      0.84      1184
#            1       0.99      0.97      0.98      1187
#            2       0.77      0.80      0.78      1206
#            3       0.88      0.90      0.89      1225
#            4       0.76      0.79      0.77      1217
#            5       0.98      0.96      0.97      1220
#            6       0.70      0.62      0.66      1186
#            7       0.92      0.95      0.94      1156
#            8       0.97      0.96      0.97      1208
#            9       0.96      0.95      0.95      1211
#
#     accuracy                           0.88     12000
#    macro avg       0.88      0.88      0.87     12000
# weighted avg       0.88      0.88      0.87     12000
