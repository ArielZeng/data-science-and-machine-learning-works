import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report
import time

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

# Initialize CatBoost Classifier
cat_model = CatBoostClassifier(
    iterations=50,
    learning_rate=0.19,
    loss_function='MultiClass',
    depth=5,
    verbose=0
)

# Create a pipeline with RBM and CatBoost Classifier
rbm_features_classifier = Pipeline([
    ("rbm", rbm),
    ("catboost", cat_model)
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
print("Classification report using RBM and CatBoost features:\n")
print(classification_report(Y_val, Y_pred))

# [BernoulliRBM] Iteration 1, pseudo-likelihood = -241.29, time = 46.05s
# [BernoulliRBM] Iteration 2, pseudo-likelihood = -227.30, time = 49.09s
# [BernoulliRBM] Iteration 3, pseudo-likelihood = -215.25, time = 32.60s
# [BernoulliRBM] Iteration 4, pseudo-likelihood = -213.67, time = 31.77s
# [BernoulliRBM] Iteration 5, pseudo-likelihood = -210.16, time = 67.42s
# [BernoulliRBM] Iteration 6, pseudo-likelihood = -208.08, time = 58.05s
# [BernoulliRBM] Iteration 7, pseudo-likelihood = -204.30, time = 31.51s
# [BernoulliRBM] Iteration 8, pseudo-likelihood = -202.40, time = 34.60s
# [BernoulliRBM] Iteration 9, pseudo-likelihood = -205.40, time = 33.81s
# [BernoulliRBM] Iteration 10, pseudo-likelihood = -203.16, time = 33.41s
# [BernoulliRBM] Iteration 11, pseudo-likelihood = -200.59, time = 31.16s
# [BernoulliRBM] Iteration 12, pseudo-likelihood = -202.04, time = 22.30s
# [BernoulliRBM] Iteration 13, pseudo-likelihood = -201.31, time = 21.90s
# [BernoulliRBM] Iteration 14, pseudo-likelihood = -200.40, time = 21.11s
# [BernoulliRBM] Iteration 15, pseudo-likelihood = -198.85, time = 20.42s
# [BernoulliRBM] Iteration 16, pseudo-likelihood = -204.52, time = 20.55s
# [BernoulliRBM] Iteration 17, pseudo-likelihood = -200.49, time = 23.47s
# [BernoulliRBM] Iteration 18, pseudo-likelihood = -201.24, time = 21.21s
# [BernoulliRBM] Iteration 19, pseudo-likelihood = -198.10, time = 23.27s
# [BernoulliRBM] Iteration 20, pseudo-likelihood = -199.58, time = 26.26s
# Training time: 656.993504524231 seconds
# Prediction time: 0.7512633800506592 seconds
# Classification report using RBM and CatBoost features:
#
#               precision    recall  f1-score   support
#
#            0       0.77      0.82      0.80      1184
#            1       0.98      0.93      0.96      1187
#            2       0.72      0.76      0.74      1206
#            3       0.82      0.87      0.84      1225
#            4       0.72      0.77      0.75      1217
#            5       0.91      0.89      0.90      1220
#            6       0.64      0.50      0.56      1186
#            7       0.88      0.88      0.88      1156
#            8       0.95      0.95      0.95      1208
#            9       0.90      0.93      0.91      1211
#
#     accuracy                           0.83     12000
#    macro avg       0.83      0.83      0.83     12000
# weighted avg       0.83      0.83      0.83     12000
#
