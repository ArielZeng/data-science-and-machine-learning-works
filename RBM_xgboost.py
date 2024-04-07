import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import time
from sklearn.metrics import classification_report

# Load the Fashion-MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train_full, y_train_full), (_, _) = fashion_mnist.load_data()

# Reshape and scale pixel values to 0-1
X_train_full = X_train_full.reshape((-1, 28*28)) / 255.0

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=0)

# Initialize RBM
rbm = BernoulliRBM(
    learning_rate=0.01,
    n_iter=20,
    n_components=200,
    random_state=0,
    verbose=True
)

# Initialize XGBClassifier
gbm = XGBClassifier(
    max_depth=3,
    n_estimators=300,
    learning_rate=0.05,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

# Create a pipeline with RBM and XGBClassifier
rbm_features_classifier = Pipeline([
    ("rbm", rbm),
    ("xgboost", gbm)
])

# Measure training time
start_train = time.time()
rbm_features_classifier.fit(X_train, Y_train)
training_time = time.time() - start_train
print(f"Training time: {training_time} seconds")

# Measure prediction time
start_predict = time.time()
Y_pred = rbm_features_classifier.predict(X_test)
prediction_time = time.time() - start_predict
print(f"Prediction time: {prediction_time} seconds")

# Print classification report
print("Classification report using RBM and XGBoost features:\n")
print(classification_report(Y_test, Y_pred))

# [BernoulliRBM] Iteration 1, pseudo-likelihood = -241.29, time = 21.63s
# [BernoulliRBM] Iteration 2, pseudo-likelihood = -227.30, time = 22.15s
# [BernoulliRBM] Iteration 3, pseudo-likelihood = -215.25, time = 21.27s
# [BernoulliRBM] Iteration 4, pseudo-likelihood = -213.67, time = 20.49s
# [BernoulliRBM] Iteration 5, pseudo-likelihood = -210.16, time = 20.53s
# [BernoulliRBM] Iteration 6, pseudo-likelihood = -208.08, time = 23.64s
# [BernoulliRBM] Iteration 7, pseudo-likelihood = -204.30, time = 21.51s
# [BernoulliRBM] Iteration 8, pseudo-likelihood = -202.40, time = 23.39s
# [BernoulliRBM] Iteration 9, pseudo-likelihood = -205.40, time = 26.48s
# [BernoulliRBM] Iteration 10, pseudo-likelihood = -203.16, time = 31.22s
# [BernoulliRBM] Iteration 11, pseudo-likelihood = -200.59, time = 31.28s
# [BernoulliRBM] Iteration 12, pseudo-likelihood = -202.04, time = 46.11s
# [BernoulliRBM] Iteration 13, pseudo-likelihood = -201.31, time = 49.20s
# [BernoulliRBM] Iteration 14, pseudo-likelihood = -200.40, time = 44.64s
# [BernoulliRBM] Iteration 15, pseudo-likelihood = -198.85, time = 47.16s
# [BernoulliRBM] Iteration 16, pseudo-likelihood = -204.52, time = 45.40s
# [BernoulliRBM] Iteration 17, pseudo-likelihood = -200.49, time = 47.89s
# [BernoulliRBM] Iteration 18, pseudo-likelihood = -201.24, time = 45.64s
# [BernoulliRBM] Iteration 19, pseudo-likelihood = -198.10, time = 42.96s
# [BernoulliRBM] Iteration 20, pseudo-likelihood = -199.58, time = 42.17s
# Training time: 716.0212318897247 seconds
# Prediction time: 0.33957767486572266 seconds
# Classification report using RBM and XGBoost features:
#
#               precision    recall  f1-score   support
#
#            0       0.79      0.84      0.82      1184
#            1       0.99      0.96      0.97      1187
#            2       0.76      0.79      0.78      1206
#            3       0.87      0.88      0.88      1225
#            4       0.76      0.77      0.77      1217
#            5       0.94      0.93      0.93      1220
#            6       0.66      0.59      0.62      1186
#            7       0.92      0.92      0.92      1156
#            8       0.96      0.97      0.97      1208
#            9       0.93      0.94      0.94      1211
#
#     accuracy                           0.86     12000
#    macro avg       0.86      0.86      0.86     12000
# weighted avg       0.86      0.86      0.86     12000
