import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import time

# Load the Fashion-MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
X_train_full = X_train_full.reshape((-1, 28*28)) / 255.0

X_train, X_val, Y_train, Y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=0)

# VAE parameters
original_dim = 28 * 28
intermediate_dim = 128
latent_dim = 5

# Build VAE model
inputs = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(inputs)
h = Dropout(0.2)(h)
z_mean = Dense(latent_dim)(h)
z_log_sigma = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.0)
    return z_mean + K.exp(z_log_sigma) * epsilon

z = Lambda(sampling)([z_mean, z_log_sigma])

# Decoder
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# VAE model
vae = Model(inputs, x_decoded_mean)

# VAE loss function
xent_loss = original_dim * binary_crossentropy(inputs, x_decoded_mean)
kl_loss = - 0.5 * K.sum(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')

# Train VAE and measure time
start_vae_train = time.time()
vae.fit(X_train, X_train, epochs=100, batch_size=256, validation_data=(X_val, X_val))
vae_training_time = time.time() - start_vae_train
print(f"VAE Training time: {vae_training_time} seconds")

# Build encoder model
encoder = Model(inputs, z_mean)

# Encode the data
X_train_encoded = encoder.predict(X_train)
X_val_encoded = encoder.predict(X_val)

# Initialize XGBoost model and train
xgb_model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss')
start_xgb_train = time.time()
xgb_model.fit(X_train_encoded, Y_train)
xgb_training_time = time.time() - start_xgb_train
print(f"XGBoost Training time: {xgb_training_time} seconds")

# Predict with XGBoost and measure time
start_xgb_predict = time.time()
y_pred = xgb_model.predict(X_val_encoded)
xgb_prediction_time = time.time() - start_xgb_predict
print(f"XGBoost Prediction time: {xgb_prediction_time} seconds")

# Evaluate accuracy
accuracy = accuracy_score(Y_val, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# XGBoost Training time: 2.3517942428588867 seconds
# XGBoost Prediction time: 0.008537769317626953 seconds
# Accuracy: 79.02%