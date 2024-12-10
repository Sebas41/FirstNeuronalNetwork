import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from datasets.data_loader import load_data
from datasets.preprocess import preprocess_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Oculta mensajes de INFO y WARNING

# Load the data from CSV
data = load_data('datasets/celsius_fahrenheit_data.csv')

# Preprocess the data
X_train, X_test, y_train, y_test = preprocess_data(data)

# Create the model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),  # Define explicitly the input shape
    tf.keras.layers.Dense(units=1)
])

# Compile the model
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

# Train the model
history = model.fit(X_train, y_train, epochs=500, verbose=False, validation_data=(X_test, y_test))
print("Finished training the model")

# Display training statistics
plt.xlabel('Epoch Number')
plt.ylabel('Loss Magnitude')
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

# Use the model to predict values
prediction = model.predict(np.array([100.0]))  # Use np.array for input
print(f"Predicted value for 100°C: {prediction[0][0]} °F")
print("These are the layer variables: {}".format(model.get_weights()))