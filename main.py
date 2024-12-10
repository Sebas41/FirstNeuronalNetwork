import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Oculta mensajes de INFO y WARNING

# Load the data from CSV
data = pd.read_csv('datasets/celsius_fahrenheit_data.csv')
celsius = data['Celsius'].values
fahrenheit = data['Fahrenheit'].values

# Create the model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),  # Define explicitly the input shape
    tf.keras.layers.Dense(units=1)
])

# Compile the model
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

# Train the model
history = model.fit(celsius, fahrenheit, epochs=500, verbose=False)
print("Finished training the model")

# Display training statistics
plt.xlabel('Epoch Number')
plt.ylabel('Loss Magnitude')
plt.plot(history.history['loss'])
plt.show()

# Use the model to predict values
prediction = model.predict(np.array([100.0]))  # Use np.array for input
print(f"Predicted value for 100°C: {prediction[0][0]} °F")
print("These are the layer variables: {}".format(model.get_weights()))