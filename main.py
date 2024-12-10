import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Oculta mensajes de INFO y WARNING


# Load the data
celsius = np.array([-40, -10, 0, 8, 15, 22, 38, 33, 57, 44, 12, 90, 76, 788, 65, 89], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100, 91.4, 134.6, 111.2, 53.6, 194, 168.8, 1450.4, 149, 192.2], dtype=float)

# Create the model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),  # Define explicitly the input shape
    tf.keras.layers.Dense(units=1)
])

# Compile the model
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

# Train the model
history = model.fit(celsius, fahrenheit, epochs=1000, verbose=False)
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